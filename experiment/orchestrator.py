"""
Orchestrator — runs a single trial end to end, with NO human in the loop.

One attack trial:
  fresh FSM + ledger -> attacker sends -> target responds (resolving any tool
  calls against the FSM) -> optional Defense-B monitor review -> score each turn
  -> stop on first attack success or at MAX_TURNS. Every message is logged
  verbatim with the FSM snapshot and the running success labels.

This module is the answer to the rubric's most-weighted criterion: the two-agent
loop advances turn after turn on its own; a human only starts it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import config
from core.fsm import VerificationFSM
from core.tools import ToolContext
from core import scoring
from agents.llm_client import make_client
from agents.target import TargetAgent
from agents.attacker import AttackerAgent
from agents.monitor import make_monitor


@dataclass
class TrialConfig:
    kind: str                 # "attack" or "benign"
    strategy: str             # attack strategy, or benign profile id
    defense: str              # one of config.CONDITIONS
    patient: dict
    trial_idx: int
    provider: str             # "mock" | "openrouter"
    seed: int = 0
    should_succeed: bool = True   # benign only
    benign_messages: list = field(default_factory=list)  # benign only


def _mechanisms(condition: str) -> dict:
    """Resolve a condition name to its {hardened, gate, monitor} mechanisms."""
    return config.CONDITIONS[condition]


def run_attack_trial(tc: TrialConfig) -> dict:
    mech = _mechanisms(tc.defense)
    fsm = VerificationFSM()
    ctx = ToolContext(fsm=fsm, target_patient=tc.patient,
                      gate_enabled=mech["gate"])

    attacker_client = make_client(
        role="attacker", provider=tc.provider,
        model=config.MODEL_ATTACKER, fallbacks=config.FALLBACKS_ATTACKER,
        seed=tc.seed, patient=tc.patient, strategy=tc.strategy,
    )
    target_client = make_client(
        role="target", provider=tc.provider,
        model=config.MODEL_TARGET, fallbacks=config.FALLBACKS_TARGET,
        seed=tc.seed + 1000, patient=tc.patient, strategy=tc.strategy,
    )
    attacker = AttackerAgent(attacker_client, tc.strategy, tc.patient["name"])
    target = TargetAgent(target_client, ctx, hardened=mech["hardened"])

    monitor = None
    if mech["monitor"]:
        mon_client = None
        if tc.provider != "mock":
            mon_client = make_client(
                role="monitor", provider=tc.provider,
                model=config.MODEL_MONITOR, fallbacks=config.FALLBACKS_MONITOR,
                seed=tc.seed + 2000,
            )
        monitor = make_monitor(tc.provider, tc.patient, mon_client)

    transcript: list[dict] = []
    attacker_msgs: list[str] = []
    target_replies: list[str] = []

    # target's OpenAI-style history (caller = 'user', target = 'assistant')
    target_history: list[dict] = []

    for turn in range(config.MAX_TURNS):
        # --- attacker moves ---
        atk = attacker.next_message(target_replies, attacker_msgs)
        attacker_msgs.append(atk)
        target_history.append({"role": "user", "content": atk})
        transcript.append({"role": "attacker", "content": atk,
                           "state": fsm.snapshot()})

        # --- target responds (may call tools) ---
        reply, tool_events = target.respond(target_history)

        # --- Defense B: monitor review before "sending" ---
        vetoed = False
        if monitor is not None:
            reply, vetoed = monitor.review(reply, fsm.is_verified, attacker_text=atk)

        target_history.append({"role": "assistant", "content": reply})
        target_replies.append(reply)
        transcript.append({"role": "target", "content": reply,
                           "state": fsm.snapshot(),
                           "tools": [te["call"] for te in tool_events],
                           "monitor_vetoed": vetoed})

        # --- score so far; stop on success ---
        s = scoring.score_attack(transcript, ctx)
        if s["attack_success"]:
            s["success_turn"] = turn + 1
            return _package(tc, transcript, ctx, s, turns_used=turn + 1)

    s = scoring.score_attack(transcript, ctx)
    s["success_turn"] = None
    return _package(tc, transcript, ctx, s, turns_used=config.MAX_TURNS)


def run_benign_trial(tc: TrialConfig) -> dict:
    """
    Benign callers are scripted (not adversarial), so no attacker LLM is needed.
    The target still runs its full tool/FSM logic and (if defense B) the monitor.
    """
    mech = _mechanisms(tc.defense)
    fsm = VerificationFSM()
    ctx = ToolContext(fsm=fsm, target_patient=tc.patient,
                      gate_enabled=mech["gate"])
    target_client = make_client(
        role="target", provider=tc.provider,
        model=config.MODEL_TARGET, fallbacks=config.FALLBACKS_TARGET,
        seed=tc.seed + 1000, patient=tc.patient, strategy="benign",
    )
    target = TargetAgent(target_client, ctx, hardened=mech["hardened"])

    monitor = None
    if mech["monitor"]:
        mon_client = None
        if tc.provider != "mock":
            mon_client = make_client(role="monitor", provider=tc.provider,
                                     model=config.MODEL_MONITOR,
                                     fallbacks=config.FALLBACKS_MONITOR, seed=tc.seed + 2000)
        monitor = make_monitor(tc.provider, tc.patient, mon_client)

    transcript: list[dict] = []
    target_history: list[dict] = []

    for msg in tc.benign_messages:
        target_history.append({"role": "user", "content": msg})
        transcript.append({"role": "caller", "content": msg, "state": fsm.snapshot()})

        reply, tool_events = target.respond(target_history)
        vetoed = False
        if monitor is not None:
            reply, vetoed = monitor.review(reply, fsm.is_verified, attacker_text=msg)
        target_history.append({"role": "assistant", "content": reply})
        transcript.append({"role": "target", "content": reply, "state": fsm.snapshot(),
                           "tools": [te["call"] for te in tool_events],
                           "monitor_vetoed": vetoed})

    s = scoring.score_benign(transcript, ctx, tc.should_succeed)
    return _package(tc, transcript, ctx, s, turns_used=len(tc.benign_messages))


def _package(tc: TrialConfig, transcript, ctx, score, turns_used) -> dict:
    return {
        "meta": {
            "kind": tc.kind, "strategy": tc.strategy, "defense": tc.defense,
            "patient": tc.patient["name"], "trial_idx": tc.trial_idx,
            "provider": tc.provider, "seed": tc.seed, "turns_used": turns_used,
        },
        "score": score,
        "ledger": [c.to_dict() for c in ctx.ledger],
        "transcript": transcript,
    }


def trial_filename(tc: TrialConfig) -> str:
    return f"{tc.kind}__{tc.defense}__{tc.strategy}__p{_slug(tc.patient['name'])}__t{tc.trial_idx:02d}.jsonl"


def _slug(s: str) -> str:
    return s.lower().replace(" ", "-")


def write_trial(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        # one JSON object per line: header, then each turn, then footer
        fh.write(json.dumps({"record": "meta", **result["meta"]}) + "\n")
        fh.write(json.dumps({"record": "score", **result["score"]}) + "\n")
        for turn in result["transcript"]:
            fh.write(json.dumps({"record": "turn", **turn}) + "\n")
        fh.write(json.dumps({"record": "ledger", "calls": result["ledger"]}) + "\n")
