"""
Run the full experiment sweep and aggregate results.

Sweep = 5 strategies x 3 defenses x TRIALS_PER_CELL attack trials, plus the
benign suite (4 profiles x 3 patients x 3 defenses). RESUMABLE: any trial whose
JSONL already exists is skipped, so a run interrupted by the free-tier daily cap
can be resumed the next day without redoing work.

Usage:
  python -m experiment.run_experiment --provider mock            # offline demo
  python -m experiment.run_experiment --provider openrouter      # real run
  python -m experiment.run_experiment --provider mock --trials 5 # quick dev run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import config
from core import patient_db
from experiment.strategies import STRATEGIES, benign_profiles
from experiment.orchestrator import (
    TrialConfig, run_attack_trial, run_benign_trial, trial_filename, write_trial,
)
from agents.llm_client import RateLimitedError

DEFENSES = ("none", "A", "B")


def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def run_sweep(provider: str, trials: int, patients_subset: int | None) -> None:
    patients = patient_db.all_patients()
    if patients_subset:
        patients = patients[:patients_subset]

    os.makedirs(config.TRIALS_DIR, exist_ok=True)
    n_done = n_run = n_skipped = 0

    # ---- attack trials --------------------------------------------------- #
    for defense in DEFENSES:
        for strategy in STRATEGIES:
            for patient in patients:
                for idx in range(trials):
                    tc = TrialConfig(
                        kind="attack", strategy=strategy, defense=defense,
                        patient=patient, trial_idx=idx, provider=provider,
                        seed=1000 * idx + hash(strategy) % 997,
                    )
                    path = os.path.join(config.TRIALS_DIR, trial_filename(tc))
                    if _exists(path):
                        n_done += 1
                        continue
                    try:
                        result = run_attack_trial(tc)
                    except RateLimitedError as e:
                        n_skipped += 1
                        print(f"  [skip] rate-limited, will resume later: "
                              f"{trial_filename(tc)}  ({e})")
                        continue
                    except Exception as e:
                        n_skipped += 1
                        print(f"  [skip] error, will resume later: "
                              f"{trial_filename(tc)}  ({type(e).__name__}: {e})")
                        continue
                    write_trial(result, path)
                    n_run += 1
                    _tick(n_run, tc, result["score"])

    # ---- benign suite ---------------------------------------------------- #
    for defense in DEFENSES:
        for patient in patients:
            for profile in benign_profiles(patient):
                tc = TrialConfig(
                    kind="benign", strategy=profile["id"], defense=defense,
                    patient=patient, trial_idx=0, provider=provider,
                    should_succeed=profile["should_succeed"],
                    benign_messages=profile["messages"],
                )
                path = os.path.join(config.TRIALS_DIR, trial_filename(tc))
                if _exists(path):
                    n_done += 1
                    continue
                try:
                    result = run_benign_trial(tc)
                except RateLimitedError as e:
                    n_skipped += 1
                    print(f"  [skip] rate-limited, will resume later: "
                          f"{trial_filename(tc)}  ({e})")
                    continue
                except Exception as e:
                    n_skipped += 1
                    print(f"  [skip] error, will resume later: "
                          f"{trial_filename(tc)}  ({type(e).__name__}: {e})")
                    continue
                write_trial(result, path)
                n_run += 1

    print(f"\nSweep complete. New: {n_run}, already done: {n_done}, "
          f"rate-limited/skipped: {n_skipped}")
    if n_skipped:
        print(f"  -> {n_skipped} trial(s) were throttled. Re-run the same command "
              f"later to fill them in (finished trials are skipped).")
    aggregate()


def _tick(n, tc, score):
    flag = "HIT " if score.get("attack_success") else "miss"
    print(f"[{n:4d}] {flag} def={tc.defense:4s} {tc.strategy:22s} "
          f"p={tc.patient['name']:14s} t{tc.trial_idx:02d}")


def aggregate() -> None:
    """Read all trial JSONLs and write results/summary.csv."""
    # attack: ASR per (strategy, defense); avg turns-to-success
    hits = defaultdict(int)
    total = defaultdict(int)
    turns_sum = defaultdict(int)
    turns_n = defaultdict(int)
    leak_hits = defaultdict(int)
    send_hits = defaultdict(int)
    # benign: over-refusal per defense
    benign_total = defaultdict(int)
    benign_overrefuse = defaultdict(int)
    benign_success = defaultdict(int)

    for fn in os.listdir(config.TRIALS_DIR):
        if not fn.endswith(".jsonl"):
            continue
        meta = score = None
        with open(os.path.join(config.TRIALS_DIR, fn), encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                if obj.get("record") == "meta":
                    meta = obj
                elif obj.get("record") == "score":
                    score = obj
        if not meta or not score:
            continue

        if meta["kind"] == "attack":
            key = (meta["strategy"], meta["defense"])
            total[key] += 1
            if score.get("attack_success"):
                hits[key] += 1
                if score.get("success_turn"):
                    turns_sum[key] += score["success_turn"]
                    turns_n[key] += 1
            if score.get("phi_leaked"):
                leak_hits[key] += 1
            if score.get("unauthorized_send"):
                send_hits[key] += 1
        else:  # benign
            d = meta["defense"]
            if score.get("should_succeed"):
                benign_total[d] += 1
                if score.get("over_refusal"):
                    benign_overrefuse[d] += 1
                if score.get("benign_success"):
                    benign_success[d] += 1

    # ---- write attack summary ------------------------------------------- #
    rows = []
    for (strategy, defense) in sorted(total):
        t = total[(strategy, defense)]
        h = hits[(strategy, defense)]
        asr = h / t if t else 0.0
        avg_turns = (turns_sum[(strategy, defense)] / turns_n[(strategy, defense)]
                     if turns_n[(strategy, defense)] else None)
        rows.append({
            "strategy": strategy, "defense": defense, "trials": t,
            "attack_success_rate": round(asr, 3),
            "phi_leak_rate": round(leak_hits[(strategy, defense)] / t, 3) if t else 0,
            "unauth_send_rate": round(send_hits[(strategy, defense)] / t, 3) if t else 0,
            "avg_turns_to_success": round(avg_turns, 2) if avg_turns else "",
        })

    with open(config.SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["strategy", "defense", "trials", "attack_success_rate",
                            "phi_leak_rate", "unauth_send_rate", "avg_turns_to_success"])
        w.writeheader()
        w.writerows(rows)

    # ---- print human-readable tables ------------------------------------ #
    _print_asr_table(rows)
    _print_benign_table(benign_total, benign_overrefuse, benign_success)
    print(f"\nWrote {config.SUMMARY_CSV}")


def _print_asr_table(rows):
    print("\n=== Attack Success Rate (ASR) by strategy x defense ===")
    strategies = sorted({r["strategy"] for r in rows})
    print(f"{'strategy':24s} {'none':>8s} {'A':>8s} {'B':>8s}")
    for s in strategies:
        cells = {}
        for r in rows:
            if r["strategy"] == s:
                cells[r["defense"]] = r["attack_success_rate"]
        print(f"{s:24s} {cells.get('none',0):>8.2f} {cells.get('A',0):>8.2f} "
              f"{cells.get('B',0):>8.2f}")


def _print_benign_table(total, overrefuse, success):
    print("\n=== Benign behaviour (should-succeed callers) by defense ===")
    print(f"{'defense':8s} {'n':>4s} {'served':>8s} {'over-refused':>14s}")
    for d in ("none", "A", "B"):
        n = total.get(d, 0)
        s = success.get(d, 0)
        o = overrefuse.get(d, 0)
        rate = f"{(o/n):.2f}" if n else "-"
        print(f"{d:8s} {n:>4d} {s:>8d} {o:>3d} ({rate:>5s})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["mock", "openrouter"], default="mock")
    ap.add_argument("--trials", type=int, default=config.TRIALS_PER_CELL)
    ap.add_argument("--patients", type=int, default=None,
                    help="limit to first N patients (default: all)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="re-build summary.csv from existing trials without running")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate()
        return
    run_sweep(args.provider, args.trials, args.patients)


if __name__ == "__main__":
    main()
