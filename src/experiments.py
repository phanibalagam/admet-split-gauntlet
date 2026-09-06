"""
Experiment harness: bootstrap confidence intervals on every reported metric, and
repeated runs across model seeds.

A single-split, single-seed ADMET number is not a result. Two sources of
variance are quantified here:

  TEST-SET VARIANCE   Bootstrap resampling of the held-out set gives a 95%
                      interval on each metric. Two arms whose intervals overlap
                      are not distinguishable on this data, however different
                      their point estimates look.

  MODEL-SEED VARIANCE Repeating the fit across seeds shows how much of a gap is
                      the model's own initialisation.

The headline claim of this project - that the winning arm changes with the split
- only means something if the gaps involved are larger than these intervals.
This module is what tests that, and it is allowed to disconfirm the claim.

    python -m src.experiments --endpoints esol tox21:SR-MMP
    python -m src.experiments --seeds 5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_endpoint
from src.featurize import FEATURIZERS, parse
from src.gauntlet import _fit_predict, _score, _better
from src.splits import random_split, scaffold_split

RESULTS = Path("results")
BASELINE = "morgan"   # the arm every other arm has to beat to ship
RNG = np.random.default_rng(0)


def bootstrap_ci(y_true, y_pred, task: str, n_boot: int = 1000,
                 seed: int = 0) -> tuple[float, float, float]:
    """
    Percentile bootstrap over the test set. Returns (point, lo, hi) for the
    primary metric. Resamples with replacement; folds that lose a class are
    skipped rather than scored, which is the standard handling for AUC.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    point = _score(y_true, y_pred, task)[1]

    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if task != "regression" and len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(_score(y_true[idx], y_pred[idx], task)[1])
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run(endpoints: list[str], featurizers: list[str], splits: list[str],
        seeds: int = 5, n_boot: int = 1000, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for ep_name in endpoints:
        ep = load_endpoint(ep_name)
        mols, keep = parse(ep.smiles)
        y = ep.y[keep].to_numpy()
        feats = {}
        for f in featurizers:
            try:
                feats[f] = FEATURIZERS[f](mols)
            except RuntimeError as e:
                print(f"  SKIP {f}: {str(e).splitlines()[0]}")

        for split in splits:
            for fname, X in feats.items():
                per_seed, per_seed_lo, per_seed_hi = [], [], []
                boot_lo = boot_hi = point = float("nan")
                for s in range(seeds):
                    if split == "random":
                        # the partition is redrawn with the model seed, so the
                        # spread across seeds carries both sources at once and
                        # cannot be decomposed into them. See METHODS.md.
                        tr, te = random_split(len(mols), seed=s)
                    elif split == "scaffold":
                        # a function of the molecules; there is nothing to seed
                        tr, te = scaffold_split(mols)
                    else:
                        continue
                    pred = _fit_predict(X[tr], y[tr], X[te], ep.task, seed=s)
                    metric, val, _, _ = _score(y[te], pred, ep.task)
                    per_seed.append(val)
                    # bootstrap every seed, not just the first: the seed-0
                    # interval alone measures test-set sampling and says nothing
                    # about the run-to-run spread the withdrawn claim was about.
                    pt, lo, hi = bootstrap_ci(y[te], pred, ep.task,
                                              n_boot=n_boot, seed=0)
                    per_seed_lo.append(lo)
                    per_seed_hi.append(hi)
                    if s == 0:
                        point, boot_lo, boot_hi = pt, lo, hi
                if not per_seed:
                    continue
                sd = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
                mean = statistics.fmean(per_seed)
                # Two uncertainty objects, reported side by side rather than
                # mixed. seed_lo/seed_hi is the spread of the metric across
                # runs; boot_lo/boot_hi is one run's test-set sampling. wide_*
                # is the union, and every separability verdict below uses it,
                # because a difference that survives the wider of the two is
                # the only kind worth calling separable.
                rows.append({
                    "endpoint": ep_name, "featurizer": fname, "split": split,
                    "metric": metric,
                    "mean_over_seeds": round(mean, 4),
                    "sd_over_seeds": round(sd, 4),
                    "seed0": round(point, 4),
                    "seed_lo": round(min(per_seed), 4),
                    "seed_hi": round(max(per_seed), 4),
                    "boot_lo": round(boot_lo, 4), "boot_hi": round(boot_hi, 4),
                    "boot_lo_mean": round(statistics.fmean(per_seed_lo), 4),
                    "boot_hi_mean": round(statistics.fmean(per_seed_hi), 4),
                    "wide_lo": round(min(min(per_seed), min(per_seed_lo)), 4),
                    "wide_hi": round(max(max(per_seed), max(per_seed_hi)), 4),
                    "n_seeds": len(per_seed),
                })

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "with_intervals.csv", index=False)

    if verbose:
        _report(df)
    return df


def _report(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 92}")
    print("  RESULTS WITH UNCERTAINTY   bootstrap 95% CI on the test set, "
          "sd across model seeds")
    print("=" * 92)
    for ep, g in df.groupby("endpoint", sort=False):
        metric = g["metric"].iloc[0]
        print(f"\n  {ep}   ({metric})")
        print(f"  {'featurizer':<14}{'split':<11}{'mean over seeds':>18}"
              f"{'sd':>8}{'bootstrap 95% CI':>26}")
        print("  " + "-" * 88)
        for _, r in g.iterrows():
            ci = f"[{r['boot_lo']:.3f}, {r['boot_hi']:.3f}]"
            print(f"  {r['featurizer']:<14}{r['split']:<11}"
                  f"{r['mean_over_seeds']:>18.4f}{r['sd_over_seeds']:>8.4f}{ci:>26}")

    # -- does the "winner changes with the split" claim survive the intervals?
    #
    # Two arms are treated as indistinguishable when their bootstrap intervals
    # overlap, and an arm counts as separable only when its interval overlaps no
    # other arm's on that endpoint and split. That is a deliberately conservative
    # reading. The looser alternative, asking only whether the leader's point
    # estimate falls inside a rival's interval, calls more arms separable on this
    # data and would let several rankings stand that the intervals do not support.
    # Both are reported below so the choice is visible rather than buried.
    print(f"\n{'-' * 92}")
    print("  DOES THE HEADLINE CLAIM SURVIVE THE INTERVALS?")
    print("-" * 92)

    def _overlap(a, b, lo="wide_lo", hi="wide_hi") -> bool:
        return a[lo] <= b[hi] and b[lo] <= a[hi]

    def _separable(rows, lo, hi) -> list[str]:
        return [r["featurizer"] for r in rows
                if all(not _overlap(r, o, lo, hi) for o in rows
                       if o["featurizer"] != r["featurizer"])]

    verdicts = []
    for ep, g in df.groupby("endpoint", sort=False):
        metric = g["metric"].iloc[0]
        better = _better(metric)
        winners, seps, sep_arms, loose, narrow = {}, {}, {}, {}, {}
        for split, gs in g.groupby("split"):
            gs = gs.reset_index(drop=True)
            best = gs.iloc[0]
            for _, r in gs.iterrows():
                if better(r["mean_over_seeds"], best["mean_over_seeds"]):
                    best = r
            winners[split] = best["featurizer"]

            rows = [r for _, r in gs.iterrows()]
            arms = _separable(rows, "wide_lo", "wide_hi")
            sep_arms[split] = arms
            seps[split] = winners[split] in arms
            narrow[split] = {
                "test_set_bootstrap_seed0": _separable(rows, "boot_lo", "boot_hi"),
                "across_seeds": _separable(rows, "seed_lo", "seed_hi"),
            }

            others = gs[gs["featurizer"] != best["featurizer"]]
            loose[split] = all(
                not (r["boot_lo"] <= best["mean_over_seeds"] <= r["boot_hi"])
                for _, r in others.iterrows())

        flips = len(set(winners.values())) > 1
        verdicts.append({"endpoint": ep, "winners": winners,
                         "winner_changes_with_split": flips,
                         "criterion": "intervals do not overlap, on the union of "
                                      "the across-seed spread and the test-set "
                                      "bootstrap (wide_lo, wide_hi)",
                         "separable_under_each_interval_alone": narrow,
                         "separable_arms": sep_arms,
                         "winner_separated_from_field": seps,
                         "any_arm_separable": {k: bool(v) for k, v in sep_arms.items()},
                         "winner_separated_point_in_interval": loose})
        w = " | ".join(f"{k}: {v}" for k, v in winners.items())
        sepstr = " | ".join(
            f"{k}: {', '.join(sep_arms[k]) if sep_arms[k] else 'NONE SEPARABLE'}"
            for k in sep_arms)
        print(f"  {ep:<18} {w}")
        print(f"  {'':<18} flips: {str(flips):<6} separable: {sepstr}")

    n_flip = sum(v["winner_changes_with_split"] for v in verdicts)
    n_sep = sum(1 for v in verdicts
                if v["winner_changes_with_split"]
                and all(v["winner_separated_from_field"].values()))
    n_none = sum(1 for v in verdicts
                 if not any(v["any_arm_separable"].values()))
    print("-" * 92)
    print(f"  Winner changes with the split on {n_flip} of {len(verdicts)} endpoints.")
    print(f"  Of those, {n_sep} have winners separated from the field by their "
          f"bootstrap intervals.")
    print(f"  No arm is separable from any other on {n_none} of {len(verdicts)} "
          f"endpoints under either split.")
    if n_flip and not n_sep:
        print("  So the flip is real as a ranking, but the arms are not statistically")
        print("  separable. The honest claim is the weaker one: on this data the arms")
        print("  are close enough that split choice decides the ranking - which is")
        print("  itself the argument against ranking on one split.")
    print("=" * 92)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "claim_check.json").write_text(json.dumps(verdicts, indent=2))

    # -- the gate, adjudicated with uncertainty --------------------------------
    #
    # An arm ships for an endpoint only if it beats the fingerprint baseline
    # under the hardest available split, AND its interval does not overlap the
    # baseline's. out/gate.json compares point estimates at one seed and says so
    # in its own header; this is the version with the uncertainty attached, and
    # it is the one the write-up should quote.
    print(f"\n{'-' * 92}")
    print("  THE GATE: does an arm beat the fingerprint under the hardest split,")
    print("            by more than the intervals allow?")
    print("-" * 92)
    gate = {}
    for ep, g in df.groupby("endpoint", sort=False):
        metric = g["metric"].iloc[0]
        better = _better(metric)
        hardest = "scaffold" if "scaffold" in set(g["split"]) else "random"
        sub = {r["featurizer"]: r for _, r in g[g["split"] == hardest].iterrows()}
        if BASELINE not in sub:
            continue
        base = sub[BASELINE]
        ships = [f for f, r in sub.items()
                 if f != BASELINE
                 and better(r["mean_over_seeds"], base["mean_over_seeds"])
                 and not _overlap(r, base)]
        gate[ep] = {"split_used": hardest, "baseline": BASELINE,
                    "baseline_score": round(float(base["mean_over_seeds"]), 4),
                    "criterion": "beats the baseline and its interval does not "
                                 "overlap the baseline's (wide_lo, wide_hi)",
                    "arms_beating_baseline_separably": ships,
                    "ship": ships[0] if ships else BASELINE,
                    "passed": bool(ships)}
        tag = (f"{', '.join(ships)} beats {BASELINE} separably" if ships
               else f"nothing beats {BASELINE} separably - SHIP THE BASELINE")
        print(f"  {ep:<18} ({hardest:<8}) {BASELINE}={base['mean_over_seeds']:.4f}"
              f"   -> {tag}")
    n_pass = sum(v["passed"] for v in gate.values())
    print("-" * 92)
    print(f"  The gate passes on {n_pass} of {len(gate)} endpoints.")
    print("=" * 92)
    (RESULTS / "gate.json").write_text(json.dumps(gate, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoints", nargs="+",
                    default=["esol", "tox21:NR-AR", "tox21:SR-MMP", "tox21:NR-AhR"])
    ap.add_argument("--featurizers", nargs="+",
                    default=["morgan", "descriptors", "combo"])
    ap.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    run(a.endpoints, a.featurizers, a.splits, seeds=a.seeds, n_boot=a.n_boot)
