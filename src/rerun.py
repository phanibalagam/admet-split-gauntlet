"""
Both curation arms, end to end, with the verdicts derived rather than asserted.

Runs the full protocol twice over the same endpoints, featurizers and splits:

  as_published   the sets exactly as the pipeline has always built them
  dedup          the same sets after src.curate.dedup_index has been applied

Each arm uses five model seeds and a 500-resample percentile bootstrap computed
at every seed, and reports the union of the across-seed spread and the bootstrap
as its interval, which is what every separability verdict is decided on.

Nothing here is hardcoded from a previous run. The reorder counts, the separable
arm pairs and the largest metric shifts are all computed from the rows this
module produces.

Partial results are checkpointed to results/_partial/ after every
(endpoint, arm) pair, so an interrupted run resumes instead of starting over.

    python run.py rerun
    python run.py rerun --seeds 5 --n-boot 500
"""

from __future__ import annotations

import itertools
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.curate import dedup_index
from src.data import load_endpoint
from src.experiments import bootstrap_ci
from src.featurize import FEATURIZERS, parse
from src.gauntlet import _better, _fit_predict, _score
from src.splits import random_split, scaffold_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
PARTIAL = RESULTS / "_partial"
ENDPOINTS = ["esol", "tox21:NR-AR", "tox21:SR-MMP", "tox21:NR-AhR"]
FEATS = ["morgan", "descriptors", "combo"]
SPLITS = ["random", "scaffold"]
ARMS = ["as_published", "dedup"]


def _one(ep_name: str, arm: str, seeds: int, n_boot: int) -> list[dict]:
    ep = load_endpoint(ep_name)
    mols, keep = parse(ep.smiles)
    y = ep.y[keep].to_numpy()
    if arm == "dedup":
        idx, _ = dedup_index(mols, y, ep.task)
        mols = [mols[i] for i in idx]
        y = y[idx]
    feats = {f: FEATURIZERS[f](mols) for f in FEATS}
    rows = []
    for split in SPLITS:
        for fname, X in feats.items():
            per, los, his, seed0 = [], [], [], None
            for s in range(seeds):
                if split == "random":
                    tr, te = random_split(len(mols), seed=s)
                else:
                    tr, te = scaffold_split(mols)
                pred = _fit_predict(X[tr], y[tr], X[te], ep.task, seed=s)
                metric, val, _, _ = _score(y[te], pred, ep.task)
                per.append(val)
                pt, lo, hi = bootstrap_ci(y[te], pred, ep.task, n_boot=n_boot, seed=0)
                los.append(lo)
                his.append(hi)
                if s == 0:
                    seed0 = (pt, lo, hi)
            rows.append({
                "arm": arm, "endpoint": ep_name, "featurizer": fname, "split": split,
                "metric": metric, "n": len(mols), "n_seeds": seeds,
                "mean_over_seeds": round(statistics.fmean(per), 4),
                "sd_over_seeds": round(statistics.stdev(per) if len(per) > 1 else 0.0, 4),
                "seed0": round(seed0[0], 4),
                "seed_lo": round(min(per), 4), "seed_hi": round(max(per), 4),
                "boot_lo": round(seed0[1], 4), "boot_hi": round(seed0[2], 4),
                "boot_lo_mean": round(statistics.fmean(los), 4),
                "boot_hi_mean": round(statistics.fmean(his), 4),
                "wide_lo": round(min(min(per), min(los)), 4),
                "wide_hi": round(max(max(per), max(his)), 4),
            })
    return rows


def _overlap(a, b, lo, hi) -> bool:
    return a[lo] <= b[hi] and b[lo] <= a[hi]


def _best(sub: pd.DataFrame, column: str) -> str:
    better = _better(sub["metric"].iloc[0])
    row = sub.iloc[0]
    for _, r in sub.iterrows():
        if better(r[column], row[column]):
            row = r
    return row["featurizer"]


def verdicts(df: pd.DataFrame) -> dict:
    """Every claim the write-up makes, derived from the rows."""
    out = {}
    for arm in sorted(df["arm"].unique()):
        a = df[df["arm"] == arm]
        per_endpoint, reorder_5, reorder_0 = [], 0, 0
        for ep in ENDPOINTS:
            g = a[a["endpoint"] == ep]
            if g.empty:
                continue
            best5 = {sp: _best(g[g["split"] == sp], "mean_over_seeds") for sp in SPLITS}
            best0 = {sp: _best(g[g["split"] == sp], "seed0") for sp in SPLITS}
            pairs = {}
            for sp in SPLITS:
                rows = [r for _, r in g[g["split"] == sp].iterrows()]
                for iv, (lo, hi) in {"union": ("wide_lo", "wide_hi"),
                                     "bootstrap_seed0": ("boot_lo", "boot_hi"),
                                     "across_seeds": ("seed_lo", "seed_hi")}.items():
                    sep = [f"{x['featurizer']} vs {o['featurizer']}"
                           for x, o in itertools.combinations(rows, 2)
                           if not _overlap(x, o, lo, hi)]
                    pairs.setdefault(iv, {})[sp] = sorted(sep)
            reorder_5 += best5["random"] != best5["scaffold"]
            reorder_0 += best0["random"] != best0["scaffold"]
            per_endpoint.append({
                "endpoint": ep, "n": int(g["n"].iloc[0]),
                "best_arm_five_seeds": best5, "best_arm_seed0": best0,
                "reorders_five_seeds": best5["random"] != best5["scaffold"],
                "reorders_seed0": best0["random"] != best0["scaffold"],
                "separable_pairs_by_interval": pairs,
            })
        out[arm] = {
            "endpoints": per_endpoint,
            "reorder_count_five_seeds": int(reorder_5),
            "reorder_count_seed0": int(reorder_0),
            "separable_pairs_union_total": sum(
                len(e["separable_pairs_by_interval"]["union"][sp])
                for e in per_endpoint for sp in SPLITS),
        }
    if set(ARMS) <= set(out):
        merged = df.pivot_table(index=["endpoint", "featurizer", "split"],
                                columns="arm", values="mean_over_seeds")
        merged["delta"] = (merged["dedup"] - merged["as_published"]).round(4)
        biggest = merged.reindex(merged["delta"].abs().sort_values(ascending=False).index)
        out["largest_shifts"] = [
            {"endpoint": i[0], "featurizer": i[1], "split": i[2],
             "as_published": float(r["as_published"]), "dedup": float(r["dedup"]),
             "delta": float(r["delta"])}
            for i, r in biggest.head(6).iterrows()]
    return out


def run(seeds: int = 5, n_boot: int = 500) -> pd.DataFrame:
    RESULTS.mkdir(exist_ok=True)
    PARTIAL.mkdir(exist_ok=True)
    t0 = time.time()
    frames, timings = [], []
    for ep_name in ENDPOINTS:
        for arm in ARMS:
            tag = f"{ep_name.replace(':', '_')}__{arm}"
            ck = PARTIAL / f"{tag}.csv"
            if ck.is_file():
                print(f"  resume {tag}", flush=True)
                frames.append(pd.read_csv(ck))
                continue
            t = time.time()
            rows = _one(ep_name, arm, seeds, n_boot)
            pd.DataFrame(rows).to_csv(ck, index=False)
            dt = time.time() - t
            timings.append({"endpoint": ep_name, "arm": arm, "seconds": round(dt, 1)})
            frames.append(pd.DataFrame(rows))
            print(f"  done {tag} in {dt / 60:.1f} min", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "dedup_compare.csv", index=False)

    total = time.time() - t0
    (RESULTS / "environment.json").write_text(json.dumps({
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "settings": {"seeds": seeds, "n_boot": n_boot, "endpoints": ENDPOINTS,
                     "featurizers": FEATS, "splits": SPLITS, "arms": ARMS},
        "runtime_seconds_this_run": round(total, 1),
        "runtime_by_endpoint_arm": timings,
    }, indent=2) + "\n")

    v = verdicts(df)
    (RESULTS / "claim_check.json").write_text(json.dumps(v, indent=2) + "\n")
    _report(df, v)
    return df


def _package_versions() -> dict:
    import importlib
    out = {}
    for mod in ("numpy", "pandas", "sklearn", "rdkit"):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = "absent"
    return out


def _report(df: pd.DataFrame, v: dict) -> None:
    print("\n" + "=" * 78)
    print("  DERIVED VERDICTS")
    print("=" * 78)
    for arm in ARMS:
        if arm not in v:
            continue
        a = v[arm]
        print(f"\n  {arm}")
        print(f"    best arm differs between splits: "
              f"{a['reorder_count_seed0']} of 4 at seed 0, "
              f"{a['reorder_count_five_seeds']} of 4 over five seeds")
        for e in a["endpoints"]:
            sep = sorted({p for sp in SPLITS
                          for p in e["separable_pairs_by_interval"]["union"][sp]})
            print(f"      {e['endpoint']:<16} n={e['n']:<6} "
                  f"5-seed {e['best_arm_five_seeds']['random']}/"
                  f"{e['best_arm_five_seeds']['scaffold']}  "
                  f"separable(union): {', '.join(sep) if sep else 'none'}")
    if "largest_shifts" in v:
        print("\n  largest shifts, dedup minus as_published")
        for s in v["largest_shifts"]:
            print(f"    {s['endpoint']:<16} {s['featurizer']:<12} {s['split']:<9} "
                  f"{s['as_published']:.4f} -> {s['dedup']:.4f}  {s['delta']:+.4f}")
    print("=" * 78)
