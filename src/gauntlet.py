"""
The gauntlet: every featurizer against every split, on every endpoint.

Output is one table. The table is the deliverable - not a model, not a
leaderboard position. It answers three questions a programme has to answer
before it funds anything:

  1. Does the fancy featurizer beat the fingerprint on OUR endpoint?
  2. How much of our accuracy disappears when the test set contains chemistry
     the model has not seen?
  3. Is the winner the same across endpoints, or are we picking noise?

Gate (from the build plan): an arm ships for an endpoint only if it beats the
morgan baseline under the hardest available split. Per endpoint, independently.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

from src.data import Endpoint, load_endpoint
from src.featurize import FEATURIZERS, parse
from src.splits import random_split, scaffold_split, temporal_split

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Result:
    endpoint: str
    featurizer: str
    split: str
    n_train: int
    n_test: int
    primary_metric: str
    primary: float
    secondary_metric: str
    secondary: float


def _fit_predict(X_tr, y_tr, X_te, task: str, seed: int = 0):
    if task == "regression":
        m = HistGradientBoostingRegressor(random_state=seed, max_iter=300,
                                          early_stopping=True)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
    m = HistGradientBoostingClassifier(random_state=seed, max_iter=300,
                                       early_stopping=True)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def _score(y_true, y_pred, task: str):
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return ("RMSE", rmse, "R2", float(r2_score(y_true, y_pred)))
    # a single positive class in test makes ROC-AUC undefined
    if len(np.unique(y_true)) < 2:
        return ("ROC_AUC", float("nan"), "PR_AUC", float("nan"))
    return ("ROC_AUC", float(roc_auc_score(y_true, y_pred)),
            "PR_AUC", float(average_precision_score(y_true, y_pred)))


def run_endpoint(ep: Endpoint, featurizers: list[str], splits: list[str],
                 seed: int = 0, verbose: bool = True) -> list[Result]:
    mols, keep = parse(ep.smiles)
    y = ep.y[keep].to_numpy()
    dates = ep.date[keep].reset_index(drop=True) if ep.date is not None else None
    if verbose:
        print(f"\n{ep.name}: {len(mols)} parsed molecules, task={ep.task}")

    feats: dict[str, np.ndarray] = {}
    for f in featurizers:
        try:
            feats[f] = FEATURIZERS[f](mols)
            if verbose:
                print(f"  featurized {f:<12} {feats[f].shape}")
        except RuntimeError as e:
            print(f"  SKIP {f}: {e}")

    results: list[Result] = []
    for split in splits:
        try:
            if split == "random":
                tr, te = random_split(len(mols), seed=seed)
            elif split == "scaffold":
                tr, te = scaffold_split(mols, seed=seed)
            elif split == "temporal":
                tr, te = temporal_split(dates)
            else:
                raise ValueError(split)
        except ValueError as e:
            if verbose:
                print(f"  SKIP split '{split}': {str(e).splitlines()[0]}")
            continue

        for fname, X in feats.items():
            pred = _fit_predict(X[tr], y[tr], X[te], ep.task, seed)
            pm, pv, sm, sv = _score(y[te], pred, ep.task)
            results.append(Result(ep.name, fname, split, len(tr), len(te),
                                  pm, round(pv, 4), sm, round(sv, 4)))
    return results


def run(endpoints: list[str], featurizers: list[str], splits: list[str],
        seed: int = 0, out_dir: str = "out") -> pd.DataFrame:
    rows: list[Result] = []
    for name in endpoints:
        rows += run_endpoint(load_endpoint(name), featurizers, splits, seed)

    df = pd.DataFrame([asdict(r) for r in rows])
    Path(out_dir).mkdir(exist_ok=True)
    df.to_csv(f"{out_dir}/gauntlet.csv", index=False)
    return df


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _better(metric: str):
    """Return a comparison function: is a better than b?"""
    lower_is_better = {"RMSE", "MAE"}
    return (lambda a, b: a < b) if metric in lower_is_better else (lambda a, b: a > b)


def report(df: pd.DataFrame, baseline: str = "morgan", out_dir: str = "out") -> dict:
    print("\n" + "=" * 88)
    print("  GAUNTLET RESULTS")
    print("=" * 88)

    for ep, g in df.groupby("endpoint", sort=False):
        metric = g["primary_metric"].iloc[0]
        print(f"\n  {ep}   primary metric: {metric}")
        piv = g.pivot_table(index="featurizer", columns="split", values="primary")
        cols = [c for c in ["random", "scaffold", "temporal"] if c in piv.columns]
        piv = piv[cols]
        print(piv.round(4).to_string())

        if "random" in cols and "scaffold" in cols:
            gap = (piv["scaffold"] - piv["random"]).abs()
            worst = gap.idxmax()
            print(f"    random -> scaffold shift, largest for '{worst}': "
                  f"{gap.max():.4f} {metric}")

    # -- the gate ----------------------------------------------------------
    verdicts = {}
    print("\n" + "-" * 88)
    print("  GATE: does an arm beat the fingerprint baseline under the hardest split?")
    print("-" * 88)
    for ep, g in df.groupby("endpoint", sort=False):
        metric = g["primary_metric"].iloc[0]
        better = _better(metric)
        hardest = ("temporal" if "temporal" in set(g["split"])
                   else "scaffold" if "scaffold" in set(g["split"]) else "random")
        sub = g[g["split"] == hardest].set_index("featurizer")["primary"]
        if baseline not in sub.index:
            continue
        base = sub[baseline]
        winners = [f for f, v in sub.items()
                   if f != baseline and better(v, base) and np.isfinite(v)]
        verdicts[ep] = {"split_used": hardest, "baseline": baseline,
                        "baseline_score": float(base),
                        "arms_beating_baseline": winners,
                        "ship": winners[0] if winners else baseline}
        tag = (f"{', '.join(winners)} beats baseline" if winners
               else "nothing beats the fingerprint - SHIP THE BASELINE")
        print(f"  {ep:<18} ({hardest:<8}) {baseline}={base:.4f}   -> {tag}")

    print("-" * 88)

    # -- ranking stability -------------------------------------------------
    # The 2026 benchmark finding, reproduced on your own numbers: if the winner
    # changes with the endpoint or with the split, you are choosing noise.
    winners = {}
    for (ep, split), g in df.groupby(["endpoint", "split"], sort=False):
        metric = g["primary_metric"].iloc[0]
        better = _better(metric)
        best, best_v = None, None
        for _, row in g.iterrows():
            v = row["primary"]
            if not np.isfinite(v):
                continue
            if best_v is None or better(v, best_v):
                best, best_v = row["featurizer"], v
        winners[(ep, split)] = best

    distinct = sorted({w for w in winners.values() if w})
    flips = [ep for ep in df["endpoint"].unique()
             if len({w for (e, _), w in winners.items() if e == ep}) > 1]
    print(f"  ranking stability: {len(distinct)} distinct winner(s) across "
          f"{len(df['endpoint'].unique())} endpoints -> {', '.join(distinct)}")
    if flips:
        print(f"  winner CHANGES with the split on: {', '.join(flips)}")
        print("  -> a leaderboard built on one split would have picked the wrong arm.")
    else:
        print("  winner is stable across splits on every endpoint.")

    print("-" * 88)
    n_base = sum(1 for v in verdicts.values() if not v["arms_beating_baseline"])
    print(f"  {n_base} of {len(verdicts)} endpoints are best served by the "
          f"fingerprint baseline.")
    print("=" * 88)

    Path(out_dir).mkdir(exist_ok=True)
    Path(f"{out_dir}/gate.json").write_text(json.dumps(
        {"gate": verdicts,
         "winners": {f"{e}|{s_}": w for (e, s_), w in winners.items()},
         "ranking_unstable_on": flips}, indent=2))
    return verdicts
