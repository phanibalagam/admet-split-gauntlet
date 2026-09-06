#!/usr/bin/env python
"""
ADMET baseline gauntlet - command line entry point.

    python run.py setup
    python run.py gauntlet
    python run.py experiments                  repeated seeds + bootstrap CIs
    python run.py gauntlet --featurizers morgan descriptors combo chemberta
    python run.py gauntlet --endpoints esol tox21:SR-MMP --splits random scaffold
    python run.py internal my_assays.csv --y-col logD --task regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_ENDPOINTS = ["esol", "tox21:NR-AR", "tox21:SR-MMP", "tox21:NR-AhR"]


def cmd_setup(a):
    from src.data import download, load_endpoint
    from src.featurize import available
    download()
    print("\nendpoints:")
    for n in DEFAULT_ENDPOINTS:
        print("  ", load_endpoint(n).describe())
    print("\nfeaturizer availability:")
    for k, v in available().items():
        print(f"   {k:<12} {v}")


def cmd_gauntlet(a):
    from src.gauntlet import run, report
    df = run(a.endpoints, a.featurizers, a.splits, seed=a.seed)
    report(df, baseline=a.baseline)
    print(f"\n  full table -> out/gauntlet.csv\n  gate verdicts -> out/gate.json")


def cmd_experiments(a):
    from src.experiments import run as run_exp
    run_exp(a.endpoints, a.featurizers, a.splits, seeds=a.seeds, n_boot=a.n_boot)


def cmd_internal(a):
    from src.data import load_internal
    from src.gauntlet import run_endpoint, report
    import pandas as pd
    from dataclasses import asdict
    ep = load_internal(a.csv, smiles_col=a.smiles_col, y_col=a.y_col,
                       date_col=a.date_col, task=a.task)
    print(ep.describe())
    rows = run_endpoint(ep, a.featurizers, a.splits, seed=a.seed)
    df = pd.DataFrame([asdict(r) for r in rows])
    Path("out").mkdir(exist_ok=True)
    df.to_csv("out/gauntlet_internal.csv", index=False)
    report(df, baseline=a.baseline)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="download datasets, show availability")
    s.set_defaults(func=cmd_setup)

    common = dict(featurizers=["morgan", "descriptors", "combo"],
                  splits=["random", "scaffold"])

    s = sub.add_parser("gauntlet", help="run every arm against every split")
    s.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    s.add_argument("--featurizers", nargs="+", default=common["featurizers"])
    s.add_argument("--splits", nargs="+", default=common["splits"])
    s.add_argument("--baseline", default="morgan")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_gauntlet)

    s = sub.add_parser("experiments",
                       help="repeated seeds + bootstrap CIs, and the claim check")
    s.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    s.add_argument("--featurizers", nargs="+", default=common["featurizers"])
    s.add_argument("--splits", nargs="+", default=common["splits"])
    s.add_argument("--seeds", type=int, default=5)
    s.add_argument("--n-boot", type=int, default=500)
    s.set_defaults(func=cmd_experiments)

    s = sub.add_parser("internal", help="run the gauntlet on your own assay export")
    s.add_argument("csv")
    s.add_argument("--smiles-col", default="smiles")
    s.add_argument("--y-col", default="value")
    s.add_argument("--date-col", default="assay_date")
    s.add_argument("--task", default="regression",
                   choices=["regression", "classification"])
    s.add_argument("--featurizers", nargs="+", default=common["featurizers"])
    s.add_argument("--splits", nargs="+",
                   default=["random", "scaffold", "temporal"])
    s.add_argument("--baseline", default="morgan")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_internal)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
