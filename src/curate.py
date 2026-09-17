"""
Data curation audit, duplicate leakage measurement, and the deduplication arm.

The evaluation pipeline as originally written applied exactly one filter: rows
with a missing label or a missing SMILES were dropped, and RDKit then discarded
what it could not parse. Nothing was deduplicated, no salt or counterion was
stripped, and no charge or tautomer standardization was applied. This module
measures what that leaves in the data, and implements the one correction whose
effect is large enough to change a reported number.

    python run.py curate      -> results/curation.json
    python run.py leakage     -> results/dedup_stats.json

DEDUPLICATION RULE
------------------
Records are grouped by RDKit canonical SMILES.

  1. A compound whose duplicate records DISAGREE on the label is removed
     entirely, every record of it. The disagreement cannot be resolved without
     the source assay, which the redistributed benchmark does not carry, and
     keeping either value would be a guess.
  2. Duplicate records that agree collapse to a single record, the first
     occurrence in file order.
  3. Nothing else is changed. Salts, counterions and mixtures are left intact,
     and no standardization is applied, because neither was done in the
     published run and adding them here would confound the comparison.

WHY LEAKAGE IS MEASURED PER SPLIT
---------------------------------
A duplicate that survives into a random partition can place the same molecule in
both training and test. A Bemis-Murcko scaffold partition cannot do that:
identical molecules share a scaffold, and the scaffold split assigns whole
scaffold groups. So the same uncurated data leaks under one protocol and not the
other, and any random-to-scaffold gap measured on it carries that asymmetry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from src.data import DATA, SOURCES, load_endpoint
from src.featurize import parse
from src.splits import random_split, scaffold_split

RDLogger.DisableLog("rdApp.*")

RESULTS = Path(__file__).resolve().parent.parent / "results"
ENDPOINTS = ["esol", "tox21:NR-AR", "tox21:SR-MMP", "tox21:NR-AhR"]
SEEDS = 5


def source_provenance() -> dict:
    """SHA-256 and origin URL of every raw input, so the numbers carry their source."""
    out = {}
    for fname, url in SOURCES.items():
        p = DATA / fname
        if not p.is_file():
            raise SystemExit(f"curate: {p} is missing. Run `python run.py setup` first.")
        out[fname] = {"url": url,
                      "bytes": p.stat().st_size,
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


def _raw_frame(name: str) -> tuple[int, int]:
    """(rows in the source file, rows surviving the missing-value filter)."""
    if name == "esol":
        df = pd.read_csv(DATA / "delaney.csv")
        col = "measured log solubility in mols per litre"
        return len(df), len(df.dropna(subset=[col, "smiles"]))
    assay = name.split(":", 1)[1]
    df = pd.read_csv(DATA / "tox21.csv.gz")
    return len(df), len(df.dropna(subset=[assay, "smiles"]))


def canonical(mols: list) -> np.ndarray:
    return np.array([Chem.MolToSmiles(m) for m in mols])


def dedup_index(mols: list, y: np.ndarray, task: str) -> tuple[np.ndarray, dict]:
    """Apply the deduplication rule. Returns (indices to keep, statistics)."""
    cs = canonical(mols)
    df = pd.DataFrame({"c": cs, "y": np.asarray(y), "i": np.arange(len(mols))})
    g = df.groupby("c")["y"]
    if task == "regression":
        conflicting = set(g.apply(lambda s: float(s.max() - s.min()) > 1e-9)
                          .pipe(lambda s: s[s]).index)
    else:
        conflicting = set(g.nunique().pipe(lambda s: s[s > 1]).index)
    kept = df[~df["c"].isin(conflicting)].drop_duplicates("c", keep="first")
    stats = {
        "n_before": int(len(df)),
        "conflicting_compounds": int(len(conflicting)),
        "conflicting_records_removed": int(df["c"].isin(conflicting).sum()),
        "agreeing_duplicate_records_collapsed":
            int(len(df) - df["c"].isin(conflicting).sum() - len(kept)),
        "n_after": int(len(kept)),
    }
    return kept["i"].to_numpy(), stats


def audit() -> list[dict]:
    """Per-endpoint curation record, computed from the source files."""
    rows = []
    for name in ENDPOINTS:
        ep = load_endpoint(name)
        n_source, n_kept = _raw_frame(name)
        mols, keep = parse(ep.smiles)
        y = ep.y[keep].to_numpy()
        cs = canonical(mols)
        counts = pd.Series(cs).value_counts()
        dup_compounds = counts[counts > 1]

        ser = pd.DataFrame({"c": cs, "y": y}).groupby("c")["y"]
        if ep.task == "regression":
            spread = ser.max() - ser.min()
            conflict = int((spread[dup_compounds.index] > 1e-9).sum()) if len(dup_compounds) else 0
            widest = float(spread[dup_compounds.index].max()) if len(dup_compounds) else 0.0
        else:
            nun = ser.nunique()
            conflict = int((nun[dup_compounds.index] > 1).sum()) if len(dup_compounds) else 0
            widest = None

        rec = {
            "endpoint": name,
            "task": ep.task,
            "source_records": int(n_source),
            "dropped_missing_label_or_smiles": int(n_source - n_kept),
            "dropped_rdkit_parse_failure": int(n_kept - len(mols)),
            "records_analyzed": int(len(mols)),
            "distinct_canonical_smiles": int(len(counts)),
            "duplicate_records": int(len(mols) - len(counts)),
            "compounds_with_a_duplicate": int(len(dup_compounds)),
            "compounds_with_conflicting_duplicates": conflict,
            "widest_label_disagreement": widest,
            "disconnected_component_records": int(sum("." in c for c in cs)),
            "records_with_formal_charge": int(sum(("+" in c or "-" in c) for c in cs)),
        }
        if ep.task == "classification":
            pos = int(np.asarray(y).sum())
            rec["positives"] = pos
            rec["negatives"] = int(len(y) - pos)
            rec["positive_rate"] = round(pos / len(y), 4)
        else:
            rec["label_min"] = round(float(np.min(y)), 4)
            rec["label_max"] = round(float(np.max(y)), 4)
            rec["label_mean"] = round(float(np.mean(y)), 4)
        _, st = dedup_index(mols, y, ep.task)
        rec["deduplication"] = st
        rows.append(rec)
        print(f"  {name:<16} analyzed={rec['records_analyzed']:>6} "
              f"distinct={rec['distinct_canonical_smiles']:>6} "
              f"dupes={rec['duplicate_records']:>4} "
              f"conflicts={rec['compounds_with_conflicting_duplicates']:>3} "
              f"after dedup={st['n_after']:>6}", flush=True)
    return rows


def leakage() -> list[dict]:
    """Test rows whose canonical SMILES is also in training, per split, per seed."""
    rows = []
    for name in ENDPOINTS:
        ep = load_endpoint(name)
        mols, keep = parse(ep.smiles)
        cs = canonical(mols)
        n = len(mols)
        for split in ("random", "scaffold"):
            for seed in range(SEEDS):
                if split == "random":
                    tr, te = random_split(n, seed=seed)
                else:
                    # deterministic: a function of the molecules, with no seed
                    tr, te = scaffold_split(mols)
                train_set = set(cs[tr])
                leaked = [i for i in te if cs[i] in train_set]
                rows.append({
                    "endpoint": name, "split": split, "seed": seed,
                    "n_train": int(len(tr)), "n_test": int(len(te)),
                    "leaked_test_rows": int(len(leaked)),
                    "leaked_test_compounds": int(len({cs[i] for i in leaked})),
                    "leaked_pct_of_test": round(100.0 * len(leaked) / len(te), 4),
                })
                if split == "scaffold":
                    break          # no seed to vary; one measurement is the answer
        done = [r for r in rows if r["endpoint"] == name]
        print(f"  {name:<16} random seed0={done[0]['leaked_test_rows']}/"
              f"{done[0]['n_test']} ({done[0]['leaked_pct_of_test']}%)  "
              f"scaffold={done[-1]['leaked_test_rows']}/{done[-1]['n_test']}", flush=True)
    return rows


def write_curation() -> None:
    RESULTS.mkdir(exist_ok=True)
    payload = {"sources": source_provenance(),
               "deduplication_rule":
                   "compounds whose duplicate records disagree on the label are "
                   "removed entirely; agreeing duplicates collapse to the first "
                   "occurrence; no salt stripping or standardization is applied",
               "endpoints": audit()}
    (RESULTS / "curation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULTS / 'curation.json'}")


def write_leakage() -> None:
    RESULTS.mkdir(exist_ok=True)
    payload = {"sources": source_provenance(),
               "seeds": SEEDS,
               "note": "the scaffold split is a deterministic function of the "
                       "molecules and has no seed, so it is measured once",
               "measurements": leakage()}
    (RESULTS / "dedup_stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULTS / 'dedup_stats.json'}")
