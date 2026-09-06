"""
Splits - where most reported ADMET accuracy quietly comes from.

random    shuffle. Optimistic: test molecules look like training molecules.
scaffold  Bemis-Murcko scaffold split. Test set contains chemistry the model
          has not seen the shape of. The honest offline proxy.
temporal  split by assay run date. The only split that answers the question you
          actually care about - will this hold up on next quarter's compounds?
          Requires a date column, which public datasets do not have. That is the
          single most important thing to fix in your own data export.

The gap between random and scaffold on the same model is the amount of your
reported accuracy that is memorisation of chemotypes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def random_split(n: int, frac_train: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * frac_train)
    return idx[:cut], idx[cut:]


def scaffold_split(mols: list, frac_train: float = 0.8):
    """
    Group by Bemis-Murcko scaffold, then fill the training set with the largest
    groups first. Standard MoleculeNet practice: deterministic, and it puts the
    rare chemotypes in test where they belong.

    There is no seed argument, deliberately. This split is a function of the
    molecules alone, so the partition cannot be varied without changing the
    protocol to a randomised or balanced scaffold split, which is a different
    method rather than a different seed. An earlier version accepted a `seed`
    and ignored it, which read as if the partition were being reseeded when it
    was not.
    """
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(mols):
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:  # noqa: BLE001
            s = ""
        groups.setdefault(s, []).append(i)

    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n_train = int(len(mols) * frac_train)
    train, test = [], []
    for g in ordered:
        (train if len(train) + len(g) <= n_train else test).extend(g)
    return np.array(train), np.array(test)


def temporal_split(dates: pd.Series | None, frac_train: float = 0.8):
    """
    Everything before the cut date trains, everything after tests. No leakage of
    future chemistry into the past, which is exactly the leak that makes offline
    ADMET numbers look better than deployed ones.
    """
    if dates is None or dates.isna().all():
        raise ValueError(
            "temporal split needs an assay run date per row.\n"
            "Public benchmark sets do not carry one. When you export internal\n"
            "assay history, include the run date - it is the difference between\n"
            "a number you can defend and a number you cannot.")
    order = np.argsort(dates.values)
    cut = int(len(order) * frac_train)
    return order[:cut], order[cut:]


SPLITS = {"random": "shuffled", "scaffold": "Bemis-Murcko", "temporal": "by run date"}
