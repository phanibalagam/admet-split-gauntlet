"""
Endpoint registry.

Two real, public MoleculeNet datasets back the project:

  esol        aqueous solubility, regression, 1128 compounds (Delaney)
  tox21:*     12 binary toxicity assays, ~8000 compounds

Both are downloaded from the DeepChem repository by `python -m src.data`.
They stand in for your internal assay history. The important difference is
called out in splits.py: public sets have no run dates, so they cannot be split
temporally, which is the split that actually predicts deployment behaviour.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

SOURCES = {
    "delaney.csv": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
    "tox21.csv.gz": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/tox21.csv.gz",
}

TOX21_ASSAYS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
                "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE",
                "SR-MMP", "SR-p53"]


@dataclass
class Endpoint:
    name: str
    smiles: list[str]
    y: pd.Series
    task: str          # "regression" | "classification"
    units: str = ""
    date: pd.Series | None = None   # populated only for internal data

    def __len__(self) -> int:
        return len(self.smiles)

    def describe(self) -> str:
        if self.task == "regression":
            return (f"{self.name:<16} regression  n={len(self):>5}  "
                    f"mean={self.y.mean():.2f} sd={self.y.std():.2f} {self.units}")
        pos = int(self.y.sum())
        return (f"{self.name:<16} binary      n={len(self):>5}  "
                f"positives={pos} ({pos / len(self):.1%})")


def download(force: bool = False) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for fname, url in SOURCES.items():
        dest = DATA / fname
        if dest.exists() and not force:
            print(f"  have {fname}")
            continue
        print(f"  downloading {fname} ...")
        urllib.request.urlretrieve(url, dest)
    print("datasets ready in", DATA)


def load_endpoint(name: str) -> Endpoint:
    """
    name is either "esol" or "tox21:<ASSAY>", e.g. "tox21:SR-MMP".
    """
    if name == "esol":
        df = pd.read_csv(DATA / "delaney.csv")
        col = "measured log solubility in mols per litre"
        df = df.dropna(subset=[col, "smiles"])
        return Endpoint(name="esol", smiles=df["smiles"].str.strip().tolist(),
                        y=df[col].reset_index(drop=True), task="regression",
                        units="log mol/L")

    if name.startswith("tox21:"):
        assay = name.split(":", 1)[1]
        if assay not in TOX21_ASSAYS:
            raise ValueError(f"unknown assay {assay!r}; choose from {TOX21_ASSAYS}")
        df = pd.read_csv(DATA / "tox21.csv.gz")
        df = df.dropna(subset=[assay, "smiles"])
        return Endpoint(name=name, smiles=df["smiles"].str.strip().tolist(),
                        y=df[assay].astype(int).reset_index(drop=True),
                        task="classification")

    raise ValueError(f"unknown endpoint {name!r}")


def load_internal(csv_path: str, smiles_col: str = "smiles",
                  y_col: str = "value", date_col: str | None = "assay_date",
                  task: str = "regression") -> Endpoint:
    """
    Load your own assay export. THIS is the function that matters in production.

    The date column is not optional in practice: without it you cannot run a
    temporal split, and without a temporal split your reported accuracy is the
    accuracy of interpolating within chemistry you have already explored.
    """
    df = pd.read_csv(csv_path).dropna(subset=[smiles_col, y_col])
    date = None
    if date_col and date_col in df.columns:
        date = pd.to_datetime(df[date_col], errors="coerce").reset_index(drop=True)
    return Endpoint(name=Path(csv_path).stem, smiles=df[smiles_col].tolist(),
                    y=df[y_col].reset_index(drop=True), task=task, date=date)


if __name__ == "__main__":
    download()
    for n in ["esol", "tox21:NR-AR", "tox21:SR-MMP", "tox21:NR-AhR"]:
        print(" ", load_endpoint(n).describe())
