"""
Featurizers - the arms of the gauntlet.

Three run with no downloads:
  morgan       2048-bit Morgan/ECFP4 fingerprint. The baseline you must beat.
  descriptors  ~200 RDKit 2D physicochemical descriptors.
  combo        morgan + descriptors concatenated.

One is optional:
  chemberta    mean-pooled ChemBERTa embeddings. Needs `pip install torch
               transformers` and network access to huggingface.co on first run.

The point of keeping `morgan` first is procedural, not sentimental: a 2026
benchmark of molecular property prediction found method rankings unstable across
evaluation protocols, with four different models taking the top score across six
ADME endpoints under temporal splitting. A foundation model that cannot beat a
2048-bit fingerprint on your own data has not earned its inference cost.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

_DESC_NAMES = [n for n, _ in Descriptors._descList]
_DESC_FUNCS = [f for _, f in Descriptors._descList]


def parse(smiles: list[str]) -> tuple[list, np.ndarray]:
    """Returns (mols, keep_mask). Unparseable SMILES are dropped, not imputed."""
    mols, keep = [], np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
            keep[i] = True
    return mols, keep


def morgan(mols: list) -> np.ndarray:
    return np.array([_MORGAN.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)


def descriptors(mols: list) -> np.ndarray:
    rows = []
    for m in mols:
        vals = []
        for f in _DESC_FUNCS:
            try:
                v = f(m)
            except Exception:  # noqa: BLE001 - a handful of descriptors throw
                v = np.nan
            vals.append(v)
        rows.append(vals)
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # a few RDKit descriptors (Ipc in particular) blow past float32 range
    lim = np.finfo(np.float32).max
    return np.clip(X, -lim, lim).astype(np.float32)


def combo(mols: list) -> np.ndarray:
    return np.hstack([morgan(mols), descriptors(mols)])


def chemberta(mols: list, model_name: str = "DeepChem/ChemBERTa-77M-MTR",
              batch_size: int = 64) -> np.ndarray:
    """
    Mean-pooled hidden states from a chemical language model.

    Optional arm. Requires: pip install torch transformers
    First run downloads ~150MB from huggingface.co.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "chemberta arm needs torch + transformers:\n"
            "    pip install torch transformers\n"
            "The other three arms run without them.") from e

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    smis = [Chem.MolToSmiles(m) for m in mols]

    out = []
    with torch.no_grad():
        for i in range(0, len(smis), batch_size):
            enc = tok(smis[i:i + batch_size], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.vstack(out).astype(np.float32)


FEATURIZERS = {
    "morgan": morgan,
    "descriptors": descriptors,
    "combo": combo,
    "chemberta": chemberta,
}


def available() -> dict[str, str]:
    status = {k: "ready" for k in ["morgan", "descriptors", "combo"]}
    try:
        import torch, transformers  # noqa: F401
        status["chemberta"] = "ready"
    except ImportError:
        status["chemberta"] = "needs: pip install torch transformers"
    return status
