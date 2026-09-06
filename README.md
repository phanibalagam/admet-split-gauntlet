# admet-split-gauntlet

Split-aware evaluation of molecular featurizers, with a gate that runs per
endpoint and is allowed to fail.

Standalone project, with no dependency on its siblings. Tier B. This is the
project that answers the strategic question the landscape poster cannot: is our
data an advantage, or are we reproducing what is already open?

## What it does

Runs every featurizer against every split on every endpoint, and prints one
table. The table is the deliverable. It is not a model, and it is not a
leaderboard position.

Arms:

| Arm | What it is | Needs |
|---|---|---|
| `morgan` | 2048-bit ECFP4 fingerprint | rdkit only |
| `descriptors` | ~217 RDKit 2D physicochemical descriptors | rdkit only |
| `combo` | both, concatenated | rdkit only |
| `chemberta` | mean-pooled ChemBERTa embeddings | `pip install torch transformers` |

Splits:

| Split | What it simulates |
|---|---|
| `random` | interpolating within chemistry you have already explored. Optimistic. |
| `scaffold` | Bemis-Murcko. Test set has chemotypes the model has not seen. |
| `temporal` | next quarter's compounds. **Needs a run-date column.** |

## Run it

```
python run.py setup
python run.py gauntlet
python run.py gauntlet --featurizers morgan descriptors combo chemberta
```

## What it found on the shipped public data

All four endpoints and all three arms. Means over 5 model seeds, with the
bootstrap 95% interval on the seed-0 test set. Every value is in
`results/with_intervals.csv`.

```
esol            RMSE       random                 scaffold
  morgan                   1.093 [1.049,1.277]    1.618 [1.469,1.794]
  descriptors              0.586 [0.504,0.625]    0.938 [0.830,1.011]
  combo                    0.584 [0.490,0.611]    0.928 [0.808,1.003]

tox21:NR-AR     ROC_AUC    random                 scaffold
  morgan                   0.801 [0.684,0.824]    0.731 [0.651,0.797]
  descriptors              0.764 [0.669,0.831]    0.746 [0.668,0.824]
  combo                    0.769 [0.680,0.831]    0.735 [0.668,0.821]

tox21:SR-MMP    ROC_AUC    random                 scaffold
  morgan                   0.871 [0.845,0.898]    0.763 [0.736,0.807]
  descriptors              0.930 [0.911,0.951]    0.842 [0.814,0.872]
  combo                    0.932 [0.908,0.950]    0.844 [0.818,0.873]

tox21:NR-AhR    ROC_AUC    random                 scaffold
  morgan                   0.891 [0.871,0.925]    0.797 [0.772,0.835]
  descriptors              0.906 [0.897,0.943]    0.841 [0.816,0.870]
  combo                    0.908 [0.893,0.943]    0.843 [0.819,0.875]
```

The split shift is the result that holds up. Scaffold splitting costs something
in all twelve endpoint-arm combinations: 0.018 to 0.108 ROC-AUC, and 48% to 60%
added RMSE on ESOL. That gap is the share of a reported accuracy that is
memorisation of chemotypes, and it is why any ADMET number quoted without naming
its split is unreadable.

The direction holds everywhere but the magnitude does not, and the random and
scaffold intervals fail to overlap in nine of twelve combinations rather than all
twelve. The three exceptions are the three NR-AR arms, which have both the
smallest shifts and much the widest intervals.

The ranking flips only where nothing is separable.

A single-seed run of this project originally reported that the winning arm
changed with the split on three of the four endpoints. Adding repeated seeds and
bootstrap intervals cut that to **one** of four, `tox21:NR-AR`. That endpoint is
also the only one where all three arms sit inside each other's intervals, in both
splits. On ESOL and SR-MMP, where `descriptors` and `combo` are cleanly separated
from `morgan`, the winner does not change at all.

So the reordering happens exactly where the arms cannot be told apart. That is
sampling noise, not a property of the splits, and the original claim is
withdrawn.

That is a weaker claim about ranking and a stronger argument for the gate. If
the arms are within each other's intervals, "which featurizer is best" is not a
well-posed question, and any leaderboard answering it is reporting noise. The
practical consequence stands and gets firmer: run the gate per endpoint and ship
per endpoint, preferring the cheaper arm when intervals overlap.

Run `python run.py experiments` to reproduce both the intervals and the check
that tests this claim, including its ability to fail.

## The gate

An arm ships for an endpoint only if it beats the `morgan` baseline under the
hardest available split, independently, on that endpoint. `out/gate.json` records
the verdict per endpoint so a reviewer can see what was compared and what won.

If nothing beats the fingerprint, ship the fingerprint. It is faster, easier to
interpret, and easier to defend.

## Using your own assay data

```
python run.py internal exports/logD_2019_2026.csv \
    --smiles-col canonical_smiles --y-col logD --date-col assay_date \
    --task regression --splits random scaffold temporal
```

The export needs three columns:

| Column | Why |
|---|---|
| SMILES | obvious |
| value | the measured endpoint |
| **assay run date** | **the one that is usually missing, and the one that matters** |

Without run dates you cannot do a temporal split, and without a temporal split
your accuracy is the accuracy of interpolating within chemistry you have already
explored. Ask for that column before you ask for the model.

Also worth doing before the first run: check that the same compound tested twice
under the same protocol agrees with itself. That reproducibility figure is the
ceiling on any model trained on the data, and it is the only honest answer to
"how good could this get?"

---

## Standalone by design

This project has no dependency on its sibling projects and no hosted-model
dependency at all. It is pure tabular modelling. Copy the folder anywhere and
it runs.

Each project in the collection is meant to be redistributable on its own, as a
repository, a post, or an attachment, so nothing is imported across project
boundaries.

## Licence and provenance

Code in this directory: MIT (see `LICENSE`).

Data: see "What it found on the shipped public data" above for what is real and what is
generated, and under which licence each part may be redistributed.

Originally candidate 06 in a fourteen-candidate portfolio assessment; renumbered
sequentially here because these three were the ones built.
