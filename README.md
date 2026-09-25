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
python run.py setup        # downloads ESOL and Tox21; they are not redistributed here
python run.py curate       # curation audit          -> results/curation.json
python run.py leakage      # duplicate leakage       -> results/dedup_stats.json
python run.py gauntlet     # one seed, point estimates
python run.py experiments  # five seeds + bootstrap  -> results/with_intervals.csv
python run.py rerun        # both curation arms      -> results/dedup_compare.csv
```

`./reproduce.sh` (or `reproduce.ps1`) runs all of it in order and writes every
reported number under `results/`. Partial results are checkpointed to
`results/_partial/`, so an interrupted run resumes instead of starting over.

### Curation

The pipeline as originally written performed no curation beyond dropping missing
values: no deduplication, no salt stripping, no standardization. `run.py curate`
measures what that leaves in the data and `run.py leakage` measures what it costs
- duplicate molecules that a random split can place on both sides of the
partition, which a scaffold split cannot. `run.py rerun` then runs the whole
protocol with and without deduplication so the difference is reported rather than
hidden. See METHODS.md sections 2.1 and 2.2.

### Runtime, measured

On one modern x86-64 container, Python 3.11.15, no GPU:

| Step | Measured |
|---|---|
| `run.py setup` | seconds (two downloads, 222 KB total) |
| `run.py curate` + `run.py leakage` | about 2 minutes |
| `run.py gauntlet` | 5.9 minutes |
| `run.py experiments --seeds 5 --n-boot 500` | 14.9 minutes |
| `run.py rerun --seeds 5 --n-boot 500` | 28.6 minutes from cold |
| **`./reproduce.sh` end to end** | **about 52 minutes** |

A rerun that resumes from `results/_partial/` finishes in about a second, which
is what the checkpoints are for. An earlier version of this README quoted about
fifteen minutes; that was the `experiments` step alone, before curation and the
two-arm rerun existed.

## What it found on the shipped public data

All four endpoints and all three arms. Means over 5 model seeds, with an
interval that is the union of two things: the spread across the five seeds, and a
500-resample test-set bootstrap computed at every seed. Every value, and the
separability verdict under each interval separately, is in `results/`.

```
esol            RMSE       random                 scaffold
  morgan                   1.093 [0.863,1.321]    1.618 [1.394,1.794]
  descriptors              0.586 [0.477,0.740]    0.938 [0.803,1.062]
  combo                    0.584 [0.490,0.723]    0.928 [0.800,1.050]

tox21:NR-AR     ROC_AUC    random                 scaffold
  morgan                   0.801 [0.684,0.905]    0.731 [0.630,0.819]
  descriptors              0.764 [0.616,0.873]    0.746 [0.627,0.835]
  combo                    0.769 [0.628,0.874]    0.735 [0.643,0.837]

tox21:SR-MMP    ROC_AUC    random                 scaffold
  morgan                   0.871 [0.832,0.903]    0.763 [0.712,0.807]
  descriptors              0.930 [0.891,0.953]    0.842 [0.805,0.875]
  combo                    0.932 [0.898,0.957]    0.844 [0.813,0.873]

tox21:NR-AhR    ROC_AUC    random                 scaffold
  morgan                   0.891 [0.858,0.925]    0.797 [0.746,0.839]
  descriptors              0.906 [0.852,0.943]    0.841 [0.807,0.874]
  combo                    0.908 [0.860,0.943]    0.843 [0.806,0.877]
```

The split shift is the result that holds up. Scaffold splitting costs something
in all twelve endpoint-arm combinations: 0.018 to 0.108 ROC-AUC, and 48% to 60%
added RMSE on ESOL. That gap is the share of a reported accuracy that is
memorization of chemotypes, and it is why any ADMET number quoted without naming
its split is unreadable.

The direction holds everywhere but the magnitude does not, and the random and
scaffold intervals fail to overlap in seven of twelve combinations rather than all
twelve. The widest exceptions are the three NR-AR arms, which have the smallest
shift averaged over arms and much the widest intervals. On the test-set bootstrap
alone the count is nine of twelve; taking the union of both uncertainty sources
moves it to seven.

The ranking flips only where nothing is separable.

A single-seed run of this project originally reported that the winning arm
changed with the split on three of the four endpoints. Adding repeated seeds and
bootstrap intervals cut that to **one** of four, `tox21:NR-AR`. That endpoint is
one of three, with `tox21:SR-MMP` and `tox21:NR-AhR`, where no arm is separable
from the field under either split. Only on ESOL is an arm separable, and it is
`morgan`, distinguishably the worst of the three; that endpoint's winner does not
change at all.

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
dependency at all. It is pure tabular modeling. Copy the folder anywhere and
it runs.

Each project in the collection is meant to be redistributable on its own, as a
repository, a post, or an attachment, so nothing is imported across project
boundaries.

## This repository in a peer-reviewed submission

This project is one of three systems evaluated in "Gates That Can Fail: A
Reproducible Evaluation Discipline for Applied Machine-Learning Systems in Drug
Development" (Phani Kumar Balagam, sole author), under review at PeerJ Computer
Science as an AI Application article. It is also the primary artifact of a
companion manuscript that reports this study in full.

The technique that paper proposes is a nine-part evaluation discipline, not the
system in this repository. This project is its Case Study II: one of three
applications the discipline was evaluated on, and the source of two of the six
corrections the paper reports, a featurizer reordering withdrawn under repeated
seeding and an uncurated duplicate leak. The model choice here is deliberately
plain for that reason. Gradient-boosted trees on molecular featurizers were
chosen because the comparison is between representations, so an untuned tree with
identical hyperparameters across every arm holds the learner constant while the
featurizer varies. It is a baseline chosen so that the evaluation discipline, and
not model sophistication, is what is under test.

The submission carries a consolidated supplemental README covering all three
repositories against PeerJ's AI Application requirements: third-party dataset
sources, preprocessing, technique selection, computing infrastructure, evaluation
method and metrics, model-type justification, limitations, and code availability.
The other two systems are:

- `citation-binding-rag` (Case Study I) --- https://github.com/phanibalagam/citation-binding-rag --- doi:10.5281/zenodo.22541458
- `capa-recurrence-triage` (Case Study III) --- https://github.com/phanibalagam/capa-recurrence-triage --- doi:10.5281/zenodo.22541462

### Reproducing the numbers this repository reports

Python 3.11 or newer. No GPU. `run.py setup` needs network access to
raw.githubusercontent.com once, to download ESOL and Tox21; neither is
redistributed here.

```
git clone https://github.com/phanibalagam/admet-split-gauntlet
cd admet-split-gauntlet
pip install -r requirements.txt
./reproduce.sh
```

Or step by step:

```
python run.py setup        # downloads ESOL and Tox21
python run.py curate       # curation audit          -> results/curation.json
python run.py leakage      # duplicate leakage       -> results/dedup_stats.json
python run.py gauntlet     # one seed, point estimates
python run.py experiments  # five seeds + bootstrap  -> results/with_intervals.csv
python run.py rerun        # both curation arms      -> results/dedup_compare.csv
```

Measured step timings are in "Runtime, measured" above. Partial results are
checkpointed to `results/_partial/`, so an interrupted run resumes rather than
starting over. `results/environment.json` records the interpreter version, the
platform string and the resolved package versions alongside the results they
produced.

## License and provenance

Code in this directory: MIT (see `LICENSE`).

Data: see "What it found on the shipped public data" above for what is real and what is
generated, and under which license each part may be redistributed.

Originally candidate 06 in a fourteen-candidate portfolio assessment; renumbered
sequentially here because these three were the ones built.

## Disclaimer

This work was carried out independently, on personal time and equipment, and is
not connected to the author's employment. The views expressed are the author's own
and do not represent the views, positions or policies of any current, former or
future employer or client. No proprietary, confidential or internal data of any
organization was used.

All data is public: ESOL (Delaney) and three Tox21 assays. Neither is
redistributed here; `run.py setup` downloads both from the DeepChem repository,
whose MIT License covers that project's software. The data itself originates
with Delaney (2004) and with the NIH/NCATS Tox21 initiative, and those sources
govern its terms of use.
