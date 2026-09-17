#!/usr/bin/env bash
# Regenerate every number reported in METHODS.md, README.md and the manuscript.
# Everything lands in results/ as JSON and CSV. Nothing is retyped from a
# write-up: each file below is produced by the step above it.
#
# Runtime is dominated by `rerun`, which fits three featurizers against two
# splits over five seeds on both curation arms, with a 500-resample bootstrap at
# every seed. See README.md for the measured figure. Partial results are
# checkpointed to results/_partial/, so an interrupted run resumes.
set -euo pipefail
cd "$(dirname "$0")"
echo "python: $(python3 --version)"
mkdir -p results

echo; echo ">>> run.py setup            (downloads ESOL and Tox21; not redistributed here)"
python3 run.py setup

echo; echo ">>> run.py curate           -> results/curation.json"
python3 run.py curate

echo; echo ">>> run.py leakage          -> results/dedup_stats.json"
python3 run.py leakage

echo; echo ">>> run.py gauntlet         -> out/gauntlet.csv, out/gate.json"
python3 run.py gauntlet

echo; echo ">>> run.py experiments      -> results/with_intervals.csv, gate.json"
python3 run.py experiments --seeds 5 --n-boot 500

echo; echo ">>> run.py rerun            -> results/dedup_compare.csv, claim_check.json,"
echo "                                   results/environment.json"
python3 run.py rerun --seeds 5 --n-boot 500

echo; echo "Done. Every reported number is in results/."
