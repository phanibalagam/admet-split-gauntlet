#!/usr/bin/env bash
# Regenerate every number reported in METHODS.md and README.md.
# Results are written to results/ as JSON and CSV.
set -euo pipefail
cd "$(dirname "$0")"
echo "python: $(python3 --version)"
mkdir -p results
echo
echo ">>> run.py setup"
python3 run.py setup
echo
echo ">>> run.py gauntlet"
python3 run.py gauntlet
echo
echo ">>> run.py experiments --seeds 5 --n-boot 500"
python3 run.py experiments --seeds 5 --n-boot 500
echo
echo "Done. Reported numbers are in results/."
