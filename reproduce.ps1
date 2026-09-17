# Regenerate every number reported in METHODS.md, README.md and the manuscript.
# Everything lands in results/ as JSON and CSV. Nothing is retyped from a
# write-up: each file below is produced by the step above it.
#
# Runtime is dominated by `rerun`. See README.md for the measured figure.
# Partial results are checkpointed to results/_partial/, so an interrupted run
# resumes rather than starting over.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python --version
New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host ""; Write-Host ">>> run.py setup            (downloads ESOL and Tox21; not redistributed here)"
python run.py setup

Write-Host ""; Write-Host ">>> run.py curate           -> results/curation.json"
python run.py curate

Write-Host ""; Write-Host ">>> run.py leakage          -> results/dedup_stats.json"
python run.py leakage

Write-Host ""; Write-Host ">>> run.py gauntlet         -> out/gauntlet.csv, out/gate.json"
python run.py gauntlet

Write-Host ""; Write-Host ">>> run.py experiments      -> results/with_intervals.csv, gate.json"
python run.py experiments --seeds 5 --n-boot 500

Write-Host ""; Write-Host ">>> run.py rerun            -> results/dedup_compare.csv, claim_check.json"
python run.py rerun --seeds 5 --n-boot 500

Write-Host ""; Write-Host "Done. Every reported number is in results/."
