# Regenerate every number reported in METHODS.md and README.md.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python --version
New-Item -ItemType Directory -Force -Path results | Out-Null
Write-Host ""; Write-Host ">>> run.py setup"
python run.py setup
Write-Host ""; Write-Host ">>> run.py gauntlet"
python run.py gauntlet
Write-Host ""; Write-Host ">>> run.py experiments --seeds 5 --n-boot 500"
python run.py experiments --seeds 5 --n-boot 500
Write-Host ""; Write-Host "Done. Reported numbers are in results/."
