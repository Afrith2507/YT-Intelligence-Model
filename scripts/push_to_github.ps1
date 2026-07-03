# Commit and push to GitHub (excludes large data via .gitignore)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "=== Smoke test ===" -ForegroundColor Cyan
python scripts/final_smoke_test.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Untrack large data (if previously committed) ===" -ForegroundColor Cyan
git rm -r --cached data/raw/ 2>$null
git rm -r --cached data/processed/ 2>$null
git rm -r --cached mlruns/ 2>$null

Write-Host "`n=== Stage changes ===" -ForegroundColor Cyan
git add .
git status

$msg = @"
Final submission: Youtube Comments Intelligence dashboard and pipeline.

Minimalist UI with light/dark mode, Groq RAG generation, and MLflow eval.
"@

Write-Host "`n=== Commit ===" -ForegroundColor Cyan
git commit -m $msg
if ($LASTEXITCODE -ne 0) {
    Write-Host "Nothing to commit or commit failed." -ForegroundColor Yellow
}

Write-Host "`n=== Push to origin/main ===" -ForegroundColor Cyan
git push -u origin main

Write-Host "`nDone." -ForegroundColor Green
