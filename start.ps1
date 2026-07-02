# Youtube Comments Intelligence - launcher
#   .\start.ps1              fast start (default)
#   .\start.ps1 -Check       run dependency + smoke tests first
#   .\start.ps1 -Install     pip install then start
#
# First-time: copy .env.example to .env and add GROQ_API_KEY

param(
    [switch]$Check,
    [switch]$Install
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Load-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return }
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            Set-Item -Path "env:$($parts[0].Trim())" -Value $parts[1].Trim().Trim('"')
        }
    }
}

# venv (optional)
$venv = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venv) { . $venv }

# config
Load-DotEnv (Join-Path $ProjectRoot ".env")
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:RAG_GEN_MODEL = "groq/llama-3.1-8b-instant"
$env:RAG_EMBED_BACKEND = "sentence-transformers"
$env:ENRICHED_CSV_PATH = Join-Path $ProjectRoot "data\processed\comments_enriched.csv"

if ($Install) {
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    python -m pip install -r requirements.txt -q
}

if ($Check) {
    Write-Host "Running checks..." -ForegroundColor Cyan
    python scripts\check_deps.py
    if ($LASTEXITCODE -ne 0) { exit 1 }
    python scripts\final_smoke_test.py
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

if (-not (Test-Path $env:ENRICHED_CSV_PATH)) {
    Write-Host "WARNING: comments_enriched.csv not found. Analytics tab will be empty." -ForegroundColor Yellow
}

if (-not $env:GROQ_API_KEY) {
    Write-Host "NOTE: GROQ_API_KEY not set - Ask/Summarize need a key in .env" -ForegroundColor Yellow
}

Write-Host "Starting dashboard -> http://localhost:8501" -ForegroundColor Green
python -m streamlit run app\dashboard.py
