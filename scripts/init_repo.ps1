# Build a readable commit history instead of one opaque "initial commit".
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts/init_repo.ps1

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) { git init -b main }

function Commit($message, [string[]]$paths) {
    $existing = @($paths | Where-Object { Test-Path $_ })
    if ($existing.Count -eq 0) { Write-Host "skip: $message"; return }
    git add -- $existing
    $staged = git diff --cached --name-only
    if (-not $staged) { Write-Host "nothing staged: $message"; return }
    git commit -q -m $message
    Write-Host "committed: $message"
}

Commit "chore: project scaffolding, packaging, linting and CI" `
    @(".gitignore", ".gitattributes", ".dockerignore", "LICENSE", "pyproject.toml",
      "requirements.txt", "Makefile", ".github/workflows/ci.yml",
      "src/alpha_engine/__init__.py", "src/alpha_engine/config.py", "src/alpha_engine/utils")

Commit "feat(data): point-in-time S&P 500 universe and resumable market data ingestion" `
    @("src/alpha_engine/data/__init__.py", "src/alpha_engine/data/universe.py",
      "src/alpha_engine/data/providers.py", "src/alpha_engine/data/factors.py",
      "src/alpha_engine/data/loader.py", "scripts/fetch_data.py")

Commit "feat(data): synthetic market simulator with plantable alpha for leakage control" `
    @("src/alpha_engine/data/simulator.py")

Commit "feat(features): 36-factor library, cross-sectional normalisation and labelling" `
    @("src/alpha_engine/features")

Commit "feat(validation): purged walk-forward and purged k-fold splitters with embargo" `
    @("src/alpha_engine/validation")

Commit "feat(models): linear, gradient boosting, sequence models and a zero-parameter factor benchmark" `
    @("src/alpha_engine/models")

Commit "feat(portfolio): construction with simultaneous dollar, beta and sector neutrality" `
    @("src/alpha_engine/portfolio")

Commit "feat(backtest): daily engine with commission, slippage, borrow costs and a turnover cap" `
    @("src/alpha_engine/backtest")

Commit "feat(evaluation): IC, deflated Sharpe, PBO, bootstrap and Fama-French attribution" `
    @("src/alpha_engine/evaluation", "src/alpha_engine/explain")

Commit "feat: end-to-end experiment orchestration, CLI and report generation" `
    @("src/alpha_engine/experiment.py", "src/alpha_engine/cli.py",
      "src/alpha_engine/reporting", "configs", "scripts/factor_diagnostics.py")

Commit "feat(api): FastAPI scoring and screening service" `
    @("src/alpha_engine/api")

Commit "feat(dashboard): Streamlit research console" `
    @("dashboard", ".streamlit")

Commit "test: leakage guards, null-alpha control and accounting invariants" `
    @("tests")

Commit "docs: methodology, README and research walkthrough notebook" `
    @("README.md", "docs", "notebooks")

Commit "chore: docker image, compose stack and demo bundling" `
    @("docker", "scripts/prepare_demo.py", "scripts/init_repo.ps1")

Commit "docs(results): committed run artifacts, figures and results report" `
    @("demo_artifacts", "reports")

Write-Host ""
Write-Host "--- history ---"
git --no-pager log --oneline
Write-Host ""
Write-Host "Untracked files remaining (should be data caches and logs only):"
git status --porcelain --untracked-files=normal | Select-Object -First 20
