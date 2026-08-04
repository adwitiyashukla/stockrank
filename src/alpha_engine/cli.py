"""Command line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from alpha_engine.config import load_config
from alpha_engine.utils.logging import get_logger, setup_logging

app = typer.Typer(add_completion=False, help="Equity alpha research engine")
logger = get_logger("cli")


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    smoke: bool = typer.Option(False, "--smoke", help="tiny run for CI"),
    report: bool = typer.Option(True, "--report/--no-report", help="write figures and RESULTS.md"),
) -> None:
    """Run the full research pipeline."""
    setup_logging()
    from alpha_engine.experiment import run_experiment

    cfg = load_config(config)
    res = run_experiment(cfg, smoke=smoke)

    if report:
        from alpha_engine.reporting import build_report

        build_report(res)
    typer.echo(f"\nArtifacts written to {res.dir}")


@app.command("rebacktest")
def rebacktest(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    report: bool = typer.Option(True, "--report/--no-report"),
) -> None:
    """Re-run backtest, evaluation and reporting from cached predictions.

    Portfolio and cost assumptions do not affect the forecasts, so changing
    leverage, the volatility target or the cost model does not need a refit.
    """
    setup_logging()
    from alpha_engine.experiment import rerun_from_predictions

    cfg = load_config(config)
    res = rerun_from_predictions(cfg)
    if report:
        from alpha_engine.reporting import build_report

        build_report(res)
    typer.echo(f"\nArtifacts refreshed in {res.dir}")


@app.command()
def fetch(config: str = typer.Option("configs/default.yaml", "--config", "-c")) -> None:
    """Download and cache market data only."""
    setup_logging()
    from alpha_engine.data.loader import load_market_data

    cfg = load_config(config)
    data = load_market_data(cfg)
    typer.echo(str(data.summary()))


@app.command()
def report(config: str = typer.Option("configs/default.yaml", "--config", "-c")) -> None:
    """Rebuild figures and RESULTS.md from artifacts already on disk."""
    setup_logging()
    from alpha_engine.reporting import build_report_from_disk

    cfg = load_config(config)
    path = build_report_from_disk(cfg)
    typer.echo(f"Report written to {path}")


@app.command()
def models() -> None:
    """List available models."""
    from alpha_engine.models.registry import available_models

    typer.echo("\n".join(available_models()))


@app.command("list-runs")
def list_runs(artifacts: str = typer.Option("artifacts", "--artifacts")) -> None:
    """Show completed runs, including the demo bundle shipped with the repository."""
    roots = [Path(artifacts), Path("demo_artifacts")]
    found = False
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            complete = (d / "model_metrics.csv").exists()
            label = f"{root.name}/{d.name}"
            typer.echo(f"{label:<32} {'ok' if complete else 'incomplete'}")
            found = True
    if not found:
        typer.echo(
            "No runs found yet. Generate one with:\n"
            "  python scripts/fetch_data.py --config configs/default.yaml\n"
            "  python -m alpha_engine.cli run --config configs/default.yaml"
        )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
