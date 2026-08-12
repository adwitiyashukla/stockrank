from __future__ import annotations

from pathlib import Path

import typer

from stockrank.config import load_config
from stockrank.utils.logging import get_logger, setup_logging

app = typer.Typer(add_completion=False, help="Equity alpha research engine")
logger = get_logger("cli")


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    smoke: bool = typer.Option(False, "--smoke", help="tiny run for CI"),
    report: bool = typer.Option(True, "--report/--no-report", help="write figures and RESULTS.md"),
) -> None:
    setup_logging()
    from stockrank.experiment import run_experiment

    cfg = load_config(config)
    res = run_experiment(cfg, smoke=smoke)

    if report:
        from stockrank.reporting import build_report

        build_report(res)
    typer.echo(f"\nArtifacts written to {res.dir}")


@app.command("rebacktest")
def rebacktest(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    report: bool = typer.Option(True, "--report/--no-report"),
) -> None:
    setup_logging()
    from stockrank.experiment import rerun_from_predictions

    cfg = load_config(config)
    res = rerun_from_predictions(cfg)
    if report:
        from stockrank.reporting import build_report

        build_report(res)
    typer.echo(f"\nArtifacts refreshed in {res.dir}")


@app.command()
def fetch(config: str = typer.Option("configs/default.yaml", "--config", "-c")) -> None:
    setup_logging()
    from stockrank.data.loader import load_market_data

    cfg = load_config(config)
    data = load_market_data(cfg)
    typer.echo(str(data.summary()))


@app.command()
def report(config: str = typer.Option("configs/default.yaml", "--config", "-c")) -> None:
    setup_logging()
    from stockrank.reporting import build_report_from_disk

    cfg = load_config(config)
    path = build_report_from_disk(cfg)
    typer.echo(f"Report written to {path}")


@app.command()
def models() -> None:
    from stockrank.models.registry import available_models

    typer.echo("\n".join(available_models()))


@app.command("list-runs")
def list_runs(artifacts: str = typer.Option("artifacts", "--artifacts")) -> None:
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
            "  python -m stockrank.cli run --config configs/default.yaml"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    app()
