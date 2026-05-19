"""Live training dashboard for the S3 detector.

Tails `live.json` written by `train_s3.train()` and re-renders every 0.5 s.
Run in a second terminal alongside training:

    uv run python -m openstetho_model.dash_s3 --run runs/s3_circor_v1

Quits cleanly on Ctrl-C. Will wait politely if the run directory does not
exist yet (so you can launch the dashboard before the trainer).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _render(live_json: dict | None, history: list[dict], run_dir: Path) -> Panel:
    if live_json is None:
        body = Text("Waiting for training to start…", style="dim")
        return Panel(body, title=f"S3 dash · {run_dir}", border_style="yellow")

    epoch = live_json["epoch"]
    epochs = live_json["epochs"]
    bn = live_json["batch_no"]
    tb = live_json["total_batches"]
    pct = bn / max(tb, 1)
    bar = ProgressBar(total=tb, completed=bn, width=40)

    head = Table.grid(padding=(0, 2))
    head.add_column(justify="right")
    head.add_column()
    head.add_row("phase",        live_json.get("phase", "?"))
    head.add_row("epoch",        f"{epoch}/{epochs}")
    head.add_row("batch",        f"{bn}/{tb}  ({pct*100:.1f}%)")
    head.add_row("seen",         f"{live_json['seen']:,}")
    head.add_row("items/sec",    f"{live_json['items_per_sec']:.1f}")
    head.add_row("ETA epoch",    _fmt_secs(live_json["eta_s"]))
    head.add_row("loss (running)", f"{live_json['running_loss']:.4f}")
    ema = live_json.get("ema_loss")
    head.add_row("loss (ema)",   f"{ema:.4f}" if ema is not None else "—")
    best = live_json.get("best_auprc")
    head.add_row("best AUPRC",   f"{best:.3f}" if best else "—")

    history_table = Table(title="Epoch metrics (validation)", show_header=True, header_style="bold cyan")
    for col in ("epoch", "train_loss", "val_loss", "auroc", "auprc", "f1@0.5", "ece"):
        history_table.add_column(col, justify="right")
    for row in history[-10:]:
        history_table.add_row(
            row["epoch"],
            f"{float(row['train_loss']):.4f}",
            f"{float(row['val_loss']):.4f}",
            f"{float(row['auroc']):.3f}",
            f"{float(row['auprc']):.3f}",
            f"{float(row['f1_at_0_5']):.3f}",
            f"{float(row['ece']):.3f}",
        )

    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_row(head)
    grid.add_row(bar)
    grid.add_row(history_table)
    return Panel(grid, title=f"S3 dash · {run_dir}", border_style="cyan")


def _read_history(metrics_csv: Path) -> list[dict]:
    if not metrics_csv.exists():
        return []
    with metrics_csv.open() as f:
        return list(csv.DictReader(f))


def _read_live(live_json: Path) -> dict | None:
    if not live_json.exists():
        return None
    try:
        return json.loads(live_json.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="training output directory")
    p.add_argument("--interval", type=float, default=0.5)
    args = p.parse_args()

    console = Console()
    live_json = args.run / "live.json"
    metrics_csv = args.run / "metrics.csv"

    with Live(_render(None, [], args.run), refresh_per_second=4, console=console) as live:
        try:
            while True:
                data = _read_live(live_json)
                history = _read_history(metrics_csv)
                live.update(_render(data, history, args.run))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
