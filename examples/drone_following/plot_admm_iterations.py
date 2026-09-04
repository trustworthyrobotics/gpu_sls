#!/usr/bin/env python3
"""
Parse GPU-SLS debug.log output and plot ADMM iteration counts across all solves.

The parser recognizes lines such as:

    ADMM done: Total Iterations=99 converged=False rho=...

and, when present, automatically detects the initialization schedule from:

    First MPC solve: 50 nominal SQP + 50 SLS SQP iterations

The resulting plot separates:
  1. initial nominal SQP solves,
  2. initial SLS SQP solves,
  3. subsequent 1-SQP RTI solves.

Usage:
    python plot_admm_iterations.py debug.log

Optional:
    python plot_admm_iterations.py debug.log \
        --output admm_iterations.png \
        --csv admm_iterations.csv \
        --show
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ADMM_RE = re.compile(
    r"ADMM done:\s*"
    r"Total Iterations=(?P<iterations>\d+)\s+"
    r"converged=(?P<converged>True|False)"
    r"(?:\s+rho=(?P<rho>[+\-0-9.eE]+))?"
    r"(?:\s+rp=(?P<rp>[+\-0-9.eE]+))?"
    r".*?"
    r"(?:\s+rd=(?P<rd>[+\-0-9.eE]+))?"
)

INIT_RE = re.compile(
    r"First MPC solve:\s*"
    r"(?P<nominal>\d+)\s+nominal SQP\s*\+\s*"
    r"(?P<sls>\d+)\s+SLS SQP iterations",
    re.IGNORECASE,
)


@dataclass
class ADMMRecord:
    call: int
    iterations: int
    converged: bool
    rho: float | None
    rp: float | None
    rd: float | None
    phase: str
    phase_index: int


def to_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def parse_log(path: Path) -> tuple[list[ADMMRecord], int | None, int | None]:
    text = path.read_text(errors="replace")

    # Detect the initialization split, if the log reports it.
    init_match = INIT_RE.search(text)
    nominal_iters = int(init_match.group("nominal")) if init_match else None
    sls_iters = int(init_match.group("sls")) if init_match else None

    raw = []
    for match in ADMM_RE.finditer(text):
        raw.append(
            {
                "iterations": int(match.group("iterations")),
                "converged": match.group("converged") == "True",
                "rho": to_float(match.group("rho")),
                "rp": to_float(match.group("rp")),
                "rd": to_float(match.group("rd")),
            }
        )

    records: list[ADMMRecord] = []
    for i, item in enumerate(raw, start=1):
        if nominal_iters is not None and sls_iters is not None:
            if i <= nominal_iters:
                phase = "Initial nominal"
                phase_index = i - 1
            elif i <= nominal_iters + sls_iters:
                phase = "Initial SLS"
                phase_index = i - nominal_iters - 1
            else:
                phase = "RTI"
                # MPC 000 is the initial full solve, so the first post-init
                # ADMM call corresponds to MPC 001.
                phase_index = i - (nominal_iters + sls_iters)
        else:
            phase = "All solves"
            phase_index = i - 1

        records.append(
            ADMMRecord(
                call=i,
                iterations=item["iterations"],
                converged=item["converged"],
                rho=item["rho"],
                rp=item["rp"],
                rd=item["rd"],
                phase=phase,
                phase_index=phase_index,
            )
        )

    return records, nominal_iters, sls_iters


def write_csv(records: list[ADMMRecord], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "admm_call",
                "phase",
                "phase_index",
                "iterations",
                "converged",
                "rho",
                "rp",
                "rd",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.call,
                    r.phase,
                    r.phase_index,
                    r.iterations,
                    r.converged,
                    r.rho,
                    r.rp,
                    r.rd,
                ]
            )


def make_plot(
    records: list[ADMMRecord],
    nominal_iters: int | None,
    sls_iters: int | None,
    output: Path,
    show: bool,
) -> None:
    if not records:
        raise RuntimeError("No 'ADMM done:' lines were found in the log.")

    x = [r.call for r in records]
    y = [r.iterations for r in records]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, y, marker="o", markersize=3, linewidth=1)

    # Overlay non-converged solves with an x marker.
    bad_x = [r.call for r in records if not r.converged]
    bad_y = [r.iterations for r in records if not r.converged]
    if bad_x:
        ax.scatter(
            bad_x,
            bad_y,
            marker="x",
            s=45,
            linewidths=1.5,
            label="ADMM did not converge",
        )

    if nominal_iters is not None and sls_iters is not None:
        init_total = nominal_iters + sls_iters

        # Boundaries are placed between integer-valued ADMM calls.
        ax.axvline(nominal_iters + 0.5, linestyle="--", linewidth=1)
        ax.axvline(init_total + 0.5, linestyle="--", linewidth=1)

        ymax = max(y)
        text_y = ymax * 0.96 if ymax > 0 else 1.0

        ax.text(
            max(1, nominal_iters / 2),
            text_y,
            "Initial nominal",
            ha="center",
            va="top",
        )
        ax.text(
            nominal_iters + max(1, sls_iters / 2),
            text_y,
            "Initial SLS",
            ha="center",
            va="top",
        )
        if len(records) > init_total:
            ax.text(
                init_total + max(1, (len(records) - init_total) / 2),
                text_y,
                "RTI",
                ha="center",
                va="top",
            )

    ax.set_xlabel("Sequential ADMM solve")
    ax.set_ylabel("ADMM iterations")
    ax.set_title("ADMM Iterations Throughout GPU-SLS Solves")
    ax.grid(True, alpha=0.3)

    if bad_x:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse debug.log and plot ADMM iteration counts."
    )
    parser.add_argument(
        "log",
        nargs="?",
        default="debug.log",
        type=Path,
        help="Path to debug.log (default: debug.log)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG path (default: <log_stem>_admm_iterations.png)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the plot interactively",
    )
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"Log file not found: {args.log}")

    output = (
        args.output
        if args.output is not None
        else args.log.with_name(f"{args.log.stem}_admm_iterations.png")
    )

    records, nominal_iters, sls_iters = parse_log(args.log)

    if not records:
        raise RuntimeError(
            f"No ADMM records found in {args.log}. "
            "Expected lines containing 'ADMM done: Total Iterations=...'."
        )

    make_plot(records, nominal_iters, sls_iters, output, args.show)

    if args.csv is not None:
        write_csv(records, args.csv)

    converged = sum(r.converged for r in records)
    not_converged = len(records) - converged

    print(f"Parsed {len(records)} ADMM solves from: {args.log}")
    if nominal_iters is not None and sls_iters is not None:
        init_total = nominal_iters + sls_iters
        rti_count = max(0, len(records) - init_total)
        print(
            f"Detected phases: nominal={nominal_iters}, "
            f"SLS={sls_iters}, RTI={rti_count}"
        )
    print(f"Converged: {converged}")
    print(f"Not converged: {not_converged}")
    print(f"Saved plot: {output}")
    if args.csv is not None:
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
