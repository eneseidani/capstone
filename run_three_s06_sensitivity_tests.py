"""
Three additional S06 sensitivity experiments (baseline 0.60/20% is already complete).

Keep this file next to S06_three_strategy_threshold065_cap25.py
in the D:/Capstone project root. It runs the same engine three times, each
in a separate Python process with its own output folder.

Run:
    python run_three_s06_sensitivity_tests.py

It never runs the existing baseline or overwrites its output.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "S06_three_strategy_threshold065_cap25.py"
OUTPUT_ROOT = ROOT / "portfolio_results"

# Baseline A (0.60, 20%) is deliberately omitted.
EXPERIMENTS = (
    ("B_threshold065_cap20", "0.65", "0.20"),
    ("C_threshold060_cap25", "0.60", "0.25"),
    ("D_threshold065_cap25", "0.65", "0.25"),
)

def main() -> None:
    if not ENGINE.is_file():
        raise FileNotFoundError(
            f"Missing allocation engine: {ENGINE}. "
            "Place the two Python files together in the Capstone project root."
        )

    for name, threshold, cap in EXPERIMENTS:
        out_dir = OUTPUT_ROOT / name

        # Prevent destroying an earlier completed run by accident.
        if out_dir.exists():
            raise FileExistsError(
                f"Experiment output folder already exists: {out_dir}\n"
                "Move or rename it first, rather than overwriting existing results."
            )

        command = [
            sys.executable, str(ENGINE),
            "--threshold", threshold,
            "--max-weight", cap,
            "--out", str(out_dir),
        ]
        print(
            f"\nRUNNING {name}: selection threshold={threshold}, "
            f"maximum allocation={float(cap):.0%}", flush=True
        )
        subprocess.run(command, cwd=ROOT, check=True)
        print(f"COMPLETED: {out_dir}", flush=True)

    print("\nAll three additional experiments completed.", flush=True)

if __name__ == "__main__":
    main()
