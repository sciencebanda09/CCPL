"""Canonical evaluation entry point for the research package."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "run_benchmark_v7.py"), run_name="__main__")
