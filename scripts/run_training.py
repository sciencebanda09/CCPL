"""Canonical training entry point preserved from the V7 runner."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "run_ccpl.py"), run_name="__main__")
