"""Compatibility command for the maintained CCPL benchmark runner.

The former multi-phase implementation targeted removed v5/v6 classes and
could not execute: it referenced undefined builders, obsolete constructor
arguments, and result keys that no current agent produces.  Keep this filename
for existing scripts while routing supported options to ``run_benchmark_v7``.
"""

from __future__ import annotations

import argparse
import warnings
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ccpl"))
sys.path.insert(0, str(ROOT / "ccpl" / "algorithms"))
sys.path.insert(0, str(ROOT / "ccpl" / "environments"))

from run_benchmark_v7 import main as run_benchmark


def _build_forwarded_arguments(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--episodes", str(args.episodes),
        "--eval-eps", str(args.eval_episodes),
        "--max-steps", str(args.max_steps),
        "--delay", str(args.delay),
        "--seed", str(args.seed),
        "--seeds", str(args.seeds),
        "--out", args.out,
    ]
    if not args.quiet:
        forwarded.append("--verbose")
    if args.ablation:
        forwarded.append("--ablation")
    if args.safety:
        forwarded.append("--safety")
    return forwarded


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for run_benchmark_v7.py")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=str, default="results_v7")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--safety", action="store_true")

    # Parse obsolete flags so old shell scripts fail gracefully and visibly.
    parser.add_argument("--log-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--custom-only", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--mechanistic", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-phase", type=int, nargs="*", default=[],
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    obsolete_used = bool(
        args.log_dir or args.custom_only or args.mechanistic or args.no_phase)
    if obsolete_used:
        warnings.warn(
            "Legacy phase/log flags are no longer supported; use "
            "ccpl_experiments.py for individual experiments.",
            RuntimeWarning, stacklevel=2,
        )
    if args.quick:
        args.episodes = min(args.episodes, 100)
        args.eval_episodes = min(args.eval_episodes, 10)
        args.seeds = min(args.seeds, 1)

    return run_benchmark(_build_forwarded_arguments(args))


if __name__ == "__main__":
    main()
