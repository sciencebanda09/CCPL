"""Run the CCPL experiment suite sequentially with bounded failure handling."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path("/kaggle/working/CCPL")
RESULTS = Path("/kaggle/working/ccpl_all_results")
REPO_URL = os.environ.get("CCPL_REPO", "https://github.com/sciencebanda09/CCPL.git")
EPISODES = os.environ.get("CCPL_EPISODES", "100")
EVAL_EPISODES = os.environ.get("CCPL_EVAL_EPISODES", "20")
SEED_VALUES = os.environ.get("CCPL_SEEDS", "42,43,44")
MAX_STEPS = os.environ.get("CCPL_MAX_STEPS", "50")
DELAY = os.environ.get("CCPL_DELAY", "5")
TIMEOUT = int(os.environ.get("CCPL_EXPERIMENT_TIMEOUT", "900"))


def run_logged(command, *, cwd=None, log_path=None, timeout=None):
    started = time.time()
    with log_path.open("w", encoding="utf-8") if log_path else open(os.devnull, "w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return {"status": "timeout", "seconds": round(time.time() - started, 2)}
    return {
        "status": "passed" if return_code == 0 else "failed",
        "return_code": return_code,
        "seconds": round(time.time() - started, 2),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()

    # Keep the numerical stack below NumPy 2 for the repository's tested path.
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy<2"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", "."], cwd=REPO, check=True)

    seeds = [int(value.strip()) for value in SEED_VALUES.split(",") if value.strip()]
    common = [
        "--episodes", EPISODES,
        "--eval-episodes", EVAL_EPISODES,
        "--seeds", str(len(seeds)),
        "--max-steps", MAX_STEPS,
        "--delay", DELAY,
        "--out", str(RESULTS),
    ]
    help_text = subprocess.check_output(
        [sys.executable, "ccpl_experiments.py", "--help"], cwd=REPO, text=True
    )
    if "--seed-values" in help_text:
        common += ["--seed-values", ",".join(str(seed) for seed in seeds)]

    protocol = {
        "repository": REPO_URL,
        "revision": revision,
        "experiments": [f"E{i}" for i in range(1, 11)],
        "episodes": int(EPISODES),
        "evaluation_episodes": int(EVAL_EPISODES),
        "seeds": seeds,
        "max_steps": int(MAX_STEPS),
        "delay": int(DELAY),
        "timeout_seconds_per_experiment": TIMEOUT,
        "note": "Sequential bounded run. E8/E9 are attempted after the base suite and may be unavailable on Kaggle Python 3.12.",
    }
    (RESULTS / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    statuses = {}
    for experiment in [f"E{i}" for i in range(1, 11)]:
        print(f"\n===== {experiment} =====")
        log_path = RESULTS / f"{experiment}.log"
        statuses[experiment] = run_logged(
            [sys.executable, "ccpl_experiments.py", "--exp", experiment, *common],
            cwd=REPO,
            log_path=log_path,
            timeout=TIMEOUT,
        )
        statuses[experiment]["log"] = str(log_path)
        (RESULTS / "status.json").write_text(
            json.dumps({"protocol": protocol, "experiments": statuses}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({experiment: statuses[experiment]}, indent=2))

    archive = Path("/kaggle/working/ccpl_all_results.tar.gz")
    subprocess.run(["tar", "-czf", str(archive), "-C", str(RESULTS.parent), RESULTS.name], check=True)
    print(json.dumps({"revision": revision, "results": str(RESULTS), "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
