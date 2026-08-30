import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path


os.environ.update({
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
})

repo = Path("/kaggle/working/CCPL")
results = Path("/kaggle/working/ccpl_e8_e9_results")
episodes = os.environ.get("CCPL_EPISODES", "100")
eval_episodes = os.environ.get("CCPL_EVAL_EPISODES", "20")
seed_values = os.environ.get("CCPL_SEEDS", "42,43,44")
max_steps = os.environ.get("CCPL_MAX_STEPS", "100")
delay = os.environ.get("CCPL_DELAY", "5")
timeout = int(os.environ.get("CCPL_TIMEOUT", "900"))


def install():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/sciencebanda09/CCPL.git", str(repo)], check=True)
    packages = [
        "numpy==1.26.4",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "gymnasium==0.28.1",
        "mujoco==2.3.3",
        "xmltodict",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", *packages], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                    "pygame==2.6.1", "gymnasium-robotics==1.2.2",
                    "safety-gymnasium==1.0.0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", "."],
                   cwd=repo, check=True)


def write_sitecustomize():
    path = Path("/kaggle/working/sitecustomize.py")
    path.write_text(
        """import dataclasses\nimport inspect\nimport numpy as np\n_source = inspect.getsource(dataclasses._get_field)\n_source = _source.replace(\n    \"if f._field_type is _FIELD and f.default.__class__.__hash__ is None:\",\n    \"if (f._field_type is _FIELD and f.default.__class__.__hash__ is None \"\n    \"and not isinstance(f.default, np.ndarray)):\"\n)\n_namespace = dataclasses.__dict__.copy()\n_namespace[\"np\"] = np\nexec(_source, _namespace)\ndataclasses._get_field = _namespace[\"_get_field\"]\n""",
        encoding="utf-8",
    )


def run_experiment(name, common, env):
    log_path = results / f"{name}.log"
    started = time.time()
    command = [sys.executable, "-u", "ccpl_experiments.py", "--exp", name, *common]
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=repo, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            return_code = process.wait(timeout=timeout)
        status = "passed" if return_code == 0 else "failed"
        return {"status": status, "return_code": return_code,
                "seconds": round(time.time() - started, 2), "log": str(log_path)}
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return {"status": "timeout", "seconds": round(time.time() - started, 2),
                "log": str(log_path)}


def main():
    results.mkdir(parents=True, exist_ok=True)
    install()
    write_sitecustomize()
    seeds = [int(value.strip()) for value in seed_values.split(",") if value.strip()]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = "/kaggle/working:/kaggle/working/CCPL"
    help_text = subprocess.check_output(
        [sys.executable, "ccpl_experiments.py", "--help"], cwd=repo, text=True
    )
    common = ["--episodes", episodes, "--eval-episodes", eval_episodes,
              "--seeds", str(len(seeds)), "--max-steps", max_steps,
              "--delay", delay, "--out", str(results)]
    if "--seed-values" in help_text:
        common += ["--seed-values", ",".join(str(seed) for seed in seeds)]

    protocol = {
        "experiments": ["E8", "E9"],
        "episodes": int(episodes),
        "evaluation_episodes": int(eval_episodes),
        "seeds": seeds,
        "max_steps": int(max_steps),
        "delay": int(delay),
        "dependencies": {
            "gymnasium": "0.28.1",
            "mujoco": "2.3.3",
            "safety_gymnasium": "1.0.0",
        },
    }
    (results / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    statuses = {}
    for name in ("E8", "E9"):
        print(f"\n===== {name} =====")
        statuses[name] = run_experiment(name, common, child_env)
        (results / "status.json").write_text(
            json.dumps({"protocol": protocol, "experiments": statuses}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({name: statuses[name]}, indent=2))

    archive = Path("/kaggle/working/ccpl_e8_e9_results.tar.gz")
    subprocess.run(["tar", "-czf", str(archive), "-C", str(results.parent), results.name], check=True)
    print(json.dumps({"results": str(results), "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
