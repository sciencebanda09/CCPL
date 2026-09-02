import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


os.environ.update({
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
})

repo = Path(os.environ.get("CCPL_SOURCE", "/kaggle/input/ccpl-source"))
results = Path("/kaggle/working/ccpl_e8_e9_results")
episodes = os.environ.get("CCPL_EPISODES", "500")
eval_episodes = os.environ.get("CCPL_EVAL_EPISODES", "20")
seed_values = os.environ.get("CCPL_SEEDS", "42,43,44,45,46")
max_steps = os.environ.get("CCPL_MAX_STEPS", "500")
delay = os.environ.get("CCPL_DELAY", "5")
delay_mode = os.environ.get("CCPL_DELAY_MODE", "fixed")
timeout = int(os.environ.get("CCPL_TIMEOUT", "3600"))
tasks = os.environ.get("CCPL_TASKS", "SafetyPointGoal1,SafetyPointGoal2")
experiments = [name.strip() for name in os.environ.get("CCPL_EXPERIMENTS", "E8").split(",")
               if name.strip()]
policies = ["CCPL", "CPO-FO", "PPO", "SAC-Lag", "CCPL-Base"]


def install():
    global repo
    if not (repo / "ccpl_experiments.py").exists():
        archive = repo / "ccpl.tar"
        if not archive.exists():
            raise FileNotFoundError(f"CCPL dataset source is missing: {repo}")
        extracted = Path("/kaggle/working/CCPL")
        extracted.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as handle:
            handle.extractall(extracted)
        shutil.copy2(repo / "ccpl_experiments.py", extracted / "ccpl_experiments.py")
        shutil.copy2(repo / "pyproject.toml", extracted / "pyproject.toml")
        shutil.copy2(repo / "README.md", extracted / "README.md")
        repo = extracted
    if not (repo / "ccpl_experiments.py").exists():
        raise FileNotFoundError(f"CCPL dataset source is missing: {repo}")
    packages = [
        "numpy==1.26.4",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "gymnasium==0.28.1",
        "mujoco==3.1.6",
        "xmltodict",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", *packages], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                    "pygame==2.6.1", "gymnasium-robotics==1.2.2",
                    "safety-gymnasium==1.0.0"], check=True)
    # The experiment runner is executed from the extracted source tree.
    # Avoid an editable build because the Kaggle Python image has a separate
    # preinstalled build stack that is incompatible with this legacy layout.


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
    revision = hashlib.sha256((repo / "ccpl_experiments.py").read_bytes()).hexdigest()
    seeds = [int(value.strip()) for value in seed_values.split(",") if value.strip()]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = "/kaggle/working:/kaggle/working/CCPL"
    help_text = subprocess.check_output(
        [sys.executable, "ccpl_experiments.py", "--help"],
        cwd=repo, env=child_env, text=True
    )
    common = ["--episodes", episodes, "--eval-episodes", eval_episodes,
              "--seeds", str(len(seeds)), "--max-steps", max_steps,
              "--delay", delay, "--delay-mode", delay_mode,
              "--tasks", tasks, "--out", str(results)]
    if "--seed-values" in help_text:
        common += ["--seed-values", ",".join(str(seed) for seed in seeds)]

    protocol = {
        "source": str(repo),
        "source_sha256_ccpl_experiments": revision,
        "experiments": experiments,
        "tasks": [name.strip() for name in tasks.split(",") if name.strip()],
        "policies": policies,
        "episodes": int(episodes),
        "evaluation_episodes": int(eval_episodes),
        "seeds": seeds,
        "max_steps": int(max_steps),
        "delay": int(delay),
        "delay_mode": delay_mode,
        "dependencies": {
            "gymnasium": "0.28.1",
            "mujoco": "3.1.6",
            "safety_gymnasium": "1.0.0",
        },
    }
    (results / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    statuses = {}
    for name in experiments:
        print(f"\n===== {name} =====")
        statuses[name] = run_experiment(name, common, child_env)
        (results / "status.json").write_text(
            json.dumps({"success": all(item.get("status") == "passed"
                                        for item in statuses.values()),
                        "protocol": protocol, "experiments": statuses}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({name: statuses[name]}, indent=2))

    archive = Path("/kaggle/working/ccpl_e8_e9_results.tar.gz")
    subprocess.run(["tar", "-czf", str(archive), "-C", str(results.parent), results.name], check=True)
    print(json.dumps({"results": str(results), "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
