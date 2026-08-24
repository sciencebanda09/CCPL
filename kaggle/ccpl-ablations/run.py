import subprocess
import sys
from pathlib import Path


repo = Path("/kaggle/working/CCPL")
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/sciencebanda09/CCPL.git", str(repo)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo, check=True)

common = ["--episodes", "2000", "--eval-episodes", "200", "--seeds", "10",
          "--max-steps", "100", "--delay", "5", "--out", "/kaggle/working/results"]
for experiment in ("E2", "E10"):
    subprocess.run([sys.executable, "ccpl_experiments.py", "--exp", experiment,
                    *common], cwd=repo, check=True)

archive = Path("/kaggle/working/ccpl_ablation_results.tar.gz")
subprocess.run(["tar", "-czf", str(archive), "-C", "/kaggle/working", "results"], check=True)
print(archive)
