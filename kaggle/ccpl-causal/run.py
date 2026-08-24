import subprocess
import sys
from pathlib import Path


repo = Path("/kaggle/working/CCPL")
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/sciencebanda09/CCPL.git", str(repo)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo, check=True)
args = ["--exp", "E3", "--episodes", "1000", "--eval-episodes", "100",
        "--seeds", "5", "--max-steps", "100", "--delay", "5",
        "--out", "/kaggle/working/results"]
subprocess.run([sys.executable, "ccpl_experiments.py", *args], cwd=repo, check=True)
archive = Path("/kaggle/working/ccpl_causal_results.tar.gz")
subprocess.run(["tar", "-czf", str(archive), "-C", "/kaggle/working", "results"], check=True)
print(archive)
