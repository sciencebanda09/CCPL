import os
import subprocess
import sys
from pathlib import Path


os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
repo = Path("/kaggle/working/CCPL")
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/sciencebanda09/CCPL.git", str(repo)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade",
                "numpy<2", "gymnasium==0.28.1", "mujoco==3.1.6",
                "safety-gymnasium==1.0.0", "xmltodict", "pygame==2.6.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo, check=True)
common = ["--episodes", "1000", "--eval-episodes", "100", "--seeds", "5",
          "--max-steps", "100", "--delay", "5", "--out", "/kaggle/working/results"]
for experiment in ("E8", "E9"):
    subprocess.run([sys.executable, "ccpl_experiments.py", "--exp", experiment,
                    *common], cwd=repo, check=True)
archive = Path("/kaggle/working/ccpl_safety_results.tar.gz")
subprocess.run(["tar", "-czf", str(archive), "-C", "/kaggle/working", "results"], check=True)
print(archive)
