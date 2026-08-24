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
                "xmltodict"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                "gymnasium-robotics==1.2.2", "safety-gymnasium==1.0.0"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo, check=True)
Path("/kaggle/working/sitecustomize.py").write_text(
    """import dataclasses, inspect, numpy as np
_source = inspect.getsource(dataclasses._get_field)
_source = _source.replace(
    "if f._field_type is _FIELD and f.default.__class__.__hash__ is None:",
    "if (f._field_type is _FIELD and f.default.__class__.__hash__ is None "
    "and not isinstance(f.default, np.ndarray)):"
)
_namespace = dataclasses.__dict__.copy()
_namespace["np"] = np
exec(_source, _namespace)
dataclasses._get_field = _namespace["_get_field"]
""",
    encoding="utf-8")
child_env = os.environ.copy()
child_env["PYTHONPATH"] = "/kaggle/working:/kaggle/working/CCPL"
common = ["--episodes", "1000", "--eval-episodes", "100", "--seeds", "5",
          "--max-steps", "100", "--delay", "5", "--out", "/kaggle/working/results"]
for experiment in ("E8", "E9"):
    subprocess.run([sys.executable, "ccpl_experiments.py", "--exp", experiment,
                    *common], cwd=repo, check=True, env=child_env)
archive = Path("/kaggle/working/ccpl_safety_results.tar.gz")
subprocess.run(["tar", "-czf", str(archive), "-C", "/kaggle/working", "results"], check=True)
print(archive)
