# Experiment entry points

Use the existing runners while the package migration is staged:

- `python run_benchmark_v7.py --help` — primary benchmark
- `python run_ccpl.py --help` — focused training run
- `python generate_plots.py --help` — derived figures
- `python -m pytest -q` — regression and numerical correctness suite

Do not edit generated `results_v7/` outputs by hand. Rerun the producing command and record its
config in the result directory.
