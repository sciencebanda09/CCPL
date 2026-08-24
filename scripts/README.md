# Experiment Entry Points

Use the existing runners from the repository root:

```bash
python run_benchmark_v7.py --help
python run_ccpl.py --help
python generate_plots.py --help
python -m pytest -q
```

Generated result directories are data products. Do not edit their files by
hand; rerun the producing command and preserve the configuration and seed
metadata.
