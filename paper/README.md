# CCPL Paper Draft

This directory contains an anonymous ICLR-style manuscript draft for CCPL.

## Contents

- `ccpl_paper.tex`: manuscript source.
- `iclr2025_conference.sty`: supplied ICLR 2025 style file.
- `figures/archived_benchmark.png`: source snapshot used to transcribe the
  benchmark and ablation tables; it is not used as a paper figure.
- `figures/01_topdown.png`: SafeRoute route visualization frame.
- `figures/04_policy_comparison.png`: SafeRoute policy comparison frame.
- `figures/07_rotating_3d.png`: SafeRoute 3D visualization frame.

The SafeRoute figures are visual artifacts. No new SafeRoute numerical claim is
made because the matching multi-seed summary was not preserved with the
figures. The benchmark numbers in the paper are transcribed into LaTeX tables
from the supplied archived snapshot and are not merged with later runs.

The Airspace Guardian, Neurocity, and Safety Audit directories are companion
artifacts. They demonstrate rollout visualization and stress-test reporting;
they are not additional benchmark evidence in this manuscript.

## Build

Install a LaTeX distribution with `pdflatex` and run from this directory:

```bash
pdflatex ccpl_paper.tex
bibtex ccpl_paper
pdflatex ccpl_paper.tex
pdflatex ccpl_paper.tex
```

The anonymous author block is intentional for an ICLR submission.
