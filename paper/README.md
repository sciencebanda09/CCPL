# CCPL Paper Draft

This directory contains an anonymous ICLR-style manuscript draft for CCPL.

## Contents

- `ccpl_paper.tex`: manuscript source.
- `iclr2025_conference.sty`: supplied ICLR 2025 style file.
- `figures/archived_benchmark.png`: supplied archived benchmark screenshot.
- `figures/02_risk_heatmap.png`: SafeRoute risk visualization frame.
- `figures/03_delayed_timeline.png`: SafeRoute delayed-cost visualization frame.
- `figures/07_rotating_3d.png`: SafeRoute 3D visualization frame.

The SafeRoute figures are visual artifacts. No new SafeRoute numerical claim is
made because the matching multi-seed summary was not preserved with the
figures. The benchmark numbers in the paper are transcribed from the supplied
archived screenshot and are not merged with later runs.

## Build

Install a LaTeX distribution with `pdflatex` and run from this directory:

```bash
pdflatex ccpl_paper.tex
bibtex ccpl_paper
pdflatex ccpl_paper.tex
pdflatex ccpl_paper.tex
```

The anonymous author block is intentional for an ICLR submission.
