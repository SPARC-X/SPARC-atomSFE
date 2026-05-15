# Paper results

LaTeX fragments, figures, and matching tests for the SPARC-atomSFE paper. Each topic has its own subdirectory under `paper/`; shared figures are in [`figures/`](figures/). Full text, equations, tables, and captions are in the linked `.tex` files.

---

## Radial Schrödinger

Verifies the spectral finite-element framework for the all-electron radial Schrödinger equation by comparing occupied eigenvalues for neutral uranium to hydrogenic reference energies, with errors below 0.1 nano-Hartree.

| | |
|--|--|
| TeX | [`schrodinger/z92_hydrogenic.tex`](schrodinger/z92_hydrogenic.tex) |
| Standalone PDF preview | [`schrodinger/z92_hydrogenic_standalone.tex`](schrodinger/z92_hydrogenic_standalone.tex) |
| Test | [`../test_z92_schrodinger.py`](../test_z92_schrodinger.py) |
| Figure | [`figures/test_z92_schrodinger_summary.pdf`](figures/test_z92_schrodinger_summary.pdf) |

Refresh the figure with `REGENERATE_SUMMARY_PDF = True`, then `python ../test_z92_schrodinger.py`. Preview TeX from `schrodinger/`: `pdflatex z92_hydrogenic_standalone.tex` (requires the PDF in `figures/`). 
---
