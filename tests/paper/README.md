# Paper results

LaTeX fragments, figures, and matching tests for the SPARC-atomSFE paper. Each topic has its own subdirectory under `paper/`; shared figures are in [`figures/`](figures/). Full text, equations, tables, and captions are in the linked `.tex` files.

---

## Radial Schrödinger

Verifies the spectral finite-element framework for the all-electron radial Schrödinger equation by comparing occupied eigenvalues for neutral Z=92 to hydrogenic reference energies, with errors below 0.1 nano-Hartree.

| Item | Path |
|------|------|
| TeX | [`schrodinger/z92_hydrogenic.tex`](schrodinger/z92_hydrogenic.tex) |
| Standalone PDF | [`schrodinger/z92_hydrogenic_standalone.pdf`](schrodinger/z92_hydrogenic_standalone.pdf) |
| Python script | [`../test_z92_schrodinger.py`](../test_z92_schrodinger.py) |
| Figure | [`figures/test_z92_schrodinger_summary.pdf`](figures/test_z92_schrodinger_summary.pdf) |

Refresh the figure with `REGENERATE_SUMMARY_PDF = True`, then `python ../test_z92_schrodinger.py`.

---

## All-electron discretization convergence

Narrative and figures for Kohn--Sham mesh convergence in the all-electron setting.

| Item | Path |
|------|------|
| TeX | [`convergence/all_electron_convergence.tex`](convergence/all_electron_convergence.tex) |
| Standalone PDF | [`convergence/all_electron_convergence_standalone.pdf`](convergence/all_electron_convergence_standalone.pdf) |

Paths below are under `tests/data/` (`../data/...` from this directory).

| XC functional | atomSFE summary | Compare script | Figure |
|---------------|-----------------|----------------|--------|
| LDA-SVWN | [`summary/all_electron/lda_svwn/`](../data/summary/all_electron/lda_svwn/) | [`compare/lda_svwn_convergence_test.py`](../data/compare/lda_svwn_convergence_test.py) | [`figures/lda_svwn_convergence_test_summary.pdf`](figures/lda_svwn_convergence_test_summary.pdf) |
| LDA-SVWN vs FeAtom | [`summary/all_electron/lda_svwn/`](../data/summary/all_electron/lda_svwn/) | [`compare/lda_svwn_convergence_test_featom.py`](../data/compare/lda_svwn_convergence_test_featom.py) | [`figures/lda_svwn_convergence_test_featom_summary.pdf`](figures/lda_svwn_convergence_test_featom_summary.pdf) |
| GGA-PBE | [`summary/all_electron/gga_pbe/`](../data/summary/all_electron/gga_pbe/) | [`compare/gga_pbe_convergence_test.py`](../data/compare/gga_pbe_convergence_test.py) | [`figures/gga_pbe_convergence_test_summary.pdf`](figures/gga_pbe_convergence_test_summary.pdf) |
| rSCAN | [`summary/all_electron/rscan/`](../data/summary/all_electron/rscan/) | [`compare/rscan_convergence_test.py`](../data/compare/rscan_convergence_test.py) | [`figures/rscan_convergence_test_summary.pdf`](figures/rscan_convergence_test_summary.pdf) |
| HF (closed-shell) | [`summary/all_electron/hf/`](../data/summary/all_electron/hf/) | [`compare/hf_closed_shell_convergence_test.py`](../data/compare/hf_closed_shell_convergence_test.py) | [`figures/hf_closed_shell_convergence_test_summary.pdf`](figures/hf_closed_shell_convergence_test_summary.pdf) |

---

## Pseudopotential discretization convergence

Narrative and figures for Kohn--Sham mesh convergence in the pseudopotential setting.

| Item | Path |
|------|------|
| TeX | [`convergence/pseudopotential_convergence.tex`](convergence/pseudopotential_convergence.tex) |
| Standalone PDF | [`convergence/pseudopotential_convergence_standalone.pdf`](convergence/pseudopotential_convergence_standalone.pdf) |

Paths below are under `tests/data/` (`../data/...` from this directory).

| XC functional | atomSFE summary | Compare script | Figure |
|---------------|-----------------|----------------|--------|
| LDA-SVWN | [`summary/pseudo_potential/lda_svwn/`](../data/summary/pseudo_potential/lda_svwn/) | [`compare/pseudo_lda_svwn_convergence_test.py`](../data/compare/pseudo_lda_svwn_convergence_test.py) | [`figures/pseudo_lda_svwn_convergence_test_summary.pdf`](figures/pseudo_lda_svwn_convergence_test_summary.pdf) |
| GGA-PBE | [`summary/pseudo_potential/gga_pbe/`](../data/summary/pseudo_potential/gga_pbe/) | [`compare/pseudo_gga_pbe_convergence_test.py`](../data/compare/pseudo_gga_pbe_convergence_test.py) | [`figures/pseudo_gga_pbe_convergence_test_summary.pdf`](figures/pseudo_gga_pbe_convergence_test_summary.pdf) |
| rSCAN | [`summary/pseudo_potential/rscan/`](../data/summary/pseudo_potential/rscan/) | [`compare/pseudo_rscan_convergence_test.py`](../data/compare/pseudo_rscan_convergence_test.py) | [`figures/pseudo_rscan_convergence_test_summary.pdf`](figures/pseudo_rscan_convergence_test_summary.pdf) |
| PBE0 | [`summary/pseudo_potential/pbe0/`](../data/summary/pseudo_potential/pbe0/) | [`compare/pseudo_pbe0_convergence_test.py`](../data/compare/pseudo_pbe0_convergence_test.py) | [`figures/pseudo_pbe0_convergence_test_summary.pdf`](figures/pseudo_pbe0_convergence_test_summary.pdf) |

---

## All-electron accuracy

Comparisons of SPARC-atomSFE against FeAtom (LDA), atomPAW (PBE, rSCAN), and Lehtola (HF); violin summary over selected atoms and functionals.

| Item | Path |
|------|------|
| TeX | [`accuracy/all_electron_accuracy.tex`](accuracy/all_electron_accuracy.tex) |
| Standalone PDF | [`accuracy/all_electron_accuracy_standalone.pdf`](accuracy/all_electron_accuracy_standalone.pdf) |
| Figure | [`figures/xc_accuracy_violin_ae.pdf`](figures/xc_accuracy_violin_ae.pdf) |

Paths below are under `tests/data/` (relative to this `paper/` directory: `../data/...`). Regenerate compare reports from `tests/data/compare/` with `python <script>.py`.

| XC functional | Reference data | atomSFE summary | Compare script |
|---------------|----------------|-----------------|----------------|
| LDA-SVWN | [`reference/all_electron/lda_svwn/featom_atoms_lda.json`](../data/reference/all_electron/lda_svwn/featom_atoms_lda.json) | [`summary/all_electron/lda_svwn/fe12_R040__z1_92.json`](../data/summary/all_electron/lda_svwn/fe12_R040__z1_92.json) | [`compare/lda_svwn_accuracy_test_featom.py`](../data/compare/lda_svwn_accuracy_test_featom.py) |
| GGA-PBE | [`reference/all_electron/gga_pbe/atompaw_atoms_gga_pbe.json`](../data/reference/all_electron/gga_pbe/atompaw_atoms_gga_pbe.json) | [`summary/all_electron/gga_pbe/fe12_R040__z1_92.json`](../data/summary/all_electron/gga_pbe/fe12_R040__z1_92.json) | [`compare/gga_pbe_accuracy_test_atompaw.py`](../data/compare/gga_pbe_accuracy_test_atompaw.py) |
| rSCAN | [`reference/all_electron/rscan/atompaw_atoms_rscan_dense.json`](../data/reference/all_electron/rscan/atompaw_atoms_rscan_dense.json) | [`summary/all_electron/rscan/fe12_R040__z1_92.json`](../data/summary/all_electron/rscan/fe12_R040__z1_92.json) | [`compare/rscan_accuracy_test_atompaw.py`](../data/compare/rscan_accuracy_test_atompaw.py) |
| HF (neutral) | [`reference/all_electron/hf/lehtola_closed_subshell_atoms_hf.json`](../data/reference/all_electron/hf/lehtola_closed_subshell_atoms_hf.json) | [`summary/all_electron/hf/fe12_R040__z1_92.json`](../data/summary/all_electron/hf/fe12_R040__z1_92.json) | [`compare/hf_accuracy_test_neural_lehtola.py`](../data/compare/hf_accuracy_test_neural_lehtola.py) |
| HF (charged) | [`reference/all_electron/hf/lehtola_charged_atoms_hf.json`](../data/reference/all_electron/hf/lehtola_charged_atoms_hf.json) | [`summary/all_electron/hf/charged/fe12_R040__charged.json`](../data/summary/all_electron/hf/charged/fe12_R040__charged.json) | [`compare/hf_accuracy_test_charged_lehtola.py`](../data/compare/hf_accuracy_test_charged_lehtola.py) |

All-electron RPA-OEP results for He, Be, and Ne are summarized in [`summary/all_electron/rpa_oep/rpa_oep_he_ne_be_summary.json`](../data/summary/all_electron/rpa_oep/rpa_oep_he_ne_be_summary.json).

---

## Pseudopotential accuracy

Comparisons of SPARC-atomSFE against M-SPARC (SPARC-atom) for seven valence atoms (He, N, O, Fe, Mn, Mo, Cs) and four functionals (LDA-SVWN, GGA-PBE, rSCAN, PBE0); violin summary over those cases.

| Item | Path |
|------|------|
| TeX | [`accuracy/pseudopotential_accuracy.tex`](accuracy/pseudopotential_accuracy.tex) |
| Standalone PDF | [`accuracy/pseudopotential_accuracy_standalone.pdf`](accuracy/pseudopotential_accuracy_standalone.pdf) |
| Figure | [`figures/xc_accuracy_violin_psp.pdf`](figures/xc_accuracy_violin_psp.pdf) |

Paths below are under `tests/data/` (relative to this `paper/` directory: `../data/...`).

| XC functional | Reference data | atomSFE summary | Compare script |
|---------------|----------------|-----------------|----------------|
| LDA-SVWN | [`reference/pseudo_potential/msparc_atoms_lda_svwn.json`](../data/reference/pseudo_potential/msparc_atoms_lda_svwn.json) | [`summary/pseudo_potential/lda_svwn/fe10_R040__z7c.json`](../data/summary/pseudo_potential/lda_svwn/fe10_R040__z7c.json) | [`compare/pseudo_accuracy_test_msparc.py`](../data/compare/pseudo_accuracy_test_msparc.py) |
| GGA-PBE | [`reference/pseudo_potential/msparc_atoms_gga_pbe.json`](../data/reference/pseudo_potential/msparc_atoms_gga_pbe.json) | [`summary/pseudo_potential/gga_pbe/fe10_R040__z7c.json`](../data/summary/pseudo_potential/gga_pbe/fe10_R040__z7c.json) | [`compare/pseudo_accuracy_test_msparc.py`](../data/compare/pseudo_accuracy_test_msparc.py) |
| rSCAN | [`reference/pseudo_potential/msparc_atoms_rscan.json`](../data/reference/pseudo_potential/msparc_atoms_rscan.json) | [`summary/pseudo_potential/rscan/fe10_R040__z7c.json`](../data/summary/pseudo_potential/rscan/fe10_R040__z7c.json) | [`compare/pseudo_accuracy_test_msparc.py`](../data/compare/pseudo_accuracy_test_msparc.py) |
| PBE0 | [`reference/pseudo_potential/msparc_atoms_pbe0.json`](../data/reference/pseudo_potential/msparc_atoms_pbe0.json) | [`summary/pseudo_potential/pbe0/fe10_R040__z7c.json`](../data/summary/pseudo_potential/pbe0/fe10_R040__z7c.json) | [`compare/pseudo_accuracy_test_msparc.py`](../data/compare/pseudo_accuracy_test_msparc.py) |

---
