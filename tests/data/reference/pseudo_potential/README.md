# Pseudopotential reference datasets (M-SPARC)

This folder stores pseudopotential-based atomic reference datasets generated with **M-SPARC** for regression tests and cross-checks.

## Files

- `msparc_atoms_lda_svwn.json` - LDA_SVWN pseudopotential atomic references.
- `msparc_atoms_gga_pbe.json` - GGA_PBE pseudopotential atomic references.
- `msparc_atoms_rscan.json` - RSCAN pseudopotential atomic references.
- `msparc_atoms_pbe0.json` - PBE0 pseudopotential atomic references.

All files use a top-level JSON array, where each entry corresponds to one element/species run.

## Per-species fields mapping (array entries)

| Physical meaning | Mathematical expression | Variable name (JSON key) |
| --- | --- | --- |
| Element symbol | - | `element` |
| Exchange-correlation label | - | `xc` |
| Pseudopotential file name | - | `psp_file` |
| Pseudopotential source directory | - | `psp_source_dir` |
| Atomic number | $Z$ | `Zatom` |
| Valence charge | $Z_{val}$ | `Zvalence` |
| NLCC flag requested in run | - | `NLCC_flag_used` |
| NLCC availability in PSP | - | `NLCC_in_psp` |
| Run timestamp | - | `timestamp` |
| Total energy | $E_{tot}$ (Ha) | `Etot` |
| Band-energy term | $E_{band}$ (Ha) | `Eband` |
| Exchange-correlation energy | $E_{xc}$ (Ha) | `Exc` |
| Exchange-correlation double counting | $E_{xc,dc}$ (Ha) | `Exc_dc` |
| Electronic double counting term | $E_{elec,dc}$ (Ha) | `Eelec_dc` |
| Number of occupied states | $N_{occ}$ | `n_occupied_states` |
| Occupied-state records | $\{(n,l,\epsilon_i,f_i)\}_{i\in occ}$ | `occupied_states` |
| Occupied spin-up eigenvalues | $\{\epsilon_i^\uparrow\}_{i\in occ}$ (Ha) | `eigenvalues_occupied_up_Ha` |
| Occupied spin-down eigenvalues | $\{\epsilon_i^\downarrow\}_{i\in occ}$ (Ha) | `eigenvalues_occupied_down_Ha` |
| Available angular-momentum channels | $\{l\}$ | `angular_momentum_channels` |
| Number of eigenpairs per channel | $N_l$ | `full_spectrum_counts` |

## References

- Xu, Q.; Sharma, A.; Suryanarayana, P. **M-SPARC: Matlab-simulation package for ab-initio real-space calculations**. *SoftwareX* **11**, 100423 (2020).
- Zhang, B.; Jing, X.; Kumar, S.; Suryanarayana, P. **Version 2.0.0-M-SPARC: Matlab-simulation package for ab-initio real-space calculations**. *SoftwareX* **21**, 101295 (2023).
