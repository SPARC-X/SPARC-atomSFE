# GGA-PBE reference dataset


## `atompaw_atoms_gga_pbe.json`

- Citation: Holzwarth, N. A. W.; Torrent, M.; Charraud, J.; Cote, M. "Cubic spline solver for generalized density functional treatments of atoms and generation of atomic datasets for use with exchange-correlation functionals including meta-GGA." *Phys. Rev. B* **105**(12), 125144 (2022).
- Source: generated locally with ATOMPAW (`atompaw_aeonly`) for a neutral-atom sweep (`Z=1..92`) using LibXC GGA-PBE (`XC_GGA_X_PBE+XC_GGA_C_PBE`) in spin-unpolarized mode; run settings are recorded once under `generation_parameters`.

At the top level, the JSON stores generation metadata (`generation_parameters`) and per-atom records (`atoms`).

The `generation_parameters` block stores the run setup used to generate this dataset: `xc_functional`, `wtau_mode`, `spin_polarization`, `relativistic_mode`, `grid_points`, `grid_rmax`, and `unit_note`.

### Per-species fields mapping (`atoms` entries)

| Physical meaning | Mathematical expression | Variable name (JSON key) |
| --- | --- | --- |
| Atomic number | $Z$ | `atomic_number` |
| Element symbol | — | `symbol` |
| Convergence flag | — | `converged` |
| Total energy | $E_{\mathrm{tot}}$ (Ha) | `total_energy_ha` |
| Occupied Kohn--Sham eigenvalue records | $\{\epsilon_i\}_{i\in\mathrm{occ}}$ (Ha) | `occupied_eigenvalues_ha` |

### Occupied-eigenvalue fields mapping (`occupied_eigenvalues_ha[]` entries)

| Physical meaning | Mathematical expression | Variable name (JSON key) |
| --- | --- | --- |
| Principal quantum number | $n$ | `n` |
| Orbital angular momentum quantum number | $l$ | `l` |
| Occupancy of each occupied state | $f_i$ | `occupancy` |
| Occupied Kohn--Sham eigenvalue | $\epsilon_i$ (Ha) | `energy_ha` |

**Notes**

- In the original ATOMPAW outputs used in this workflow, energies are treated as Ry-like values; in `atompaw_atoms_gga_pbe.json`, energies are converted to Ha (`Ry -> Ha`) consistently.
- For atoms that did not converge in the production run, `converged` is `false`, and `total_energy_ha` / `occupied_eigenvalues_ha` use placeholder values (`null` / empty list) for downstream tabulation.
- This dataset is intended as a GGA-PBE ATOMPAW reference set for regression/accuracy comparisons.


## `lehtola_even_z_2_to_20_gga_pbe.json`

- Citation: Cinal, M. "Highly accurate numerical solution of Hartree--Fock equation with pseudospectral method for closed-shell atoms." *Journal of Mathematical Chemistry* 58(8), 1571--1600 (2020).
- Source: neutral closed-shell selected even-$Z$ atoms ($Z \in \{2,4,10,12,18,20\}$) exported from HelFEM spin-unpolarized restricted GGA-PBE atomic calculations; grid and SCF settings are recorded once under `generation_parameters` in the JSON.

Field layout for this file differs from `atompaw_atoms_gga_pbe.json`: atomic number key is `z`, and occupied eigenvalues are split into alpha/beta arrays under `eigenvalues_ha`.


## `oncvpsp_atoms_gga_pbe.json`

- Citation: Hamann, D. R. "ONCVPSP pseudopotential generation code." [www.mat-simresearch.com](http://www.mat-simresearch.com) (accessed 2023-05-08).
- Source: generated locally for selected ten elements (H, Be, C, Ne, Na, Si, Fe, Kr, Gd, U).
- Physics/setup: ONCVPSP all-electron (AE), non-relativistic radial mode, LibXC GGA-PBE.

### Per-species fields mapping (`atoms` entries, ONCVPSP selected-10 file)

| Physical meaning | Mathematical expression | Variable name (JSON key) |
| --- | --- | --- |
| Atomic number | $Z$ | `atomic_number` |
| Element symbol | — | `symbol` |
| Convergence flag | — | `converged` |
| Total energy | $E_{\mathrm{tot}}$ (Ha) | `total_energy_ha` |
| Occupied Kohn--Sham eigenvalue records | $\{\epsilon_i\}_{i\in\mathrm{occ}}$ (Ha) | `occupied_eigenvalues_ha` |
| SCF iteration count | — | `scf_iterations` |
| Wall-clock time | $t$ (s) | `wall_time_seconds` |
