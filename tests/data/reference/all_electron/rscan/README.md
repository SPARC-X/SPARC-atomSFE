# RSCAN reference dataset


## `atompaw_atoms_rscan.json`

- Citation: Holzwarth, N. A. W.; Torrent, M.; Charraud, J.; Cote, M. "Cubic spline solver for generalized density functional treatments of atoms and generation of atomic datasets for use with exchange-correlation functionals including meta-GGA." *Phys. Rev. B* **105**(12), 125144 (2022).
- Source: generated locally with ATOMPAW (`atompaw_aeonly`) for a neutral-atom sweep (`Z=1..92`) using LibXC RSCAN (`XC_MGGA_X_RSCAN+XC_MGGA_C_RSCAN`) in spin-unpolarized mode; run settings are recorded once under `generation_parameters`.

At the top level, the JSON stores generation metadata (`generation_parameters`) and per-atom records (`atoms`).

The `generation_parameters` block stores the run setup used to generate this dataset: `xc_functional`, `wtau_mode`, `spin_polarization`, `relativistic_mode`, `grid_points`, `grid_rmax`, and `unit_note`.

### Per-species fields mapping (`atoms` entries)

| Physical meaning | Mathematical expression | Variable name (JSON key) |
| --- | --- | --- |
| Atomic number | $Z$ | `atomic_number` |
| Element symbol | — | `symbol` |
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

- In the original literature/program outputs, energies are reported in Ry. In `atompaw_atoms_rscan.json`, energies are converted to Ha (`Ry -> Ha`) while preserving consistent significant digits.
- This dataset is intended as an RSCAN reference set for regression/accuracy comparisons.


## `atompaw_atoms_rscan_dense.json`

- **Purpose:** same JSON schema as `atompaw_atoms_rscan.json`, but only the **selected ten elements** (H, Be, C, Ne, Na, Si, Fe, Kr, Gd, U).
- **What differs:** these runs use an **ultradense radial grid** (`grid_points=50001`, `grid_rmax=40`) and **finer spline settings** (`splr0`, `splns` in `generation_parameters`) compared with the bulk `atompaw_atoms_rscan.json` sweep (`grid_points=12001`, `grid_rmax=260`). The intent is a **higher-resolution numerical reference** for regression checks on that subset.

Field layout (`generation_parameters`, `atoms`, `occupied_eigenvalues_ha`) matches the tables in the section above.
