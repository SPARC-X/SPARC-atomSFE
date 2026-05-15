# SPARC-atomSFE

**Atomic density functional theory with spectral finite elements in real space**

**Material Physics & Mechanics Group**  
Georgia Institute of Technology

**PI:** Phanish Suryanarayana

**Contributors**

- Qihao Cheng (qcheng61@gatech.edu)  
- Shubhang Trivedi (strivedi44@gatech.edu)  
- Phanish Suryanarayana (phanish.suryanarayana@ce.gatech.edu)

---

## Introduction

**SPARC-atomSFE** is a Python package for **spherical (atomic) Kohn–Sham DFT** discretized with **high-order spectral finite elements** on a real-space radial grid. It supports all-electron and norm-conserving pseudopotential (e.g. psp8) workflows, multiple exchange–correlation families (LDA, GGA, meta-GGA, hybrids, OEP-related routes, RPA, etc., subject to the functional list in the code), and an SCF stack with mixing, preconditioning, and optional outer loops where the functional requires it. For runnable snippets, see **[cookbook.md](cookbook.md)**.

---

## `AtomicDFTSolver.__init__(...)`

Only **`atomic_number`** is required; other arguments use documented defaults after `set_and_check_initial_parameters()`.

### Physical / model

- **`atomic_number`**: nuclear charge \(Z\) (may be fractional in supported modes; pseudopotential runs require integer \(Z\)).
- **`n_electrons`**: number of electrons; default `atomic_number`. For pseudopotentials, must match `atomic_number` in this build.
- **`all_electron_flag`**: `True` for all-electron, `False` for valence-only with a PSP (default).
- **`xc_functional`**: XC label string, e.g. `GGA_PBE`, `LDA_PZ`, `SCAN`, `PBE0`, `RPA`, … (must be in the package’s allowed list).
- **`use_oep`**: enable the OEP-related workflow when the functional and settings require it (default rules depend on `xc_functional`).

### Grid / FE mesh

- **`domain_size`**: radial simulation box extent (Bohr).
- **`finite_element_number`**: number of finite elements along the radius.
- **`polynomial_order`**: polynomial degree on the reference element.
- **`quadrature_point_number`**: quadrature points per element (integration accuracy).
- **`oep_basis_number`**: auxiliary basis size for OEP when `use_oep` is active (otherwise defaulted / unused as appropriate).
- **`mesh_type`**: `exponential`, `polynomial`, or `uniform` radial mesh law.
- **`mesh_concentration`**: clustering / concentration parameter for the mesh generator.
- **`mesh_spacing`**: target spacing for **output** uniform grids (does not replace the FE quadrature during SCF).

### SCF / mixing

- **`scf_tolerance`**: density (or driver) residual tolerance; meta-GGA defaults may relax slightly vs LDA/GGA.
- **`max_scf_iterations`**: maximum **inner** SCF iterations.
- **`max_scf_iterations_outer`**: maximum **outer** iterations for functionals that need an outer loop (HF, PBE0, EXX, RPA, …).
- **`use_pulay_mixing`**: `True` for Pulay mixing, `False` for linear mixing.
- **`use_preconditioner`**: preconditioner on/off (defaults coupled to mixing choice).
- **`pulay_mixing_parameter`**, **`pulay_mixing_history`**, **`pulay_mixing_frequency`**: Pulay mixer knobs.
- **`linear_mixing_alpha1`**, **`linear_mixing_alpha2`**: linear mixer \(\alpha_1,\alpha_2\) when Pulay is off.

### Pseudopotentials (when `all_electron_flag=False`)

- **`psp_dir_path`**: directory containing PSP files.
- **`psp_file_name`**: PSP filename (default pattern uses `atomic_number`).

### Advanced XC / RPA / OEP knobs

- **`hybrid_mixing_parameter`**: hybrid mixing (e.g. PBE0 EXX fraction); functional-dependent defaults.
- **`frequency_quadrature_point_number`**: RPA frequency quadrature count (RPA only).
- **`angular_momentum_cutoff`**: angular momentum cutoff used in the RPA path (RPA only).
- **`oep_mixing_parameter`**: scales OEP potentials in the OEP workflow (\(\lambda\)).
- **`enable_parallelization`**: RPA parallelization toggle (RPA path; environment may force off).

### Other

- **`verbose`**: print banners, parameters, and progress.
- **`print_debug`**: **Deprecated**; use `verbose` instead.
- **`number_of_finite_elements`**: **Deprecated**; use `finite_element_number`.

---

## `solve(...)`

Self-consistent Kohn–Sham solution; returns a **`dict`** of arrays and metadata (keys such as `energy`, `rho`, `orbitals`, `converged`, `iterations`, …).

- **`save_intermediate`**: if `True`, retain per-iteration diagnostics (e.g. density/residual traces) inside the returned structure where implemented.
- **`save_energy_density`**: if `True`, retain local XC **energy density** fields needed for analysis/plots.
- **`save_full_spectrum`**: if `True`, retain **full-spectrum** eigen information along the iteration history (heavier memory).
- **`rho_initial`**: optional initial **density** on quadrature nodes, length = number of quadrature points; overrides the default guess when given.
- **`orbitals_initial`**: optional initial **occupied orbitals** on the quadrature grid, shape `(n_quadrature_points, n_occupied)`; fed into the SCF driver (and warm-start path when applicable).
- **`use_warm_start`**: if `True` (default), may run a **warm-start** pre-calculation (e.g. GGA_PBE) for meta-GGA / certain OEP setups before the main SCF. Set `False` to skip and start from `rho_initial` / `orbitals_initial` only.
- **`evaluate_basis_on_uniform_grid`**: if `True`, interpolate orbitals and local XC quantities (and, when `save_energy_density` is `True`, energy densities) onto a **uniform visualization grid**; if `False`, related `*_on_uniform_grid` entries in the result are `None`.

---

## `forward(...)`

A **single** potential/energy evaluation **without** SCF iteration: builds density data from supplied orbitals, evaluates XC (and related) contributions, and returns a **`dict`** in the same style as `solve()` where applicable.

- **`orbitals`**: Kohn–Sham orbitals on the quadrature grid, shape **`(n_quadrature_points, n_states)`** (same layout as `solve()`’s `orbitals` / `orbitals_initial`): each column is one radial orbital \(R_{nl}(r)\) sampled on the quadrature nodes.
- **`full_eigen_energies`**: optional full eigenvalue vector for the same state ordering, used when the XC path needs orbital energies.
- **`full_orbitals`**: optional full orbital set beyond the occupied block, if required by the functional implementation.
- **`full_l_terms`**: optional angular-momentum channel data aligned with the “full” spectrum when needed.
- **`compute_energy_density`**: if `True`, also assemble energy-density fields needed for decomposed energy-density output.

Typical use: post-process or differentiate **around** a converged `solve()` state by reusing returned `orbitals` and, if needed, `full_eigen_energies` / `full_orbitals` / `full_l_terms`.

---

**Citation**

```bibtex
@software{sparc_atomsfe_placeholder,
  author = {TBD},
  title = {TBD},
  url = {TBD},
  version = {TBD},
  year = {TBD},
}
```

**Acknowledgements**

- U.S. Department of Energy (DOE), Office of Science (SC): **DE-SC0023445**

---

**User guide:** **[cookbook.md](cookbook.md)**
