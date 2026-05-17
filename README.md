<div align="center">
<img src="logo.png" alt="logo" width="250"></img>
</div>


# SPARC-atomSFE — Atomic DFT with Spectral Finite Elements

![build](https://img.shields.io/badge/build-passing-brightgreen)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)


[**Features**](#features)
| [**Quick start**](#quick-start)
| [**Installation**](#installation)
| [**Change log**](ChangeLog)
| [**Documentation**](docs/cookbook.md)


## What is SPARC-atomSFE?

**SPARC-atomSFE** is a Python library for atomic (spherical) density functional theory (DFT) with a Spectral Finite Element (SFE) discretization in real space. It is heavily optimized and aims for high numerical accuracy.

SPARC-atomSFE solves the Kohn-Sham equations self-consistently and supports a wide range of exchange-correlation functionals. Calculations can be run in all-electron mode or by reading norm-conserving pseudopotential (PSP) files.

Advanced capabilities include the optimized effective potential (OEP) method, hybrid functionals with exact exchange (e.g. HF), RPA with parallelization, configurable parameters and advanced options, and more.

This is a research code. Please try it out, [report issues](https://github.com/SPARC-X/SPARC-atomSFE/issues), and share feedback.

```python
from atom import AtomicDFTSolver

# Single-atom DFT with GGA-PBE
solver = AtomicDFTSolver(atomic_number=13, xc_functional="GGA_PBE")
results = solver.solve()

# Access total energy, density, eigenvalues, etc.
print(results["energy"])
```

### Contents

- [SPARC-atomSFE — Atomic DFT with Spectral Finite Elements](#sparc-atomsfe--atomic-dft-with-spectral-finite-elements)
  - [What is SPARC-atomSFE?](#what-is-sparc-atomsfe)
    - [Contents](#contents)
  - [Features](#features)
  - [Quick start](#quick-start)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Instructions](#instructions)
  - [Project structure](#project-structure)
  - [Optional dependencies](#optional-dependencies)
  - [Citing SPARC-atomSFE](#citing-sparc-atomsfe)
  - [License](#license)
  - [Reference documentation](#reference-documentation)
  - [Acknowledgement](#acknowledgement)


## Features

* **Finite-element discretization** — Real-space mesh and operators in `atom.mesh`.
* **Pseudopotentials** — Norm-conserving pseudopotential support (e.g. psp8) in `atom.pseudo`.
* **SCF driver** — Density, Hamiltonian, eigensolver, Poisson, mixing, and convergence in `atom.scf`.
* **Exchange–correlation** — LDA, GGA-PBE, hybrid (HF), and ML-XC in `atom.xc`.


## Quick start

```python
from atom import AtomicDFTSolver

# xc_functional can be any supported functional (e.g. GGA_PBE, LDA_PZ, PBE0, ...)
solver = AtomicDFTSolver(atomic_number=29, xc_functional="GGA_PBE")
results = solver.solve()

# Many options available: domain_size, mesh, grid, SCF settings, verbose, etc.
solver = AtomicDFTSolver(
    atomic_number=6,
    xc_functional="LDA_PZ",
    domain_size=15.0,
    verbose=True,
)
results = solver.solve()
```


## Installation

### Requirements

* Python ≥ 3.8
* NumPy ≥ 1.20
* SciPy ≥ 1.7

### Instructions

| Use case        | Command |
|-----------------|---------|
| Core (CPU)      | `pip install -e .` or `pip install atom` |
| With viz        | `pip install -e ".[viz]"` |
| Dev + tests     | `pip install -e ".[dev]"` |
| All optional    | `pip install -e ".[all]"` |

From the repository root:

```bash
pip install -e .
```


## Project structure

| Directory / module | Description |
|--------------------|-------------|
| `src/mesh`         | Grid construction and operators |
| `src/pseudo`       | Pseudopotential reading and evaluation (local / non-local) |
| `src/scf`          | SCF loop: density, Hamiltonian, eigensolver, Poisson, mixer |
| `src/xc`           | XC functionals: LDA, GGA, HF, ML-XC, etc. |
| `src/data`         | Data generation, loading, and processing |
| `src/utils`        | Occupation states, periodicity helpers |
| `tests`            | Unit and integration tests |
| `docs`             | Cookbook (`cookbook.md`) and short `docs/README.md` |


## Optional dependencies

| Extra   | Purpose |
|---------|---------|
| `ml`    | PyTorch, scikit-learn for ML-XC |
| `viz`   | Matplotlib for plotting |
| `dev`   | pytest, Jupyter for development |
| `threadpool` | threadpoolctl for RPA/thread control |


## Citing SPARC-atomSFE

If you use this code in your research, please cite the repository:

```
@software{sparc_atomsfe_placeholder,
  author = {TBD},
  title = {TBD},
  url = {TBD},
  version = {TBD},
  year = {TBD},
}
```


## License

SPARC-atomSFE is licensed under **GNU GPLv3**.


## Reference documentation

For usage examples, see **[docs/cookbook.md](docs/cookbook.md)** (and [docs/README.md](docs/README.md) for a one-line pointer).

For development and contribution guidelines, see the [repository](https://github.com/SPARC-X/SPARC-atomSFE).



## Acknowledgement
  
* **U.S. Department of Energy (DOE), Office of Science (SC): DE-SC0023445**