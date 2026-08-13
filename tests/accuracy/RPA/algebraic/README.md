# RPA accuracy / timing / memory tests -- algebraic ω mapping

Test suite for RPA: self-consistent RPA (`RPA-OEP`, both all-electron and
pseudopotential) and non-self-consistent RPA (`RPA@GGA_PBE`, all-electron only),
across 5 elements (He, Be, Al, Ar, Au) -- 15 cases total.

## ⚠ These tests FAIL against the current rpa.py default

`reference.json` was generated when `rpa.py` defaulted to the **algebraic** frequency
map with **c = 1** (then named `inverse_linear`):

    ω(ξ) = c (1 + ξ)/(1 - ξ),   ξ = Gauss-Legendre nodes on (-1, 1),   c = 1

`rpa.py` now defaults to the **sinh** map with the **midpoint** base rule and
`ω_ceiling = 1e8`. Nothing in these case definitions pins the grid -- it is a default on
`RPACorrelation.__init__`, not a solver argument -- so a live run silently evaluates a
different quadrature and compares it against these numbers.

**Expect hard failures on every element except possibly He.** The two maps do not agree
at ω = 24: the algebraic map reaches only ω ≈ 290 there and is nowhere near converged
for anything heavier than He, so the gap is ~5e-3 on E_c for Ar against a 1e-5
tolerance. This is a *grid mismatch*, not a physics regression.

### How to make it pass

Neither `frequency_grid_type` nor `algebraic_scale` can be passed through
`AtomicDFTSolver` -- they are defaults on `RPACorrelation.__init__`, and
`RPACorrelation` is constructed inside the solver (`src/scf/driver.py` for `RPA@DFT`,
and inherited as a mixin by `OEPCalculator` in `src/xc/oep.py` for `RPA-OEP`). So the
grid has to be pinned on the class default, before the solver is built.

**File:** `tests/accuracy/RPA/algebraic/_run_one.py`
**Where:** immediately after the existing `from src.solver import AtomicDFTSolver`, i.e.
before the `AtomicDFTSolver(**kwargs)` call further down.

```python
from src.xc.rpa import RPACorrelation  # noqa: E402

# reference.json was generated on the algebraic map with c = 1, while rpa.py now
# defaults to sinh.  Neither is a solver argument, so pin them on the class default.
# Matched BY NAME rather than by index, so adding a parameter to __init__ cannot
# silently repoint this at the wrong one.
_code     = RPACorrelation.__init__.__code__
_defaults = list(RPACorrelation.__init__.__defaults__)
_names    = list(_code.co_varnames[_code.co_argcount - len(_defaults):_code.co_argcount])
_defaults[_names.index("frequency_grid_type")] = "algebraic"
_defaults[_names.index("algebraic_scale")]     = 1.0
RPACorrelation.__init__.__defaults__ = tuple(_defaults)
```

`_names` is `['radial_coulomb_kernel_apply', 'frequency_grid_type',
'frequency_base_rule', 'omega_ceiling', 'algebraic_scale']`, so this turns the defaults
from `('differential_equation', 'sinh', 'midpoint', 1e8, 1.0)` into
`('differential_equation', 'algebraic', 'midpoint', 1e8, 1.0)`. `algebraic_scale` is
already `1.0`; it is set explicitly so the file states the c these references assume
rather than relying on a default that could move.

`frequency_base_rule` and `omega_ceiling` are irrelevant here -- both apply to the sinh
map only -- and are left alone.

This is a test-only device, the same one `../sinh/_run_one.py` uses to *report* the grid
in force. The alternative is to **regenerate** `reference.json` against whatever
`rpa.py` currently defaults to, after which the numbers describe the production path and
this folder stops being an algebraic-map suite.

For the production grid as it stands, use `../sinh/` instead -- that suite pins and
asserts its grid identity (`EXPECTED_GRID_TYPE` / `EXPECTED_BASE_RULE` /
`EXPECTED_OMEGA_CEILING`) as a hard check, precisely so this cannot happen again.

## Running

```bash
python -m pytest tests/accuracy/RPA/algebraic -m rpa_algebraic -v
```

`-m rpa_algebraic` selects this suite only; `-m rpa_sinh` selects `../sinh/`;
`-m rpa_accuracy` runs both, since every case carries that marker too.

Run from the repo root (SPARC-atomSFE folder, outside the tests directory),
in an environment with this package installed (`pip install -e .`) 
and pytest available.

**Strongly recommended: run with a 24-core allocation.** The stored reference
data (`reference.json` and each case's `result.json`) was generated at
`omega=24` (24 RPA frequency workers, one per core). Running with fewer cores
does not affect the hard accuracy checks -- total energy, RPA correlation
energy, exact exchange energy, and ionization potential (RPA-OEP only) 
are core-count-independent -- but it will make the timing/memory
checks to generate warnings, since wall time and peak memory both
scale with the number of RPA workers actually available. `conftest.py` warns
automatically if fewer than 24 cores are detected.
If the memory and timings are off by 20% of the reference results memory and timings,
the test will generate a warning. 
Only accuracy checks are used to determine if the tests passes or fails.

To run just a subset, select by element or by full case ID:

```bash
python -m pytest tests/accuracy/RPA/algebraic -m rpa_algebraic -k He -v
python -m pytest "tests/accuracy/RPA/algebraic/test_rpa_algebraic.py::test_rpa_case[RPA-OEP/AE/He]" -v
```

List every valid case ID:

```bash
python -m pytest tests/accuracy/RPA/algebraic --collect-only -q
```

## What's checked

Each case is re-run fresh, in its own subprocess (`_run_one.py`, invoked via
`cases.run_subprocess_case`), and compared two ways:

- **Hard** (fails the test, 1e-5 absolute tolerance): total energy, RPA
  correlation energy, exact exchange energy, and -- self-consistent RPA
  only -- ionization potential (`-HOMO` eigenvalue). These four are the only
  quantities stored in `reference.json`.
- **Soft** (warns, never fails, 20% relative tolerance): wall time and peak
  RSS, read from each case's own `result.json` (not from `reference.json`).

A case with no `reference.json` entry, or no baseline `result.json`, is
skipped, not failed -- it means the data hasn't been produced yet.

## Files in this folder

- `cases.py` -- the 15 case definitions, plus the shared subprocess runner
  used by both the tests and the reference-data generator
- `_run_one.py` -- runs one case in isolation; reports total energy, RPA
  correlation energy, exact exchange energy, HOMO, wall time, peak RSS
- `conftest.py` -- warns once per session if fewer than 24 cores are available
- `test_rpa_algebraic.py` -- the actual pytest tests. Loads `cases.py` by path under a
  unique module name, because `../sinh/` also has a `cases.py` and a plain
  `import cases` would bind whichever collected first
- `reference.json` -- the four physics quantities per case; the only thing the
  hard checks compare against
- `RPA-OEP/<AE|psp>/<element>/result.json` and
  `RPA@GGA_PBE/AE/<element>/result.json` -- per-case baseline (also the source
  of the timing/memory soft-check baseline)

`cases.py` and `_run_one.py` are required to run the tests -- they are not
generation-only tools. The reference-data generator itself
(`generate_reference_data.py`), its SLURM launcher
(`generate_reference.sbatch`), a pytest SLURM launcher
(`run_rpa_accuracy_tests.sbatch`), and the raw `run.log` files for each case
are kept outside this folder (not needed to run the tests, only to regenerate
the reference data) -- ask whoever maintains this suite for their current
location if you need to add elements or regenerate after a code change.
