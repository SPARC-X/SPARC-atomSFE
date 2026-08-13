# RPA accuracy — sinh ω mapping

Reference suite for the production frequency grid. 15 cases: 5 elements ×
{RPA-OEP AE, RPA-OEP psp, RPA@GGA_PBE AE}.

## The grid

`ω = c·sinh(y)` on `y ∈ [0, arcsinh(ω_max/c)]`, with `c = Δ_min` rebuilt from the
Kohn–Sham spectrum at every `compute_*` call.

| | |
|---|---|
| grid type | `sinh` |
| **base rule** | **`midpoint`** |
| ω_ceiling | `1e8` |

**The base rule is `midpoint`** — rpa.py's default, and nothing here patches it.
`rpa.py` implements four rules on this map (`midpoint`, `trapezoid`,
`clenshaw_curtis`, `gauss_legendre`); midpoint and trapezoid are equispaced in `y`
and converge as `exp(−π²n/y_max)`, which is the rate the pole-strip argument predicts,
while Clenshaw–Curtis and Gauss–Legendre cluster at the ends and are optimal for a
Bernstein ellipse instead. Measured at Lmax=5 across He→Au, all four reach the same
limit and need the same node count to within one ladder step, so midpoint is chosen for
being the default rather than for being faster.

Only the closed rules (`trapezoid`, `clenshaw_curtis`) place a node at ω = 0. Midpoint
does not, so these references never evaluate the static limit.

All three settings are defaults on `RPACorrelation.__init__`, not solver arguments, so
no case can request them. `_run_one.py` reports what was actually in force and the test
compares against `EXPECTED_GRID_TYPE` / `EXPECTED_BASE_RULE` / `EXPECTED_OMEGA_CEILING`
as a **hard** check — a moved default fails naming the cause instead of silently
reinterpreting the file.

## ω per mode

| mode | ω | why |
|---|---|---|
| RPA-OEP | **48** | also scored on the ionization potential, i.e. an eigenvalue |
| RPA@GGA_PBE | **24** | scored on energies only |

`E_total` and `E_c` are variational in `v_c`, so a quadrature error in the potential
reaches them at second order and ~24 nodes holds them to ~1e-5. Eigenvalues respond at
first order and need about two more ladder steps — ~50 nodes for the same 1e-5. 48
keeps it a multiple of the 24-core allocation.

## Lmax

5 (He, Be), 10 (Al, Ar, Au). Held **fixed, not converged** — Lmax is a convergence
parameter of the angular sum and is independent of the frequency grid, so it only has to
be held constant for the comparison to mean anything. Lowering it is what makes
regenerating all 15 references practical.

These are therefore a **regression baseline for the ω mapping**, not converged atomic
energies. Do not compare them against literature values, or against `../algebraic/`.

## Running

```bash
# from the repo root
python -m pytest tests/accuracy/RPA/sinh      -m rpa_sinh        # this suite only
python -m pytest tests/accuracy/RPA/algebraic -m rpa_algebraic   # the algebraic suite
python -m pytest tests/accuracy/RPA           -m rpa_accuracy    # both
python -m pytest tests                        -m "not rpa_accuracy"   # neither
```

Use `python -m pytest` rather than bare `pytest`: it puts the repo root on `sys.path`,
which `tests/unit/test_pulay_mixing_trajectories.py` needs. Requires the package
installed (`python -m pip install -e .`).

24 cores strongly recommended — see `conftest.py`. Fewer cores does not affect the
accuracy checks, only the warning-only timing and memory ones.

## Files in this folder

- `cases.py` — the 15 case definitions, the Lmax/ω table, the expected grid identity,
  and the shared subprocess runner used by both the tests and the generator
- `_run_one.py` — runs one case in isolation; reports the energies, HOMO, wall time,
  peak RSS, and the frequency grid actually in force
- `conftest.py` — warns once per session if fewer than 24 cores are available
- `test_rpa_sinh.py` — the pytest tests. Loads `cases.py` by path under a unique module
  name, because `../algebraic/` also has a `cases.py` and a plain `import cases` would
  bind whichever collected first
- `reference.json` — the four physics quantities per case; the only thing the hard
  checks compare against
- `<mode>/<electrons>/<element>/result.json` — per-case baseline, and the source of the
  timing/memory soft-check baseline

`cases.py` and `_run_one.py` are required to RUN the tests — they are not
generation-only tools.

## Regenerating

The generator (`generate_reference_data.py`) and the raw per-case `run.log` files are
kept **outside this folder**, matching `../algebraic/` — this directory holds only what
is needed to run the tests. Ask whoever maintains this suite for their current location.

It takes `--suite` (this folder) and writes `reference.json` and each
`result.json` back into it, while `run.log` stays with the generator:

```bash
python generate_reference_data.py --suite <repo>/tests/accuracy/RPA/sinh

# subsets, resumable — cases already present are skipped unless --force
python generate_reference_data.py --mode RPA@GGA_PBE
python generate_reference_data.py --element He Be
python generate_reference_data.py --cases RPA-OEP/AE/Au --force
```

## What is checked

**Hard** (fails, 1e-5 absolute): `total_energy`, `rpa_correlation_energy`,
`exact_exchange_energy`, and — self-consistent only — `ionization_potential` (= −HOMO).
Plus the grid identity above.

**Soft** (warns, 20% relative): wall time and peak RSS, from each case's own
`result.json`. Both scale with core count, which is why they are not in
`reference.json`.

A case with no `reference.json` entry is skipped, not failed.
