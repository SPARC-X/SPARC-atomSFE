# Data Folder Tutorial

This folder contains dataset generation scripts, summary extraction, and cleanup tools.
The raw datasets can be very large, so the typical workflow is:

1. Generate dataset(s).
2. Build summary JSON files from `out.txt`.
3. Keep scripts + summaries, clean heavy raw files when needed.

---

## Comparison figures (`compare/`)

Plotting scripts that **compare** or **convergence-test** sweep results (read under `summary/<functional>/…`) live here:

- `compare/lda_svwn_accuracy_test_featom.py` — FEATOM reference vs LDA_SVWN sweep (per-eigenvalue errors); run with no arguments (reads ``summary/lda_svwn/``, writes ``lda_svwn_accuracy_test_featom.png``).
- `compare/hf_accuracy_test_neural_lehtola.py` — HF ``summary/hf`` vs ``reference/hf`` closed-subshell reference (total energy, HOMO, exchange); prints detail and writes ``hf_accuracy_test_neural_lehtola_summary.txt`` (see ``--out-txt``).
- `compare/gga_pbe_convergence_test.py` — GGA-PBE sweep convergence test: max energy / eigenvalue error vs the finest reference case.

---

## File-by-file behavior

### `generate_dataset.py`

Single-case dataset generator.

- Generates exactly one dataset per run (no FE/radius sweep).
- Writes a one-entry manifest.
- Useful for quick tests or one fixed parameter setup.

Common usage:

```bash
# From repository root
# Plan only
python tests/data/generate_dataset.py --dry-run

# Actually generate
python tests/data/generate_dataset.py --regenerate-data

# Or run locally from this folder
cd tests/data
python generate_dataset.py --dry-run
```

---

### `generate_sweep_dataset.py`

Multi-case generator for finite-element and domain-radius sweeps (discretization batches).

- Supports FE sweep and domain-radius sweep.
- Supports `use_oep` job selection (`false/true/both`).
- Writes a multi-entry manifest (`convergence_dataset_manifest.json`).

Common usage:

```bash
# From repository root
# Plan only
python tests/data/generate_sweep_dataset.py --dry-run

# Generate all selected jobs
python tests/data/generate_sweep_dataset.py --regenerate-data

# Example: both OEP and non-OEP job sets
python tests/data/generate_sweep_dataset.py --use-oep-options both --regenerate-data

# Or run locally from this folder
cd tests/data
python generate_sweep_dataset.py --dry-run
```

---

### `build_summary_from_out.py`

Extracts summary JSON from each configuration `out.txt`.

- Creates a `summary/` folder under this directory.
- Mirrors original dataset path to case level:
  `summary/<functional>/<sweep>/<case>/configuration_energy_summary.json`
- Parses final energy block from `out.txt` and includes:
  - total energy
  - all energy components found in that block
- Does **not** compute error metrics.

Common usage:

```bash
# From repository root
# Scan only, no writes
python tests/data/build_summary_from_out.py --dry-run

# Write summary JSON files
python tests/data/build_summary_from_out.py

# Custom output folder and filename
python tests/data/build_summary_from_out.py --summary-dir-name summary --output-name configuration_energy_summary.json

# Or run locally from this folder
cd tests/data
python build_summary_from_out.py --dry-run
```

---

### `cleanup_dataset.py`

Cleanup tool for large raw datasets.

- Default mode: pattern-based keep rules from `keep_list.txt`.
- Aggressive mode: `--summary-only` keeps only summary-like outputs plus control files.
- Default behavior is dry-run; add `--apply` to actually delete.

Common usage:

```bash
# From repository root
# Dry-run with keep_list patterns
python tests/data/cleanup_dataset.py

# Dry-run aggressive summary-only keep mode
python tests/data/cleanup_dataset.py --summary-only

# Apply deletion (pattern-based)
python tests/data/cleanup_dataset.py --apply

# Apply deletion (summary-only)
python tests/data/cleanup_dataset.py --summary-only --apply

# Or run locally from this folder
cd tests/data
python cleanup_dataset.py --summary-only
```

Runtime behavior:

- Prints startup configuration immediately.
- Shows live-refresh scan line for current file.
- In default mode, prints `[DEL] ...` when a file is marked for deletion.
- In `--summary-only` mode, prints `[KEEP] ...` for retained files.

---

### `keep_list.txt`

Pattern list used by `cleanup_dataset.py` in default mode.

- Glob-style patterns relative to `tests/data`.
- Supports comments with `#`.
- Edit this file to tune what remains in pattern-based cleanup.

---

### `functional_dataset_manifest.json` (and similar manifest JSON files)

Manifests created by generation scripts.

- Store planned/generated jobs and key parameters.
- Useful for reproducibility and reruns.
- Usually lightweight; recommended to keep.

---

### `.gitkeep`

Keeps this folder tracked when no data files exist.

---

## Recommended end-to-end workflow

```bash
# 1) Generate data
python tests/data/generate_sweep_dataset.py --regenerate-data

# 2) Build summaries from out.txt
python tests/data/build_summary_from_out.py

# 3) Preview cleanup strategy
python tests/data/cleanup_dataset.py --summary-only

# 4) Apply cleanup when confirmed
python tests/data/cleanup_dataset.py --summary-only --apply
```
