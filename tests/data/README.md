# Data Folder Tutorial

This folder contains dataset generation scripts, summary extraction, and cleanup tools.
The raw datasets can be very large, so the typical workflow is:

1. Generate a **single-case** dataset with `generate_dataset.py` (optional; for new XC presets).
2. Build or refresh summary JSON from `out.txt` with `build_summary_from_out.py`.
3. Keep scripts + committed summaries; use `cleanup_dataset.py` on large raw trees when needed.

---

## Comparison figures (`compare/`)

Scripts that compare committed summaries to reference data and regenerate paper figures are listed in [`../paper/README.md`](../paper/README.md) (paths under `compare/`, outputs under `../paper/figures/`).

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

### `summary_naming.py`

Shared helpers for summary JSON basenames (`fe12_R040__z1_92.json`, `fe10_R040__z7c.json`, etc.),
flat sweep layout (no `subset_*` subfolders in `summary/`), and glob paths used by compare scripts.

---

### `build_summary_from_out.py`

Extracts summary JSON from each configuration `out.txt`.

- Creates a `summary/` folder under this directory.
- Writes flat files under each sweep folder, e.g.
  `summary/all_electron/gga_pbe/finite_element_sweep/fe12_R040__z1_92.json`
  (see `summary_naming.py`; `subset_*` dataset dirs collapse to the sweep parent)
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

# Custom output folder or fixed basename (default: auto fe##_R###__z1_92.json from input_parameters)
python tests/data/build_summary_from_out.py --summary-dir-name summary --output-name fe12_R040__z1_92.json

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
# 1) Optional: one fixed mesh / Z batch (e.g. new XC)
python tests/data/generate_dataset.py --dry-run
python tests/data/generate_dataset.py --regenerate-data

# 2) Build summaries from out.txt (point --base-dir at raw dataset root if not tests/data)
python tests/data/build_summary_from_out.py --dry-run
python tests/data/build_summary_from_out.py

# 3) Convergence figures read committed summary/ only (no regeneration step here)
python tests/data/compare/gga_pbe_convergence_test.py
python tests/data/compare/hf_closed_shell_convergence_test.py
python tests/data/compare/lda_svwn_convergence_test.py
python tests/data/compare/rscan_convergence_test.py
python tests/data/compare/pseudo_gga_pbe_convergence_test.py
python tests/data/compare/pseudo_pbe0_convergence_test.py

# 4) Preview / apply cleanup on large raw trees
python tests/data/cleanup_dataset.py --summary-only
python tests/data/cleanup_dataset.py --summary-only --apply
```
