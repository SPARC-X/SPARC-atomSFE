"""Cleanup helper for large dataset folders.

Default behavior is dry-run (preview only). Use --apply to actually delete.
Keep rules are read from keep_list.txt in this folder.
The entire ``compare/`` and ``reference/`` trees are always skipped (scripts, figures,
and literature JSON under ``reference/hf``, etc.), regardless of keep-list or
``--summary-only``.

How to run:
- Preview deletions (default; no files removed):
  python delta/atom/tests/data/cleanup_dataset.py

- Actually delete matched files and prune empty directories:
  python delta/atom/tests/data/cleanup_dataset.py --apply

- Aggressive mode (keep only paths/filenames involving ``summary``, plus control files):
  python delta/atom/tests/data/cleanup_dataset.py --summary-only
  python delta/atom/tests/data/cleanup_dataset.py --summary-only --apply

- Another dataset root or keep-list path:
  python delta/atom/tests/data/cleanup_dataset.py --base-dir path/to/tests/data --keep-list path/to/keep_list.txt --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_KEEP_LIST = "keep_list.txt"
CONTROL_FILES = {
    ".gitkeep",
    "README.md",
    "keep_list.txt",
    "cleanup_dataset.py",
    "generate_dataset.py",
    "summary_naming.py",
    "build_summary_from_out.py",
}

# Never delete anything under these top-level dirs (even in ``--summary-only``).
_COMPARE_DIR_NAME = "compare"
_REFERENCE_DIR_NAME = "reference"


def _is_under_compare(relative: Path) -> bool:
    return bool(relative.parts) and relative.parts[0] == _COMPARE_DIR_NAME


def _is_under_reference(relative: Path) -> bool:
    return bool(relative.parts) and relative.parts[0] == _REFERENCE_DIR_NAME


def _truncate_path(path_text: str, max_len: int = 120) -> str:
    if len(path_text) <= max_len:
        return path_text
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return f"{path_text[:left]}...{path_text[-right:]}"


def _print_live_status(scanned: int, files_seen: int, candidates: int, relative_path: Path) -> None:
    status = (
        f"[SCAN] visited={scanned:,} files={files_seen:,} candidates={candidates:,} "
        f"checking={_truncate_path(relative_path.as_posix())}"
    )
    print(f"\r{status}", end="", flush=True)


def _clear_live_status_line() -> None:
    print("\r" + (" " * 200) + "\r", end="", flush=True)


def _log_candidate(relative_path: Path) -> None:
    _clear_live_status_line()
    print(f"[DEL] {_truncate_path(relative_path.as_posix(), max_len=180)}", flush=True)


def _read_keep_patterns(keep_list_path: Path) -> list[str]:
    patterns: list[str] = []
    lines = keep_list_path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _is_kept(relative_path: Path, patterns: list[str]) -> bool:
    rel_str = relative_path.as_posix()
    for pattern in patterns:
        if relative_path.match(pattern):
            return True
        # also allow pure filename patterns like "*.json" for nested files
        if "/" not in pattern and Path(rel_str).name and Path(rel_str).match(pattern):
            return True
    return False


def _collect_delete_candidates(
    base_dir: Path,
    patterns: list[str],
) -> list[Path]:
    candidates: list[Path] = []
    scanned = 0
    files_seen = 0
    for path in base_dir.rglob("*"):
        scanned += 1
        _print_live_status(scanned, files_seen, len(candidates), path.relative_to(base_dir))
        if not path.is_file():
            continue
        files_seen += 1
        relative = path.relative_to(base_dir)
        if _is_under_compare(relative) or _is_under_reference(relative):
            continue
        if _is_kept(relative, patterns):
            continue
        candidates.append(path)
        _log_candidate(relative)
    _clear_live_status_line()
    return candidates


def _is_summary_file(relative_path: Path) -> bool:
    return "summary" in relative_path.name.lower()


def _is_python_file(relative_path: Path) -> bool:
    return relative_path.suffix.lower() == ".py"


def _collect_delete_candidates_summary_only(base_dir: Path) -> list[Path]:
    """Same traversal contract as pattern mode: log and return delete candidates only."""
    candidates: list[Path] = []
    scanned = 0
    files_seen = 0
    for path in base_dir.rglob("*"):
        scanned += 1
        _print_live_status(scanned, files_seen, len(candidates), path.relative_to(base_dir))
        if not path.is_file():
            continue
        files_seen += 1
        relative = path.relative_to(base_dir)
        if _is_under_compare(relative) or _is_under_reference(relative):
            continue
        if relative.name in CONTROL_FILES:
            continue
        if _is_python_file(relative):
            continue
        if _is_summary_file(relative):
            continue
        candidates.append(path)
        _log_candidate(relative)
    _clear_live_status_line()
    return candidates


def _cleanup_empty_dirs(base_dir: Path, apply: bool) -> int:
    removed = 0
    dirs = sorted([p for p in base_dir.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        if d == base_dir:
            continue
        if any(d.iterdir()):
            continue
        if apply:
            d.rmdir()
        removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete large dataset files while preserving key scripts/summaries."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Target dataset folder (default: current script folder).",
    )
    parser.add_argument(
        "--keep-list",
        type=Path,
        default=None,
        help=f"Path to keep patterns file (default: {DEFAULT_KEEP_LIST} under base-dir).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, only print preview.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Aggressive mode: keep only files with 'summary' in filename "
            "(plus control scripts/files), delete everything else."
        ),
    )
    args = parser.parse_args()

    base_dir: Path = args.base_dir.resolve()
    keep_list_path = args.keep_list.resolve() if args.keep_list else (base_dir / DEFAULT_KEEP_LIST)
    if not keep_list_path.is_file():
        raise FileNotFoundError(f"Keep list not found: {keep_list_path}")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] base_dir={base_dir}")
    print(f"[{mode}] keep_list={keep_list_path}")
    print(f"[{mode}] keep_mode={'summary-only' if args.summary_only else 'pattern-list'}")
    print(f"[{mode}] Scanning files... this can take a while on first pass.")

    patterns = _read_keep_patterns(keep_list_path)
    if args.summary_only:
        candidates = _collect_delete_candidates_summary_only(base_dir)
    else:
        candidates = _collect_delete_candidates(base_dir, patterns)

    total_bytes = sum(p.stat().st_size for p in candidates)
    print(f"[{mode}] keep_patterns={len(patterns)}")
    print(f"[{mode}] delete_candidates={len(candidates)} files ({total_bytes / (1024**3):.3f} GiB)")

    preview_limit = 40
    print(f"[{mode}] delete preview (up to {preview_limit} lines):")
    for p in candidates[:preview_limit]:
        print(f"  - {p.relative_to(base_dir).as_posix()}")
    if len(candidates) > preview_limit:
        print(f"  ... and {len(candidates) - preview_limit} more")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to delete.")
        return

    for p in candidates:
        p.unlink(missing_ok=True)

    removed_empty_dirs = _cleanup_empty_dirs(base_dir, apply=True)
    print(f"\nDeleted {len(candidates)} files.")
    print(f"Removed {removed_empty_dirs} empty directories.")


if __name__ == "__main__":
    main()

