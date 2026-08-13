"""
Session-wide warning for this folder: the reference data is generated with omega = 48
for the self-consistent runs and 24 for the non-self-consistent ones, so a 24-core
allocation reproduces the stored timing and memory figures -- two frequency batches per
core for RPA-OEP, one for RPA@GGA_PBE.

Running with fewer cores does NOT affect the hard accuracy checks -- total energy, RPA
correlation energy, exact exchange energy and ionization potential are all
core-count-independent -- but it will make the soft timing/memory checks fire their
20%-tolerance warnings, since wall time and peak memory both scale with the number of
RPA workers actually available.

Peak RSS is unavailable on some platforms (the 'resource' module is Unix-only;
_run_one.py falls back to the Win32 counters and then to psutil, and reports None if
none of those work).  A None simply skips the memory warning.
"""

import os
import warnings

REQUIRED_CORES = 24


def _available_cores():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def pytest_collection_modifyitems(config, items):
    # Matched on the folder name rather than a slash-joined path fragment: on Windows
    # fspath uses backslashes, so an "accuracy/RPA" substring test never fires there.
    if not any("rpa_oep_mapping" in str(item.fspath) for item in items):
        return
    n = _available_cores()
    if n < REQUIRED_CORES:
        warnings.warn(
            "RPA accuracy tests (sinh mapping): only %d cores available, but the "
            "reference data was generated with %d (omega=48 self-consistent, 24 "
            "otherwise). "
            "STRONGLY recommended: run with a %d-core allocation (e.g. "
            "`sbatch --cpus-per-task=%d` around the pytest invocation), or expect the "
            "timing/memory checks to warn -- they will not fail the accuracy checks, "
            "but the numbers will not be meaningful for judging performance "
            "regressions." % (n, REQUIRED_CORES, REQUIRED_CORES, REQUIRED_CORES),
            stacklevel=1,
        )
