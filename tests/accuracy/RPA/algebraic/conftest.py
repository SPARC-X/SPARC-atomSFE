"""
Session-wide warning for this folder: the RPA accuracy reference data was
generated with 24 cores (omega=24, one RPA frequency worker per core).  Running
these tests with fewer cores does not affect the hard accuracy checks (total
energy, RPA correlation energy, exact exchange energy, ionization potential are
core-count-independent), but it WILL make the timing/memory checks fire their
20%-tolerance warnings, since wall time and peak memory both scale with the
number of RPA workers actually available.
"""

import os
import warnings

REQUIRED_CORES = 24


def _available_cores():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def pytest_collection_modifyitems(config, items):
    if not any("accuracy/RPA" in str(item.fspath) for item in items):
        return
    n = _available_cores()
    if n < REQUIRED_CORES:
        warnings.warn(
            "RPA accuracy tests: only %d cores available, but the reference data "
            "was generated with %d (omega=24, one thread worker per core). "
            "STRONGLY recommended: run this with a %d-core allocation (e.g. "
            "`sbatch --cpus-per-task=%d` on whatever job wraps this pytest "
            "invocation), or expect the timing/memory checks below to warn -- "
            "they will not fail the accuracy checks, but the numbers will not be "
            "meaningful for judging performance regressions."
            % (n, REQUIRED_CORES, REQUIRED_CORES, REQUIRED_CORES),
            stacklevel=1,
        )
