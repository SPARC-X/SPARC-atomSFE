"""
Run ONE RPA accuracy case as a child process and print a RESULT_JSON block.

    python -u _run_one.py <mode> <electrons> <element>

Invoked as a subprocess by generate_reference_data.py and by the pytest tests, so a
crash in one case cannot take down the rest and peak RSS is measured per case.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as C


def peak_rss_mb():
    """
    Peak resident set size in MiB, or None if unavailable.

    'resource' is Unix-only, so on Windows this falls back to the Win32 process-memory
    counters via ctypes and finally to psutil.  Returning None rather than raising
    keeps the accuracy checks -- the point of this suite -- working on a platform where
    only the soft timing/memory warnings are unavailable.
    """
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
    except Exception:                                   # noqa: BLE001 best effort only
        pass
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / (1024.0 * 1024.0)
    except Exception:                                   # noqa: BLE001
        return None


mode, electrons, element = sys.argv[1], sys.argv[2], sys.argv[3]
params = C.params_for(mode, electrons, element)

sys.path.insert(0, C.CODE_ROOT)
os.chdir(C.CODE_ROOT)  # psp path defaults resolve relative to the package
from src.solver import AtomicDFTSolver  # noqa: E402
from src.xc.rpa import RPACorrelation   # noqa: E402

kwargs = dict(params)
kwargs.setdefault("verbose", True)

result = {"ok": False, "error": None}
try:
    # The grid is NOT a solver argument -- it is a default on RPACorrelation.__init__,
    # so a case cannot request it and the reference numbers silently depend on it.
    # Record what was actually in force, and let the test compare.
    signature_defaults = RPACorrelation.__init__.__defaults__ or ()
    grid_type = next((v for v in signature_defaults
                      if v in ("sinh", "algebraic")), None)
    base_rule = next((v for v in signature_defaults
                      if v in ("midpoint", "trapezoid", "clenshaw_curtis",
                               "gauss_legendre")), None)
    omega_ceiling = next((v for v in signature_defaults
                          if isinstance(v, float) and v >= 1.0e3), None)

    t0 = time.time()
    solver = AtomicDFTSolver(**kwargs)
    out = solver.solve(save_full_spectrum=False)
    wall_s = time.time() - t0

    eig = [float(v) for v in out["eigen_energies"]]
    comps = out.get("energy_components")
    comps = ({k: float(v) for k, v in vars(comps).items() if isinstance(v, (int, float))}
             if comps is not None else {})
    exact_exchange = comps.get("oep_exchange", 0.0) or comps.get("hf_exchange", 0.0)

    result.update(
        ok                  = True,
        params              = params,
        frequency_grid_type = grid_type,
        frequency_base_rule = base_rule,
        omega_ceiling       = omega_ceiling,
        wall_s              = wall_s,
        total_energy        = float(out["energy"]),
        rpa_correlation     = comps.get("rpa_correlation"),
        exact_exchange      = exact_exchange,
        homo                = eig[-1] if eig else None,
        eigen_energies      = eig,
        converged           = bool(out.get("converged", False)),
        scf_iterations      = int(out.get("iterations", -1) or -1),
        peak_rss_mb         = peak_rss_mb(),
    )
except BaseException as exc:  # noqa: BLE001 - report, never mask
    import traceback
    result["error"] = "%s: %s" % (type(exc).__name__, exc)
    result["traceback"] = traceback.format_exc()
    result["peak_rss_mb"] = peak_rss_mb()

print("\nRESULT_JSON_BEGIN")
print(json.dumps(result))
print("RESULT_JSON_END")
sys.exit(0 if result["ok"] else 1)
