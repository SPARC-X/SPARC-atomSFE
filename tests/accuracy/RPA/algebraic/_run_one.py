"""
Run ONE RPA accuracy case as a child process and print a RESULT_JSON block.

    python -u _run_one.py <mode> <electrons> <element>

Invoked as a subprocess by generate_reference_data.py so a crash in one case can't
take down the rest, and so peak RSS can be sampled from the parent without
instrumenting the solver.
"""

import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as C

mode, electrons, element = sys.argv[1], sys.argv[2], sys.argv[3]
params = C.params_for(mode, electrons, element)

sys.path.insert(0, C.CODE_ROOT)
os.chdir(C.CODE_ROOT)  # psp path defaults resolve relative to the package
from src.solver import AtomicDFTSolver  # noqa: E402

kwargs = dict(params)
kwargs.setdefault("verbose", True)

result = {"ok": False, "error": None}
try:
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
        wall_s              = wall_s,
        total_energy        = float(out["energy"]),
        rpa_correlation     = comps.get("rpa_correlation"),
        exact_exchange      = exact_exchange,
        homo                = eig[-1] if eig else None,
        eigen_energies      = eig,
        converged           = bool(out.get("converged", False)),
        scf_iterations      = int(out.get("iterations", -1) or -1),
        peak_rss_mb         = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    )
except BaseException as exc:  # noqa: BLE001 - report, never mask
    import traceback
    result["error"] = "%s: %s" % (type(exc).__name__, exc)
    result["traceback"] = traceback.format_exc()
    result["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

print("\nRESULT_JSON_BEGIN")
print(json.dumps(result))
print("RESULT_JSON_END")
sys.exit(0 if result["ok"] else 1)
