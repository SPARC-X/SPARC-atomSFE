"""
Case definitions for the RPA accuracy/timing/memory reference suite.

Not a pytest file (doesn't match test_*.py) -- imported by generate_reference_data.py
and by the pytest tests in this folder.

    RPA-OEP       self-consistent RPA        xc_functional='RPA',     use_oep=True
    RPA@GGA_PBE   non-self-consistent RPA    xc_functional='RPA@DFT', ground_state_functional='GGA_PBE'

5 elements x {AE, psp} for RPA-OEP, 5 elements x {AE only} for RPA@GGA_PBE = 15 cases.
Lmax is 20 for He/Be, 30 for Al/Ar, 40 for Au; everything else fixed and shared.

Reference numbers were generated with enable_parallelization=True on a 24-core
allocation (24 RPA frequency workers, matching omega=24) -- running with fewer
cores changes wall time and peak memory substantially, which is exactly what the
timing/memory checks (20% tolerance, warning-only) are watching for.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# tests/accuracy/RPA/algebraic_omega_mapping/ -> four levels to SPARC-atomSFE/
CODE_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

ELEMENTS = {"He": 2, "Be": 4, "Al": 13, "Ar": 18, "Au": 79}
LMAX     = {"He": 20, "Be": 20, "Al": 30, "Ar": 30, "Au": 40}

FIXED = dict(
    polynomial_order         = 20,
    quadrature_point_number  = 45,
    domain_size               = 13.0,
    mesh_type                 = "polynomial",
    mesh_concentration        = 2,
    scf_tolerance              = 1e-8,
    frequency_quadrature_point_number = 24,
)

MODES = {
    "RPA-OEP": dict(
        xc_functional="RPA", use_oep=True, ground_state_functional=None,
        electrons=("AE", "psp"),
    ),
    "RPA@GGA_PBE": dict(
        xc_functional="RPA@DFT", use_oep=False, ground_state_functional="GGA_PBE",
        electrons=("AE",),
    ),
}
MODE_ORDER = ["RPA-OEP", "RPA@GGA_PBE"]


def outdir(mode, electrons, element):
    return os.path.join(HERE, mode, electrons, element)


def params_for(mode, electrons, element):
    m = MODES[mode]
    p = dict(FIXED)
    p.update(
        xc_functional            = m["xc_functional"],
        use_oep                  = m["use_oep"],
        ground_state_functional  = m["ground_state_functional"],
        atomic_number            = ELEMENTS[element],
        finite_element_number    = 10,
        angular_momentum_cutoff  = LMAX[element],
        all_electron_flag        = (electrons == "AE"),
        enable_parallelization   = True,
        # use_preconditioner left at its default (True) -- driver.py's dielectric
        # preconditioner used to assert incorrectly for any non-OEP functional with
        # angular_momentum_cutoff set (e.g. 'RPA@DFT'); fixed at the source (three
        # call sites in driver.py gating angular_momentum_cutoff on switches.use_oep,
        # matching how unique_l_values is already built a few lines above each one).
    )
    return {k: v for k, v in p.items() if v is not None}


def all_cases():
    out = []
    for mode in MODE_ORDER:
        for electrons in MODES[mode]["electrons"]:
            for element in ELEMENTS:
                out.append(dict(mode=mode, electrons=electrons, element=element,
                               outdir=outdir(mode, electrons, element)))
    return out


ALL_CASES = all_cases()


def case_key(case):
    return "%s/%s/%s" % (case["mode"], case["electrons"], case["element"])


def run_subprocess_case(case, timeout=None):
    """
    Run one case in an isolated subprocess via _run_one.py and return the parsed
    result dict (plus raw stdout/stderr, for callers that want to keep a log).
    Used by BOTH generate_reference_data.py and the pytest tests, so a live test
    run and the stored reference were measured the same way -- in particular,
    peak RSS is only meaningful when each case gets its own process (a shared
    long-lived process's ru_maxrss is a running high-water mark across every case
    run in it, not a per-case figure).
    """
    cmd = [sys.executable, "-u", os.path.join(HERE, "_run_one.py"),
           case["mode"], case["electrons"], case["element"]]
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=timeout)
    m = re.search(r"RESULT_JSON_BEGIN\s*\n(.*?)\nRESULT_JSON_END", proc.stdout, re.S)
    result = (json.loads(m.group(1)) if m else
              {"ok": False, "error": "no RESULT_JSON block"})
    result["_stdout"] = proc.stdout
    result["_stderr"] = proc.stderr
    return result


if __name__ == "__main__":
    for c in ALL_CASES:
        print("%-12s %-5s %-3s -> %s" % (c["mode"], c["electrons"], c["element"], c["outdir"]))
    print("%d cases" % len(ALL_CASES))
