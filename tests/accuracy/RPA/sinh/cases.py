"""
Case definitions for the RPA accuracy suite on the SINH omega mapping.

Not a pytest file (doesn't match test_*.py) -- imported by generate_reference_data.py
and by the pytest tests in this folder.

    RPA-OEP       self-consistent RPA        xc_functional='RPA',     use_oep=True
    RPA@GGA_PBE   non-self-consistent RPA    xc_functional='RPA@DFT', ground_state_functional='GGA_PBE'

5 elements x {AE, psp} for RPA-OEP, 5 elements x {AE only} for RPA@GGA_PBE = 15 cases.

HOW THIS DIFFERS FROM ../algebraic/
-----------------------------------
Same 15 cases, three differences:

  * THE GRID.  That suite's reference numbers were generated when rpa.py defaulted to
    the algebraic map omega = c (1 + xi)/(1 - xi) on Gauss-Legendre nodes, which at
    n = 24 reaches only omega ~ 290 and is nowhere near converged for anything heavier
    than He.  rpa.py now defaults to 'sinh' with the midpoint base rule, so this folder
    measures the production path.  The two sets of numbers are NOT comparable and
    deliberately live in separate folders with separate reference.json files.

  * omega.  24 throughout there; here 48 for the self-consistent runs and 24 for the
    non-self-consistent ones -- see MODES below.

  * Lmax.  20/20/30/30/40 there, 5/5/10/10/10 here -- see LMAX below.

WHY Lmax IS LOWER HERE
----------------------
Lmax is a convergence parameter of the ANGULAR sum and is independent of the frequency
grid, so it only has to be held FIXED for the comparison to mean anything, not
converged.  Lower Lmax cuts the cost of every case by a large factor, which is what
makes regenerating all 15 references practical.  These numbers are therefore a
regression baseline for the omega mapping, not converged atomic energies -- do not
compare them against literature values or against ../algebraic_omega_mapping/.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# tests/accuracy/RPA/rpa_oep_mapping/ -> four levels to SPARC-atomSFE/
CODE_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

ELEMENTS = {"He": 2, "Be": 4, "Al": 13, "Ar": 18, "Au": 79}

#: 5 for the two light elements, 10 for the rest.  Held fixed, not converged -- see the
#: module docstring.
LMAX = {"He": 5, "Be": 5, "Al": 10, "Ar": 10, "Au": 10}

FIXED = dict(
    polynomial_order         = 20,
    quadrature_point_number  = 45,
    domain_size              = 13.0,
    mesh_type                = "polynomial",
    mesh_concentration       = 2,
    scf_tolerance            = 1e-8,
)

#: The grid rpa.py must be using for these references to mean anything.  None of the
#: three is a solver argument -- they are defaults on RPACorrelation.__init__ -- so a
#: case cannot pass them.  _run_one.py reports what was actually in force and the test
#: compares, so a moved default fails naming the cause rather than silently
#: reinterpreting the file.  All three are rpa.py's own defaults; nothing is patched.
EXPECTED_GRID_TYPE     = "sinh"
EXPECTED_BASE_RULE     = "midpoint"
EXPECTED_OMEGA_CEILING = 1.0e8

MODES = {
    # omega differs by mode because the two are scored on different things.  E_total and
    # E_c are variational in v_c, so a quadrature error in the potential reaches them at
    # second order and ~24 nodes holds them to ~1e-5.  The self-consistent runs are also
    # scored on the ionization potential, i.e. an EIGENVALUE, which responds at first
    # order and needs about two more ladder steps -- ~50 nodes for the same 1e-5.
    # 48 rather than 50 keeps it a multiple of the 24-core allocation.
    "RPA-OEP": dict(
        xc_functional="RPA", use_oep=True, ground_state_functional=None,
        electrons=("AE", "psp"), n_omega=48,
    ),
    "RPA@GGA_PBE": dict(
        xc_functional="RPA@DFT", use_oep=False, ground_state_functional="GGA_PBE",
        electrons=("AE",), n_omega=24,
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
        frequency_quadrature_point_number = m["n_omega"],
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
    Run one case in an isolated subprocess via _run_one.py and return the parsed result
    dict (plus raw stdout/stderr, for callers that want to keep a log).

    Used by BOTH generate_reference_data.py and the pytest tests, so a live run and the
    stored reference were measured the same way -- in particular peak RSS is only
    meaningful per-process; in one long-lived process it is a running high-water mark
    across every case, not a per-case figure.
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
        print("%-12s %-5s %-3s  Lmax=%-3d -> %s"
              % (c["mode"], c["electrons"], c["element"],
                 LMAX[c["element"]], c["outdir"]))
    print("%d cases" % len(ALL_CASES))
