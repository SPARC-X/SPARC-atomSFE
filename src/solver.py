"""
Atomic Density Functional Theory (DFT) Solver

    This module provides a comprehensive implementation of Atomic DFT solver using
    finite element method for solving the Kohn-Sham equations for atomic systems.

    The solver supports:
    - Multiple exchange-correlation functionals (LDA, GGA, Meta-GGA, Hybrid, etc.)
    - Both all-electron and pseudopotential calculations
    - Self-consistent field (SCF) iterations with convergence control
    - High-order finite element discretization with Legendre-Gauss-Lobatto nodes
    - Various mesh types (exponential, polynomial, uniform)

    @file    solver.py
    @brief   Atomic DFT Solver using finite element method
    @authors Qihao Cheng <qcheng61@gatech.edu>
             Shubhang Trivedi <strivedi44@gatech.edu>
             Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

    Copyright (c) 2025 Material Physics & Mechanics Group, Georgia Tech.
"""


from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import re
import time
import warnings
from pathlib import Path
from datetime import datetime


# Fix the relative import issue when running as a script
try:
    __package__
except NameError:
    __package__ = None

if __package__ is None:
    # Set the package name, so the relative import can work
    __package__ = 'atom'
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))


import numpy as np
np.set_printoptions(precision=20) 
np.set_printoptions(threshold=sys.maxsize)

from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple

# Mesh & operators
from .mesh.builder import Quadrature1D, Mesh1D
from .mesh.operators import GridData, RadialOperatorsBuilder

# Typing imports
from .pseudo.local import LocalPseudopotential
from .scf.energy import EnergyComponents
from .scf.driver import SCFResult, SwitchesFlags
from .xc.functional_requirements import get_functional_requirements, VALID_XC_FUNCTIONAL_LIST
from .utils.occupation_states import OccupationInfo
from .utils.periodic import atomic_number_to_name


# Get parallelization-related variables from package __init__ (avoid circular import)
# These are defined in atom/__init__.py but we access them via sys.modules to avoid import issues
_pkg_module = sys.modules.get(__package__)
if _pkg_module is not None:
    _NUMPY_IMPORTED_BEFORE_ATOMIC = getattr(_pkg_module, '_NUMPY_IMPORTED_BEFORE_ATOMIC', False)
    _BLAS_ENV_SINGLE_THREADED     = getattr(_pkg_module, '_BLAS_ENV_SINGLE_THREADED'    , False)
    _THREADPOOLCTL_INSTALLED      = getattr(_pkg_module, '_THREADPOOLCTL_INSTALLED'     , False)
else:
    # Fallback: try direct import (may fail during circular import)
    try:
        from . import _NUMPY_IMPORTED_BEFORE_ATOMIC, _BLAS_ENV_SINGLE_THREADED, _THREADPOOLCTL_INSTALLED
    except ImportError:
        # Safe defaults if import fails
        _NUMPY_IMPORTED_BEFORE_ATOMIC = False
        _BLAS_ENV_SINGLE_THREADED     = False
        _THREADPOOLCTL_INSTALLED      = False

# SCF stack
from .scf import (
    HamiltonianBuilder,
    DensityCalculator,
    PoissonSolver,
    SCFDriver,
    EnergyCalculator,
    EigenSolver,
    Mixer,
    format_wall_duration,
)


# Valid XC Functional (imported from functional_requirements for single source of truth)
# Valid XC Functional for OEP
VALID_XC_FUNCTIONAL_FOR_OEP_LIST = ['EXX', 'RPA', 'PBE0']

# XC Functionals which need outer loop
VALID_XC_FUNCTIONAL_FOR_OUTER_LOOP_LIST = ['HF', 'PBE0', 'EXX', 'RPA']

# Valid Mesh Type
VALID_MESH_TYPE_LIST = ['exponential', 'polynomial', 'uniform']


# Error Messages for basic physical parameters
ATOMIC_NUMBER_NOT_INTEGER_OR_FLOAT_ERROR = \
    "parameter 'atomic_number' must be a integer or float, get {} instead."
ATOMIC_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'atomic_number' must be greater than 0, get {} instead."
ATOMIC_NUMBER_LARGER_THAN_119_ERROR = \
    "parameter 'atomic_number' must be smaller than 119, get {} instead."
ATOMIC_NUMBER_NOT_INTEGER_VALUED_FOR_PSEUDOPOTENTIAL_CALCULATION_ERROR = \
    "parameter 'atomic_number' must be integer-valued for pseudopotential calculations, get {} instead."
N_ELECTRONS_NOT_INTEGER_OR_FLOAT_ERROR = \
    "parameter 'n_electrons' must be an integer or float, get {} instead."
N_ELECTRONS_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'n_electrons' must be greater than 0, get {} instead."
CHARGE_SYSTEMS_NOT_SUPPORTED_FOR_PSEUDOPOTENTIAL_CALCULATION_ERROR = \
    "Charged systems are not supported with pseudopotentials. Use all-electron calculations for non-neutral systems."
ALL_ELECTRON_FLAG_NOT_BOOL_ERROR = \
    "parameter 'all_electron_flag' must be a boolean, get {} instead."
SPIN_POLARIZED_FLAG_NOT_BOOL_ERROR = \
    "parameter 'spin_polarized_flag' must be a boolean, get {} instead."
XC_FUNCTIONAL_NOT_STRING_ERROR = \
    "parameter 'xc_functional' must be a string, get {} instead."
XC_FUNCTIONAL_TYPE_ERROR_MESSAGE = \
    "parameter 'xc_functional' must be a string, get type {} instead."
XC_FUNCTIONAL_NOT_IN_VALID_LIST_ERROR = \
    "parameter 'xc_functional' must be in {}, get {} instead."
USE_OEP_NOT_BOOL_ERROR = \
    "parameter 'use_oep' must be a boolean, get {} instead."
USE_OEP_NOT_TRUE_FOR_OEP_FUNCTIONAL_ERROR = \
    "parameter 'use_oep' must be True for OEP functional '{}', get {} instead."
USE_OEP_NOT_FALSE_FOR_NON_OEP_FUNCTIONAL_ERROR = \
    "parameter 'use_oep' must be False for non-OEP functional '{}', get {} instead."

# Error Messages for grid, basis, and mesh parameters
DOMAIN_SIZE_NOT_FLOAT_ERROR = \
    "parameter 'domain_size' must be a float, get {} instead."
DOMAIN_SIZE_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'domain_size' must be greater than 0, get {} instead."
NUMBER_OF_FINITE_ELEMENTS_NOT_INTEGER_ERROR = \
    "parameter 'number_of_finite_elements' must be an integer, get {} instead."
NUMBER_OF_FINITE_ELEMENTS_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'number_of_finite_elements' must be greater than 0, get {} instead."
FINITE_ELEMENT_NUMBER_NOT_INTEGER_ERROR = \
    "parameter 'finite_element_number' must be an integer, get {} instead."
FINITE_ELEMENT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'finite_element_number' must be greater than 0, get {} instead."
POLYNOMIAL_ORDER_NOT_INTEGER_ERROR = \
    "parameter 'polynomial_order' must be an integer, get {} instead."
POLYNOMIAL_ORDER_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'polynomial_order' must be greater than 0, get {} instead."
QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR = \
    "parameter 'quadrature_point_number' must be an integer, get {} instead."
QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'quadrature_point_number' must be greater than 0, get {} instead."
QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_2_POLYNOMIAL_ORDER_PLUS_3_ERROR = (
    "parameter 'quadrature_point_number' must be greater than '2 * polynomial_order + 3', "
    "i.e. at least 2 * {} + 3 = {}, get {} instead."
)
OEP_BASIS_NUMBER_NOT_INTEGER_ERROR = \
    "parameter 'oep_basis_number' must be an integer, get {} instead."
OEP_BASIS_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'oep_basis_number' must be greater than 0, get {} instead."
MESH_TYPE_NOT_STRING_ERROR = \
    "parameter 'mesh_type' must be a string, get {} instead."
MESH_TYPE_NOT_IN_VALID_LIST_ERROR = \
    "parameter 'mesh_type' must be in {}, get {} instead."
MESH_CONCENTRATION_NOT_FLOAT_ERROR = \
    "parameter 'mesh_concentration' must be a float, get {} instead."
MESH_CONCENTRATION_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'mesh_concentration' must be greater than 0, get {} instead."
MESH_SPACING_NOT_FLOAT_ERROR = \
    "parameter 'mesh_spacing' must be a float, get {} instead."
MESH_SPACING_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'mesh_spacing' must be greater than 0, get {} instead."

# Error Messages for self-consistent field (SCF) convergence parameters
SCF_TOLERANCE_NOT_FLOAT_ERROR = \
    "parameter 'scf_tolerance' must be a float, get {} instead."
SCF_TOLERANCE_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'scf_tolerance' must be greater than 0, get {} instead."
MAX_SCF_ITERATIONS_NOT_INTEGER_ERROR = \
    "parameter 'max_scf_iterations' must be an integer, get {} instead."
MAX_SCF_ITERATIONS_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'max_scf_iterations' must be greater than 0, get {} instead."
MAX_SCF_ITERATIONS_OUTER_NOT_INTEGER_ERROR = \
    "parameter 'max_scf_iterations_outer' must be an integer, get {} instead."
MAX_SCF_ITERATIONS_OUTER_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'max_scf_iterations_outer' must be greater than 0, get {} instead."
USE_PULAY_MIXING_NOT_BOOL_ERROR = \
    "parameter 'use_pulay_mixing' must be a boolean, get {} instead."
USE_PRECONDITIONER_NOT_BOOL_ERROR = \
    "parameter 'use_preconditioner' must be a boolean, get {} instead."
PULAY_MIXING_PARAMETER_NOT_FLOAT_ERROR = \
    "parameter 'pulay_mixing_parameter' must be a float, get {} instead."
PULAY_MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR = \
    "parameter 'pulay_mixing_parameter' must be in [0, 1], get {} instead."
PULAY_MIXING_HISTORY_NOT_INTEGER_ERROR = \
    "parameter 'pulay_mixing_history' must be an integer, get {} instead."
PULAY_MIXING_HISTORY_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'pulay_mixing_history' must be greater than 0, get {} instead."
PULAY_MIXING_FREQUENCY_NOT_INTEGER_ERROR = \
    "parameter 'pulay_mixing_frequency' must be an integer, get {} instead."
PULAY_MIXING_FREQUENCY_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'pulay_mixing_frequency' must be greater than 0, get {} instead."
LINEAR_MIXING_ALPHA1_NOT_FLOAT_ERROR = \
    "parameter 'linear_mixing_alpha1' must be a float, get {} instead."
LINEAR_MIXING_ALPHA1_NOT_IN_ZERO_ONE_ERROR = \
    "parameter 'linear_mixing_alpha1' must be in [0, 1], get {} instead."
LINEAR_MIXING_ALPHA2_NOT_FLOAT_ERROR = \
    "parameter 'linear_mixing_alpha2' must be a float, get {} instead."
LINEAR_MIXING_ALPHA2_NOT_IN_ZERO_ONE_ERROR = \
    "parameter 'linear_mixing_alpha2' must be in [0, 1], get {} instead."

# Error Messages for pseudopotential parameters
PSP_DIR_PATH_NOT_STRING_ERROR = \
    "parameter 'psp_dir_path' must be a string, get {} instead."
PSP_DIR_PATH_NOT_EXISTS_ERROR = \
    "parameter 'psp_dir_path' default directory path {} does not exist, please provide a valid psp directory path."
PSP_FILE_NAME_NOT_STRING_ERROR = \
    "parameter 'psp_file_name' must be a string, get {} instead."
PSP_FILE_NAME_NOT_EXISTS_ERROR = \
    "parameter 'psp_file_name' '{}' does not exist in the psp file path '{}', please provide a valid psp file name."

# Error Messages for advanced functional parameters
HYBRID_MIXING_PARAMETER_NOT_FLOAT_ERROR = \
    "parameter 'hybrid_mixing_parameter' must be a float, get {} instead."
HYBRID_MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR = \
    "parameter 'hybrid_mixing_parameter' must be in [0, 1], get {} instead."
HYBRID_MIXING_PARAMETER_NOT_ONE_FOR_NON_HYBRID_FUNCTIONAL_ERROR = \
    "parameter 'hybrid_mixing_parameter' must be 1.0 for non-hybrid functional, get {} instead."
HYBRID_MIXING_PARAMETER_NOT_ONE_ERROR = \
    "parameter 'hybrid_mixing_parameter' must be 1.0 for functional {}, get {} instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR = \
    "parameter 'frequency_quadrature_point_number' must be an integer, get {} instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'frequency_quadrature_point_number' must be greater than 0, get {} instead."
ANGULAR_MOMENTUM_CUTOFF_NOT_INTEGER_ERROR = \
    "parameter 'angular_momentum_cutoff' must be an integer, get {} instead."
ANGULAR_MOMENTUM_CUTOFF_NEGATIVE_ERROR = \
    "parameter 'angular_momentum_cutoff' must be non-negative, get {} instead."
OEP_MIXING_PARAMETER_NOT_FLOAT_ERROR = \
    "parameter 'oep_mixing_parameter' must be a float, get {} instead."
OEP_MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR = \
    "parameter 'oep_mixing_parameter' must be in [0, 1], get {} instead."
ENABLE_PARALLELIZATION_NOT_BOOL_ERROR = \
    "parameter 'enable_parallelization' must be a boolean, get {} instead."

# Error Messages for debugging and verbose parameters
VERBOSE_NOT_BOOL_ERROR = \
    "parameter 'verbose' must be a boolean, get {} instead."

# Error Messages for machine learning model parameters
ML_XC_CALCULATOR_NOT_MLXCCALCULATOR_ERROR = \
    "parameter 'ml_xc_calculator' must be a MLXCCalculator, get {} instead."
ML_EACH_SCF_STEP_NOT_BOOL_ERROR = \
    "parameter 'ml_each_scf_step' must be a boolean, get {} instead."
ML_SMOOTH_RHO_FOR_FEATURES_NOT_BOOL_ERROR = \
    "parameter 'ml_smooth_rho_for_features' must be a boolean, get {} instead."

ML_XC_CALCULATOR_TARGET_FUNCTIONAL_NOT_EQUAL_TO_XC_FUNCTIONAL_ERROR = \
    """
    [ATOM ERROR] Machine Learning Exchange-Correlation (MLXC) target functional mismatch.

    The MLXC calculator was created for target functional '{}', but the solver
    is configured with xc_functional '{}'. These must match so the ML correction
    is applied to the intended target functional.

    How this path works:
        1) The solver validates that ml_xc_calculator.target_functional matches
           xc_functional.
        2) If valid, it switches xc_functional to
           ml_xc_calculator.reference_functional to run the reference SCF.
        3) The ML model provides a delta correction to approximate the target.

    How to fix:
        • Set xc_functional to the same value as ml_xc_calculator.target_functional.
        • Or create the MLXCCalculator with a target that matches xc_functional.
    """

# Error Messages for output file
OUTPUT_FILE_NOT_FOUND_ERROR = \
    "output file not found: {}."
OUTPUT_FILE_NO_INPUT_BLOCK_ERROR = \
    "no INPUT PARAMETERS block found in output file."
OUTPUT_FILE_MULTIPLE_INPUT_BLOCKS_ERROR = \
    "multiple INPUT PARAMETERS blocks found; cannot determine which one to use."
OUTPUT_FILE_MISSING_ATOMIC_NUMBER_ERROR = \
    "failed to parse atomic_number from output file."

# Error Messages for solve() methods
SAVE_INTERMEDIATE_NOT_BOOL_ERROR = \
    "parameter 'save_intermediate' must be a boolean, get {} instead."
SAVE_ENERGY_DENSITY_NOT_BOOL_ERROR = \
    "parameter 'save_energy_density' must be a boolean, get {} instead."
SAVE_FULL_SPECTRUM_NOT_BOOL_ERROR = \
    "parameter 'save_full_spectrum' must be a boolean, get {} instead."
RHO_INITIAL_NOT_NUMPY_ARRAY_ERROR = \
    "parameter 'rho_initial' must be a numpy array, get {} instead."
RHO_INITIAL_LENGTH_MISMATCH_ERROR = \
    "parameter 'rho_initial' must have length {}, get {} instead."
ORBITALS_INITIAL_NOT_NUMPY_ARRAY_ERROR = \
    "parameter 'orbitals_initial' must be a numpy array, get {} instead."
ORBITALS_INITIAL_FIRST_DIM_MISMATCH_ERROR = \
    "parameter 'orbitals_initial' first dimension must be {} (quadrature points), get {} instead."
USE_WARM_START_NOT_BOOL_ERROR = \
    "parameter 'use_warm_start' must be a boolean, get {} instead."
EVALUATE_BASIS_ON_UNIFORM_GRID_NOT_BOOL_ERROR = \
    "parameter 'evaluate_basis_on_uniform_grid' must be a boolean, get {} instead."


# WARNING Messages
MESH_CONCENTRATION_NOT_NONE_FOR_UNIFORM_MESH_TYPE_WARNING = \
    "WARNING: parameter 'mesh_concentration' is not None for uniform mesh type, so it will be ignored."
MAX_SCF_ITERATIONS_OUTER_NOT_NONE_AND_NOT_ONE_FOR_XC_FUNCTIONAL_OTHER_THAN_OUTER_LOOP_LIST_WARNING = \
    "WARNING: parameter 'max_scf_iterations_outer' is not None and not 1 for XC functional '{}' which does not require outer loop, so it will be ignored."
PULAY_MIXING_PARAMETER_NOT_NONE_WHEN_USE_PULAY_MIXING_IS_FALSE_WARNING = \
    "WARNING: parameter 'pulay_mixing_parameter' is not None when 'use_pulay_mixing' is False, so it will be ignored."
PULAY_MIXING_HISTORY_NOT_NONE_WHEN_USE_PULAY_MIXING_IS_FALSE_WARNING = \
    "WARNING: parameter 'pulay_mixing_history' is not None when 'use_pulay_mixing' is False, so it will be ignored."
PULAY_MIXING_FREQUENCY_NOT_NONE_WHEN_USE_PULAY_MIXING_IS_FALSE_WARNING = \
    "WARNING: parameter 'pulay_mixing_frequency' is not None when 'use_pulay_mixing' is False, so it will be ignored."

PSP_DIR_PATH_NOT_NONE_FOR_ALL_ELECTRON_CALCULATION_WARNING = \
    "WARNING: parameter 'psp_dir_path' is not None for all-electron calculation, so it will be ignored."
PSP_FILE_NAME_NOT_NONE_FOR_ALL_ELECTRON_CALCULATION_WARNING = \
    "WARNING: parameter 'psp_file_name' is not None for all-electron calculation, so it will be ignored."
NO_HYBRID_MIXING_PARAMETER_PROVIDED_FOR_HYBRID_FUNCTIONAL_WARNING = \
    "WARNING: 'hybrid_mixing_parameter' not provided for {} functional, using default value {}."
HYBRID_MIXING_PARAMETER_NOT_IN_ZERO_ONE_WARNING = \
    "WARNING: 'hybrid_mixing_parameter' for {} should be in [0, 1], get {} instead."
HYBRID_MIXING_PARAMETER_NOT_ONE_FOR_NON_HYBRID_FUNCTIONAL_WARNING = \
    "WARNING: 'hybrid_mixing_parameter' for {} must be 1.0 for non-hybrid functional, get {} instead."
HYBRID_MIXING_PARAMETER_NOT_FLOAT_WARNING = \
    "WARNING: 'hybrid_mixing_parameter' for {} must be a float, get {} instead."
HYBRID_MIXING_PARAMETER_NOT_IN_ZERO_ONE_WARNING = \
    "WARNING: 'hybrid_mixing_parameter' for {} must be in [0, 1], get {} instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_NONE_FOR_OEPX_AND_NONE_XC_FUNCTIONAL_WARNING = \
    "WARNING: parameter 'frequency_quadrature_point_number' is not None for XC functional '{}', so it will be ignored."
ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_FOR_XC_FUNCTIONAL_OTHER_THAN_RPA_WARNING = \
    "WARNING: parameter 'angular_momentum_cutoff' is not None for XC functional '{}', so it will be ignored."
ENABLE_PARALLELIZATION_NOT_NONE_FOR_XC_FUNCTIONAL_OTHER_THAN_RPA_WARNING = \
    "WARNING: parameter 'enable_parallelization' is not None for XC functional '{}', so it will be ignored."
ML_EACH_SCF_STEP_NOT_NONE_FOR_ML_XC_CALCULATOR_NOT_NONE_WARNING = \
    "WARNING: parameter 'ml_each_scf_step' is not None for machine learning model, so it will be ignored."
ML_SMOOTH_RHO_FOR_FEATURES_NOT_NONE_WITHOUT_ML_XC_WARNING = \
    "WARNING: parameter 'ml_smooth_rho_for_features' is not None without ml_xc_calculator; it will be set to False."
WARM_START_NOT_CONVERGED_WARNING = \
    "WARNING: warm start calculation for '{}' did not converge, using intermediate result."

# This warning message is only raised when user wants to enable parallel execution of RPA calculations
NUMPY_IMPORTED_BEFORE_ATOMIC_WARNING = \
    """
    [ATOM WARNING] NumPy was imported before the 'atom' package.

    This prevents the package from forcing BLAS libraries (MKL / OpenBLAS / NumExpr)
    into single-thread mode. When parallel RPA calculations attempt to use
    multiple Python threads or processes, each NumPy/SciPy linear algebra call may
    internally spawn many BLAS threads.

    This can lead to:
        • Severe CPU oversubscription (N_threads x BLAS_threads >> CPU cores),
        • Very slow execution or apparent freezing/hanging,
        • Poor scaling in parallel regions.

    To safely enable parallel execution, please choose ONE of the following:

    1) Import 'atom' BEFORE importing NumPy/SciPy, e.g.:

            import atom
            import numpy as np

    2) Configure BLAS to single-thread mode *before importing NumPy*, e.g.:

            # In your shell (recommended):
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export NUMEXPR_NUM_THREADS=1

        or in Python BEFORE NumPy is imported:

            import os
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"

    3) Install 'threadpoolctl' in your environment, which allows the package to 
        dynamically limit BLAS threads even if NumPy was already imported:

            pip install threadpoolctl

        After installation, 'atom' will automatically detect it and  
        apply safe single-thread limits for parallel execution.

    Parallel mode is disabled for this run to avoid deadlocks or CPU thrashing.
    """


# Deprecated parameters
PRINT_DEBUG_DEPRECATED_WARNING = \
    "WARNING: parameter 'print_debug' is now deprecated, use 'verbose' instead."
PRINT_DEBUG_AND_VERBOSE_BOTH_SPECIFIED_ERROR = \
    "parameter 'print_debug' and 'verbose' cannot be specified at the same time."
NUMBER_OF_FINITE_ELEMENTS_DEPRECATED_WARNING = \
    "WARNING: parameter 'number_of_finite_elements' is now deprecated, use 'finite_element_number' instead."
FINITE_ELEMENT_NUMBER_AND_NUMBER_OF_FINITE_ELEMENTS_BOTH_SPECIFIED_ERROR = \
    "Cannot specify both 'finite_element_number' and deprecated 'number_of_finite_elements'. Use 'finite_element_number' only."


def get_sparc_time_string() -> str:
    """
    Generate a time string in SPARC format: "Tue Oct 14 13:54:05 2025"
    
    Returns:
        str: Formatted time string matching SPARC output format
    """
    return datetime.now().strftime("%a %b %d %H:%M:%S %Y")


class AtomicDFTSolver:
    """
    Atomic Density Functional Theory (DFT) Solver using finite element method.
    
    This class provides a comprehensive interface for solving the Kohn-Sham equations
    for atomic systems using various exchange-correlation functionals and computational
    parameters. It supports both all-electron and pseudopotential calculations.
    """

    # Basic physical parameters
    atomic_number                     : float # Atomic number of the element to calculate (e.g., 13 for Aluminum), can be fractional
    n_electrons                       : float # Number of electrons in the system, can be fractional
    all_electron_flag                 : bool  # True for all-electron calculation, False for pseudopotential calculation
    spin_polarized_flag               : bool  # internal: always False (constructor toggle hidden in this build)
    xc_functional                     : str   # XC functional type: 'GGA_PBE', 'RPA', 'EXX', 'LDA_SVWN', 'LDA_PZ', 'LDA_PW', 'SCAN', 'RSCAN', 'R2SCAN'
    use_oep                           : bool  # Enable optimized effective potential (OEP) workflow in SCF

    # Grid, basis, and mesh parameters
    domain_size                       : float # Radial computational domain size in atomic units (typically 10-30 Bohr)
    finite_element_number             : int   # Number of finite elements in the computational domain
    polynomial_order                  : int   # Polynomial order of basis functions within each finite element
    quadrature_point_number           : int   # Number of quadrature points for numerical integration (recommended: 3-4x polynomial_order)
    oep_basis_number                  : int   # Basis size used in OEP calculations when enabled
    mesh_type                         : str   # Mesh distribution type: 'exponential' (higher density near nucleus), 'polynomial', or 'uniform'
    mesh_concentration                : float # Mesh concentration parameter (controls point density distribution)
    mesh_spacing                      : float # Used to set the output uniform mesh spacing, irrelevant during SCF calculation

    # Self-consistent field (SCF) convergence parameters
    scf_tolerance                     : float # SCF convergence tolerance (typically 1e-8)
    max_scf_iterations                : int   # Maximum number of inner SCF iterations
    max_scf_iterations_outer          : int   # Maximum number of outer SCF iterations (for functionals requiring outer loop like HF, EXX, RPA, PBE0)
    use_pulay_mixing                  : bool  # True for Pulay mixing for SCF convergence, False for linear mixing (True by default)
    use_preconditioner                : bool  # Flag for using preconditioner for SCF convergence (True by default)
    pulay_mixing_parameter            : float # Pulay mixing parameter 
    pulay_mixing_history              : int   # Pulay mixing history   
    pulay_mixing_frequency            : int   # Pulay mixing frequency 
    linear_mixing_alpha1              : float # Linear mixing parameter (alpha_1 in linear mixing)
    linear_mixing_alpha2              : float # Linear mixing parameter (alpha_2 in linear mixing)

    # Pseudopotential parameters
    psp_dir_path                      : str   # Path to pseudopotential files directory (required when all_electron_flag=False)
    psp_file_name                     : str   # Name of the pseudopotential file (required when all_electron_flag=False)
    
    # Advanced functional parameters (for EXX, RPA, etc.)
    hybrid_mixing_parameter           : float # Mixing parameter for hybrid/double-hybrid functionals (e.g., 0.25 for PBE0)
    frequency_quadrature_point_number : int   # Number of frequency quadrature points for RPA calculations
    angular_momentum_cutoff           : int   # Maximum angular momentum quantum number to include
    double_hybrid_flag                : bool  # internal: always False (constructor toggle hidden in this build)
    oep_mixing_parameter              : float # Scaling parameter (λ) for OEP exchange/correlation potentials
    enable_parallelization            : bool  # Flag for parallelization of RPA calculations
    
    # Debugging and verbose parameters
    verbose                           : bool  # Flag for printing information during execution

    # ML-XC hooks (scf/energy internals); not exposed on AtomicDFTSolver in this build — always inactive.
    ml_xc_calculator                  : Any
    ml_each_scf_step                  : bool
    ml_smooth_rho_for_features        : bool


    def __init__(self, 
        atomic_number                     : int | float,                       # Only atomic_number is required, all other parameters have default values
        n_electrons                       : Optional[int | float]    = None,   # Number of electrons in the system, by default, set to atomic_number
        all_electron_flag                 : Optional[bool]           = None,   # False by default
        xc_functional                     : Optional[str]            = None,   # 'GGA_PBE' by default
        use_oep                           : Optional[bool]           = None,   # False by default

        domain_size                       : Optional[float]          = None,   # 20.0 by default
        finite_element_number             : Optional[int]            = None,   # 12 by default
        polynomial_order                  : Optional[int]            = None,   # 20 by default
        quadrature_point_number           : Optional[int]            = None,   # 60 by default
        oep_basis_number                  : Optional[int]            = None,   # not needed by default, if needed, int(polynomial_order * 0.25) by default
        mesh_type                         : Optional[str]            = None,   # 'exponential' by default
        mesh_concentration                : Optional[float]          = None,   # 61.0 by default
        mesh_spacing                      : Optional[float]          = None,   # 0.1 by default

        scf_tolerance                     : Optional[float]          = None,   # 1e-8 by default (1e-6 for SCAN/RSCAN/R2SCAN functionals)
        max_scf_iterations                : Optional[int]            = None,   # 500 by default, maximum number of inner SCF iterations
        max_scf_iterations_outer          : Optional[int]            = None,   # 50   by default for certain functionals, otherwise not needed
        use_pulay_mixing                  : Optional[bool]           = None,   # True by default
        use_preconditioner                : Optional[bool]           = None,   # True by default if use_pulay_mixing=True, False by default if use_pulay_mixing=False
        pulay_mixing_parameter            : Optional[float]          = None,   # 1.0  by default if use_preconditioner=True, 0.45 by default if use_preconditioner=False
        pulay_mixing_history              : Optional[int]            = None,   # 7    by default if use_preconditioner=True, 11   by default if use_preconditioner=False
        pulay_mixing_frequency            : Optional[int]            = None,   # 3    by default if use_preconditioner=True, 1    by default if use_preconditioner=False
        linear_mixing_alpha1              : Optional[float]          = None,   # 0.75 by default if use_pulay_mixing=True  , 0.7  by default if use_pulay_mixing=False
        linear_mixing_alpha2              : Optional[float]          = None,   # 0.95 by default if use_pulay_mixing=True  , 1.0  by default if use_pulay_mixing=False

        psp_dir_path                      : Optional[str]            = None,   # ../psps by default
        psp_file_name                     : Optional[str]            = None,   # {atomic_number}.psp8 by default

        hybrid_mixing_parameter           : Optional[float]          = None,   # 1.0 by default (0.25 for PBE0, variable for RPA)
        frequency_quadrature_point_number : Optional[int]            = None,   # for RPA, 25 by default, otherwise not needed
        angular_momentum_cutoff           : Optional[int]            = None,   # for RPA, 4 by default, otherwise not needed
        oep_mixing_parameter              : Optional[float]          = None,   # 1.0 by default (scales OEP potentials), used for double hybrid functional only
        enable_parallelization            : Optional[bool]           = None,   # for RPA, False by default, otherwise not needed

        verbose                           : Optional[bool]           = None,   # False by default

        # deprecated parameters
        print_debug                       : Optional[bool]           = None,   # Deprecated: use verbose instead
        number_of_finite_elements         : Optional[int]            = None,   # Deprecated: use finite_element_number instead
    ):   

        """
        Initialize the AtomicDFTSolver.
        

        Basic physical parameters
        --------------------------
        `atomic_number` : float
            Atomic number of the element (e.g., 13 for Aluminum), can be fractional.
        `n_electrons` : float
            Number of electrons in the system, can also be fractional. Defaults to atomic_number.
        `all_electron_flag` : bool
            True for all-electron, False for pseudopotential. Defaults to False.
        `xc_functional` : str
            Exchange-correlation functional ('GGA_PBE', 'RPA', 'EXX', etc.). Defaults to 'GGA_PBE'.
        `use_oep` : bool
            Enable optimized effective potential calculations. Defaults to False.

        Grid, basis, and mesh parameters
        --------------------------------
        `domain_size` : float
            Radial domain size in atomic units (typically 10-30). Defaults to 20.0.
        `finite_element_number` : int
            Number of finite elements in the domain. Defaults to 12.
        `polynomial_order` : int
            Polynomial order of basis functions (typically 20-40). Defaults to 20.
        `quadrature_point_number` : int
            Quadrature points for integration (3-4x polynomial_order). Defaults to 60.
        `oep_basis_number` : int
            Size of OEP auxiliary basis. Defaults to int(polynomial_order * 0.25) when use_oep=True.
        `mesh_type` : str
            Mesh type ('exponential', 'polynomial', 'uniform'). Defaults to 'exponential'.
        `mesh_concentration` : float
            Mesh concentration parameter (controls point density). Defaults based on mesh_type.
        `mesh_spacing` : float
            Mesh spacing for the uniform grid, used to set the output mesh spacing. Defaults to 0.1.

        Self-consistent field (SCF) convergence parameters
        --------------------------------------------------
        `scf_tolerance` : float
            SCF convergence tolerance (typically 1e-8). Defaults to 1e-8 (1e-6 for SCAN/RSCAN/R2SCAN functionals).
        `max_scf_iterations` : int
            Maximum number of inner SCF iterations. Defaults to 500.
        `max_scf_iterations_outer` : int
            Maximum number of outer SCF iterations (for functionals requiring outer loop like HF, EXX, RPA, PBE0). 
            Defaults to 50 when needed, otherwise not used.
        `use_pulay_mixing` : bool
            True for Pulay mixing for SCF convergence, False for linear mixing. Defaults to True.
        `use_preconditioner` : bool
            Flag for using preconditioner for SCF convergence. Defaults to True if use_pulay_mixing=True, False if use_pulay_mixing=False.
            Can be used with both Pulay mixing and linear mixing.
        `pulay_mixing_parameter` : float
            Pulay mixing parameter. Defaults to 1.0 if use_preconditioner=True, 0.45 if False.
        `pulay_mixing_history` : int
            Pulay mixing history. Defaults to 7 if use_preconditioner=True, 11 if False.
        `pulay_mixing_frequency` : int
            Pulay mixing frequency. Defaults to 3 if use_preconditioner=True, 1 if False.
        `linear_mixing_alpha1` : float
            Linear mixing parameter (alpha_1). Defaults to 0.75 if use_pulay_mixing=True, 0.7 if False.
        `linear_mixing_alpha2` : float
            Linear mixing parameter (alpha_2). Defaults to 0.95 if use_pulay_mixing=True, 1.0 if False.

        Pseudopotential parameters
        ---------------------------
        `psp_dir_path` : str
            Path to pseudopotential directory (required if all_electron_flag=False). Defaults to '../psps'.
        `psp_file_name` : str
            Name of pseudopotential file (required if all_electron_flag=False). Defaults to '{atomic_number}.psp8'.

        Advanced functional parameters (for EXX, RPA, etc.)
        --------------------------------------------------
        `hybrid_mixing_parameter` : float
            Mixing parameter for hybrid functionals (e.g., 0.25 for PBE0). Defaults based on functional.
        `frequency_quadrature_point_number` : int
            Frequency quadrature points for RPA calculations, used for RPA functional only. Defaults to 25.
        `angular_momentum_cutoff` : int
            Maximum angular momentum quantum number, used for RPA functional only. Defaults to 4.
        `oep_mixing_parameter` : float
            Mixing parameter for OEP functionals (lambda in OEP). Defaults to 1.0.
        `enable_parallelization` : bool
            Enable parallelization for RPA calculations. Defaults to False.

        Debugging and verbose parameters
        --------------------------------
        `verbose` : bool
            Whether to print information during execution. Defaults to False.
        """

        # handle deprecated parameters
        if print_debug is not None:
            if verbose is not None:
                raise ValueError(PRINT_DEBUG_AND_VERBOSE_BOTH_SPECIFIED_ERROR)
            verbose = print_debug
            warnings.warn(PRINT_DEBUG_DEPRECATED_WARNING, DeprecationWarning, stacklevel=2)
        
        # Handle deprecated number_of_finite_elements parameter
        if number_of_finite_elements is not None:
            if finite_element_number is not None:
                raise ValueError(FINITE_ELEMENT_NUMBER_AND_NUMBER_OF_FINITE_ELEMENTS_BOTH_SPECIFIED_ERROR)
            finite_element_number = number_of_finite_elements
            warnings.warn(NUMBER_OF_FINITE_ELEMENTS_DEPRECATED_WARNING, DeprecationWarning, stacklevel=2)

        # Initialize the class attributes
        self.atomic_number                     = atomic_number
        self.n_electrons                       = n_electrons
        self.all_electron_flag                 = all_electron_flag
        self.spin_polarized_flag               = False
        self.xc_functional                     = xc_functional
        self.use_oep                           = use_oep

        self.domain_size                       = domain_size
        self.finite_element_number             = finite_element_number
        self.polynomial_order                  = polynomial_order
        self.quadrature_point_number           = quadrature_point_number
        self.oep_basis_number                  = oep_basis_number
        self.mesh_type                         = mesh_type
        self.mesh_concentration                = mesh_concentration
        self.mesh_spacing                      = mesh_spacing

        self.scf_tolerance                     = scf_tolerance
        self.max_scf_iterations                = max_scf_iterations
        self.max_scf_iterations_outer          = max_scf_iterations_outer
        self.use_pulay_mixing                  = use_pulay_mixing
        self.use_preconditioner                = use_preconditioner
        self.pulay_mixing_parameter            = pulay_mixing_parameter
        self.pulay_mixing_history              = pulay_mixing_history
        self.pulay_mixing_frequency            = pulay_mixing_frequency
        self.linear_mixing_alpha1              = linear_mixing_alpha1
        self.linear_mixing_alpha2              = linear_mixing_alpha2

        self.psp_dir_path                      = psp_dir_path
        self.psp_file_name                     = psp_file_name

        self.hybrid_mixing_parameter           = hybrid_mixing_parameter 
        self.frequency_quadrature_point_number = frequency_quadrature_point_number
        self.angular_momentum_cutoff           = angular_momentum_cutoff
        self.double_hybrid_flag                = False
        self.oep_mixing_parameter              = oep_mixing_parameter 
        self.enable_parallelization            = enable_parallelization

        self.verbose                           = verbose
        self.ml_xc_calculator                  = None
        self.ml_each_scf_step                  = False
        self.ml_smooth_rho_for_features        = False

        # set the default parameters, if not provided
        self.set_and_check_initial_parameters()
        if self.verbose:
            self.print_input_parameters()

        # initialize the psuedopotential data
        self.pseudo = LocalPseudopotential(
            atomic_number    = self.atomic_number, 
            n_electrons      = self.n_electrons,
            path             = self.psp_dir_path, 
            filename         = self.psp_file_name)
        if self.verbose:
            self.pseudo.print_info()

        # initialize the occupation information
        self.occupation_info = OccupationInfo(
            z_nuclear         = self.pseudo.z_nuclear, 
            z_valence         = self.pseudo.z_valence,
            all_electron_flag = self.all_electron_flag,
            n_electrons       = self.n_electrons)
        if self.verbose:
            self.occupation_info.print_info()

        # Grid data and operators (initialized in __init__)
        self.grid_data_standard    : Optional[GridData] = None
        self.grid_data_dense       : Optional[GridData] = None
        self.grid_data_oep         : Optional[GridData] = None
        self.ops_builder_standard  : Optional[RadialOperatorsBuilder] = None
        self.ops_builder_dense     : Optional[RadialOperatorsBuilder] = None
        self.ops_builder_oep       : Optional[RadialOperatorsBuilder] = None

        # SCF components (initialized in __init__)
        self.hamiltonian_builder   : Optional[HamiltonianBuilder] = None
        self.density_calculator    : Optional[DensityCalculator]  = None
        self.poisson_solver        : Optional[PoissonSolver]      = None
        self.energy_calculator     : Optional[EnergyCalculator]   = None
        self.scf_driver            : Optional[SCFDriver]          = None

        # Initialize grids and operators
        self.grid_data_standard, self.grid_data_dense, self.grid_data_oep = self._initialize_grids()
        
        self.ops_builder_standard = RadialOperatorsBuilder.from_grid_data(
            self.grid_data_standard, verbose=self.verbose, builder_label="Standard"
        )
        self.ops_builder_dense = RadialOperatorsBuilder.from_grid_data(
            self.grid_data_dense, verbose=self.verbose, builder_label="Dense"
        )

        if self.use_oep:
            # Initialize OEP operators builder
            self.ops_builder_oep = RadialOperatorsBuilder.from_grid_data(
                self.grid_data_oep, verbose=self.verbose, builder_label="OEP"
            )

        # Initialize SCF components
        self._initialize_scf_components(
            ops_builder_standard = self.ops_builder_standard,
            ops_builder_dense    = self.ops_builder_dense,
        )


    @classmethod
    def from_output_file(
        cls,
        out_file_path : str,
        verbose       : Optional[bool] = None,
    ) -> "AtomicDFTSolver":
        """
        Initialize AtomicDFTSolver from a single output file that includes
        the input parameter block printed by function `self.print_input_parameters()`.
        """
        out_path = Path(out_file_path)
        if not out_path.is_file():
            raise FileNotFoundError(OUTPUT_FILE_NOT_FOUND_ERROR.format(out_file_path))

        text = out_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        # Locate the unique INPUT PARAMETERS block in the output file
        block_indices = [idx for idx, line in enumerate(lines) if line.strip() == "INPUT PARAMETERS"]
        if len(block_indices) == 0:
            raise ValueError(OUTPUT_FILE_NO_INPUT_BLOCK_ERROR)
        if len(block_indices) > 1:
            raise ValueError(OUTPUT_FILE_MULTIPLE_INPUT_BLOCKS_ERROR)

        key_map = {
            # Basic physical parameters
            "atomic_number"                     : "atomic_number",
            "n_electrons"                       : "n_electrons",
            "all_electron_flag"                 : "all_electron_flag",
            "xc_functional"                     : "xc_functional",
            "use_oep"                           : "use_oep",

            # Grid, basis, and mesh parameters
            "domain_size"                       : "domain_size",
            "finite_element_number"             : "finite_element_number",
            "number_of_finite_elements"         : "finite_element_number",  # Map deprecated name to new name
            "polynomial_order"                  : "polynomial_order",
            "quadrature_point_number"           : "quadrature_point_number",
            "oep_basis_number"                  : "oep_basis_number",
            "mesh_type"                         : "mesh_type",
            "mesh_concentration"                : "mesh_concentration",
            "mesh_spacing"                      : "mesh_spacing",

            # Self-consistent field (SCF) convergence parameters
            "scf_tolerance"                     : "scf_tolerance",
            "max_scf_iterations"                : "max_scf_iterations",
            "max_scf_iterations_outer"          : "max_scf_iterations_outer",
            "use_pulay_mixing"                  : "use_pulay_mixing",
            "use_preconditioner"                : "use_preconditioner",
            "pulay_mixing_parameter"            : "pulay_mixing_parameter",
            "pulay_mixing_history"              : "pulay_mixing_history",
            "pulay_mixing_frequency"            : "pulay_mixing_frequency",
            "linear_mixing_alpha1"              : "linear_mixing_alpha1",
            "linear_mixing_alpha2"              : "linear_mixing_alpha2",

            # Pseudopotential parameters
            "psp_dir_path"                      : "psp_dir_path",
            "psp_file_name"                     : "psp_file_name",

            # Advanced functional parameters (for EXX, RPA, etc.)
            "hybrid_mixing_parameter"           : "hybrid_mixing_parameter",
            "frequency_quadrature_point_number" : "frequency_quadrature_point_number",
            "angular_momentum_cutoff"           : "angular_momentum_cutoff",
            "oep_mixing_parameter"              : "oep_mixing_parameter",
            "enable_parallelization"            : "enable_parallelization",
        }

        def parse_value(raw_value: str) -> Any:
            # Best-effort parser for values printed by `print_input_parameters`
            value = raw_value.strip()
            if "(" in value and ")" in value:
                value = value.split("(", 1)[0].strip()
            if value.lower() in ["none", "null"]:
                return None
            if value.lower() in ["true", "false"]:
                return value.lower() == "true"
            if re.match(r"^-?\d+$", value):
                return int(value)
            try:
                if any(ch in value.lower() for ch in [".", "e"]):
                    return float(value)
            except ValueError:
                pass
            return value

        params: Dict[str, Any] = {}
        ml_detected = False
        start_idx = block_indices[0]
        for line in lines[start_idx + 1:]:
            if line.strip() == "" and params:
                break
            match = re.match(r"^\s*([A-Za-z0-9_ ]+?)\s*:\s*(.*)$", line)
            if not match:
                continue
            raw_key = match.group(1).strip()
            raw_value = match.group(2).strip()

            normalized_key = raw_key.replace(" ", "_")
            if normalized_key.startswith("ml_") or normalized_key.startswith("use_machine_learning"):
                ml_detected = True
            if normalized_key not in key_map:
                continue
            params[key_map[normalized_key]] = parse_value(raw_value)

        if ml_detected:
            warnings.warn(
                "Output file mentions ML-XC settings; ML-XC is not active in this AtomSFE build. "
                "Those lines are ignored.",
                UserWarning,
                stacklevel=2,
            )

        if "atomic_number" not in params:
            raise ValueError(OUTPUT_FILE_MISSING_ATOMIC_NUMBER_ERROR)

        if verbose is not None:
            params["verbose"] = verbose

        return cls(**params)



    def set_and_check_initial_parameters(self):
        """
        set and check the default parameters, if not provided
        """
        # atomic number
        try:
            self.atomic_number = float(self.atomic_number)
        except:
            raise ValueError(ATOMIC_NUMBER_NOT_INTEGER_OR_FLOAT_ERROR.format(type(self.atomic_number)))
        assert self.atomic_number > 0, \
            ATOMIC_NUMBER_NOT_GREATER_THAN_0_ERROR.format(self.atomic_number)
        assert self.atomic_number < 119, \
            ATOMIC_NUMBER_LARGER_THAN_119_ERROR.format(self.atomic_number)

        # number of electrons
        if self.n_electrons is None:
            # by default, set to atomic number
            self.n_electrons = self.atomic_number
        assert isinstance(self.n_electrons, (int, float)), \
            N_ELECTRONS_NOT_INTEGER_OR_FLOAT_ERROR.format(type(self.n_electrons))
        assert self.n_electrons > 0, \
            N_ELECTRONS_NOT_GREATER_THAN_0_ERROR.format(self.n_electrons)
        self.n_electrons = float(self.n_electrons)

        # all electron flag
        if self.all_electron_flag is None:
            self.all_electron_flag = False
        if self.all_electron_flag in [0, 1]:
            self.all_electron_flag = False if self.all_electron_flag == 0 else True
        assert isinstance(self.all_electron_flag, bool), \
            ALL_ELECTRON_FLAG_NOT_BOOL_ERROR.format(type(self.all_electron_flag))
        
        if not self.all_electron_flag:
            if self.n_electrons != self.atomic_number:
                raise ValueError(CHARGE_SYSTEMS_NOT_SUPPORTED_FOR_PSEUDOPOTENTIAL_CALCULATION_ERROR)
            if not self.atomic_number.is_integer():
                raise ValueError(ATOMIC_NUMBER_NOT_INTEGER_VALUED_FOR_PSEUDOPOTENTIAL_CALCULATION_ERROR.format(self.atomic_number))

        # Spin-polarized runs are not exposed on AtomicDFTSolver in this build.
        self.spin_polarized_flag = False

        # xc functional (ML-XC path disabled at solver boundary in this build)
        if self.xc_functional is None:
            self.xc_functional = 'GGA_PBE'

        assert isinstance(self.xc_functional, str), \
            XC_FUNCTIONAL_NOT_STRING_ERROR.format(type(self.xc_functional))
        assert self.xc_functional in VALID_XC_FUNCTIONAL_LIST, \
            XC_FUNCTIONAL_NOT_IN_VALID_LIST_ERROR.format(VALID_XC_FUNCTIONAL_LIST, self.xc_functional)


        # use OEP flag
        if self.use_oep in [0, 1]:
            self.use_oep = False if self.use_oep == 0 else True
        if self.xc_functional in VALID_XC_FUNCTIONAL_FOR_OEP_LIST:
            # OEP functional must be used with OEP flag
            if self.xc_functional == 'PBE0':
                if self.use_oep is None:
                    self.use_oep = False
            else:
                if self.use_oep is None:
                    self.use_oep = True
                assert self.use_oep is True, \
                    USE_OEP_NOT_TRUE_FOR_OEP_FUNCTIONAL_ERROR.format(self.xc_functional, self.use_oep)
        else:
            # Other functionals must not be used with OEP flag, otherwise raise error
            if self.use_oep is None:
                self.use_oep = False
            assert self.use_oep is False, \
                USE_OEP_NOT_FALSE_FOR_NON_OEP_FUNCTIONAL_ERROR.format(self.xc_functional, self.use_oep)

        assert isinstance(self.use_oep, bool), \
            USE_OEP_NOT_BOOL_ERROR.format(type(self.use_oep))


        # domain size
        if self.domain_size is None:
            self.domain_size = 20.0
        try:
            self.domain_size = float(self.domain_size)
        except:
            raise ValueError(DOMAIN_SIZE_NOT_FLOAT_ERROR.format(type(self.domain_size)))
        assert isinstance(self.domain_size, float), \
            DOMAIN_SIZE_NOT_FLOAT_ERROR.format(type(self.domain_size))
        assert self.domain_size > 0, \
            DOMAIN_SIZE_NOT_GREATER_THAN_0_ERROR.format(self.domain_size)


        # finite element number
        if self.finite_element_number is None:
            self.finite_element_number = 12
        assert isinstance(self.finite_element_number, int), \
            FINITE_ELEMENT_NUMBER_NOT_INTEGER_ERROR.format(type(self.finite_element_number))
        assert self.finite_element_number > 0, \
            FINITE_ELEMENT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(self.finite_element_number)


        # polynomial order
        if self.polynomial_order is None:
            self.polynomial_order = 20
        assert isinstance(self.polynomial_order, int), \
            POLYNOMIAL_ORDER_NOT_INTEGER_ERROR.format(type(self.polynomial_order))
        assert self.polynomial_order > 0, \
            POLYNOMIAL_ORDER_NOT_GREATER_THAN_0_ERROR.format(self.polynomial_order)


        # quadrature point number
        if self.quadrature_point_number is None:
            self.quadrature_point_number = 60
        assert isinstance(self.quadrature_point_number, int), \
            QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(self.quadrature_point_number))
        assert self.quadrature_point_number > 0, \
            QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(self.quadrature_point_number)
        assert self.quadrature_point_number >= 2 * self.polynomial_order + 3, \
            QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_2_POLYNOMIAL_ORDER_PLUS_3_ERROR.\
                format(self.polynomial_order, 2 * self.polynomial_order + 3, self.quadrature_point_number)

        # OEP auxiliary basis size
        if self.oep_basis_number is None:
            if self.use_oep:
                default_oep_basis = max(1, int(self.polynomial_order * 0.25))
                self.oep_basis_number = default_oep_basis
        else:
            if not isinstance(self.oep_basis_number, int):
                try:
                    self.oep_basis_number = int(self.oep_basis_number)
                except Exception:
                    raise ValueError(OEP_BASIS_NUMBER_NOT_INTEGER_ERROR.format(type(self.oep_basis_number)))
            assert isinstance(self.oep_basis_number, int), \
                OEP_BASIS_NUMBER_NOT_INTEGER_ERROR.format(type(self.oep_basis_number))
            assert self.oep_basis_number > 0, \
                OEP_BASIS_NUMBER_NOT_GREATER_THAN_0_ERROR.format(self.oep_basis_number)


        # mesh type
        if self.mesh_type is None:
            self.mesh_type = 'exponential'
        assert isinstance(self.mesh_type, str), \
            MESH_TYPE_NOT_STRING_ERROR.format(type(self.mesh_type))
        assert self.mesh_type in ['exponential', 'polynomial', 'uniform'], \
            MESH_TYPE_NOT_IN_VALID_LIST_ERROR.format(VALID_MESH_TYPE_LIST, self.mesh_type)


        # mesh concentration
        if self.mesh_concentration is None: # default value
            if self.mesh_type == 'exponential':
                self.mesh_concentration = 100.0
            elif self.mesh_type == 'polynomial':
                self.mesh_concentration = 2.0
            elif self.mesh_type == 'uniform':
                self.mesh_concentration = None
        if self.mesh_type in ['exponential', 'polynomial']: # type check
            try:
                self.mesh_concentration = float(self.mesh_concentration)
            except:
                raise ValueError(MESH_CONCENTRATION_NOT_FLOAT_ERROR.format(type(self.mesh_concentration)))
            assert isinstance(self.mesh_concentration, float), \
                MESH_CONCENTRATION_NOT_FLOAT_ERROR.format(type(self.mesh_concentration))
            assert self.mesh_concentration > 0., \
                MESH_CONCENTRATION_NOT_GREATER_THAN_0_ERROR.format(self.mesh_concentration)
        elif self.mesh_type == 'uniform':
            if self.mesh_concentration is not None:
                print(MESH_CONCENTRATION_NOT_NONE_FOR_UNIFORM_MESH_TYPE_WARNING)
                self.mesh_concentration = None
        else:
            raise ValueError("This error should never be raised.")
                    

        # mesh spacing
        if self.mesh_spacing is None:
            self.mesh_spacing = 0.1
        assert isinstance(self.mesh_spacing, float), \
            MESH_SPACING_NOT_FLOAT_ERROR.format(type(self.mesh_spacing))
        assert self.mesh_spacing > 0., \
            MESH_SPACING_NOT_GREATER_THAN_0_ERROR.format(self.mesh_spacing)


        # scf tolerance
        if self.scf_tolerance is None:
            # For most functionals, the default tolerance is 1e-8
            self.scf_tolerance = 1e-8
            if self.xc_functional in ['SCAN', 'RSCAN', 'R2SCAN']:
                # SCAN, RSCAN, R2SCAN functionals suffer from convergence issues, so we use a higher tolerance
                self.scf_tolerance = 1e-6
        try:
            self.scf_tolerance = float(self.scf_tolerance)
        except:
            raise ValueError(SCF_TOLERANCE_NOT_FLOAT_ERROR.format(type(self.scf_tolerance)))
        assert isinstance(self.scf_tolerance, float), \
            SCF_TOLERANCE_NOT_FLOAT_ERROR.format(type(self.scf_tolerance))
        assert self.scf_tolerance > 0., \
            SCF_TOLERANCE_NOT_GREATER_THAN_0_ERROR.format(self.scf_tolerance)


        # max scf iterations
        if self.max_scf_iterations is None:
            self.max_scf_iterations = 500
        assert isinstance(self.max_scf_iterations, int), \
            MAX_SCF_ITERATIONS_NOT_INTEGER_ERROR.format(type(self.max_scf_iterations))
        assert self.max_scf_iterations > 0, \
            MAX_SCF_ITERATIONS_NOT_GREATER_THAN_0_ERROR.format(self.max_scf_iterations)


        # max scf iterations outer
        if self.xc_functional in VALID_XC_FUNCTIONAL_FOR_OUTER_LOOP_LIST:
            if self.max_scf_iterations_outer is None:
                self.max_scf_iterations_outer = 50
            assert isinstance(self.max_scf_iterations_outer, int), \
                MAX_SCF_ITERATIONS_OUTER_NOT_INTEGER_ERROR.format(type(self.max_scf_iterations_outer))
            assert self.max_scf_iterations_outer > 0, \
                MAX_SCF_ITERATIONS_OUTER_NOT_GREATER_THAN_0_ERROR.format(self.max_scf_iterations_outer)
        else:
            if self.max_scf_iterations_outer is not None and self.max_scf_iterations_outer != 1:
                print(MAX_SCF_ITERATIONS_OUTER_NOT_NONE_AND_NOT_ONE_FOR_XC_FUNCTIONAL_OTHER_THAN_OUTER_LOOP_LIST_WARNING.format(self.xc_functional))
                self.max_scf_iterations_outer = None


        # use pulay mixing flag
        if self.use_pulay_mixing is None:
            self.use_pulay_mixing = True # default is True
        if self.use_pulay_mixing in [0, 1]:
            self.use_pulay_mixing = False if self.use_pulay_mixing == 0 else True
        assert isinstance(self.use_pulay_mixing, bool), \
            USE_PULAY_MIXING_NOT_BOOL_ERROR.format(type(self.use_pulay_mixing))


        # use preconditioner flag
        if self.use_preconditioner is None:
            self.use_preconditioner = True if self.use_pulay_mixing else False
        if self.use_preconditioner in [0, 1]:
            self.use_preconditioner = False if self.use_preconditioner == 0 else True
        assert isinstance(self.use_preconditioner, bool), \
            USE_PRECONDITIONER_NOT_BOOL_ERROR.format(type(self.use_preconditioner))


        # pulay mixing parameter
        if self.use_pulay_mixing:
            if self.pulay_mixing_parameter is None:
                self.pulay_mixing_parameter = 1.0 if self.use_preconditioner else 0.45
            try:
                self.pulay_mixing_parameter = float(self.pulay_mixing_parameter)
            except:
                raise ValueError(PULAY_MIXING_PARAMETER_NOT_FLOAT_ERROR.format(type(self.pulay_mixing_parameter)))
            assert isinstance(self.pulay_mixing_parameter, float), \
                PULAY_MIXING_PARAMETER_NOT_FLOAT_ERROR.format(type(self.pulay_mixing_parameter))
            assert 0.0 < self.pulay_mixing_parameter <= 1.0, \
                PULAY_MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR.format(self.pulay_mixing_parameter)
        else:
            if self.pulay_mixing_parameter is not None:
                print(PULAY_MIXING_PARAMETER_NOT_NONE_WHEN_USE_PULAY_MIXING_IS_FALSE_WARNING)
                self.pulay_mixing_parameter = None


        # pulay mixing history
        if self.use_pulay_mixing:
            if self.pulay_mixing_history is None:
                self.pulay_mixing_history = 7 if self.use_preconditioner else 11
            assert isinstance(self.pulay_mixing_history, int), \
                PULAY_MIXING_HISTORY_NOT_INTEGER_ERROR.format(type(self.pulay_mixing_history))
            assert self.pulay_mixing_history > 0, \
                PULAY_MIXING_HISTORY_NOT_GREATER_THAN_0_ERROR.format(self.pulay_mixing_history)
        else:
            if self.pulay_mixing_history is not None:
                print(PULAY_MIXING_HISTORY_NOT_NONE_WHEN_USE_PULAY_MIXING_IS_FALSE_WARNING)
                self.pulay_mixing_history = None


        # pulay mixing frequency
        if self.use_pulay_mixing:
            if self.pulay_mixing_frequency is None:
                self.pulay_mixing_frequency = 3 if self.use_preconditioner else 1
            assert isinstance(self.pulay_mixing_frequency, int), \
                PULAY_MIXING_FREQUENCY_NOT_INTEGER_ERROR.format(type(self.pulay_mixing_frequency))
            assert self.pulay_mixing_frequency > 0, \
                PULAY_MIXING_FREQUENCY_NOT_GREATER_THAN_0_ERROR.format(self.pulay_mixing_frequency)
        else:   
            if self.pulay_mixing_frequency is not None:
                print(PULAY_MIXING_FREQUENCY_NOT_NONE_WHEN_USE_PULAY_MIXING_IS_FALSE_WARNING)
                self.pulay_mixing_frequency = None


        # linear mixing alpha1
        if self.linear_mixing_alpha1 is None:
            self.linear_mixing_alpha1 = 0.75 if self.use_pulay_mixing else 0.7
        assert isinstance(self.linear_mixing_alpha1, float), \
            LINEAR_MIXING_ALPHA1_NOT_FLOAT_ERROR.format(type(self.linear_mixing_alpha1))
        assert 0.0 <= self.linear_mixing_alpha1 <= 1.0, \
            LINEAR_MIXING_ALPHA1_NOT_IN_ZERO_ONE_ERROR.format(self.linear_mixing_alpha1)


        # linear mixing alpha2
        if self.linear_mixing_alpha2 is None:
            self.linear_mixing_alpha2 = 0.95 if self.use_pulay_mixing else 1.0
        assert isinstance(self.linear_mixing_alpha2, float), \
            LINEAR_MIXING_ALPHA2_NOT_FLOAT_ERROR.format(type(self.linear_mixing_alpha2))
        assert 0.0 <= self.linear_mixing_alpha2 <= 1.0, \
            LINEAR_MIXING_ALPHA2_NOT_IN_ZERO_ONE_ERROR.format(self.linear_mixing_alpha2)

    
        # psp directory path
        if self.all_electron_flag == False:
            if self.psp_dir_path is None:
                self.psp_dir_path = os.path.join(os.path.dirname(__file__), "..", "psps")
            if not os.path.exists(self.psp_dir_path):
                # if the psp directory path is not absolute path, make it absolute path
                self.psp_dir_path = os.path.join(os.path.dirname(__file__), "..", self.psp_dir_path) 
            assert isinstance(self.psp_dir_path, str), \
                PSP_DIR_PATH_NOT_STRING_ERROR.format(type(self.psp_dir_path))
            assert os.path.exists(self.psp_dir_path), \
                PSP_DIR_PATH_NOT_EXISTS_ERROR.format(self.psp_dir_path)

        elif self.all_electron_flag == True:
            if self.psp_dir_path is not None:
                print(PSP_DIR_PATH_NOT_NONE_FOR_ALL_ELECTRON_CALCULATION_WARNING)
                self.psp_dir_path = None
        else:
            raise ValueError("This error should never be raised.")


        # psp file name
        if self.all_electron_flag == False:
            if self.psp_file_name is None:
                # default value
                if self.atomic_number < 10:
                    self.psp_file_name = "0" + str(int(self.atomic_number)) + ".psp8"
                else:
                    self.psp_file_name = str(int(self.atomic_number)) + ".psp8"
            assert isinstance(self.psp_file_name, str), \
                PSP_FILE_NAME_NOT_STRING_ERROR.format(type(self.psp_file_name))
            assert os.path.exists(os.path.join(self.psp_dir_path, self.psp_file_name)), \
                PSP_FILE_NAME_NOT_EXISTS_ERROR.format(self.psp_file_name, self.psp_dir_path)
        elif self.all_electron_flag == True:
            if self.psp_file_name is not None:
                print(PSP_FILE_NAME_NOT_NONE_FOR_ALL_ELECTRON_CALCULATION_WARNING)
                self.psp_file_name = None
        else:
            raise ValueError("This error should never be raised.")


        # hybrid mixing parameter
        # Only validate for hybrid functionals (PBE0, HF)
        if self.xc_functional in ['PBE0', 'HF', 'EXX', 'RPA']:
            if self.hybrid_mixing_parameter is None:
                # Use default values based on functional
                if self.xc_functional == 'PBE0':
                    self.hybrid_mixing_parameter = 0.25
                    # print(NO_HYBRID_MIXING_PARAMETER_PROVIDED_FOR_HYBRID_FUNCTIONAL_WARNING.format(self.xc_functional, 0.25))
                elif self.xc_functional in ['HF', 'EXX', 'RPA']:
                    self.hybrid_mixing_parameter = 1.0
                
            # If the hybrid mixing parameter is provided, check the type and value
            assert isinstance(self.hybrid_mixing_parameter, (float, int)), \
                HYBRID_MIXING_PARAMETER_NOT_FLOAT_ERROR.format(type(self.hybrid_mixing_parameter))
            assert 0.0 <= self.hybrid_mixing_parameter <= 1.0, \
                HYBRID_MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR.format(self.hybrid_mixing_parameter)
            if self.xc_functional in ["HF", "EXX", "RPA"]:
                assert self.hybrid_mixing_parameter == 1.0, \
                HYBRID_MIXING_PARAMETER_NOT_ONE_ERROR.format(self.xc_functional, self.hybrid_mixing_parameter)
        else:
            # For non-hybrid functionals, hybrid_mixing_parameter is not used
            # Set it to None to avoid confusion
            self.hybrid_mixing_parameter = None


        # frequency integration point number
        if self.xc_functional in ['RPA', ]:
            if self.frequency_quadrature_point_number is None:
                self.frequency_quadrature_point_number = 25
            assert isinstance(self.frequency_quadrature_point_number, int), \
                FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(self.frequency_quadrature_point_number))
            assert self.frequency_quadrature_point_number > 0, \
                FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(self.frequency_quadrature_point_number)
        else:
            if self.frequency_quadrature_point_number is not None:
                print(FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_NONE_FOR_OEPX_AND_NONE_XC_FUNCTIONAL_WARNING.format(self.xc_functional))
                self.frequency_quadrature_point_number = None


        # angular_momentum_cutoff, used for RPA functional only 
        if self.xc_functional in ['RPA']:
            if self.angular_momentum_cutoff is None:
                self.angular_momentum_cutoff = 4
            assert isinstance(self.angular_momentum_cutoff, int), \
                ANGULAR_MOMENTUM_CUTOFF_NOT_INTEGER_ERROR.format(type(self.angular_momentum_cutoff))
            assert self.angular_momentum_cutoff >= 0., \
                ANGULAR_MOMENTUM_CUTOFF_NEGATIVE_ERROR.format(self.angular_momentum_cutoff)
        else:
            if self.angular_momentum_cutoff is not None:
                print(ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_FOR_XC_FUNCTIONAL_OTHER_THAN_RPA_WARNING.format(self.xc_functional))
                self.angular_momentum_cutoff = None


        # Double-hybrid workflow is not exposed on AtomicDFTSolver in this build.
        self.double_hybrid_flag = False

        # OEP mixing parameter (λ scaling for OEP potentials)
        if self.oep_mixing_parameter is None:
            if self.use_oep:
                self.oep_mixing_parameter = 1.0
        else:
            if not isinstance(self.oep_mixing_parameter, float):
                try:
                    self.oep_mixing_parameter = float(self.oep_mixing_parameter)
                except:
                    raise ValueError(OEP_MIXING_PARAMETER_NOT_FLOAT_ERROR.format(type(self.oep_mixing_parameter)))
            assert isinstance(self.oep_mixing_parameter, float), \
                OEP_MIXING_PARAMETER_NOT_FLOAT_ERROR.format(type(self.oep_mixing_parameter))
            assert self.oep_mixing_parameter > 0.0 and self.oep_mixing_parameter <= 1.0, \
                OEP_MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR.format(self.oep_mixing_parameter)

        # enable parallelization flag
        if self.xc_functional in ['RPA']:
            if self.enable_parallelization is None:
                self.enable_parallelization = False
            assert isinstance(self.enable_parallelization, bool), \
                ENABLE_PARALLELIZATION_NOT_BOOL_ERROR.format(type(self.enable_parallelization))
            if self.enable_parallelization:
                if _NUMPY_IMPORTED_BEFORE_ATOMIC and not _BLAS_ENV_SINGLE_THREADED and not _THREADPOOLCTL_INSTALLED:
                    print(NUMPY_IMPORTED_BEFORE_ATOMIC_WARNING)
                    self.enable_parallelization = False

        else:
            if self.enable_parallelization is not None:
                print(ENABLE_PARALLELIZATION_NOT_NONE_FOR_XC_FUNCTIONAL_OTHER_THAN_RPA_WARNING.format(self.xc_functional))


        # verbose flag
        if self.verbose is None:
            self.verbose = False
        if self.verbose in [0, 1]:
            self.verbose = False if self.verbose == 0 else True
        assert isinstance(self.verbose, bool), \
            VERBOSE_NOT_BOOL_ERROR.format(type(self.verbose))

        self.ml_xc_calculator = None
        self.ml_each_scf_step = False
        self.ml_smooth_rho_for_features = False
        self.double_hybrid_flag = False


    def print_input_parameters(self):

        def _format_number(value: float) -> str:
            if abs(value - round(value)) < 1e-8:
                return str(int(round(value))) + "   "
            return f"{value:.2f}"

        # format atomic number and n_electrons' display mode
        # - for integer valued atomic number, display as atomic_number (atomic_name), e.g. 13 (Al)
        # - for fractional valued atomic number, display as atomic_number (fractional), e.g. 13.5 (fractional)
        atomic_is_int = abs(self.atomic_number - round(self.atomic_number)) < 1e-8
        if atomic_is_int:
            atomic_label = atomic_number_to_name(int(round(self.atomic_number)))
            atomic_display = f"{_format_number(self.atomic_number)} ({atomic_label})"
        else:
            atomic_display = f"{_format_number(self.atomic_number)} (fractional)"

        n_electrons_display = _format_number(self.n_electrons)


        # Display relative path for psp_dir_path
        if self.psp_dir_path is not None:
            try:
                # Try to get relative path from current working directory
                psp_path_display = os.path.relpath(self.psp_dir_path)
            except ValueError:
                # If relative path fails (e.g., different drives on Windows), use absolute path
                psp_path_display = self.psp_dir_path
        else:
            psp_path_display = self.psp_dir_path

        # print the input parameters
        # Be careful! This output can also be used to initialize the AtomicDFTSolver from output files!
        #     So, do not change the format of this output! Or if you want to change, please update the from_output_file method!
        print("===========================================================================")
        print("*                  SPARC-atomSFE  (version May 16, 2026)                  *")
        print("*   Copyright (c) 2026 Material Physics & Mechanics Group, Georgia Tech   *")
        print("*           Distributed under GNU General Public License 3 (GPL)          *")
        print("*                   Start time: {}                  *".format(get_sparc_time_string())) # Do not change the length for this line
        print("===========================================================================")
        print("                              INPUT PARAMETERS                             ")
        print("===========================================================================")
        
        # Basic physical parameters
        print("\t atomic_number                     : {}".format(atomic_display))
        print("\t n_electrons                       : {}".format(n_electrons_display))
        print("\t all_electron_flag                 : {}".format(self.all_electron_flag))
        print("\t xc_functional                     : {}".format(self.xc_functional))
        print("\t use_oep                           : {}".format(self.use_oep))

        # Grid, basis, and mesh parameters
        print("\t domain_size                       : {}".format(self.domain_size))
        print("\t finite_element_number             : {}".format(self.finite_element_number))
        print("\t polynomial_order                  : {}".format(self.polynomial_order))
        print("\t quadrature_point_number           : {}".format(self.quadrature_point_number))
        print("\t oep_basis_number                  : {}".format(self.oep_basis_number))
        print("\t mesh_type                         : {}".format(self.mesh_type))
        print("\t mesh_concentration                : {}".format(self.mesh_concentration))
        print("\t mesh_spacing                      : {}".format(self.mesh_spacing))

        # Self-consistent field (SCF) convergence parameters
        print("\t scf_tolerance                     : {}".format(self.scf_tolerance))
        print("\t max_scf_iterations                : {}".format(self.max_scf_iterations))
        print("\t max_scf_iterations_outer          : {}".format(self.max_scf_iterations_outer))
        print("\t use_pulay_mixing                  : {}".format(self.use_pulay_mixing))
        print("\t use_preconditioner                : {}".format(self.use_preconditioner))
        print("\t pulay_mixing_parameter            : {}".format(self.pulay_mixing_parameter))
        print("\t pulay_mixing_history              : {}".format(self.pulay_mixing_history))
        print("\t pulay_mixing_frequency            : {}".format(self.pulay_mixing_frequency))
        print("\t linear_mixing_alpha1              : {}".format(self.linear_mixing_alpha1))
        print("\t linear_mixing_alpha2              : {}".format(self.linear_mixing_alpha2))

        # Pseudopotential parameters
        print("\t psp_dir_path                      : {}".format(psp_path_display))
        print("\t psp_file_name                     : {}".format(self.psp_file_name))

        # Advanced functional parameters (for EXX, RPA, etc.)
        print("\t hybrid_mixing_parameter           : {}".format(self.hybrid_mixing_parameter))
        print("\t frequency_quadrature_point_number : {}".format(self.frequency_quadrature_point_number))
        print("\t angular_momentum_cutoff           : {}".format(self.angular_momentum_cutoff))
        print("\t oep_mixing_parameter              : {}".format(self.oep_mixing_parameter))
        print("\t enable_parallelization            : {}".format(self.enable_parallelization))

        print()


    def print_footer(self):
        print("===========================================================================")
        print("*             Material Physics & Mechanics Group, Georgia Tech            *")
        print("*                       PI: Phanish Suryanarayana                         *")
        print("*               List of contributors: See the documentation               *")
        print("*         Citation: See README.md or the documentation for details        *")
        print("*                Acknowledgements: U.S. DOE SC (DE-SC0023445)             *")
        print("===========================================================================")


    def set_time_print_decimal_places(self, n: int) -> None:
        from .scf.convergence import set_time_print_decimal_places
        set_time_print_decimal_places(n)


    def _initialize_grids(self) -> Tuple[GridData, GridData, Optional[GridData]]:
        """
        Initialize finite element grids and quadrature.
        
        Generates two or three grid configurations:
        - Standard grid: for most operators and wavefunctions
        - Dense grid: refined mesh for Hartree potential solver (double density)
        - OEP grid: for OEP solver
        
        Returns
        -------
        grid_data_standard : GridData
            Standard grid data for operators and wavefunctions
        grid_data_dense : GridData
            Dense grid data for Hartree solver
        grid_data_oep : Optional[GridData]
            Grid data for OEP solver
        """
        # Generate Lobatto interpolation nodes on reference interval [-1, 1]
        interp_nodes_ref, _ = Quadrature1D.lobatto(self.polynomial_order)
    
        # Generate mesh boundaries
        mesh1d = Mesh1D(
            domain_size            = self.domain_size,
            finite_elements_number = self.finite_element_number,
            mesh_type              = self.mesh_type,
            clustering_param       = self.mesh_concentration,
            exp_shift              = getattr(self, 'exp_shift', None)
        )

        boundaries_nodes, _ = mesh1d.generate_mesh_nodes_and_width()
        
        # Generate standard FE nodes
        global_nodes = Mesh1D.generate_fe_nodes(
            boundaries_nodes = boundaries_nodes,
            interp_nodes     = interp_nodes_ref
        )

        # Generate refined FE nodes (for Hartree potential solver)
        refined_interp_nodes_ref = Mesh1D.refine_interpolation_nodes(interp_nodes_ref)
        refined_global_nodes = Mesh1D.generate_fe_nodes(
            boundaries_nodes = boundaries_nodes,
            interp_nodes     = refined_interp_nodes_ref
        )
        
        # Generate Gauss-Legendre quadrature nodes and weights
        quadrature_nodes_ref, quadrature_weights_ref = Quadrature1D.gauss_legendre(
            self.quadrature_point_number
        )
        
        # Map quadrature to physical elements
        quadrature_nodes, quadrature_weights = Mesh1D.map_quadrature_to_physical_elements(
            boundaries_nodes = boundaries_nodes,
            interp_nodes     = quadrature_nodes_ref,
            interp_weights   = quadrature_weights_ref,
            flatten          = True
        )

        # Create grid data objects
        grid_data_standard = GridData(
            finite_element_number = self.finite_element_number,
            physical_nodes        = global_nodes,
            quadrature_nodes      = quadrature_nodes,
            quadrature_weights    = quadrature_weights
        )
        
        grid_data_dense = GridData(
            finite_element_number = self.finite_element_number,
            physical_nodes        = refined_global_nodes,
            quadrature_nodes      = quadrature_nodes,
            quadrature_weights    = quadrature_weights
        )


        # For OEP method, extra set of grids are needed for solving the OEP equation
        grid_data_oep : Optional[GridData] = None

        if self.use_oep:
            # Generate Lobatto interpolation nodes for OEP basis
            oep_interp_nodes_ref, _ = Quadrature1D.lobatto(
                self.oep_basis_number
            )

            # Generate OEP basis nodes
            oep_global_nodes = Mesh1D.generate_fe_nodes(
                boundaries_nodes = boundaries_nodes,
                interp_nodes     = oep_interp_nodes_ref,
            )

            grid_data_oep = GridData(
                finite_element_number = self.finite_element_number,
                physical_nodes        = oep_global_nodes,
                quadrature_nodes      = quadrature_nodes,
                quadrature_weights    = quadrature_weights
            )

        return grid_data_standard, grid_data_dense, grid_data_oep

    @property
    def xc_requirements(self):
        """Requirements for the current ``xc_functional`` (e.g. ``needs_tau``, ``needs_gradient``)."""
        return get_functional_requirements(self.xc_functional)


    def _initialize_scf_components(
        self, 
        ops_builder_standard : RadialOperatorsBuilder,
        ops_builder_dense    : RadialOperatorsBuilder,
    ) -> None:
        """
        Initialize all SCF components.
        
        This method creates and configures all the modular SCF components:
        - HamiltonianBuilder : constructs Hamiltonian matrices (uses standard grid)
        - DensityCalculator  : computes density from orbitals (uses standard grid)
        - PoissonSolver      : solves for Hartree potential (uses dense grid)
        - EnergyCalculator   : computes total energy (uses standard grid)
        - SCFDriver          : manages SCF iterations
        
        Note: Only PoissonSolver uses the dense grid for accurate Hartree potential.
              All other components use the standard grid.
        """

        grid_data_standard = ops_builder_standard.grid_data

        # Hamiltonian builder (uses standard grid)
        self.hamiltonian_builder = HamiltonianBuilder(
            ops_builder     = ops_builder_standard,
            pseudo          = self.pseudo,
            occupation_info = self.occupation_info,
            all_electron    = self.all_electron_flag
        )
        
        # Density calculator (uses standard grid)
        self.density_calculator = DensityCalculator(
            grid_data         = grid_data_standard,
            occupation_info   = self.occupation_info,
            derivative_matrix = ops_builder_standard.derivative_matrix,
        )

        # Poisson solver for Hartree potential (uses dense grid for accuracy)
        self.poisson_solver = PoissonSolver(
            ops_builder      = ops_builder_dense,
            n_free_electrons = self.occupation_info.n_free_electrons
        )
        
        # EigenSolver for Kohn-Sham equation
        eigensolver = EigenSolver(xc_functional = self.xc_functional)

        # Mixer for density mixing (uses Pulay mixing or linear mixing)
        mixer = Mixer(
            use_pulay_mixing       = self.use_pulay_mixing,
            use_preconditioner     = self.use_preconditioner,
            pulay_mixing_parameter = self.pulay_mixing_parameter, 
            pulay_mixing_history   = self.pulay_mixing_history, 
            pulay_mixing_frequency = self.pulay_mixing_frequency,
            linear_mixing_alpha1   = self.linear_mixing_alpha1,
            linear_mixing_alpha2   = self.linear_mixing_alpha2,
        )


        # SCF Driver (create first to get xc_calculator)
        self.scf_driver = SCFDriver(
            hamiltonian_builder               = self.hamiltonian_builder,
            density_calculator                = self.density_calculator,
            poisson_solver                    = self.poisson_solver,
            eigensolver                       = eigensolver,
            mixer                             = mixer,
            occupation_info                   = self.occupation_info,
            xc_functional                     = self.xc_functional,
            spin_polarized_flag               = self.spin_polarized_flag,
            hybrid_mixing_parameter           = self.hybrid_mixing_parameter,
            use_oep                           = self.use_oep,
            ops_builder_oep                   = self.ops_builder_oep,
            oep_mixing_parameter              = self.oep_mixing_parameter,
            frequency_quadrature_point_number = self.frequency_quadrature_point_number,
            angular_momentum_cutoff           = self.angular_momentum_cutoff,
            enable_parallelization            = self.enable_parallelization,
            ml_xc_calculator                  = self.ml_xc_calculator,
            ml_each_scf_step                  = self.ml_each_scf_step,
            ml_smooth_rho_for_features        = self.ml_smooth_rho_for_features,
        )
        
        # Get XC calculator and HF calculator from scf_driver
        xc_calculator  = self.scf_driver.xc_calculator  if hasattr(self.scf_driver, 'xc_calculator')  else None
        hf_calculator  = self.scf_driver.hf_calculator  if hasattr(self.scf_driver, 'hf_calculator')  else None
        oep_calculator = self.scf_driver.oep_calculator if hasattr(self.scf_driver, 'oep_calculator') else None
        
        # Energy calculator (uses standard grid data and ops_builder, but dense derivative matrix)
        self.energy_calculator = EnergyCalculator(
            switches           = self.scf_driver.switches,
            grid_data          = grid_data_standard,
            occupation_info    = self.occupation_info,
            ops_builder        = ops_builder_standard,
            poisson_solver     = self.poisson_solver,
            pseudo             = self.pseudo,
            xc_calculator      = xc_calculator,
            hf_calculator      = hf_calculator,   # Pass HF calculator from SCFDriver
            oep_calculator     = oep_calculator,  # Pass OEP calculator from SCFDriver
            ml_xc_calculator   = self.ml_xc_calculator,  # Pass ML XC calculator from solver
            derivative_matrix  = ops_builder_dense.derivative_matrix,  # Use dense grid derivative for accuracy
            density_calculator = self.density_calculator,  # For ML feature construction (grad_rho, etc.)
        )
        

    def _get_scf_settings(self, xc_functional: str) -> Dict[str, Any]:
        assert isinstance(xc_functional, str), \
            XC_FUNCTIONAL_TYPE_ERROR_MESSAGE.format(type(xc_functional))
        assert xc_functional in VALID_XC_FUNCTIONAL_LIST, \
            XC_FUNCTIONAL_NOT_IN_VALID_LIST_ERROR.format(VALID_XC_FUNCTIONAL_LIST, xc_functional)
        
        """Get SCF settings based on XC functional."""
        settings = {
            'inner_max_iter' : self.max_scf_iterations,
            'outer_max_iter' : 1,  # Default: no outer loop for LDA/GGA etc.
            'rho_tol'        : self.scf_tolerance,
            'outer_rho_tol'  : self.scf_tolerance,
            'n_consecutive'  : 1,
            'verbose'        : self.verbose
        }
        
        # For functionals requiring outer loop (HF, EXX, RPA, PBE0)
        if xc_functional in VALID_XC_FUNCTIONAL_FOR_OUTER_LOOP_LIST:
            settings['outer_max_iter'] = self.max_scf_iterations_outer
        
        return settings

    def _get_initial_orbitals_from_density_with_precursor_xc(
        self,
        xc_functional: str,
        rho_initial: np.ndarray,
        orbitals_initial: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Like ``_get_initial_density_and_orbitals_with_warm_start`` but only orbitals: run a
        single non-self-consistent step at frozen ``rho_initial`` using ``xc_functional``
        (typically ``'GGA_PBE'``) — build Hartree + XC from that density, diagonalize KS
        once per ``l``, return occupied orbitals. No density mixing (``inner_max_iter`` is 1
        and ``rho_tol`` is ``inf`` so the driver exits after one pass). If ``orbitals_initial``
        is already provided, it is returned unchanged.
        """
        assert isinstance(xc_functional, str), \
            XC_FUNCTIONAL_TYPE_ERROR_MESSAGE.format(type(xc_functional))
        assert xc_functional in VALID_XC_FUNCTIONAL_LIST, \
            XC_FUNCTIONAL_NOT_IN_VALID_LIST_ERROR.format(VALID_XC_FUNCTIONAL_LIST, xc_functional)
        if orbitals_initial is not None:
            return orbitals_initial

        settings: Dict[str, Any] = {
            "inner_max_iter": 1,
            "outer_max_iter": 1,
            "rho_tol"       : float("inf"),
            "outer_rho_tol" : float("inf"),
            "n_consecutive" : 1,
            "verbose"       : False,
        }
        eigensolver_prec = EigenSolver(xc_functional=xc_functional)
        scf_driver_prec = SCFDriver(
            hamiltonian_builder     = self.scf_driver.hamiltonian_builder,
            density_calculator      = self.scf_driver.density_calculator,
            poisson_solver          = self.scf_driver.poisson_solver,
            eigensolver             = eigensolver_prec,
            mixer                   = self.scf_driver.mixer,
            occupation_info         = self.scf_driver.occupation_info,
            xc_functional           = xc_functional,
            spin_polarized_flag     = self.spin_polarized_flag,
            hybrid_mixing_parameter = self.scf_driver.hybrid_mixing_parameter,
        )
        scf_result_prec = scf_driver_prec.run(
            rho_initial      = rho_initial,
            settings         = settings,
            orbitals_initial = None,
        )
        return scf_result_prec.orbitals


    def _evaluate_basis_on_uniform_grid(
        self, 
        ops_builder_standard: RadialOperatorsBuilder,
        orbitals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate all orbitals on a uniform evaluation grid.
        
        This function generates a uniform grid spanning the domain and evaluates
        each orbital state on that grid using Lagrange interpolation. The result
        is useful for:
        - Visualization and analysis (uniform spacing for plotting)
        - Output formatting (matching reference data formats)
        - Further post-processing (interpolation to different grids)
        
        Parameters
        ----------
        ops_builder_standard : RadialOperatorsBuilder
            Operators builder containing mesh information and interpolation methods.
            Used to evaluate orbitals on the given grid using finite element basis functions.
        orbitals : np.ndarray
            Orbital coefficients at physical nodes, shape (n_physical_nodes, n_states).
            Each column represents one orbital state (eigenvector).
        
        Returns
        -------
        orbitals_on_given_grid : np.ndarray
            Orbital values evaluated on the uniform grid, shape (n_grid_points, n_states).
            Each column contains the values of one orbital state on the uniform grid.
        
        Notes
        -----
        - The uniform grid is generated with spacing `self.mesh_spacing` over `[0, domain_size]`.
        - Each orbital is evaluated independently using `evaluate_single_field_on_grid`.
        - The evaluation uses Lagrange polynomial interpolation within each finite element.
        
        Example
        -------
        >>> uniform_grid_values = solver._evaluate_basis_on_uniform_grid(
        ...     ops_builder_standard=ops_builder,
        ...     orbitals=eigenvectors  # shape: (n_nodes, n_states)
        ... )
        >>> # uniform_grid_values.shape = (n_grid_points, n_states)
        """
        # Generate uniform evaluation grid with specified spacing
        uniform_eval_grid = np.linspace(
            start=0.0, 
            stop=self.domain_size, 
            num=int(self.domain_size / self.mesh_spacing) + 1, 
            endpoint=True
        )

        # Evaluate each orbital state on the uniform grid
        n_states     = orbitals.shape[1]
        n_grid_given = len(uniform_eval_grid)
        orbitals_on_given_grid = np.zeros((n_grid_given, n_states))

        for state_index in range(n_states):
            # Evaluate single orbital on the uniform grid using Lagrange interpolation
            orbitals_on_given_grid[:, state_index] = ops_builder_standard.evaluate_single_field_on_grid(
                given_grid   = uniform_eval_grid,
                field_values = orbitals[:, state_index]
            )
        return uniform_eval_grid, orbitals_on_given_grid


    def _get_initial_density_and_orbitals_with_warm_start(
        self, 
        xc_functional    : str, 
        rho_initial      : np.ndarray, 
        orbitals_initial : Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warm start calculation using specified XC functional to obtain initial density and orbitals.
        
        This function is used for initializing meta-GGA functionals (e.g., SCAN) by running
        a simpler functional (e.g., GGA_PBE) first to obtain better initial guesses, which
        improves convergence.
        
        Parameters
        ----------
        xc_functional : str
            XC functional name for warm start (e.g., "GGA_PBE" or "RSCAN")
        rho_initial : np.ndarray
            Initial density guess to start the warm calculation
        orbitals_initial : np.ndarray, optional
            Initial orbitals guess (if available), None otherwise
        
        Returns
        -------
        rho_initial : np.ndarray
            Initial density from warm start calculation
        orbitals_initial : np.ndarray
            Initial orbitals from warm start calculation
        
        Notes
        -----
        - Creates a temporary SCFDriver based on existing components
        - Uses relaxed convergence criteria to accelerate warm start
        - Warm start calculation does not require full convergence, only a reasonable initial guess
        """
        # Create temporary eigensolver with specified functional type
        eigensolver_warm = EigenSolver(xc_functional=xc_functional)


        # Create temporary SCFDriver with specified xc_functional
        # Reuse existing hamiltonian_builder, density_calculator, poisson_solver in the main SCFDriver
        scf_driver_warm = SCFDriver(
            hamiltonian_builder     = self.scf_driver.hamiltonian_builder,
            density_calculator      = self.scf_driver.density_calculator,
            poisson_solver          = self.scf_driver.poisson_solver,
            eigensolver             = eigensolver_warm,
            mixer                   = self.scf_driver.mixer,
            occupation_info         = self.scf_driver.occupation_info,
            xc_functional           = xc_functional,  # Use specified functional
            spin_polarized_flag     = self.spin_polarized_flag,
            hybrid_mixing_parameter = self.scf_driver.hybrid_mixing_parameter,
        )
        
        
        # Run warm start SCF calculation
        if self.verbose:
            print(f"[Warm Start] Running {xc_functional} pre-calculation for initial guess")
        
        scf_result_warm = scf_driver_warm.run(
            rho_initial      = rho_initial,
            settings         = self._get_scf_settings(xc_functional),
            orbitals_initial = orbitals_initial
        )
        
        if self.verbose:
            if not scf_result_warm.converged:
                print(WARM_START_NOT_CONVERGED_WARNING.format(xc_functional))
        
        # Extract results: density and orbitals
        rho_final      = scf_result_warm.density_data.rho
        orbitals_final = scf_result_warm.orbitals
        
        return rho_final, orbitals_final


    def forward(
        self, 
        orbitals               : np.ndarray,
        full_eigen_energies    : Optional[np.ndarray] = None,
        full_orbitals          : Optional[np.ndarray] = None,
        full_l_terms           : Optional[np.ndarray] = None,
        compute_energy_density : bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass of the atomic DFT solver.
        
        This method performs a single forward pass without SCF iteration:
        - Takes rho and orbitals as input
        - Computes XC potential and energy
        - Returns results in the same format as solve()
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions R_nl(r))
            Shape: (n_states, n_quad_points)
        full_eigen_energies : Optional[np.ndarray]
            Full eigenvalues of the Kohn-Sham orbitals
        full_orbitals : Optional[np.ndarray]
            Full orbitals of the Kohn-Sham orbitals
        full_l_terms : Optional[np.ndarray]
            Full l terms of the Kohn-Sham orbitals
        compute_energy_density : bool
            Whether to compute energy density (default: False)
        
        Returns
        -------
        final_result : Dict[str, Any]
            Dictionary containing:
            - eigen_energies: None (not computed in forward pass)
            - orbitals: input orbitals
            - rho: valence density computed from orbitals
            - density_data: DensityData computed from orbitals (without NLCC)
            - rho_nlcc: non-linear core correction density
            - energy: total energy
            - energy_components: EnergyComponents object
            - grid_data: GridData for standard grid
            - occupation_info: OccupationInfo
            - xc_potential: XCPotentialData
            ... other information from the forward pass
        """
        _forward_wall_t0 = time.perf_counter()

        # Phase 1: Get XC functional requirements
        # Note: Grids and SCF components are already initialized in __init__
        switches = SwitchesFlags(
            xc_functional           = self.xc_functional,
            use_oep                 = self.use_oep,
            hybrid_mixing_parameter = self.hybrid_mixing_parameter,
        )
        xc_requirements = get_functional_requirements(self.xc_functional)
        
        # Phase 2: Calculate rho_nlcc (non-linear core correction for pseudopotentials)
        rho_nlcc = self.pseudo.get_rho_core_correction(self.grid_data_standard.quadrature_nodes)
        
        # Phase 3: Create density_data from orbitals (with NLCC for XC potential calculation)
        # Note: For XC potential calculation, we need density_data with NLCC
        # For energy calculation, we use density_data without NLCC
        density_data_with_nlcc = self.density_calculator.create_density_data_from_orbitals(
            orbitals         = orbitals,
            compute_gradient = xc_requirements.needs_gradient,
            compute_tau      = xc_requirements.needs_tau,
            normalize        = True,
            rho_nlcc         = rho_nlcc
        )
        
        # Phase 4: Compute localized XC potential data (using density_data with NLCC)
        # Note: compute_local_xc_potential already handles the mixing of v_x and v_x_oep
        # based on hybrid_mixing_parameter, so we don't need to mix again here
        n_grid = len(self.grid_data_standard.quadrature_nodes)
        v_x_local = np.zeros(n_grid)
        v_c_local = np.zeros(n_grid)

        if switches.use_xc_functional or switches.use_oep:
            v_x_local, v_c_local = self.energy_calculator.compute_local_xc_potential(
                density_data           = density_data_with_nlcc,
                full_eigen_energies    = full_eigen_energies,
                full_orbitals          = full_orbitals,
                full_l_terms           = full_l_terms,
                enable_parallelization = self.enable_parallelization,
            )

        # Phase 5: Create density_data without NLCC for energy calculation
        # Energy calculation uses valence density only (without NLCC)
        density_data_valence = self.density_calculator.create_density_data_from_orbitals(
            orbitals         = orbitals,
            compute_gradient = xc_requirements.needs_gradient,
            compute_tau      = xc_requirements.needs_tau,
            normalize        = True,
            rho_nlcc         = None  # No NLCC for energy calculation
        )
        
        # Phase 6: Compute final energy (using valence density only)
        energy_components : EnergyComponents = self.energy_calculator.compute_energy(
            orbitals               = orbitals,
            density_data           = density_data_valence,
            mixing_parameter       = self.hybrid_mixing_parameter,
            full_eigen_energies    = full_eigen_energies,
            full_orbitals          = full_orbitals,
            full_l_terms           = full_l_terms,
            enable_parallelization = self.enable_parallelization,
        )

        if self.verbose:
            energy_components.print_info(title = f"Total Energy ({self.xc_functional})")

        # Phase 7: Evaluate basis functions on uniform grid
        uniform_grid, orbitals_on_uniform_grid = self._evaluate_basis_on_uniform_grid(
            ops_builder_standard = self.ops_builder_standard,
            orbitals             = orbitals
        )

        # evaluate local potentials on uniform grid
        v_x_local_on_uniform_grid = self.ops_builder_standard.evaluate_single_field_on_grid(
            given_grid   = uniform_grid,
            field_values = v_x_local,
        )
        v_c_local_on_uniform_grid = self.ops_builder_standard.evaluate_single_field_on_grid(
            given_grid   = uniform_grid,
            field_values = v_c_local,
        )


        # Phase 8: Compute final energy density
        if compute_energy_density:
            e_x_local, e_c_local = self.energy_calculator.compute_local_xc_energy_density(
                density_data           = density_data_valence,
                full_eigen_energies    = full_eigen_energies,
                full_orbitals          = full_orbitals,
                full_l_terms           = full_l_terms,
                enable_parallelization = self.enable_parallelization,
            )

            # evaluate local energy density on uniform grid
            e_x_local_on_uniform_grid = self.ops_builder_standard.evaluate_single_field_on_grid(
                given_grid   = uniform_grid,
                field_values = e_x_local,
            )
            e_c_local_on_uniform_grid = self.ops_builder_standard.evaluate_single_field_on_grid(
                given_grid   = uniform_grid,
                field_values = e_c_local,
            )
        else:
            e_x_local, e_c_local = None, None
            e_x_local_on_uniform_grid, e_c_local_on_uniform_grid = None, None

        if self.verbose:
            print("===========================================================================")
            print("                         FORWARD PASS COMPLETE                             ")
            print("===========================================================================")
            _elapsed = time.perf_counter() - _forward_wall_t0
            print(f"\t Elapsed wall time: {format_wall_duration(_elapsed)}")
            print()
            self.print_footer()

        final_result = {
            'eigen_energies'            : None,                      # Not computed in forward pass
            'orbitals'                  : orbitals,                  # Input orbitals
            'rho'                       : density_data_valence.rho,  # Valence density
            'density_data'              : density_data_valence,      # Density data without NLCC
            'rho_nlcc'                  : rho_nlcc,                  # Non-linear core correction density
            'energy'                    : energy_components.total,   # Total energy
            'energy_components'         : energy_components,         # Energy components, instance of the classEnergyComponents
            'converged'                 : None,                      # Forward pass doesn't iterate, so always "converged"
            'iterations'                : None,                      # No iterations in forward pass
            'rho_residual'              : None,                      # No residual in forward pass
            'grid_data'                 : self.grid_data_standard,   # Standard grid data
            'quadrature_nodes'          : self.grid_data_standard.quadrature_nodes,   # Global quadrature nodes
            'quadrature_weights'        : self.grid_data_standard.quadrature_weights, # Global quadrature weights
            'occupation_info'           : self.occupation_info,      # Occupation info
            'v_x_local'                 : v_x_local,                 # Local XC potential
            'v_c_local'                 : v_c_local,                 # Local XC potential
            'e_x_local'                 : e_x_local,                 # Local XC energy density
            'e_c_local'                 : e_c_local,                 # Local XC energy density
            'uniform_grid'              : uniform_grid,              # Uniform grid
            'orbitals_on_uniform_grid'  : orbitals_on_uniform_grid,  # Orbitals on uniform grid
            'v_x_local_on_uniform_grid' : v_x_local_on_uniform_grid, # Local XC potential on uniform grid
            'v_c_local_on_uniform_grid' : v_c_local_on_uniform_grid, # Local XC potential on uniform grid
            'e_x_local_on_uniform_grid' : e_x_local_on_uniform_grid, # Local XC energy density on uniform grid
            'e_c_local_on_uniform_grid' : e_c_local_on_uniform_grid, # Local XC energy density on uniform grid
            'full_eigen_energies'       : full_eigen_energies,       # Full eigenvalues
            'full_orbitals'             : full_orbitals,             # Full orbitals
            'full_l_terms'              : full_l_terms,              # Full l terms
            'intermediate_info'         : None,                      # Intermediate information from SCF iterations
        }


        return final_result        


    def solve(
        self, 
        save_intermediate  : bool = False, 
        save_energy_density: bool = False,
        save_full_spectrum : bool = False,
        rho_initial        : Optional[np.ndarray] = None,
        orbitals_initial   : Optional[np.ndarray] = None,
        use_warm_start     : bool = True,
        evaluate_basis_on_uniform_grid: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve the Kohn-Sham equations using modular SCF architecture.
        
        Parameters
        ----------
        save_intermediate : bool, optional
            If True, save intermediate information from each SCF iteration.
            This includes density residuals and density values at each iteration.
            Default is False.
        save_energy_density : bool, optional
            If True, save energy density information.
            Default is False.
        save_full_spectrum : bool, optional
            If True, save full spectrum information from each iteration.
            This includes eigenvalues, eigenvectors, density, energy at each iteration.
            Default is False.
        rho_initial : np.ndarray, optional
            Initial density guess. If provided, this will be used instead of
            the default guess from pseudopotential. Should have the same length
            as the quadrature nodes. Default is None.
        orbitals_initial : np.ndarray, optional
            Initial occupied Kohn--Sham orbitals on the quadrature grid,
            shape ``(n_quadrature_points, n_occupied)``. Passed to the SCF
            driver (and into the meta-GGA / OEP warm-start pre-calculation when
            ``use_warm_start`` is True). Default is None.
        use_warm_start : bool, optional
            If True (default), run the optional GGA_PBE warm-start
            pre-calculation for SCAN / RSCAN / R2SCAN and for OEP functionals.
            If False, ``rho_initial`` is left as-is; when the functional ``needs_tau`` and
            no ``orbitals_initial`` is given, one non-self-consistent precursor-XC
            diagonalization at that density is still run
            (``_get_initial_orbitals_from_density_with_precursor_xc``) for τ.
        evaluate_basis_on_uniform_grid : bool, optional
            If True, interpolate orbitals and local XC potentials (and,
            when ``save_energy_density`` is True, energy densities) onto the
            uniform visualization grid. If False, skip that work and set
            ``uniform_grid``, ``orbitals_on_uniform_grid``, and all
            ``*_on_uniform_grid`` result entries to ``None``. Default is False.
        
        Returns
        -------
        final_result : Dict[str, Any]
            A dictionary containing the final results of the calculation.
            The keys are the names of the results, and the values are the results themselves.
            - eigen_energies            : Full eigenvalues
            - orbitals                  : Orbitals
            - rho                       : Density
            - density_data              : Density data
            - rho_nlcc                  : Non-linear core correction density
            - energy                    : Total energy
            - energy_components         : Energy components
            - converged                 : Whether the calculation converged
            - iterations                : Number of iterations
            - rho_residual              : Density residual
            - grid_data                 : Grid data
            - quadrature_nodes          : Quadrature nodes
            - quadrature_weights        : Quadrature weights
            - occupation_info           : Occupation info
            - v_x_local                 : Local XC potential
            - v_c_local                 : Local XC potential
            - e_x_local                 : Local XC energy density
            - e_c_local                 : Local XC energy density
            - uniform_grid              : Uniform grid (``None`` if ``evaluate_basis_on_uniform_grid`` is False)
            - orbitals_on_uniform_grid  : Orbitals on uniform grid (``None`` if disabled)
            - v_x_local_on_uniform_grid : Local XC potential on uniform grid (``None`` if disabled)
            - v_c_local_on_uniform_grid : Local XC potential on uniform grid (``None`` if disabled)
            - e_x_local_on_uniform_grid : Local XC energy density on uniform grid (``None`` if disabled)
            - e_c_local_on_uniform_grid : Local XC energy density on uniform grid (``None`` if disabled)
            - full_eigen_energies       : Full eigenvalues
            - full_orbitals             : Full orbitals
            - full_l_terms              : Full l terms
            - intermediate_info         : Intermediate information from SCF iterations
        """
        # Type check
        assert isinstance(save_intermediate, bool), \
            SAVE_INTERMEDIATE_NOT_BOOL_ERROR.format(type(save_intermediate))
        assert isinstance(save_energy_density, bool), \
            SAVE_ENERGY_DENSITY_NOT_BOOL_ERROR.format(type(save_energy_density))
        assert isinstance(save_full_spectrum, bool), \
            SAVE_FULL_SPECTRUM_NOT_BOOL_ERROR.format(type(save_full_spectrum))        
        assert isinstance(use_warm_start, bool), \
            USE_WARM_START_NOT_BOOL_ERROR.format(type(use_warm_start))
        assert isinstance(evaluate_basis_on_uniform_grid, bool), \
            EVALUATE_BASIS_ON_UNIFORM_GRID_NOT_BOOL_ERROR.format(type(evaluate_basis_on_uniform_grid))
        if rho_initial is not None:
            assert isinstance(rho_initial, np.ndarray), \
                RHO_INITIAL_NOT_NUMPY_ARRAY_ERROR.format(type(rho_initial))
            expected_length = len(self.grid_data_standard.quadrature_nodes)
            actual_length = len(rho_initial)
            assert actual_length == expected_length, \
                RHO_INITIAL_LENGTH_MISMATCH_ERROR.format(expected_length, actual_length)
        if orbitals_initial is not None:
            assert isinstance(orbitals_initial, np.ndarray), \
                ORBITALS_INITIAL_NOT_NUMPY_ARRAY_ERROR.format(type(orbitals_initial))
            n_q = len(self.grid_data_standard.quadrature_nodes)
            assert int(orbitals_initial.shape[0]) == n_q, \
                ORBITALS_INITIAL_FIRST_DIM_MISMATCH_ERROR.format(n_q, orbitals_initial.shape[0])

        _solve_wall_t0 = time.perf_counter()

        # Phase 1: Initial density guess
        # Note: Grids and SCF components are already initialized in __init__
        if rho_initial is None:
            rho_initial = self.pseudo.get_rho_guess(self.grid_data_standard.quadrature_nodes)
        rho_nlcc = self.pseudo.get_rho_core_correction(self.grid_data_standard.quadrature_nodes)
        orbitals_for_scf : Optional[np.ndarray] = orbitals_initial

        # Warm start calculation for relatively expensive meta-GGA functionals
        if use_warm_start and (
            self.xc_functional in ['SCAN', 'RSCAN', 'R2SCAN', 'HF', 'PBE0', 'EXX'] or self.use_oep
        ):
            rho_initial, orbitals_for_scf = self._get_initial_density_and_orbitals_with_warm_start(
                xc_functional    = "GGA_PBE", 
                rho_initial      = rho_initial, 
                orbitals_initial = orbitals_for_scf)
        elif (
            not use_warm_start
            and orbitals_for_scf is None
            and self.xc_requirements.needs_tau
        ):
            orbitals_for_scf = self._get_initial_orbitals_from_density_with_precursor_xc(
                xc_functional    = "GGA_PBE",
                rho_initial      = rho_initial,
                orbitals_initial = None,
            )

        # Phase 2: Run SCF
        scf_result : SCFResult = self.scf_driver.run(
            rho_initial        = rho_initial,
            settings           = self._get_scf_settings(self.xc_functional),
            orbitals_initial   = orbitals_for_scf,
            save_intermediate  = save_intermediate,
            save_full_spectrum = save_full_spectrum,
        )

        # Phase 3: Compute final xc potential data
        v_x_local, v_c_local = self.energy_calculator.compute_local_xc_potential(
            density_data           = scf_result.density_data,
            full_eigen_energies    = scf_result.full_eigen_energies,
            full_orbitals          = scf_result.full_orbitals,
            full_l_terms           = scf_result.full_l_terms,
            enable_parallelization = self.enable_parallelization,
        )

        # Phase 4: Compute final energy        
        energy_components : EnergyComponents = self.energy_calculator.compute_energy(
            orbitals               = scf_result.orbitals,
            density_data           = scf_result.density_data,
            mixing_parameter       = self.hybrid_mixing_parameter,
            full_eigen_energies    = scf_result.full_eigen_energies,
            full_orbitals          = scf_result.full_orbitals,
            full_l_terms           = scf_result.full_l_terms,
            enable_parallelization = self.enable_parallelization,
        )

        # Phase 5: Evaluate basis functions on uniform grid (optional)
        if evaluate_basis_on_uniform_grid:
            uniform_grid, orbitals_on_uniform_grid = self._evaluate_basis_on_uniform_grid(
                ops_builder_standard = self.ops_builder_standard,
                orbitals             = scf_result.orbitals
            )

            v_x_local_on_uniform_grid = self.ops_builder_standard.evaluate_single_field_on_grid(
                given_grid   = uniform_grid,
                field_values = v_x_local,
            )
            v_c_local_on_uniform_grid = self.ops_builder_standard.evaluate_single_field_on_grid(
                given_grid   = uniform_grid,
                field_values = v_c_local,
            )
        else:
            uniform_grid = None
            orbitals_on_uniform_grid = None
            v_x_local_on_uniform_grid = None
            v_c_local_on_uniform_grid = None

        # Phase 6: Compute final energy density
        if save_energy_density:
            e_x_local, e_c_local = self.energy_calculator.compute_local_xc_energy_density(
                density_data           = scf_result.density_data,
                full_eigen_energies    = scf_result.full_eigen_energies,
                full_orbitals          = scf_result.full_orbitals,
                full_l_terms           = scf_result.full_l_terms,
                enable_parallelization = self.enable_parallelization,
            )

            if evaluate_basis_on_uniform_grid:
                e_x_local_on_uniform_grid = self.ops_builder_standard.evaluate_single_field_on_grid(
                    given_grid   = uniform_grid,
                    field_values = e_x_local,
                )
                e_c_local_on_uniform_grid = self.ops_builder_standard.evaluate_single_field_on_grid(
                    given_grid   = uniform_grid,
                    field_values = e_c_local,
                )
            else:
                e_x_local_on_uniform_grid = None
                e_c_local_on_uniform_grid = None
        else:
            e_x_local, e_c_local = None, None
            e_x_local_on_uniform_grid, e_c_local_on_uniform_grid = None, None
        
        # Print debug information
        if self.verbose:
            energy_components.print_info(title = f"Total Energy ({self.xc_functional})")
            print("===========================================================================")
            print("                          CALCULATION COMPLETE                             ")
            print("===========================================================================")
            _elapsed = time.perf_counter() - _solve_wall_t0
            print(f"\t Elapsed wall time: {format_wall_duration(_elapsed)}")
            print()
            self.print_footer()

        # Phase 7: Pack and return results
        final_result = {
            'eigen_energies'            : scf_result.eigen_energies,
            'orbitals'                  : scf_result.orbitals,
            'rho'                       : scf_result.density_data.rho,  # Interpolate over psi and calculate rho at that site
            'density_data'              : scf_result.density_data,
            'rho_nlcc'                  : rho_nlcc,
            'energy'                    : energy_components.total,
            'energy_components'         : energy_components,
            'converged'                 : scf_result.converged,
            'iterations'                : scf_result.iterations,
            'rho_residual'              : scf_result.rho_residual,
            'grid_data'                 : self.grid_data_standard,
            'quadrature_nodes'          : self.grid_data_standard.quadrature_nodes,   # Global quadrature nodes
            'quadrature_weights'        : self.grid_data_standard.quadrature_weights, # Global quadrature weights
            'occupation_info'           : self.occupation_info,
            'v_x_local'                 : v_x_local,
            'v_c_local'                 : v_c_local,
            'e_x_local'                 : e_x_local,
            'e_c_local'                 : e_c_local,
            'uniform_grid'              : uniform_grid,
            'orbitals_on_uniform_grid'  : orbitals_on_uniform_grid,
            'v_x_local_on_uniform_grid' : v_x_local_on_uniform_grid, # Local XC potential on uniform grid
            'v_c_local_on_uniform_grid' : v_c_local_on_uniform_grid, # Local XC potential on uniform grid
            'e_x_local_on_uniform_grid' : e_x_local_on_uniform_grid, # Local XC energy density on uniform grid
            'e_c_local_on_uniform_grid' : e_c_local_on_uniform_grid, # Local XC energy density on uniform grid
            'full_eigen_energies'       : scf_result.full_eigen_energies,
            'full_orbitals'             : scf_result.full_orbitals,
            'full_l_terms'              : scf_result.full_l_terms,
            'intermediate_info'         : scf_result.intermediate_info,  # Intermediate information from SCF iterations
        }
        
        return final_result


    def evaluate_single_field_on_grid(
        self,
        given_grid   : np.ndarray,
        field_values : np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate a single field on a given grid using Lagrange interpolation.

        This is a thin wrapper around
        `RadialOperatorsBuilder.evaluate_single_field_on_grid`, provided for
        convenience when working with an `AtomicDFTSolver`.

        Parameters
        ----------
        given_grid : np.ndarray
            Grid points where the field should be evaluated.
        field_values : np.ndarray
            Field values at quadrature points, shape (n_elem * n_quad,).

        Returns
        -------
        field_on_grid : np.ndarray
            Field values evaluated on the given grid.
        """
        return self.ops_builder_standard.evaluate_single_field_on_grid(
            given_grid   = given_grid,
            field_values = field_values,
        )



    @property
    def quadrature_nodes(self) -> np.ndarray:
        """
        Global quadrature nodes.
        """
        return self.grid_data_standard.quadrature_nodes
    

    @property
    def quadrature_weights(self) -> np.ndarray:
        """
        Global quadrature weights.
        """
        return self.grid_data_standard.quadrature_weights



if __name__ == "__main__":
    """
    Example usage of AtomicDFTSolver.
    
    For comprehensive tests including comparison with reference values from
    the featom paper (Čertík et al., Comput. Phys. Commun. 297, 109051, 2024),
    see: atom/tests/solver_uranium_lda_testcase.py
    """

    import time
    start_time = time.time()
    atomic_dft_solver = AtomicDFTSolver(
        atomic_number             = 95,
        xc_functional             = 'GGA_PBE',
        domain_size               = 20,
        finite_element_number     = 4,
        polynomial_order          = 20,
        quadrature_point_number   = 60,
        mesh_type                 = "polynomial",
        mesh_concentration        = 2,
        scf_tolerance             = 1e-9,
        verbose                   = True, 
        all_electron_flag         = True,
        use_oep                   = False,
        use_preconditioner        = True,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to initialize the solver: {elapsed_time:.2f} seconds")

    start_time = time.time()
    results  = atomic_dft_solver.solve(save_energy_density=True, save_full_spectrum=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to solve the problem: {elapsed_time:.2f} seconds")


    rho      = results['rho']
    orbitals = results['orbitals']
    # n_occ = results
    print("rho.shape      = ", rho.shape)      # (n_grid_points,)
    print("orbitals.shape = ", orbitals.shape) # (n_grid_points, n_orbitals)
    # print("density data =  ", dir(density_data))
    print("grid_coord = ", results['quadrature_nodes'].shape)
    print(f"full eigen energy shape: {results['full_eigen_energies'].shape}")
    results['occupation_info'].print_info()
    n_occ = results['occupation_info'].n_states
    print(n_occ)
    print(f"full eigen energy in occupied states: {results['full_eigen_energies'][:n_occ]}")
    # print(f"full eigen energy: {results['full_eigen_energies']}")
    # np.save("full_eigen_energy.npy", np.array(results['full_eigen_energies']))
    print(f" eigen energy: {results['eigen_energies']}")

    print("difference = ", results['full_eigen_energies'][:n_occ] - results['eigen_energies'])
