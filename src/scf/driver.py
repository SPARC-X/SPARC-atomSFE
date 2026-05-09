from __future__ import annotations
import numpy as np
import copy
from typing import Dict, Any, Optional, Union, List, Tuple, Iterable, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .driver import SCFResult
    from ..xc.ml_xc import MLXCCalculator

from .hamiltonian import HamiltonianBuilder
from .density import DensityCalculator, DensityData
from .poisson import PoissonSolver
from .convergence import ConvergenceChecker
from .eigensolver import EigenSolver
from .mixer import Mixer
from .response import ResponseCalculator
from ..xc.evaluator import XCEvaluator, create_xc_evaluator
from ..xc.functional_requirements import get_functional_requirements, FunctionalRequirements
from ..xc.hf import HartreeFockExchange
from ..xc.oep import OEPCalculator
from ..mesh.operators import RadialOperatorsBuilder, GridData
from ..utils.occupation_states import OccupationInfo
from ..data.data_loading import VALID_FEATURES_LIST_FOR_ENERGY_DENSITY, VALID_FEATURES_LIST_FOR_POTENTIAL
from ..data.data_processing import DataProcessor



# Switches Flags Error Messages
XC_FUNCTIONAL_NOT_STRING_ERROR = \
    "parameter 'xc_functional' must be a string, get type {} instead."
HYBRID_MIXING_PARAMETER_NOT_A_FLOAT_ERROR = \
    "parameter 'hybrid_mixing_parameter' must be a float, get type {} instead."
USE_XC_FUNCTIONAL_TYPE_ERROR = \
    "parameter 'use_xc_functional' must be a boolean, get type {} instead."
USE_HF_EXCHANGE_TYPE_ERROR = \
    "parameter 'use_hf_exchange' must be a boolean, get type {} instead."
USE_OEP_EXCHANGE_TYPE_ERROR = \
    "parameter 'use_oep_exchange' must be a boolean, get type {} instead."
USE_OEP_CORRELATION_TYPE_ERROR = \
    "parameter 'use_oep_correlation' must be a boolean, get type {} instead."

USE_OEP_NOT_BOOL_ERROR = \
    "parameter 'use_oep' must be a boolean, get type {} instead."
USE_OEP_SHOULD_NOT_BE_NONE_FOR_PBE0_ERROR = \
    "parameter 'use_oep' should not be None for PBE0 functional, get None instead."
USE_OEP_NOT_TRUE_FOR_OEP_FUNCTIONAL_ERROR = \
    "parameter 'use_oep' must be True for OEP functional '{}', get {} instead."
USE_OEP_NOT_FALSE_FOR_NON_OEP_FUNCTIONAL_ERROR = \
    "parameter 'use_oep' must be False for non-OEP functional '{}', get {} instead."


# SCF Settings Error Messages
INNER_MAX_ITER_TYPE_ERROR = \
    "parameter 'inner_max_iter' must be an integer, get type {} instead."
RHO_TOL_TYPE_ERROR = \
    "parameter 'rho_tol' must be a float, get type {} instead."
N_CONSECUTIVE_TYPE_ERROR = \
    "parameter 'n_consecutive' must be an integer, get type {} instead."
OUTER_MAX_ITER_TYPE_ERROR = \
    "parameter 'outer_max_iter' must be an integer, get type {} instead."
OUTER_RHO_TOL_TYPE_ERROR = \
    "parameter 'outer_rho_tol' must be a float, get type {} instead."
VERBOSE_TYPE_ERROR = \
    "parameter 'verbose' must be a boolean, get type {} instead."

# SCF Result Error Messages
EIGENVALUES_TYPE_ERROR = \
    "parameter 'eigenvalues' must be a numpy array, get type {} instead."
EIGENVECTORS_TYPE_ERROR = \
    "parameter 'eigenvectors' must be a numpy array, get type {} instead."
DENSITY_DATA_TYPE_ERROR = \
    "parameter 'density_data' must be a DensityData instance, get type {} instead."
FULL_EIGENVALUES_TYPE_ERROR = \
    "parameter 'full_eigenvalues' must be a numpy array, get type {} instead."
FULL_EIGENVECTORS_TYPE_ERROR = \
    "parameter 'full_eigenvectors' must be a numpy array, get type {} instead."
FULL_L_TERMS_TYPE_ERROR = \
    "parameter 'full_l_terms' must be a numpy array, get type {} instead."

RHO_TYPE_ERROR = \
    "parameter 'rho' must be a numpy array, get type {} instead."
CONVERGED_TYPE_ERROR = \
    "parameter 'converged' must be a boolean, get type {} instead."
ITERATIONS_TYPE_ERROR = \
    "parameter 'iterations' must be an integer, get type {} instead."
RHO_RESIDUAL_TYPE_ERROR = \
    "parameter 'rho_residual' must be a float, get type {} instead."
RESIDUAL_TYPE_ERROR = \
    "parameter 'residual' must be a float, get type {} instead."
TOTAL_ENERGY_TYPE_ERROR = \
    "parameter 'total_energy' must be a float, get type {} instead."
ENERGY_COMPONENTS_TYPE_ERROR = \
    "parameter 'energy_components' must be a dictionary, get type {} instead."
OUTER_ITERATIONS_TYPE_ERROR = \
    "parameter 'outer_iterations' must be an integer, get type {} instead."
OUTER_CONVERGED_TYPE_ERROR = \
    "parameter 'outer_converged' must be a boolean, get type {} instead."
ENABLE_PARALLELIZATION_NOT_BOOL_ERROR = \
    "parameter 'enable_parallelization' must be a boolean, get type {} instead."

ML_XC_CALCULATOR_TYPE_ERROR = \
    "parameter 'ml_xc_calculator' must be a MLXCCalculator, get type {} instead."
ML_EACH_SCF_STEP_NOT_BOOL_ERROR = \
    "parameter 'ml_each_scf_step' must be a boolean, get type {} instead."
ML_SMOOTH_RHO_FOR_FEATURES_NOT_BOOL_ERROR = \
    "parameter 'ml_smooth_rho_for_features' must be a boolean, get type {} instead."
# IntermediateInfo Error Messages
INNER_ITERATION_TYPE_ERROR = \
    "parameter 'inner_iteration' must be an integer, get type {} instead."
RHO_RESIDUAL_TYPE_INTERMEDIATE_ERROR = \
    "parameter 'rho_residual' must be a float, get type {} instead."
RHO_TYPE_INTERMEDIATE_ERROR = \
    "parameter 'rho' must be a numpy array, get type {} instead."
RHO_NORM_TYPE_ERROR = \
    "parameter 'rho_norm' must be a float, get type {} instead."
OUTER_ITERATION_TYPE_ERROR = \
    "parameter 'outer_iteration' must be an integer, get type {} instead."
OUTER_RHO_RESIDUAL_TYPE_INTERMEDIATE_ERROR = \
    "parameter 'outer_rho_residual' must be a float, get type {} instead."
INNER_RESULT_TYPE_ERROR = \
    "parameter 'inner_result' must be a SCFResult instance, get type {} instead."
CURRENT_OUTER_ITERATION_TYPE_ERROR = \
    "parameter 'current_outer_iteration' must be an integer, get type {} instead."

# SCF Driver Error Messages
HAMILTONIAN_BUILDER_TYPE_ERROR = \
    "parameter 'hamiltonian_builder' must be a HamiltonianBuilder, get type {} instead."
DENSITY_CALCULATOR_TYPE_ERROR = \
    "parameter 'density_calculator' must be a DensityCalculator, get type {} instead."
POISSON_SOLVER_TYPE_ERROR = \
    "parameter 'poisson_solver' must be a PoissonSolver, get type {} instead."
EIGENSOLVER_TYPE_ERROR = \
    "parameter 'eigensolver' must be a EigenSolver, get type {} instead."
MIXER_TYPE_ERROR = \
    "parameter 'mixer' must be a Mixer, get type {} instead."
OCCUPATION_INFO_TYPE_ERROR = \
    "parameter 'occupation_info' must be a OccupationInfo, get type {} instead."
HYBRID_MIXING_PARAMETER_TYPE_ERROR = \
    "parameter 'hybrid_mixing_parameter' must be a float, get type {} instead."
OPS_BUILDER_OEP_TYPE_ERROR = \
    "parameter 'ops_builder_oep' must be a RadialOperatorsBuilder, get type {} instead."
OEP_MIXING_PARAMETER_TYPE_ERROR = \
    "parameter 'oep_mixing_parameter' must be a float, get type {} instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_TYPE_ERROR = \
    "parameter 'frequency_quadrature_point_number' must be an integer, get type {} instead."
ANGULAR_MOMENTUM_CUTOFF_TYPE_ERROR = \
    "parameter 'angular_momentum_cutoff' must be an integer, get type {} instead."
ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_ERROR = \
    "parameter 'angular_momentum_cutoff' must be not None, get None instead."


RHO_INITIAL_TYPE_ERROR = \
    "parameter 'rho_initial' must be a numpy array, get type {} instead."
SETTINGS_TYPE_ERROR = \
    "parameter 'settings' must be a SCFSettings or a dictionary, get type {} instead."
V_X_OEP_TYPE_ERROR = \
    "parameter 'v_x_oep' must be a numpy array, get type {} instead."
V_C_OEP_TYPE_ERROR = \
    "parameter 'v_c_oep' must be a numpy array, get type {} instead."
H_HF_EXCHANGE_DICT_BY_L_TYPE_ERROR = \
    "provided parameter 'H_hf_exchange_dict_by_l' must be a dictionary, get type {} instead."
SYMMETRIZE_TYPE_ERROR = \
    "parameter 'symmetrize' must be a boolean, get type {} instead."
SAVE_FULL_SPECTRUM_TYPE_ERROR = \
    "parameter 'save_full_spectrum' must be a boolean, get type {} instead."
ORBITALS_INITIAL_TYPE_ERROR = \
    "parameter 'orbitals_initial' must be a numpy array, get type {} instead."
XC_FUNCTIONAL_TYPE_ERROR = \
    "parameter 'xc_functional' must be a string, get type {} instead."
SPIN_POLARIZED_FLAG_NOT_BOOL_ERROR = \
    "parameter 'spin_polarized_flag' must be a boolean, get type {} instead."
SWITCHES_TYPE_ERROR = \
    "parameter 'switches' must be a SwitchesFlags, get type {} instead."
XC_REQUIREMENTS_TYPE_ERROR = \
    "parameter 'xc_requirements' must be a FunctionalRequirements, get type {} instead."
XC_CALCULATOR_TYPE_ERROR = \
    "parameter 'xc_calculator' must be a XCEvaluator, get type {} instead."

RHO_TYPE_ERROR = \
    "parameter 'rho' must be a numpy array, get type {} instead."
ORBITALS_TYPE_ERROR = \
    "parameter 'orbitals' must be a numpy array, get type {} instead."


OCC_EIGENVALUES_LIST_NOT_LIST_ERROR = \
    "parameter 'occ_eigenvalues_list' must be a list, get type {} instead."
OCC_EIGENVECTORS_LIST_NOT_LIST_ERROR = \
    "parameter 'occ_eigenvectors_list' must be a list, get type {} instead."
UNOCC_EIGENVALUES_LIST_NOT_LIST_ERROR = \
    "parameter 'unocc_eigenvalues_list' must be a list, get type {} instead."
UNOCC_EIGENVECTORS_LIST_NOT_LIST_ERROR = \
    "parameter 'unocc_eigenvectors_list' must be a list, get type {} instead."
OCC_EIGENVALUES_LIST_LENGTH_ERROR = \
    "Number of occupied eigenvalues in 'occ_eigenvalues_list' must match number of unique l values, get {} instead."
OCC_EIGENVECTORS_LIST_LENGTH_ERROR = \
    "Number of occupied eigenvectors in 'occ_eigenvectors_list' must match number of unique l values, get {} instead."
UNOCC_EIGENVALUES_LIST_LENGTH_ERROR = \
    "Number of unoccupied eigenvalues in 'unocc_eigenvalues_list' must match number of unique l values, get {} instead."
UNOCC_EIGENVECTORS_LIST_LENGTH_ERROR = \
    "Number of unoccupied eigenvectors in 'unocc_eigenvectors_list' must match number of unique l values, get {} instead."
OCC_EIGENVALUES_LIST_NDIM_ERROR = \
    "Occupied eigenvalues in 'occ_eigenvalues_list' must be 1D arrays, get dimension {} instead."
OCC_EIGENVECTORS_LIST_NDIM_ERROR = \
    "Occupied eigenvectors in 'occ_eigenvectors_list' must be 2D arrays, get dimension {} instead."
UNOCC_EIGENVALUES_LIST_NDIM_ERROR = \
    "Unoccupied eigenvalues in 'unocc_eigenvalues_list' must be 1D arrays, get dimension {} instead."
UNOCC_EIGENVECTORS_LIST_NDIM_ERROR = \
    "Unoccupied eigenvectors in 'unocc_eigenvectors_list' must be 2D arrays, get dimension {} instead."
OCC_EIGENVALUES_AND_EIGENVECTORS_LIST_SHAPE_MISMATCH_ERROR = \
    "Occupied eigenvalues and eigenvectors in 'occ_eigenvalues_list' and 'occ_eigenvectors_list' must have the same number of rows, get {} and {} instead."
UNOCC_EIGENVALUES_AND_EIGENVECTORS_LIST_SHAPE_MISMATCH_ERROR = \
    "Unoccupied eigenvalues and eigenvectors in 'unocc_eigenvalues_list' and 'unocc_eigenvectors_list' must have the same number of rows, get {} and {} instead."
OCC_EIGENVECTORS_LIST_SHAPE_ERROR = \
    "Occupied eigenvectors in 'occ_eigenvectors_list' must have the same number of rows as the number of interior nodes, get {} and {} instead."
UNOCC_EIGENVECTORS_LIST_SHAPE_ERROR = \
    "Unoccupied eigenvectors in 'unocc_eigenvectors_list' must have the same number of rows as the number of interior nodes, get {} and {} instead."
INVALID_FEATURES_IN_ML_MODEL_ERROR = \
    "Invalid feature in ML model detected: '{}'."
ML_FEATURES_QUADRATURE_RHO_LENGTH_MISMATCH_ERROR = \
    "Quadrature node count {} does not match rho length {} for ML features."
ML_XC_CALCULATOR_NOT_A_MLXCCALCULATOR_ERROR = \
    "parameter 'ml_xc_calculator' must be a MLXCCalculator, get type {} instead."

# Switches Flags Warning Messages
HYBRID_MIXING_PARAMETER_NOT_PROVIDED_WARNING = \
    "WARNING: 'hybrid_mixing_parameter' not provided for {} functional, using default value {}"
HYBRID_MIXING_PARAMETER_NOT_1_0_WARNING = \
    "WARNING: 'hybrid_mixing_parameter' for {} should be 1.0, got {}"
HYBRID_MIXING_PARAMETER_NOT_0_0_WARNING = \
    "WARNING: 'hybrid_mixing_parameter' for {} should be 0.0, got {}"
HYBRID_MIXING_PARAMETER_NOT_0_25_WARNING = \
    "WARNING: 'hybrid_mixing_parameter' for {} should be 0.25, got {}"

# SCF Driver Warning Messages
INNER_SCF_DID_NOT_CONVERGE_WARNING = \
    "WARNING: Inner SCF did not converge after {} iterations"
HF_CALCULATOR_NOT_AVAILABLE_WARNING = \
    "WARNING: Hartree-Fock calculator is not available, please initialize the HF calculator first"



class SwitchesFlags:
    """
    Internal helper class for HamiltonianBuilder to manage functional switches.
    Determines which Hamiltonian components to include based on the XC functional.
    
    Attributes:
        use_xc_functional   (bool) : Whether to use XC functional
        use_hf_exchange     (bool) : Whether to include Hartree-Fock exchange mixing
        use_oep_exchange    (bool) : Whether to use OEP exchange potential
        use_oep_correlation (bool) : Whether to use OEP correlation potential, for RPA
        use_metagga         (bool) : Whether the functional requires meta-GGA terms
    """
    def __init__(
        self, 
        xc_functional          : str, 
        use_oep                : Optional[bool]  = None,
        hybrid_mixing_parameter: Optional[float] = None,
    ):
        """
        Initialize switches based on XC functional and hybrid mixing parameter.
        
        Parameters
        ----------
        xc_functional : str
            Name of XC functional
        use_oep : bool, optional
            Whether to use OEP exchange potential, useful only for PBE0 functional, 
            otherwise it has to agree with use_oep_exchange flag
        hybrid_mixing_parameter : float, optional
            Mixing parameter for hybrid functionals (0-1). Required only for hybrid functionals.
            - For PBE0: should be 0.25
            - For HF  : should be 1.0
        """
        # Type checking
        assert isinstance(xc_functional, str), \
            XC_FUNCTIONAL_NOT_STRING_ERROR.format(type(xc_functional))

        if use_oep is not None:
            assert isinstance(use_oep, bool), \
            USE_OEP_NOT_BOOL_ERROR.format(type(use_oep))
        if hybrid_mixing_parameter is not None:
            assert isinstance(hybrid_mixing_parameter, (float, int)), \
                HYBRID_MIXING_PARAMETER_NOT_A_FLOAT_ERROR.format(type(hybrid_mixing_parameter))
            hybrid_mixing_parameter = float(hybrid_mixing_parameter)
        
        self.check_xc_functional_and_use_oep_consistency(use_oep, xc_functional)
        
        
        self.xc_functional           = xc_functional
        self.hybrid_mixing_parameter = hybrid_mixing_parameter
        
        # Initialize all flags to False, except use_xc_functional which is True by default
        self.use_xc_functional   = True
        self.use_hartree         = True
        self.use_hf_exchange     = False
        self.use_oep_exchange    = False
        self.use_oep_correlation = False
        self.use_metagga         = False
        
        # Set flags based on functional
        if xc_functional == 'None':
            self.use_xc_functional   = False
        
        # Direct Schrodinger mode: no XC, no Hartree
        elif xc_functional == 'Schrodinger':
            self.use_xc_functional   = False
            self.use_hartree         = False

        # Exchange-only functionals
        elif xc_functional == 'EXX':
            self.use_oep_exchange    = True
            self.use_oep_correlation = False # EXX does not use correlation potential
            self.use_xc_functional   = False
            if hybrid_mixing_parameter is not None:
                print(HYBRID_MIXING_PARAMETER_NOT_PROVIDED_WARNING.format('EXX', 1.0))
                self.hybrid_mixing_parameter = 1.0
            elif not np.isclose(hybrid_mixing_parameter, 1.0):
                print(HYBRID_MIXING_PARAMETER_NOT_1_0_WARNING.format('EXX', hybrid_mixing_parameter))
        
        # Exchange-correlation functionals, for RPA
        elif xc_functional == 'RPA':
            self.use_oep_exchange    = True
            self.use_oep_correlation = True
            self.use_xc_functional   = False
            if hybrid_mixing_parameter is not None and not np.isclose(hybrid_mixing_parameter, 1.0):
                print(HYBRID_MIXING_PARAMETER_NOT_1_0_WARNING.format('RPA', hybrid_mixing_parameter))
            self.hybrid_mixing_parameter = 1.0


        # Hartree-Fock exchange functionals
        elif xc_functional == 'HF':
            self.use_hf_exchange     = True
            self.use_xc_functional   = False
            # Check if hybrid_mixing_parameter is provided for hybrid functionals
            if hybrid_mixing_parameter is None:
                print(HYBRID_MIXING_PARAMETER_NOT_PROVIDED_WARNING.format('HF', 1.0))
                self.hybrid_mixing_parameter = 1.0
            elif not np.isclose(hybrid_mixing_parameter, 1.0):
                print(HYBRID_MIXING_PARAMETER_NOT_1_0_WARNING.format('HF', hybrid_mixing_parameter))
        
        # Hybrid GGA functionals
        elif xc_functional == 'PBE0':
            # Check if hybrid_mixing_parameter is provided for hybrid functionals
            if hybrid_mixing_parameter is None:
                print(HYBRID_MIXING_PARAMETER_NOT_PROVIDED_WARNING.format('PBE0', 0.25))
                self.hybrid_mixing_parameter = 0.25
            elif not np.isclose(hybrid_mixing_parameter, 0.25):
                print(HYBRID_MIXING_PARAMETER_NOT_0_25_WARNING.format('PBE0', hybrid_mixing_parameter))

            if use_oep is False:
                self.use_hf_exchange  = True
                self.use_oep_exchange = False
            else:
                self.use_oep_exchange = True
                self.use_hf_exchange  = False

        # Set meta-GGA flag for functionals that require de_xc_dtau
        elif xc_functional in ['SCAN', 'RSCAN', 'R2SCAN']:
            self.use_metagga = True
        
        # LDA/GGA functionals: no special flags (default False)
        elif xc_functional in ['LDA_SVWN', 'LDA_PZ', 'LDA_PW', 'GGA_PBE']:
            pass

        # Invalid XC functional
        else:
            raise ValueError(f"Invalid XC functional: {xc_functional}")



    @property
    def use_oep(self) -> bool:
        return self.use_oep_exchange or self.use_oep_correlation


    @staticmethod
    def check_xc_functional_and_use_oep_consistency(use_oep: bool, xc_functional: str):
        """
        Check if the consistency between use_oep and xc_functional is correct
        """
        # PBE0 functional must be used with OEP flag
        if xc_functional == 'PBE0':
            assert use_oep is not None, \
                USE_OEP_SHOULD_NOT_BE_NONE_FOR_PBE0_ERROR.format(xc_functional)
            return 
        
        # For other functionals, use_oep can be None, and no extra check is needed
        if use_oep is None:
            return
        

        if xc_functional in ['EXX', 'RPA']:
            assert use_oep is True, \
                USE_OEP_NOT_TRUE_FOR_OEP_FUNCTIONAL_ERROR.format(xc_functional)
        elif xc_functional in ['None', 'Schrodinger', 'LDA_SVWN', 'LDA_PZ', 'LDA_PW', 'GGA_PBE', 'SCAN', 'RSCAN', 'R2SCAN', 'HF']:
            assert use_oep is False, \
                USE_OEP_NOT_FALSE_FOR_NON_OEP_FUNCTIONAL_ERROR.format(xc_functional)
        else:
            raise ValueError(f"Invalid XC functional: {xc_functional}")



    def print_info(self):
        """
        Print functional switch information summary.
        """
        print("=" * 60)
        print("\t\t FUNCTIONAL SWITCHES")
        print("=" * 60)
        print(f"\t XC Functional              : {self.xc_functional}")
        if self.hybrid_mixing_parameter is not None:
            print(f"\t Hybrid Mixing Parameter    : {self.hybrid_mixing_parameter}")
        else:
            print(f"\t Hybrid Mixing Parameter    : None (not applicable)")
        print()
        print("\t FUNCTIONAL FLAGS:")
        print(f"\t use_xc_functional          : {self.use_xc_functional}")
        print(f"\t use_hartree                : {self.use_hartree}")
        print(f"\t use_hf_exchange            : {self.use_hf_exchange}")
        print(f"\t use_oep_exchange           : {self.use_oep_exchange}")
        print(f"\t use_oep_correlation        : {self.use_oep_correlation}")
        print(f"\t use_metagga                : {self.use_metagga}")
        print()



@dataclass
class InnerIterationInfo:
    """
    Information from a single inner SCF iteration.
    
    Attributes
    ----------
    outer_iteration : int
        Which outer iteration this inner iteration belongs to (0 if no outer loop)
    inner_iteration : int
        Inner iteration number within the current outer iteration
    rho_residual : float
        Density residual at this iteration
    rho : np.ndarray
        Density array at this iteration
    rho_norm : float
        L2 norm of density
    """
    outer_iteration : int
    inner_iteration : int
    rho_residual    : float
    rho             : np.ndarray
    rho_norm        : float
    
    def __post_init__(self):
        # Type checking
        assert isinstance(self.outer_iteration, (int, np.integer)), \
            OUTER_ITERATION_TYPE_ERROR.format(type(self.outer_iteration))
        assert isinstance(self.inner_iteration, (int, np.integer)), \
            INNER_ITERATION_TYPE_ERROR.format(type(self.inner_iteration))
        assert isinstance(self.rho_residual, (float, np.floating)), \
            RHO_RESIDUAL_TYPE_INTERMEDIATE_ERROR.format(type(self.rho_residual))
        assert isinstance(self.rho, np.ndarray), \
            RHO_TYPE_INTERMEDIATE_ERROR.format(type(self.rho))
        assert isinstance(self.rho_norm, (float, np.floating)), \
            RHO_NORM_TYPE_ERROR.format(type(self.rho_norm))


@dataclass
class OuterIterationInfo:
    """
    Information from a single outer SCF iteration.
    
    Attributes
    ----------
    outer_iteration : int
        Outer iteration number
    outer_rho_residual : float
        Outer loop density residual
    converged : bool
        Whether inner SCF loop converged
    iterations : int
        Number of inner SCF iterations performed
    eigen_energies : np.ndarray
        Kohn-Sham eigenvalues (orbital energies) for all states
    orbitals : np.ndarray
        Converged Kohn-Sham orbitals (radial wavefunctions R_nl(r))
    density_data : DensityData
        Converged electron density and related quantities
    full_eigen_energies : Optional[np.ndarray]
        Full eigenvalues (occupied + unoccupied) if available
    full_orbitals : Optional[np.ndarray]
        Full orbitals if available
    full_l_terms : Optional[np.ndarray]
        Angular momentum index for each entry in full_eigen_energies
    inner_iterations : List[InnerIterationInfo]
        List of inner iteration info for this outer iteration
    """
    outer_iteration     : int
    outer_rho_residual  : float
    converged           : bool
    iterations          : int
    eigen_energies      : np.ndarray
    orbitals            : np.ndarray
    density_data        : DensityData
    full_eigen_energies : Optional[np.ndarray]
    full_orbitals       : Optional[np.ndarray]
    full_l_terms        : Optional[np.ndarray]
    inner_iterations    : List[InnerIterationInfo]
    
    def __post_init__(self):
        # Type checking
        assert isinstance(self.outer_iteration, (int, np.integer)), \
            OUTER_ITERATION_TYPE_ERROR.format(type(self.outer_iteration))
        assert isinstance(self.outer_rho_residual, (float, np.floating)), \
            OUTER_RHO_RESIDUAL_TYPE_INTERMEDIATE_ERROR.format(type(self.outer_rho_residual))
        assert isinstance(self.converged, (bool, np.bool_)), \
            CONVERGED_TYPE_ERROR.format(type(self.converged))
        assert isinstance(self.iterations, (int, np.integer)), \
            ITERATIONS_TYPE_ERROR.format(type(self.iterations))
        assert isinstance(self.eigen_energies, np.ndarray), \
            EIGENVALUES_TYPE_ERROR.format(type(self.eigen_energies))
        assert isinstance(self.orbitals, np.ndarray), \
            EIGENVECTORS_TYPE_ERROR.format(type(self.orbitals))
        assert isinstance(self.density_data, DensityData), \
            DENSITY_DATA_TYPE_ERROR.format(type(self.density_data))
        if self.full_eigen_energies is not None:
            assert isinstance(self.full_eigen_energies, np.ndarray), \
                FULL_EIGENVALUES_TYPE_ERROR.format(type(self.full_eigen_energies))
        if self.full_orbitals is not None:
            assert isinstance(self.full_orbitals, np.ndarray), \
                FULL_EIGENVECTORS_TYPE_ERROR.format(type(self.full_orbitals))
        if self.full_l_terms is not None:
            assert isinstance(self.full_l_terms, np.ndarray), \
                FULL_L_TERMS_TYPE_ERROR.format(type(self.full_l_terms))
        assert isinstance(self.inner_iterations, list), \
            "parameter 'inner_iterations' must be a list, get type {} instead".format(type(self.inner_iterations))
        for inner_iter in self.inner_iterations:
            assert isinstance(inner_iter, InnerIterationInfo), \
                "all elements in inner_iterations must be InnerIterationInfo instances, get type {} instead".format(type(inner_iter))


@dataclass
class IntermediateInfo:
    """
    Stores intermediate information during SCF iterations for debugging and analysis.
    
    This class collects data from each iteration of the inner and outer SCF loops,
    including density residuals, density values, and other convergence-related metrics.
    
    Attributes
    ----------
    inner_iterations : List[InnerIterationInfo]
        List of inner iteration info objects, one per inner SCF iteration.
    outer_iterations : List[OuterIterationInfo]
        List of outer iteration info objects, one per outer SCF iteration.
    current_outer_iteration : int
        Current outer iteration number (0 if no outer loop)
    """
    inner_iterations: List[InnerIterationInfo] = field(default_factory=list)
    outer_iterations: List[OuterIterationInfo] = field(default_factory=list)
    current_outer_iteration: int = 0
    
    def add_inner_iteration(
        self,
        inner_iteration: int,
        rho_residual: float,
        rho: np.ndarray,
        rho_norm: Optional[float] = None
    ):
        """
        Add information from an inner SCF iteration.
        
        Parameters
        ----------
        inner_iteration : int
            Inner iteration number within the current outer iteration
        rho_residual : float
            Density residual at this iteration
        rho : np.ndarray
            Density array at this iteration
        rho_norm : float, optional
            L2 norm of density. If None, will be computed from rho.
        """
        # Type checking
        assert isinstance(inner_iteration, (int, np.integer)), \
            INNER_ITERATION_TYPE_ERROR.format(type(inner_iteration))
        assert isinstance(rho_residual, (float, np.floating)), \
            RHO_RESIDUAL_TYPE_INTERMEDIATE_ERROR.format(type(rho_residual))
        assert isinstance(rho, np.ndarray), \
            RHO_TYPE_INTERMEDIATE_ERROR.format(type(rho))
        if rho_norm is not None:
            assert isinstance(rho_norm, (float, np.floating)), \
                RHO_NORM_TYPE_ERROR.format(type(rho_norm))
        
        if rho_norm is None:
            rho_norm = np.linalg.norm(rho)
        
        self.inner_iterations.append(InnerIterationInfo(
            outer_iteration = self.current_outer_iteration,
            inner_iteration = inner_iteration,
            rho_residual    = rho_residual,
            rho             = rho.copy(),  # Store a copy to avoid reference issues
            rho_norm        = rho_norm,
        ))
    
    def add_outer_iteration(
        self,
        outer_iteration: int,
        outer_rho_residual: float,
        inner_result: 'SCFResult'
    ):
        """
        Add information from an outer SCF iteration.
        
        Parameters
        ----------
        outer_iteration : int
            Outer iteration number
        outer_rho_residual : float
            Outer loop density residual
        inner_result : SCFResult
            Result from the inner SCF loop, containing all relevant information
        """
        # Type checking
        assert isinstance(outer_iteration, (int, np.integer)), \
            OUTER_ITERATION_TYPE_ERROR.format(type(outer_iteration))
        assert isinstance(outer_rho_residual, (float, np.floating)), \
            OUTER_RHO_RESIDUAL_TYPE_INTERMEDIATE_ERROR.format(type(outer_rho_residual))
        assert isinstance(inner_result, SCFResult), \
            INNER_RESULT_TYPE_ERROR.format(type(inner_result))
        
        # Copy current inner iterations
        inner_iterations_copy = self.inner_iterations.copy()
        
        self.outer_iterations.append(OuterIterationInfo(
            outer_iteration     = outer_iteration,
            outer_rho_residual  = outer_rho_residual,
            converged           = inner_result.converged,
            iterations          = inner_result.iterations,
            eigen_energies      = inner_result.eigen_energies.copy(),
            orbitals            = inner_result.orbitals.copy(),
            density_data        = copy.deepcopy(inner_result.density_data),  # Deep copy to avoid reference issues
            full_eigen_energies = inner_result.full_eigen_energies.copy() if inner_result.full_eigen_energies is not None else None,
            full_orbitals       = inner_result.full_orbitals.copy()       if inner_result.full_orbitals       is not None else None,
            full_l_terms        = inner_result.full_l_terms.copy()        if inner_result.full_l_terms        is not None else None,
            inner_iterations    = inner_iterations_copy,
        ))
        
        # Clear inner_iterations for next outer iteration
        self.inner_iterations.clear()
        # Update current outer iteration counter
        self.current_outer_iteration = outer_iteration
    
    def clear(self):
        """Clear all stored intermediate information."""
        self.inner_iterations.clear()
        self.outer_iterations.clear()
        self.current_outer_iteration = 0


@dataclass
class SCFSettings:
    """
    Configuration settings for SCF calculation
    
    Attributes
    ----------
    inner_max_iter : int
        Maximum number of inner SCF iterations
    rho_tol : float
        Convergence tolerance for density residual
    n_consecutive : int
        Number of consecutive converged iterations required
    outer_max_iter : int
        Maximum number of outer SCF iterations (for HF/OEP/RPA)
    outer_rho_tol : float
        Convergence tolerance for outer loop density residual
    verbose : bool
        Whether to output the information during SCF
    """
    
    # Inner loop settings
    inner_max_iter : int   = 200
    rho_tol        : float = 1e-6
    n_consecutive  : int   = 1
    
    # Outer loop settings
    outer_max_iter : int   = 1
    outer_rho_tol  : float = 1e-5
    
    # Output settings
    verbose        : bool  = False


    def __post_init__(self):
        # type check (allow int, float and numpy types)
        assert isinstance(self.inner_max_iter, (int, np.integer)), \
            INNER_MAX_ITER_TYPE_ERROR.format(type(self.inner_max_iter))
        assert isinstance(self.rho_tol, (int, float, np.floating)), \
            RHO_TOL_TYPE_ERROR.format(type(self.rho_tol))
        assert isinstance(self.n_consecutive, (int, np.integer)), \
            N_CONSECUTIVE_TYPE_ERROR.format(type(self.n_consecutive))
        assert isinstance(self.outer_max_iter, (int, np.integer)), \
            OUTER_MAX_ITER_TYPE_ERROR.format(type(self.outer_max_iter))
        assert isinstance(self.outer_rho_tol, (int, float, np.floating)), \
            OUTER_RHO_TOL_TYPE_ERROR.format(type(self.outer_rho_tol))
        assert isinstance(self.verbose, (bool, np.bool_)), \
            VERBOSE_TYPE_ERROR.format(type(self.verbose))

    
    @classmethod
    def from_dict(cls, settings_dict: dict) -> SCFSettings:
        """
        Create SCFSettings from dictionary
        
        Parameters
        ----------
        settings_dict : dict
            Dictionary containing settings
            
        Returns
        -------
        SCFSettings
            Settings object
        """
        return cls(
            inner_max_iter = settings_dict.get('inner_max_iter' , 200),
            rho_tol        = settings_dict.get('rho_tol'        , 1e-6),
            n_consecutive  = settings_dict.get('n_consecutive'  , 1),
            outer_max_iter = settings_dict.get('outer_max_iter' , 1),
            outer_rho_tol  = settings_dict.get('outer_rho_tol'  , 1e-5),
            verbose        = settings_dict.get('verbose'        , False)
        )
    

    def to_dict(self) -> dict:
        """
        Convert to dictionary format
        
        Returns
        -------
        dict
            Dictionary representation of settings
        """
        return {
            'inner_max_iter' : self.inner_max_iter,
            'rho_tol'        : self.rho_tol,
            'n_consecutive'  : self.n_consecutive,
            'outer_max_iter' : self.outer_max_iter,
            'outer_rho_tol'  : self.outer_rho_tol,
            'verbose'        : self.verbose
        }


@dataclass
class SCFResult:
    """
    Results from SCF calculation
    
    Attributes
    ----------
    eigen_energies : np.ndarray
        Kohn-Sham eigenvalues (orbital energies) for all states, shape (n_states,)

    orbitals : np.ndarray
        Converged Kohn-Sham orbitals (radial wavefunctions R_nl(r))
        Shape: (n_states, n_quad_points)

    density_data : DensityData
        Converged electron density and related quantities (rho, grad_rho, tau)

    full_eigen_energies : np.ndarray, optional
        Optional storage of the complete KS spectrum (occupied + unoccupied).
        Shape: (n_full_states,). 

    full_orbitals : np.ndarray, optional
        Optional storage of all radial orbitals associated with full_eigen_energies.
        Shape: (n_full_states, n_quad_points). 

    full_l_terms : np.ndarray, optional
        Optional storage of the angular momentum index for each entry in full_eigen_energies.
        Shape: (n_full_states,). 

    converged : bool
        Whether inner SCF loop converged

    iterations : int
        Number of inner SCF iterations performed

    rho_residual : float
        Final density residual of inner loop (L2 norm)

    outer_iterations : int, optional
        Number of outer SCF iterations (for HF/OEP/RPA methods)

    outer_converged : bool, optional
        Whether outer SCF loop converged

    total_energy : float, optional
        Total energy of the system

    energy_components : dict, optional
        Breakdown of energy components (kinetic, Hartree, XC, etc.)
    """
    
    # Core results
    eigen_energies : np.ndarray
    orbitals       : np.ndarray
    density_data   : DensityData
    
    # Inner loop convergence info
    converged      : bool
    iterations     : int
    rho_residual   : float
    
    # Outer loop info (optional)
    full_eigen_energies : Optional[np.ndarray] = None
    full_orbitals       : Optional[np.ndarray] = None
    full_l_terms        : Optional[np.ndarray] = None
    
    outer_iterations    : Optional[int]   = None
    outer_converged     : Optional[bool]  = None
    
    # Energy info (optional)
    total_energy        : Optional[float] = None
    energy_components   : Optional[dict]  = field(default=None)
    
    # Intermediate information (optional, for debugging)
    intermediate_info   : Optional[IntermediateInfo] = None
    
    def __post_init__(self):
        # type check for required fields
        assert isinstance(self.eigen_energies, np.ndarray), \
            EIGENVALUES_TYPE_ERROR.format(type(self.eigen_energies))
        assert isinstance(self.orbitals, np.ndarray), \
            EIGENVECTORS_TYPE_ERROR.format(type(self.orbitals))
        assert isinstance(self.density_data, DensityData), \
            DENSITY_DATA_TYPE_ERROR.format(type(self.density_data))
        assert isinstance(self.converged, bool), \
            CONVERGED_TYPE_ERROR.format(type(self.converged))
        assert isinstance(self.iterations, int), \
            ITERATIONS_TYPE_ERROR.format(type(self.iterations))
        assert isinstance(self.rho_residual, (float, np.floating)), \
            RHO_RESIDUAL_TYPE_ERROR.format(type(self.rho_residual))
        
        # type check for optional fields (only if not None)
        if self.full_eigen_energies is not None:
            assert isinstance(self.full_eigen_energies, np.ndarray), \
                FULL_EIGENVALUES_TYPE_ERROR.format(type(self.full_eigen_energies))
        if self.full_orbitals is not None:
            assert isinstance(self.full_orbitals, np.ndarray), \
                FULL_EIGENVECTORS_TYPE_ERROR.format(type(self.full_orbitals))
        if self.full_l_terms is not None:
            assert isinstance(self.full_l_terms, np.ndarray), \
                FULL_L_TERMS_TYPE_ERROR.format(type(self.full_l_terms))
        if self.outer_iterations is not None:
            assert isinstance(self.outer_iterations, int), \
                OUTER_ITERATIONS_TYPE_ERROR.format(type(self.outer_iterations))
        if self.outer_converged is not None:
            assert isinstance(self.outer_converged, bool), \
                OUTER_CONVERGED_TYPE_ERROR.format(type(self.outer_converged))
        if self.total_energy is not None:
            assert isinstance(self.total_energy, (float, np.floating)), \
                TOTAL_ENERGY_TYPE_ERROR.format(type(self.total_energy))
        if self.energy_components is not None:
            assert isinstance(self.energy_components, dict), \
                ENERGY_COMPONENTS_TYPE_ERROR.format(type(self.energy_components))
        if self.intermediate_info is not None:
            assert isinstance(self.intermediate_info, IntermediateInfo), \
                "parameter 'intermediate_info' must be an IntermediateInfo instance, get type {} instead".format(type(self.intermediate_info))


    @classmethod
    def from_dict(cls, result_dict: dict) -> SCFResult:
        """
        Create SCFResult from dictionary
        
        Parameters
        ----------
        result_dict : dict
            Dictionary containing results
            
        Returns
        -------
        SCFResult
            Result object
        """
        return cls(
            eigenvalues         = result_dict['eigenvalues'],
            eigenvectors        = result_dict['eigenvectors'],
            rho                 = result_dict['rho'],
            converged           = result_dict['converged'],
            iterations          = result_dict['iterations'],
            residual            = result_dict['residual'],
            full_eigen_energies = result_dict.get('full_eigen_energies'),
            full_orbitals       = result_dict.get('full_orbitals'),
            full_l_terms        = result_dict.get('full_l_terms'),
            outer_iterations    = result_dict.get('outer_iterations'),
            outer_converged     = result_dict.get('outer_converged'),
            total_energy        = result_dict.get('total_energy'),
            energy_components   = result_dict.get('energy_components')
        )
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary format
        
        Returns
        -------
        dict
            Dictionary representation of results
        """
        result = {
            'eigenvalues' : self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'rho'         : self.rho,
            'converged'   : self.converged,
            'iterations'  : self.iterations,
            'residual'    : self.residual
        }
        
        # Add optional fields if present
        if self.full_eigen_energies is not None:
            result['full_eigen_energies'] = self.full_eigen_energies
        if self.full_orbitals is not None:
            result['full_orbitals'] = self.full_orbitals
        if self.full_l_terms is not None:
            result['full_l_terms'] = self.full_l_terms
        if self.outer_iterations is not None:
            result['outer_iterations'] = self.outer_iterations
        if self.outer_converged is not None:
            result['outer_converged'] = self.outer_converged
        if self.total_energy is not None:
            result['total_energy'] = self.total_energy
        if self.energy_components is not None:
            result['energy_components'] = self.energy_components
            
        return result
    
    def summary(self) -> str:
        """
        Get a summary string of the SCF results
        
        Returns
        -------
        str
            Summary of results
        """
        lines = [
            "=" * 60,
            "SCF Results Summary",
            "=" * 60,
            f"Inner SCF converged: {self.converged}",
            f"Inner iterations: {self.iterations}",
            f"Final residual: {self.residual:.6e}",
        ]
        
        if self.outer_iterations is not None:
            lines.extend([
                f"Outer SCF converged: {self.outer_converged}",
                f"Outer iterations: {self.outer_iterations}",
            ])
        
        if self.total_energy is not None:
            lines.append(f"Total energy: {self.total_energy:.10f} Ha")
        
        if self.energy_components is not None:
            lines.append("\nEnergy components:")
            for key, value in self.energy_components.items():
                lines.append(f"  {key}: {value:.10f} Ha")
        
        lines.extend([
            f"Number of states: {len(self.eigenvalues)}",
            f"Lowest eigenvalue: {self.eigenvalues[0]:.6f} Ha",
            "=" * 60
        ])
        
        return "\n".join(lines)

class SCFResultWithSpin(SCFResult):
    """
    SCF result for spin-polarized case
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eigenvalues_up = None
        self.eigenvalues_dn = None
        self.orbitals_up = None
        self.orbitals_dn = None
        self.density_data_up = None
        self.density_data_dn = None




class SCFDriver:
    """
    Self-consistent field driver for Kohn-Sham DFT
    
    Manages both inner and outer SCF loops:
    - Inner loop: standard KS self-consistency (rho → V → H → solve → rho')
    - Outer loop: for methods requiring orbital-dependent potentials (HF, OEP, RPA)
    """
    
    def __init__(
        self,
        hamiltonian_builder               : HamiltonianBuilder,
        density_calculator                : DensityCalculator,
        poisson_solver                    : PoissonSolver,
        eigensolver                       : EigenSolver,
        mixer                             : Mixer,
        occupation_info                   : OccupationInfo,
        xc_functional                     : str,
        spin_polarized_flag               : bool,
        use_oep                           : Optional[bool]                   = None,
        hybrid_mixing_parameter           : Optional[float]                  = None,
        ops_builder_oep                   : Optional[RadialOperatorsBuilder] = None,
        oep_mixing_parameter              : Optional[float]                  = None,
        frequency_quadrature_point_number : Optional[int]                    = None,  # parameter for RPA correlation potential
        angular_momentum_cutoff           : Optional[int]                    = None,  # parameter for RPA functional only
        enable_parallelization            : Optional[bool]                   = None,  # parameter for RPA calculations only
        ml_xc_calculator                  : Optional["MLXCCalculator"]       = None,
        ml_each_scf_step                  : Optional[bool]                   = None,
        ml_smooth_rho_for_features        : Optional[bool]                   = None,
    ):
        """
        Parameters
        ----------
        hamiltonian_builder : HamiltonianBuilder
            Constructs Hamiltonian matrices for each angular momentum channel
        density_calculator : DensityCalculator
            Computes electron density from Kohn-Sham orbitals
        poisson_solver : PoissonSolver
            Solves Poisson equation for Hartree potential
        eigensolver : EigenSolver
            Solves eigenvalue problems (H ψ = ε S ψ)
        mixer : Mixer
            Density mixing strategy for SCF convergence (linear, Pulay, etc.)
        occupation_info : OccupationInfo
            Occupation numbers and quantum numbers for atomic states
        xc_functional : str
            Name of XC functional (e.g., 'LDA_SVWN', 'LDA_PZ', 'GGA_PBE', 'SCAN')
            Used to determine what density-related quantities to compute
            Also used to initialize the XC calculator internally
        use_oep : bool, optional
            Whether to use OEP exchange potential, useful only for PBE0 functional, 
            otherwise it has to agree with use_oep_exchange flag
        hybrid_mixing_parameter : float, optional
            Mixing parameter for hybrid functionals (HF exchange fraction)
            Required only for hybrid functionals (PBE0, HF)
            - For PBE0: typically 0.25
            - For HF: 1.0
            For non-hybrid functionals, this parameter is ignored
            This parameter is designed to be autodiff-compatible for delta learning
        ops_builder_oep : RadialOperatorsBuilder, optional
            Dedicated operators builder for OEP basis/projectors. It must be provided when OEP is enabled, otherwise it will be ignored.
        oep_mixing_parameter : float, optional
            Scaling parameter (λ) applied to OEP exchange/correlation potentials.
        frequency_quadrature_point_number : int, optional
            Number of frequency quadrature points for RPA correlation potential.
        angular_momentum_cutoff : int, optional
            Maximum angular momentum quantum number to include for RPA functional.
        enable_parallelization : bool, optional
            Whether to enable parallelization for RPA calculations
            If True, the RPA calculations will be parallelized
            If False, the RPA calculations will be sequential
        ml_xc_calculator : MLXCCalculator, optional
            ML XC calculator for ML XC energy correction
            Should be the same instance as used in SCFDriver
        ml_each_scf_step : bool, optional
            Whether to apply ML XC at each SCF step
            If True, ML XC is used inside the SCF loop
            If False, ML XC is only used at final evaluation
        ml_smooth_rho_for_features : bool, optional
            If True, apply DataLoader-equivalent log-linear asymptote smoothing to ρ before
            building ML input features (default radii 7/10/13 Bohr, ``rho_floor=1e-30``).
            Typically set via ``AtomicDFTSolver``; ignored unless ``ml_xc_calculator`` is set.
        """

        self.hamiltonian_builder               = hamiltonian_builder
        self.density_calculator                = density_calculator
        self.poisson_solver                    = poisson_solver
        self.eigensolver                       = eigensolver
        self.mixer                             = mixer
        self.occupation_info                   = occupation_info
        self.xc_functional                     = xc_functional
        self.spin_polarized_flag               = spin_polarized_flag
        self.hybrid_mixing_parameter           = hybrid_mixing_parameter 
        self.use_oep                           = use_oep
        self.ops_builder_oep                   = ops_builder_oep
        self.oep_mixing_parameter              = oep_mixing_parameter 
        self.frequency_quadrature_point_number = frequency_quadrature_point_number
        self.angular_momentum_cutoff           = angular_momentum_cutoff
        self.enable_parallelization            = enable_parallelization
        self.ml_xc_calculator                  = ml_xc_calculator
        self.ml_each_scf_step                  = ml_each_scf_step
        if ml_smooth_rho_for_features is None:
            ml_smooth_rho_for_features = False
        self.ml_smooth_rho_for_features        = ml_smooth_rho_for_features
        self._check_initialization()


        # Create SwitchesFlags instance (handles validation internally)
        self.switches = SwitchesFlags(
            xc_functional           = xc_functional,
            use_oep                 = use_oep,
            hybrid_mixing_parameter = hybrid_mixing_parameter
        )
        
        # Get functional requirements (what to compute for this functional)
        self.xc_requirements : FunctionalRequirements = get_functional_requirements(xc_functional)
        
        # Initialize XC calculator internally based on functional
        self.xc_calculator : Optional[XCEvaluator] = self._initialize_xc_calculator(
            derivative_matrix = density_calculator.derivative_matrix,
            r_quad            = density_calculator.quadrature_nodes
        )
        
        # Initialize HF exchange calculator for hybrid functionals
        self.hf_calculator : Optional[HartreeFockExchange] = self._initialize_hf_calculator()

        # Initialize OEP calculator for OEP calculations
        self.oep_calculator : Optional[OEPCalculator] = self._initialize_oep_calculator()

        # Initialize response calculator for dielectric matrix computation
        self.response_calculator : Optional[ResponseCalculator] = self._initialize_response_calculator()

        # Extract NLCC density from pseudopotential (if using pseudopotential)
        self.rho_nlcc : Optional[np.ndarray] = self._initialize_nlcc_density()
        
        # Initialize convergence checkers (will be configured in run method)
        self.inner_convergence_checker : Optional[ConvergenceChecker] = None
        self.outer_convergence_checker : Optional[ConvergenceChecker] = None

    
    def _check_initialization(self):
        """
        Check the initialization of the SCFDriver for required parameters.
        """
        assert isinstance(self.hamiltonian_builder, HamiltonianBuilder), \
            HAMILTONIAN_BUILDER_TYPE_ERROR.format(type(self.hamiltonian_builder))
        assert isinstance(self.density_calculator, DensityCalculator), \
            DENSITY_CALCULATOR_TYPE_ERROR.format(type(self.density_calculator))
        assert isinstance(self.poisson_solver, PoissonSolver), \
            POISSON_SOLVER_TYPE_ERROR.format(type(self.poisson_solver))
        assert isinstance(self.eigensolver, EigenSolver), \
            EIGENSOLVER_TYPE_ERROR.format(type(self.eigensolver))
        assert isinstance(self.mixer, Mixer), \
            MIXER_TYPE_ERROR.format(type(self.mixer))
        assert isinstance(self.occupation_info, OccupationInfo), \
            OCCUPATION_INFO_TYPE_ERROR.format(type(self.occupation_info))
        assert isinstance(self.xc_functional, str), \
            XC_FUNCTIONAL_TYPE_ERROR.format(type(self.xc_functional))
        assert isinstance(self.spin_polarized_flag, bool), \
            SPIN_POLARIZED_FLAG_NOT_BOOL_ERROR.format(type(self.spin_polarized_flag))
        if self.hybrid_mixing_parameter is not None:
            assert isinstance(self.hybrid_mixing_parameter, (float, np.floating)), \
                HYBRID_MIXING_PARAMETER_TYPE_ERROR.format(type(self.hybrid_mixing_parameter))
        if self.use_oep is not None:
            assert isinstance(self.use_oep, bool), \
                USE_OEP_NOT_BOOL_ERROR.format(type(self.use_oep))
        if self.ops_builder_oep is not None:
            assert isinstance(self.ops_builder_oep, RadialOperatorsBuilder), \
                OPS_BUILDER_OEP_TYPE_ERROR.format(type(self.ops_builder_oep))
        if self.oep_mixing_parameter is not None:
            assert isinstance(self.oep_mixing_parameter, (float, np.floating)), \
                OEP_MIXING_PARAMETER_TYPE_ERROR.format(type(self.oep_mixing_parameter))
        if self.frequency_quadrature_point_number is not None:
            assert isinstance(self.frequency_quadrature_point_number, int), \
                FREQUENCY_QUADRATURE_POINT_NUMBER_TYPE_ERROR.format(type(self.frequency_quadrature_point_number))
        if self.angular_momentum_cutoff is not None:
            assert isinstance(self.angular_momentum_cutoff, int), \
                ANGULAR_MOMENTUM_CUTOFF_TYPE_ERROR.format(type(self.angular_momentum_cutoff))
        if self.enable_parallelization is not None:
            assert isinstance(self.enable_parallelization, bool), \
                ENABLE_PARALLELIZATION_NOT_BOOL_ERROR.format(type(self.enable_parallelization))
        
        if self.ml_xc_calculator is not None:
            from ..xc.ml_xc import MLXCCalculator
            assert isinstance(self.ml_xc_calculator, MLXCCalculator), \
                ML_XC_CALCULATOR_TYPE_ERROR.format(type(self.ml_xc_calculator))
            assert isinstance(self.ml_each_scf_step, bool), \
                ML_EACH_SCF_STEP_NOT_BOOL_ERROR.format(type(self.ml_each_scf_step))
            assert isinstance(self.ml_smooth_rho_for_features, bool), \
                ML_SMOOTH_RHO_FOR_FEATURES_NOT_BOOL_ERROR.format(type(self.ml_smooth_rho_for_features))


    def _initialize_xc_calculator(
        self, 
        derivative_matrix : np.ndarray,
        r_quad            : np.ndarray
    ) -> Optional[XCEvaluator]:
        """
        Initialize XC calculator based on the functional name.
        
        For functional 'None' (pure kinetic energy), no XC calculator is needed.
        For all other functionals, create the appropriate evaluator instance.
        
        Parameters
        ----------
        derivative_matrix : np.ndarray
            Finite element derivative matrix (from DensityCalculator)
            Required for GGA and meta-GGA to transform gradients to spherical form
        r_quad : np.ndarray
            Radial quadrature nodes (coordinates)
            Required for spherical coordinate transformations
        
        Returns
        -------
        xc_calculator : XCEvaluator or None
            Specific XC functional evaluator (e.g., LDA_SVWN, LDA_PZ, GGA_PBE).
            Returns None if xc_functional is 'None'.
        
        Raises
        ------
        ValueError
            If the specified functional is not implemented
        """
        if self.switches.use_xc_functional:
            return create_xc_evaluator(
                functional_name   = self.xc_functional,
                derivative_matrix = derivative_matrix,
                r_quad            = r_quad
            )
        
    
    def _initialize_hf_calculator(self) -> Optional[HartreeFockExchange]:
        """
        Initialize Hartree-Fock exchange calculator for hybrid functionals.
        
        Returns
        -------
        Optional[HartreeFockExchange]
            HF exchange calculator if functional requires it, None otherwise
            Now using poisson_solver.ops_builder, which has denser grids for the HF exchange calculation
        """
        # Only create HF calculator for hybrid functionals
        if not self.switches.use_hf_exchange:
            return None
        
        # Create HF exchange calculator with ops_builder and occupation_info
        hf_calculator = HartreeFockExchange(
            ops_builder       = self.hamiltonian_builder.ops_builder,
            ops_builder_dense = self.poisson_solver.ops_builder,
            occupation_info   = self.occupation_info
        )
        
        return hf_calculator
    
    
    def _initialize_oep_calculator(self) -> Optional[OEPCalculator]:
        """
        Initialize OEP calculator for OEP calculations.
        
        Returns
        -------
        Optional[OEPCalculator]
            OEP calculator if functional requires it, None otherwise
        """
        # Only create OEP calculator for OEP calculations
        if not self.switches.use_oep:
            return None
        else:
            assert self.ops_builder_oep is not None, \
                "ops_builder_oep must be provided when OEP is enabled"
        
        # Create OEP calculator with ops_builder and occupation_info
        oep_calculator = OEPCalculator(
            ops_builder                       = self.hamiltonian_builder.ops_builder,
            ops_builder_dense                 = self.poisson_solver.ops_builder,
            ops_builder_oep                   = self.ops_builder_oep,
            occupation_info                   = self.occupation_info,
            use_rpa_correlation               = self.switches.use_oep_correlation,
            frequency_quadrature_point_number = self.frequency_quadrature_point_number,
            angular_momentum_cutoff           = self.angular_momentum_cutoff,
        )

        return oep_calculator


    def _initialize_response_calculator(self) -> ResponseCalculator:
        """
        Initialize response calculator for dielectric matrix computation.
        """
        if not self.mixer.use_preconditioner:
            return None
        
        return ResponseCalculator(
            occupation_info = self.occupation_info,
            ops_builder     = self.hamiltonian_builder.ops_builder,
        )


    def _initialize_nlcc_density(self) -> np.ndarray:
        """
        Initialize non-linear core correction (NLCC) density from pseudopotential.
        
        NLCC is used to improve the accuracy of exchange-correlation energy
        in pseudopotential calculations by including core electron effects.
        
        Returns
        -------
        rho_nlcc : np.ndarray
            Core correction density at quadrature points.
            Returns zeros for all-electron calculations.
        """
        # Get quadrature nodes from hamiltonian builder
        quadrature_nodes = self.hamiltonian_builder.ops_builder.quadrature_nodes
        
        # Check if using pseudopotential or all-electron
        if self.hamiltonian_builder.all_electron:
            # No NLCC for all-electron calculations
            return np.zeros_like(quadrature_nodes)
        else:
            # Extract NLCC density from pseudopotential
            pseudo = self.hamiltonian_builder.pseudo
            rho_nlcc = pseudo.get_rho_core_correction(quadrature_nodes)
            return rho_nlcc
    

    def get_total_density_for_xc(self, rho_valence: np.ndarray) -> np.ndarray:
        """
        Get total density for exchange-correlation calculations.
        
        For pseudopotential calculations, XC functionals should use:
            rho_total = rho_valence + rho_nlcc
        
        For all-electron calculations:
            rho_total = rho_valence
        
        Parameters
        ----------
        rho_valence : np.ndarray
            Valence electron density from KS orbitals
        
        Returns
        -------
        rho_total : np.ndarray
            Total density including NLCC correction
        """
        return rho_valence + self.rho_nlcc
    
    
    def _get_zero_hf_exchange_matrices_dict(self) -> Dict[int, np.ndarray]:
        """
        Create zero HF exchange matrices dictionary for all l channels.
        
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping l values to zero matrices
            Keys are unique l values from occupation_info
            Values are zero matrices of shape (n_physical_nodes, n_physical_nodes)
        """
        zero_matrices_dict = {}
        
        # Get matrix size from kinetic energy matrix
        H_kinetic = self.hamiltonian_builder.H_kinetic
        matrix_size = H_kinetic.shape[0]
        
        # Create zero matrices for all unique l values
        for l in self.occupation_info.unique_l_values:
            zero_matrices_dict[l] = np.zeros((matrix_size, matrix_size))
        
        return zero_matrices_dict
    

    def _compute_hf_exchange_matrices_dict(self, orbitals: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute Hartree-Fock exchange matrices for all l channels.
        
        This method delegates the calculation to the hf_calculator.
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping l values to HF exchange matrices
        """
        if self.hf_calculator is None:
            # Return zero matrices for all l channels
            print(HF_CALCULATOR_NOT_AVAILABLE_WARNING)
            return self._get_zero_hf_exchange_matrices_dict()
        
        # Delegate to hf_calculator
        return self.hf_calculator.compute_exchange_matrices_dict(orbitals)
    
    def run(
        self,
        rho_initial        : np.ndarray,
        settings           : Union[SCFSettings, Dict[str, Any]],
        orbitals_initial   : Optional[np.ndarray | Tuple[np.ndarray, np.ndarray]] = None,
        save_intermediate  : bool = False,
        save_full_spectrum : bool = False,
    ) -> SCFResult:
        """
        Run SCF calculation
        
        Parameters
        ----------
        rho_initial : np.ndarray
            Initial density guess
        settings : SCFSettings or dict
            SCF settings (max iterations, tolerances, etc.)
            Can be a SCFSettings object or a dictionary
        orbitals_initial : np.ndarray | Tuple[np.ndarray, np.ndarray], optional
            Initial orbitals guess. 
            If not spin-polarized, it should be a single np.ndarray.
            If spin-polarized, it should be a tuple of two np.ndarray, one for spin-up and one for spin-down.
        save_intermediate : bool, optional
            If True, save intermediate information from each iteration.
            Default is False.
        save_full_spectrum : bool, optional
            If True, save full spectrum information from each iteration.
            Default is False.
        
        Returns
        -------
        result : SCFResult
            SCF solution including eigenvalues, eigenvectors, density, energy
        """
        assert isinstance(rho_initial, np.ndarray), \
            RHO_INITIAL_TYPE_ERROR.format(type(rho_initial))
        
        # Convert dict to SCFSettings if needed
        if isinstance(settings, dict):
            settings = SCFSettings.from_dict(settings)
        elif not isinstance(settings, SCFSettings):
            raise TypeError(SETTINGS_TYPE_ERROR.format(type(settings)))

        # Initialize intermediate info if requested
        intermediate_info = IntermediateInfo() if save_intermediate else None

        # Determine if outer loop is needed
        needs_outer_loop = (settings.outer_max_iter > 1)

        # Configure convergence checkers
        self.inner_convergence_checker = ConvergenceChecker(
            tolerance     = settings.rho_tol,
            n_consecutive = settings.n_consecutive,
            loop_type     = "Inner"
        )
        
        # Initialize outer convergence checker if outer loop is needed
        if needs_outer_loop:
            self.outer_convergence_checker = ConvergenceChecker(
                tolerance         = settings.outer_rho_tol,
                n_consecutive     = settings.n_consecutive,  # Outer loop typically converges in 1 iteration
                loop_type         = "Outer",
                footer_blank_line = True,
            )
        else:
            self.inner_convergence_checker.set_footer_blank_line(True)

        
        if hasattr(settings, 'verbose') and settings.verbose:
            print("===========================================================================")
            print("                          Self-Consistent Field                            ")
            print("===========================================================================")

        if not self.spin_polarized_flag:
            if needs_outer_loop:
                return self._outer_loop(rho_initial, settings, orbitals_initial, intermediate_info = intermediate_info, save_full_spectrum = save_full_spectrum)
            else:
                return self._inner_loop(rho_initial, settings, orbitals_initial, intermediate_info = intermediate_info, save_full_spectrum = save_full_spectrum)
        else:
            if needs_outer_loop:
                return self._outer_loop_spin_polarized(rho_initial, settings, orbitals_initial, intermediate_info = intermediate_info, save_full_spectrum = save_full_spectrum)
            else:
                return self._inner_loop_spin_polarized(rho_initial, settings, orbitals_initial, intermediate_info = intermediate_info, save_full_spectrum = save_full_spectrum)
    
    def _reorder_eigenstates_by_occupation(
        self, 
        eigenvalues_all : List[np.ndarray], 
        eigenvectors_all: List[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reorder eigenvalues and eigenvectors from l-channel lists to match occupation order.
        
        When solving each l-channel separately, results are grouped by l:
            [all l=0 states] [all l=1 states] [all l=2 states] ...
        
        But occupation list may have interleaved l values:
            [1s(l=0), 2s(l=0), 2p(l=1), 3s(l=0), 3p(l=1)]
        
        This function reorders the results to match the occupation list order.
        
        Parameters
        ----------
        eigenvalues_all : List[np.ndarray]
            List of eigenvalue arrays, one per unique l value
            eigenvalues_all[i] has shape (n_states_for_l[i],)
        eigenvectors_all : List[np.ndarray]
            List of eigenvector arrays, one per unique l value
            eigenvectors_all[i] has shape (n_grid_points, n_states_for_l[i])
        
        Returns
        -------
        eigenvalues : np.ndarray
            Reordered eigenvalues in occupation list order, shape (n_total_states,)
        eigenvectors : np.ndarray
            Reordered eigenvectors in occupation list order, shape (n_grid_points, n_total_states)
        
        Examples
        --------
        For Al (Z=13): occupation = [1s(l=0), 2s(l=0), 2p(l=1), 3s(l=0), 3p(l=1)]
        - Input: eigenvalues_all = [eigvals_l0, eigvals_l1] with l0=[1s,2s,3s], l1=[2p,3p]
        - Output: eigenvalues = [1s, 2s, 2p, 3s, 3p] (correctly interleaved)
        """
        # Preallocate output arrays
        n_total_states = len(self.occupation_info.occupations)
        n_grid_points = eigenvectors_all[0].shape[0]
        eigenvalues = np.zeros(n_total_states)
        eigenvectors = np.zeros((n_grid_points, n_total_states))
        
        # Fill arrays according to occupation list order
        for i_l, l in enumerate(self.occupation_info.unique_l_values):
            # Find all indices in occupation list where l matches
            sort_index = np.where(self.occupation_info.l_values == l)[0]
            n_states = len(sort_index)
            
            # Place eigenstates at correct positions
            eigenvalues[sort_index] = eigenvalues_all[i_l][:n_states]
            eigenvectors[:, sort_index] = eigenvectors_all[i_l][:, :n_states]
        
        return eigenvalues, eigenvectors


    def _inner_loop(
        self,
        rho_initial             : np.ndarray,
        settings                : SCFSettings,
        orbitals_initial        : Optional[np.ndarray]            = None,
        v_x_oep                 : Optional[np.ndarray]            = None,
        v_c_oep                 : Optional[np.ndarray]            = None,
        H_hf_exchange_dict_by_l : Optional[Dict[int, np.ndarray]] = None,
        intermediate_info       : Optional[IntermediateInfo]      = None,
        symmetrize              : bool                            = False,
        save_full_spectrum      : bool                            = False,
    ) -> SCFResult:
        r"""
        Inner SCF loop for spin-polarized Kohn-Sham self-consistency.

        Fixed: external potential and channel-specific HF exchange (if any).
        Iterate per channel: (rho_up, rho_dn) -> V_H, V_xc(up/dn) -> H_up/H_dn
        -> solve -> orbitals_up/orbitals_dn -> (rho_up_new, rho_dn_new).
        
        Parameters
        ----------
        rho_initial : np.ndarray
            Initial density guess
        settings : SCFSettings
            SCF settings
        orbitals_initial : np.ndarray, optional
            Initial orbitals guess for debugging
            Shape: (n_grid, n_orbitals)
            If provided, will be used as initial orbitals instead of solving eigenvalue problem
        v_x_oep : np.ndarray, optional
            OEP exchange potential
        v_c_oep : np.ndarray, optional
            OEP correlation potential
        H_hf_exchange_dict_by_l : dict, optional
            Hartree-Fock exchange matrices dictionary (from outer loop)
        intermediate_info : IntermediateInfo, optional
            Intermediate information for debugging and analysis
        symmetrize : bool, optional
            Whether to symmetrize the eigenvectors (default: False)
            - If False (default), the eigenvectors are solved using the generalized eigenvalue problem: Hx = λSx.
            - If True, the eigenvectors are symmetrized using the overlap matrix S: S^(-1/2) @ H @ S^(-1/2) @ S^(1/2) x = λS^(1/2)x.
                - Symmetrize Hamiltonian: H → S^(-1/2) @ H @ S^(-1/2)
                - Solve the eigenvalue problem: Hy = λy.
                - Symmetrized eigenvectors: x = S^(-1/2) @ y.
        save_full_spectrum : bool, optional
            If True, save full spectrum information from each iteration.
            Default is False.
        
        Returns
        -------
        result : SCFResult
            Converged SCF state
        """

        # type check for required fields
        assert isinstance(rho_initial, np.ndarray), \
            RHO_INITIAL_TYPE_ERROR.format(type(rho_initial))
        assert isinstance(settings, SCFSettings), \
            SETTINGS_TYPE_ERROR.format(type(settings))
        if orbitals_initial is not None:
            assert isinstance(orbitals_initial, np.ndarray), \
            ORBITALS_INITIAL_TYPE_ERROR.format(type(orbitals_initial))
        if v_x_oep is not None:
            assert isinstance(v_x_oep, np.ndarray), \
            V_X_OEP_TYPE_ERROR.format(type(v_x_oep))
        if v_c_oep is not None:
            assert isinstance(v_c_oep, np.ndarray), \
            V_C_OEP_TYPE_ERROR.format(type(v_c_oep))
        if H_hf_exchange_dict_by_l is not None:
            assert isinstance(H_hf_exchange_dict_by_l, dict), \
            H_HF_EXCHANGE_DICT_BY_L_TYPE_ERROR.format(type(H_hf_exchange_dict_by_l))
        assert isinstance(symmetrize, bool), \
            SYMMETRIZE_TYPE_ERROR.format(type(symmetrize))
        assert isinstance(save_full_spectrum, bool), \
            SAVE_FULL_SPECTRUM_TYPE_ERROR.format(type(save_full_spectrum))

        
        # initialize variables
        max_iter = settings.inner_max_iter
        verbose  = settings.verbose

        # rho = rho_initial.copy()
        rho = self.density_calculator.normalize_density(rho_initial.copy())
        density_data = self.density_calculator.create_density_data_from_mixed(
            rho_mixed        = rho,
            orbitals         = orbitals_initial,
            compute_gradient = self.xc_requirements.needs_gradient,
            compute_tau      = self.xc_requirements.needs_tau and (orbitals_initial is not None),
            rho_nlcc         = self.rho_nlcc
        )

        # Reset mixer and convergence checker
        self.mixer.reset()
        self.inner_convergence_checker.reset()

        # Set HF exchange if provided
        if H_hf_exchange_dict_by_l is not None:
            self.hamiltonian_builder.set_hf_exchange_matrices(H_hf_exchange_dict_by_l)
        
        # Print convergence table header
        if verbose:
            self.inner_convergence_checker.print_header(prefix="")

        # Main inner SCF loop
        for iteration in range(max_iter):
            
            # ===== Step 1: Compute potentials =====
            # Hartree potential
            if self.switches.use_hartree:
                v_hartree = self.poisson_solver.solve_hartree(rho)
            else:
                v_hartree = np.zeros_like(rho)

            # XC potential
            # For pseudopotentials: use rho_total = rho_valence + rho_nlcc
            # For all-electron: rho_nlcc is zero, so rho_total = rho_valence
            v_x = np.zeros_like(rho)  # Default: no exchange potential
            v_c = np.zeros_like(rho)  # Default: no correlation potential
            de_xc_dtau = None         # Default: no meta-GGA derivative term

            # Original XC calculation
            if self.xc_calculator is not None:
                # Compute XC using new interface: DensityData → XCPotentialData
                xc_potential_data = self.xc_calculator.compute_xc(density_data)
                v_x = xc_potential_data.v_x
                v_c = xc_potential_data.v_c
                de_xc_dtau = xc_potential_data.de_xc_dtau
            
            # Correction from ML model
            if self.ml_xc_calculator is not None and self.ml_each_scf_step:
                features = self._construct_input_features_for_ml_model(
                    features_list = self.ml_xc_calculator.features_list,
                    density_data  = density_data
                )

                v_xc_ml = self.ml_xc_calculator.predict_vxc(
                    features          = features,
                    derivative_matrix = self.hamiltonian_builder.ops_builder.derivative_matrix,
                    quadrature_nodes  = self.hamiltonian_builder.ops_builder.quadrature_nodes,
                )
            else:
                v_xc_ml = None
            
            # ===== Step 2: Build and solve for each l channel =====
            occ_eigenvalues_list    : List[np.ndarray] = []
            occ_eigenvectors_list   : List[np.ndarray] = []
            unocc_eigenvalues_list  : List[np.ndarray] = []  # needed only for OEP
            unocc_eigenvectors_list : List[np.ndarray] = []  # needed only for OEP

            # Determine which l values to iterate over
            # If angular_momentum_cutoff is set (e.g., for RPA), iterate over all l values from 0 to cutoff
            # Otherwise, only iterate over occupied l values
            if self.angular_momentum_cutoff is not None:
                unique_l_values = list(range(self.angular_momentum_cutoff + 1))
            else:
                unique_l_values = self.occupation_info.unique_l_values

            for l in unique_l_values:

                # Build Hamiltonian for this l
                H_l = self.hamiltonian_builder.build_for_l_channel(
                    l                = l,
                    v_hartree        = v_hartree,
                    v_x              = v_x,
                    v_c              = v_c,
                    switches         = self.switches,
                    v_x_oep          = v_x_oep,
                    v_c_oep          = v_c_oep,
                    v_xc_ml          = v_xc_ml,
                    de_xc_dtau       = de_xc_dtau,
                    symmetrize       = symmetrize,
                    exclude_boundary = True,
                )
                
                # Number of states to solve for this l
                n_states = self.occupation_info.n_states_for_l(l)
                
                if self.switches.use_oep or self.mixer.use_preconditioner:
                    if symmetrize:
                        # solve the eigenvalue problem Hx = λx
                        full_eigvals_l, full_eigvecs_l = self.eigensolver.solve_full(H_l)
                    else:
                        # solve the generalized eigenvalue problem Hx = λSx
                        S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                        full_eigvals_l, full_eigvecs_l = self.eigensolver.solve_generalized_full(H_l, S)

                    occ_eigvals_l = full_eigvals_l[:n_states]
                    occ_eigvecs_l = full_eigvecs_l[:, :n_states]

                    # Store unoccupied eigenvalues and eigenvectors, used only for OEP and at final readout
                    #     or preconditioning, where we need to compute dielectric matrix as preconditioner for Pulay mixing.
                    unocc_eigenvalues_list.append(full_eigvals_l[n_states:])
                    unocc_eigenvectors_list.append(full_eigvecs_l[:, n_states:])
                else:
                    # Solve eigenvalue problem
                    if symmetrize:
                        # solve the eigenvalue problem Hx = λx
                        occ_eigvals_l, occ_eigvecs_l = self.eigensolver.solve_lowest(H_l, n_states)
                    else:
                        # solve the generalized eigenvalue problem Hx = λSx
                        S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                        occ_eigvals_l, occ_eigvecs_l = self.eigensolver.solve_generalized_lowest(H_l, S, n_states)                

                # Store occupied eigenvalues and eigenvectors
                occ_eigenvalues_list.append(occ_eigvals_l)
                occ_eigenvectors_list.append(occ_eigvecs_l)

            # Reorder eigenstates to match occupation list order
            occ_eigenvalues, occ_eigenvectors = self._reorder_eigenstates_by_occupation(
                occ_eigenvalues_list, occ_eigenvectors_list
            )
            
            # Interpolate eigenvectors to quadrature points, also symmetrize the eigenvectors
            occ_orbitals = self.hamiltonian_builder.interpolate_eigenvectors_to_quadrature(
                eigenvectors = occ_eigenvectors,
                symmetrize   = symmetrize,
                pad_width    = 1,
            )
            
            # ===== Step 3: Compute new density =====
            # Compute new density from orbitals
            rho_new = self.density_calculator.compute_density(occ_orbitals, normalize=True)

            # ===== Step 4: Check convergence =====
            converged, residual = self.inner_convergence_checker.check(
                rho, rho_new, iteration + 1, 
                print_status = verbose, prefix = ""
            )
            
            # Save intermediate information if requested
            if intermediate_info is not None:
                intermediate_info.add_inner_iteration(
                    inner_iteration = iteration + 1,
                    rho_residual    = residual,
                    rho             = rho_new,
                )
            
            if converged:
                break
            
            # ===== Step 5: Mix densities and update density_data =====
            if self.mixer.use_preconditioner:
                # compute dielectric matrix for preconditioning
                full_eigenvalues, full_orbitals, full_l_terms = \
                    self._construct_full_eigenvalues_and_orbitals_and_l_terms(
                        occ_eigenvalues_list    = occ_eigenvalues_list,
                        occ_eigenvectors_list   = occ_eigenvectors_list,
                        unocc_eigenvalues_list  = unocc_eigenvalues_list,
                        unocc_eigenvectors_list = unocc_eigenvectors_list,
                        symmetrize              = False,
                        angular_momentum_cutoff = self.angular_momentum_cutoff,
                    )
                dielectric_matrix = self.response_calculator.compute_dielectric_matrix(
                    full_eigenvalues,
                    full_orbitals,
                    full_l_terms
                )
                preconditioner = dielectric_matrix
            else:
                preconditioner = None

            rho = self.mixer.mix(rho, rho_new, preconditioner)
            
            # Update density_data for next iteration using mixed density
            density_data = self.density_calculator.create_density_data_from_mixed(
                rho_mixed        = rho,
                orbitals         = occ_orbitals,
                compute_gradient = self.xc_requirements.needs_gradient,
                compute_tau      = self.xc_requirements.needs_tau,
                rho_nlcc         = self.rho_nlcc
            )
        
        # Print convergence footer
        if verbose:
            self.inner_convergence_checker.print_footer(converged, iteration + 1, prefix="")
        
        if not converged:
            print(INNER_SCF_DID_NOT_CONVERGE_WARNING.format(max_iter))


        # Update properties with or without MLXC
        occ_eigenvalues, occ_orbitals, v_xc_ml = self._update_properties(
            rho = rho,
            v_x_oep = v_x_oep,
            v_c_oep = v_c_oep,
            density_data = density_data,
            symmetrize   = symmetrize,
        )


        # Create final density_data from converged orbitals, do not include NLCC
        final_density_data : DensityData = \
            self.density_calculator.create_density_data_from_orbitals(
                orbitals         = occ_orbitals,
                compute_gradient = self.xc_requirements.needs_gradient,
                compute_tau      = self.xc_requirements.needs_tau,
                rho_nlcc         = None
            )
        

        # Construct full eigenvalues/orbitals/l-terms at the *final SCF state*.
        # This is a single-shot full-spectrum diagonalization at the converged density.
        if self.switches.use_oep or save_full_spectrum:
            full_eigen_energies, full_orbitals, full_l_terms = \
                self._compute_full_orbitals_and_eigenvalues(
                    rho             = final_density_data.rho,
                    orbitals        = occ_orbitals,
                    v_x_oep         = v_x_oep,
                    v_c_oep         = v_c_oep,
                    v_xc_ml         = v_xc_ml,
                    switches        = self.switches,
                    xc_requirements = self.xc_requirements,
                    xc_calculator   = self.xc_calculator,
                    symmetrize      = symmetrize,
                )
        else:
            full_eigen_energies = None
            full_orbitals       = None
            full_l_terms        = None
        

        result = SCFResult(
            eigen_energies      = occ_eigenvalues,
            orbitals            = occ_orbitals,
            density_data        = final_density_data,
            converged           = converged,
            iterations          = iteration + 1,
            rho_residual        = residual,
            full_eigen_energies = full_eigen_energies,
            full_orbitals       = full_orbitals,
            full_l_terms        = full_l_terms,
            intermediate_info   = intermediate_info,
        )

        return result


    def _outer_loop(
        self,
        rho_initial        : np.ndarray,
        settings           : SCFSettings,
        orbitals_initial   : Optional[np.ndarray] = None,
        intermediate_info  : Optional[IntermediateInfo] = None,
        save_full_spectrum : bool = False,
    ) -> SCFResult:
        """
        Outer SCF loop: for orbital-dependent functionals
        
        Used for:
        - Hartree-Fock exchange (requires orbitals)
        - OEP methods (requires full spectrum)
        - RPA correlation (requires response functions)
        
        Parameters
        ----------
        rho_initial : np.ndarray
            Initial density
        settings : SCFSettings
            SCF settings
        orbitals_initial : np.ndarray, optional
            Initial orbitals guess for debugging
            Shape: (n_grid, n_orbitals)
            If provided, will be used as initial orbitals instead of solving eigenvalue problem
        intermediate_info : IntermediateInfo, optional
            Intermediate information for debugging and analysis
        save_full_spectrum : bool, optional
            If True, save full spectrum information from each iteration.
            Default is False.
        
        Returns
        -------
        result : SCFResult
            Converged SCF state from outer loop
        """
        # type check for required fields
        assert isinstance(rho_initial, np.ndarray), \
            RHO_INITIAL_TYPE_ERROR.format(type(rho_initial))
        assert isinstance(settings, SCFSettings), \
            SETTINGS_TYPE_ERROR.format(type(settings))
        if orbitals_initial is not None:
            assert isinstance(orbitals_initial, np.ndarray), \
            ORBITALS_INITIAL_TYPE_ERROR.format(type(orbitals_initial))
        assert isinstance(save_full_spectrum, bool), \
            SAVE_FULL_SPECTRUM_TYPE_ERROR.format(type(save_full_spectrum))

        # initialize variables
        max_outer_iter = settings.outer_max_iter
        verbose        = settings.verbose
        
        rho            = rho_initial.copy()
        orbitals       = orbitals_initial

        # Compute HF exchange matrices from initial orbitals if needed
        if self.switches.use_hf_exchange and (orbitals_initial is not None):
            H_hf_exchange_dict_by_l = self._compute_hf_exchange_matrices_dict(orbitals_initial)
        else:
            H_hf_exchange_dict_by_l = self._get_zero_hf_exchange_matrices_dict()

        # Reset outer convergence checker
        self.outer_convergence_checker.reset()
        
        # Initialize intermediate_info current_outer_iteration if needed
        if intermediate_info is not None:
            intermediate_info.current_outer_iteration = 0

        # Compute default full orbitals and eigenvalues if use_oep is True
        if self.switches.use_oep:
            full_eigen_energies, full_orbitals, full_l_terms = \
                self._compute_default_full_orbitals_and_eigenvalues(
                    rho           = rho_initial,
                    orbitals      = orbitals_initial,
                    xc_functional = 'GGA_PBE'
                )

        # Track outer convergence status
        outer_converged = False
        
        for outer_iter in range(max_outer_iter):
            # Update current outer iteration in intermediate_info
            if intermediate_info is not None:
                intermediate_info.current_outer_iteration = outer_iter + 1
            
            if verbose:
                print(f"Outer iteration {outer_iter + 1}")

            # Compute OEP potentials from initial orbitals if use_oep is True
            if self.switches.use_oep:
                v_x_oep, v_c_oep = self.oep_calculator.compute_oep_potentials(
                    full_eigen_energies = full_eigen_energies,
                    full_orbitals       = full_orbitals,
                    full_l_terms        = full_l_terms,
                    enable_parallelization = self.enable_parallelization,
                )
            else:
                v_x_oep = None
                v_c_oep = None
            
            # Run inner SCF with fixed HF exchange
            inner_result : SCFResult = self._inner_loop(
                rho_initial             = rho,
                settings                = settings,
                H_hf_exchange_dict_by_l = H_hf_exchange_dict_by_l,
                v_x_oep                 = v_x_oep,
                v_c_oep                 = v_c_oep,
                intermediate_info       = intermediate_info,
                save_full_spectrum      = save_full_spectrum,
            )
            
            # update rho and orbitals
            rho_new  = inner_result.density_data.rho
            orbitals = inner_result.orbitals

            # update HF exchange dictionary
            if self.switches.use_hf_exchange:
                H_hf_exchange_dict_by_l = self._compute_hf_exchange_matrices_dict(orbitals) 

            # Check outer loop convergence
            outer_converged, outer_residual = self.outer_convergence_checker.check(
                rho, rho_new, outer_iter + 1,
                print_status = verbose
            )
            
            # Save intermediate information if requested
            if intermediate_info is not None:
                intermediate_info.add_outer_iteration(
                    outer_iteration    = outer_iter + 1,
                    outer_rho_residual = outer_residual,
                    inner_result       = inner_result,
                )
            
            if outer_converged:
                break
            
            # Update for next outer iteration
            rho = rho_new
            if self.switches.use_oep:
                # if OEP is used, also update full orbitals and eigenvalues from inner result
                full_eigen_energies = inner_result.full_eigen_energies
                full_orbitals       = inner_result.full_orbitals
                full_l_terms        = inner_result.full_l_terms
        

        # Update outer loop info in result
        outer_iterations = outer_iter + 1
        # outer_converged is already set correctly from the convergence check above
        
        # Print outer loop footer if verbose is True
        if verbose:
            self.outer_convergence_checker.print_footer(outer_converged, outer_iterations)
        
        outer_result = SCFResult(
            eigen_energies      = inner_result.eigen_energies,
            orbitals            = inner_result.orbitals,
            density_data        = inner_result.density_data,
            converged           = outer_converged,
            iterations          = outer_iterations,
            rho_residual        = outer_residual,
            full_eigen_energies = inner_result.full_eigen_energies,
            full_orbitals       = inner_result.full_orbitals,
            full_l_terms        = inner_result.full_l_terms,
            intermediate_info   = intermediate_info,
        )

        return outer_result



    def _inner_loop_spin_polarized(
        self,
        rho_initial             : np.ndarray,
        settings                : SCFSettings,
        orbitals_initial        : Optional[Tuple[np.ndarray, np.ndarray]] = None, # (orbitals_up, orbitals_dn)
        v_x_oep                 : Optional[Tuple[np.ndarray, np.ndarray]] = None, # (v_x_oep_up, v_x_oep_dn)
        v_c_oep                 : Optional[Tuple[np.ndarray, np.ndarray]] = None, # (v_c_oep_up, v_c_oep_dn)
        H_hf_exchange_dict_by_l : Optional[Tuple[Dict, Dict]]             = None, # (H_hf_exchange_dict_by_l_up, H_hf_exchange_dict_by_l_dn)
        intermediate_info       : Optional[IntermediateInfo]              = None,
        symmetrize              : bool                                    = False, # whether to symmetrize the eigenvectors (default: False)
        save_full_spectrum      : bool                                    = False, # whether to save full spectrum information from each iteration (default: False)
    ) -> SCFResult:
        r"""
        Inner SCF loop: standard Kohn-Sham self-consistency
        
        Fixed: external potential, HF exchange (if any)
        Iterate: rho → V_H, V_xc → H → solve → orbitals → rho'
        
        Parameters
        ----------
        rho_initial : np.ndarray
            Initial total density guess on quadrature points. This routine
            initializes spin-channel densities internally from this input.
        settings : SCFSettings
            SCF settings
        orbitals_initial : Tuple[np.ndarray, np.ndarray], optional
            Optional initial orbital guess for spin channels. (orbitals_up, orbitals_dn). 
            If provided, will be used as initial orbitals instead of solving eigenvalue problem.
        v_x_oep : Tuple[np.ndarray, np.ndarray], optional
            Spin-channel OEP exchange potentials: (v_x_oep_up, v_x_oep_dn).
        v_c_oep : Tuple[np.ndarray, np.ndarray], optional
            Spin-channel OEP correlation potentials: (v_c_oep_up, v_c_oep_dn).
        H_hf_exchange_dict_by_l : Tuple[Dict, Dict], optional
            Spin-channel HF exchange matrices by angular momentum:
            (H_hf_exchange_dict_by_l_up, H_hf_exchange_dict_by_l_dn).
        intermediate_info : IntermediateInfo, optional
            Intermediate information for debugging and analysis
        symmetrize : bool, optional
            Whether to symmetrize the eigenvectors (default: False)
            - If False (default), the eigenvectors are solved using the generalized eigenvalue problem: Hx = λSx.
            - If True, the eigenvectors are symmetrized using the overlap matrix S: S^(-1/2) @ H @ S^(-1/2) @ S^(1/2) x = λS^(1/2)x.
                - Symmetrize Hamiltonian: H → S^(-1/2) @ H @ S^(-1/2)
                - Solve the eigenvalue problem: Hy = λy.
                - Symmetrized eigenvectors: x = S^(-1/2) @ y.
        save_full_spectrum : bool, optional
            If True, save full spectrum information from each iteration.
            Default is False.
        
        Returns
        -------
        result : SCFResult
            Converged SCF state
        """
        # initialize variables to None
        orbitals_initial_up = None
        orbitals_initial_dn = None
        v_x_oep_up = None
        v_x_oep_dn = None
        v_c_oep_up = None
        v_c_oep_dn = None
        H_hf_exchange_dict_by_l_up = None
        H_hf_exchange_dict_by_l_dn = None

        # type check for required fields
        assert isinstance(rho_initial, np.ndarray), \
            RHO_INITIAL_TYPE_ERROR.format(type(rho_initial))
        assert isinstance(settings, SCFSettings), \
            SETTINGS_TYPE_ERROR.format(type(settings))
        if orbitals_initial is not None:
            assert isinstance(orbitals_initial, tuple) and len(orbitals_initial) == 2, \
                ORBITALS_INITIAL_TYPE_ERROR.format(type(orbitals_initial))
            assert isinstance(orbitals_initial[0], np.ndarray), \
                ORBITALS_INITIAL_TYPE_ERROR.format(type(orbitals_initial[0]))
            assert isinstance(orbitals_initial[1], np.ndarray), \
                ORBITALS_INITIAL_TYPE_ERROR.format(type(orbitals_initial[1]))
            orbitals_initial_up, orbitals_initial_dn = orbitals_initial
        if v_x_oep is not None:
            assert isinstance(v_x_oep, tuple) and len(v_x_oep) == 2, \
                V_X_OEP_TYPE_ERROR.format(type(v_x_oep))
            assert isinstance(v_x_oep[0], np.ndarray), \
                V_X_OEP_TYPE_ERROR.format(type(v_x_oep[0]))
            assert isinstance(v_x_oep[1], np.ndarray), \
                V_X_OEP_TYPE_ERROR.format(type(v_x_oep[1]))
            v_x_oep_up, v_x_oep_dn = v_x_oep
        if v_c_oep is not None:
            assert isinstance(v_c_oep, tuple) and len(v_c_oep) == 2, \
                V_C_OEP_TYPE_ERROR.format(type(v_c_oep))
            assert isinstance(v_c_oep[0], np.ndarray), \
                V_C_OEP_TYPE_ERROR.format(type(v_c_oep[0]))
            assert isinstance(v_c_oep[1], np.ndarray), \
                V_C_OEP_TYPE_ERROR.format(type(v_c_oep[1]))
            v_c_oep_up, v_c_oep_dn = v_c_oep
        if H_hf_exchange_dict_by_l is not None:
            assert isinstance(H_hf_exchange_dict_by_l, tuple) and len(H_hf_exchange_dict_by_l) == 2, \
                H_HF_EXCHANGE_DICT_BY_L_TYPE_ERROR.format(type(H_hf_exchange_dict_by_l))
            assert isinstance(H_hf_exchange_dict_by_l[0], dict), \
                H_HF_EXCHANGE_DICT_BY_L_TYPE_ERROR.format(type(H_hf_exchange_dict_by_l[0]))
            assert isinstance(H_hf_exchange_dict_by_l[1], dict), \
                H_HF_EXCHANGE_DICT_BY_L_TYPE_ERROR.format(type(H_hf_exchange_dict_by_l[1]))
            H_hf_exchange_dict_by_l_up, H_hf_exchange_dict_by_l_dn = H_hf_exchange_dict_by_l
        assert isinstance(symmetrize, bool), \
            SYMMETRIZE_TYPE_ERROR.format(type(symmetrize))
        assert isinstance(save_full_spectrum, bool), \
            SAVE_FULL_SPECTRUM_TYPE_ERROR.format(type(save_full_spectrum))
        
        # initialize variables
        max_iter = settings.inner_max_iter
        verbose  = settings.verbose

        # rho = rho_initial.copy()
        rho = self.density_calculator.normalize_density(rho_initial.copy())
        rho_up = rho * self.occupation_info.n_free_electrons_up / self.occupation_info.n_free_electrons
        rho_dn = rho * self.occupation_info.n_free_electrons_dn / self.occupation_info.n_free_electrons
        rho_extended = np.hstack((rho_up, rho_dn))

        density_data_up = self.density_calculator.create_density_data_from_mixed(
            rho_mixed        = rho_up,
            orbitals         = orbitals_initial_up,
            compute_gradient = self.xc_requirements.needs_gradient,
            compute_tau      = self.xc_requirements.needs_tau and (orbitals_initial_up is not None),
            rho_nlcc         = self.rho_nlcc / 2.0 if self.rho_nlcc is not None else None,
            occupations      = self.occupation_info.occupations_up,
        )
        density_data_dn = self.density_calculator.create_density_data_from_mixed(
            rho_mixed        = rho_dn,
            orbitals         = orbitals_initial_dn,
            compute_gradient = self.xc_requirements.needs_gradient,
            compute_tau      = self.xc_requirements.needs_tau and (orbitals_initial_dn is not None),
            rho_nlcc         = self.rho_nlcc / 2.0 if self.rho_nlcc is not None else None,
            occupations      = self.occupation_info.occupations_dn,
        )

        # Reset mixer and convergence checker
        self.mixer.reset()
        self.inner_convergence_checker.reset()

        # # Set HF exchange if provided
        # Since there are two spin channels, the HF exchange matrices will be set for each spin channel separately
        # if H_hf_exchange_dict_by_l is not None:
        #     self.hamiltonian_builder.set_hf_exchange_matrices(H_hf_exchange_dict_by_l)
        
        # Print convergence table header
        if verbose:
            self.inner_convergence_checker.print_header(prefix="")

        # Main inner SCF loop
        for iteration in range(max_iter):
            
            # ===== Step 1: Compute potentials =====
            # Hartree potential
            if self.switches.use_hartree:
                v_hartree = self.poisson_solver.solve_hartree(rho)
            else:
                v_hartree = np.zeros_like(rho)

            # XC potential
            # For pseudopotentials: use rho_total = rho_valence + rho_nlcc
            # For all-electron: rho_nlcc is zero, so rho_total = rho_valence
            v_x_up, v_x_dn = np.zeros_like(rho), np.zeros_like(rho)  # Default: no exchange potential
            v_c_up, v_c_dn = np.zeros_like(rho), np.zeros_like(rho)  # Default: no correlation potential
            de_xc_dtau_up, de_xc_dtau_dn = None, None                # Default: no meta-GGA derivative term

            # Original XC calculation
            if self.xc_calculator is not None:
                # Compute XC using new interface: DensityData → XCPotentialData
                xc_potential_data_up, xc_potential_data_dn = self.xc_calculator.compute_xc_with_spin(density_data_up, density_data_dn)
                v_x_up, v_x_dn = xc_potential_data_up.v_x, xc_potential_data_dn.v_x
                v_c_up, v_c_dn = xc_potential_data_up.v_c, xc_potential_data_dn.v_c
                de_xc_dtau_up, de_xc_dtau_dn = xc_potential_data_up.de_xc_dtau, xc_potential_data_dn.de_xc_dtau
            
            # Correction from ML model
            if self.ml_xc_calculator is not None and self.ml_each_scf_step:
                features_up = self._construct_input_features_for_ml_model(
                    features_list = self.ml_xc_calculator.features_list,
                    density_data  = density_data_up
                )
                features_dn = self._construct_input_features_for_ml_model(
                    features_list = self.ml_xc_calculator.features_list,
                    density_data  = density_data_dn
                )
                v_xc_ml_up, v_xc_ml_dn = \
                    self.ml_xc_calculator.predict_vxc_with_spin(features_up, features_dn)
            else:
                v_xc_ml_up, v_xc_ml_dn = None, None
            
            # ===== Step 2: Build and solve for each l channel =====
            occ_eigenvalues_up_list    : List[np.ndarray] = []
            occ_eigenvalues_dn_list    : List[np.ndarray] = []
            occ_eigenvectors_up_list   : List[np.ndarray] = []
            occ_eigenvectors_dn_list   : List[np.ndarray] = []
            unocc_eigenvalues_up_list  : List[np.ndarray] = []  # needed only for OEP
            unocc_eigenvalues_dn_list  : List[np.ndarray] = []  # needed only for OEP
            unocc_eigenvectors_up_list : List[np.ndarray] = []  # needed only for OEP
            unocc_eigenvectors_dn_list : List[np.ndarray] = []  # needed only for OEP


            # Determine which l values to iterate over
            # If angular_momentum_cutoff is set (e.g., for RPA), iterate over all l values from 0 to cutoff
            # Otherwise, only iterate over occupied l values
            if self.angular_momentum_cutoff is not None:
                unique_l_values = list(range(self.angular_momentum_cutoff + 1))
            else:
                unique_l_values = self.occupation_info.unique_l_values
        
            ######################### Spin up channel #########################

            if H_hf_exchange_dict_by_l_up is not None:
                self.hamiltonian_builder.set_hf_exchange_matrices(H_hf_exchange_dict_by_l_up)

            for l in unique_l_values:

                # Build Hamiltonian for this l
                H_l_up = self.hamiltonian_builder.build_for_l_channel(
                    l                = l,
                    v_hartree        = v_hartree,
                    v_x              = v_x_up,
                    v_c              = v_c_up,
                    switches         = self.switches,
                    v_x_oep          = v_x_oep_up,
                    v_c_oep          = v_c_oep_up,
                    v_xc_ml          = v_xc_ml_up,
                    de_xc_dtau       = de_xc_dtau_up,
                    symmetrize       = symmetrize,
                    exclude_boundary = True,
                )
                
                # Number of states to solve for this l
                n_states = self.occupation_info.n_states_for_l(l)
                
                if self.switches.use_oep or self.mixer.use_preconditioner:
                    if symmetrize:
                        # solve the eigenvalue problem Hx = λx
                        full_eigvals_l_up, full_eigvecs_l_up = self.eigensolver.solve_full(H_l_up)
                    else:
                        # solve the generalized eigenvalue problem Hx = λSx
                        S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                        full_eigvals_l_up, full_eigvecs_l_up = self.eigensolver.solve_generalized_full(H_l_up, S)

                    occ_eigvals_l_up = full_eigvals_l_up[:n_states]
                    occ_eigvecs_l_up = full_eigvecs_l_up[:, :n_states]

                    # Store unoccupied eigenvalues and eigenvectors, used only for OEP and at final readout
                    #     or preconditioning, where we need to compute dielectric matrix as preconditioner for Pulay mixing.
                    unocc_eigenvalues_up_list.append(full_eigvals_l_up[n_states:])
                    unocc_eigenvectors_up_list.append(full_eigvecs_l_up[:, n_states:])
                else:
                    # Solve eigenvalue problem
                    if symmetrize:
                        # solve the eigenvalue problem Hx = λx
                        occ_eigvals_l_up, occ_eigvecs_l_up = self.eigensolver.solve_lowest(H_l_up, n_states)
                    else:
                        # solve the generalized eigenvalue problem Hx = λSx
                        S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                        occ_eigvals_l_up, occ_eigvecs_l_up = self.eigensolver.solve_generalized_lowest(H_l_up, S, n_states)                

                # Store occupied eigenvalues and eigenvectors
                occ_eigenvalues_up_list.append(occ_eigvals_l_up)
                occ_eigenvectors_up_list.append(occ_eigvecs_l_up)


            ######################### Spin down channel #########################

            if H_hf_exchange_dict_by_l_dn is not None:
                self.hamiltonian_builder.set_hf_exchange_matrices(H_hf_exchange_dict_by_l_dn)

            for l in unique_l_values:

                # Build Hamiltonian for this l
                H_l_dn = self.hamiltonian_builder.build_for_l_channel(
                    l                = l,
                    v_hartree        = v_hartree,
                    v_x              = v_x_dn,
                    v_c              = v_c_dn,
                    switches         = self.switches,
                    v_x_oep          = v_x_oep_dn,
                    v_c_oep          = v_c_oep_dn,
                    v_xc_ml          = v_xc_ml_dn,
                    de_xc_dtau       = de_xc_dtau_dn,
                    symmetrize       = symmetrize,
                    exclude_boundary = True,
                )
                
                # Number of states to solve for this l
                n_states = self.occupation_info.n_states_for_l(l)
                
                if self.switches.use_oep or self.mixer.use_preconditioner:
                    if symmetrize:
                        # solve the eigenvalue problem Hx = λx
                        full_eigvals_l_dn, full_eigvecs_l_dn = self.eigensolver.solve_full(H_l_dn)
                    else:
                        # solve the generalized eigenvalue problem Hx = λSx
                        S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                        full_eigvals_l_dn, full_eigvecs_l_dn = self.eigensolver.solve_generalized_full(H_l_dn, S)

                    occ_eigvals_l_dn = full_eigvals_l_dn[:n_states]
                    occ_eigvecs_l_dn = full_eigvecs_l_dn[:, :n_states]

                    # Store unoccupied eigenvalues and eigenvectors, used only for OEP and at final readout
                    #     or preconditioning, where we need to compute dielectric matrix as preconditioner for Pulay mixing.
                    unocc_eigenvalues_dn_list.append(full_eigvals_l_dn[n_states:])
                    unocc_eigenvectors_dn_list.append(full_eigvecs_l_dn[:, n_states:])
                else:
                    # Solve eigenvalue problem
                    if symmetrize:
                        # solve the eigenvalue problem Hx = λx
                        occ_eigvals_l_dn, occ_eigvecs_l_dn = self.eigensolver.solve_lowest(H_l_dn, n_states)
                    else:
                        # solve the generalized eigenvalue problem Hx = λSx
                        S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                        occ_eigvals_l_dn, occ_eigvecs_l_dn = self.eigensolver.solve_generalized_lowest(H_l_dn, S, n_states)                

                # Store occupied eigenvalues and eigenvectors
                occ_eigenvalues_dn_list.append(occ_eigvals_l_dn)
                occ_eigenvectors_dn_list.append(occ_eigvecs_l_dn)


            # Reorder eigenstates to match occupation list order
            occ_eigenvalues_up, occ_eigenvectors_up = self._reorder_eigenstates_by_occupation(
                occ_eigenvalues_up_list, occ_eigenvectors_up_list
            )
            occ_eigenvalues_dn, occ_eigenvectors_dn = self._reorder_eigenstates_by_occupation(
                occ_eigenvalues_dn_list, occ_eigenvectors_dn_list
            )
            # Interpolate eigenvectors to quadrature points, also symmetrize the eigenvectors
            occ_orbitals_up = self.hamiltonian_builder.interpolate_eigenvectors_to_quadrature(
                eigenvectors = occ_eigenvectors_up,
                symmetrize   = symmetrize,
                pad_width    = 1,
            )
            occ_orbitals_dn = self.hamiltonian_builder.interpolate_eigenvectors_to_quadrature(
                eigenvectors = occ_eigenvectors_dn,
                symmetrize   = symmetrize,
                pad_width    = 1,
            )
            
            # ===== Step 3: Compute new density =====
            # Compute new density from orbitals
            rho_new_up = self.density_calculator.compute_density(occ_orbitals_up, normalize=True, occupations=self.occupation_info.occupations_up)
            rho_new_dn = self.density_calculator.compute_density(occ_orbitals_dn, normalize=True, occupations=self.occupation_info.occupations_dn)
            rho_new_extended = np.hstack((rho_new_up, rho_new_dn))

            # ===== Step 4: Check convergence =====
            converged, residual = self.inner_convergence_checker.check(
                rho_extended, rho_new_extended, iteration + 1, 
                print_status = verbose, prefix = ""
            )
            
            # Save intermediate information if requested
            if intermediate_info is not None:
                intermediate_info.add_inner_iteration(
                    inner_iteration = iteration + 1,
                    rho_residual    = residual,
                    rho             = rho_new_extended,
                )
            
            if converged:
                break
            
            # ===== Step 5: Mix densities and update density_data =====
            if self.mixer.use_preconditioner:
                # compute dielectric matrix for preconditioning
                full_eigenvalues_up, full_orbitals_up, full_l_terms_up = \
                    self._construct_full_eigenvalues_and_orbitals_and_l_terms(
                        occ_eigenvalues_list    = occ_eigenvalues_up_list,
                        occ_eigenvectors_list   = occ_eigenvectors_up_list,
                        unocc_eigenvalues_list  = unocc_eigenvalues_up_list,
                        unocc_eigenvectors_list = unocc_eigenvectors_up_list,
                        symmetrize              = False,
                        angular_momentum_cutoff = self.angular_momentum_cutoff,
                    )
                full_eigenvalues_dn, full_orbitals_dn, full_l_terms_dn = \
                    self._construct_full_eigenvalues_and_orbitals_and_l_terms(
                        occ_eigenvalues_list    = occ_eigenvalues_dn_list,
                        occ_eigenvectors_list   = occ_eigenvectors_dn_list,
                        unocc_eigenvalues_list  = unocc_eigenvalues_dn_list,
                        unocc_eigenvectors_list = unocc_eigenvectors_dn_list,
                        symmetrize              = False,
                        angular_momentum_cutoff = self.angular_momentum_cutoff,
                    )
                dielectric_matrix_up = self.response_calculator.compute_dielectric_matrix(
                    full_eigenvalues_up,
                    full_orbitals_up,
                    full_l_terms_up
                )
                dielectric_matrix_dn = self.response_calculator.compute_dielectric_matrix(
                    full_eigenvalues_dn,
                    full_orbitals_dn,
                    full_l_terms_dn
                )
                n_grid = dielectric_matrix_up.shape[0]
                dielectric_matrix_extended = np.zeros((2*n_grid, 2*n_grid))
                dielectric_matrix_extended[:n_grid, :n_grid] = dielectric_matrix_up
                dielectric_matrix_extended[n_grid:, n_grid:] = dielectric_matrix_dn
                preconditioner = dielectric_matrix_extended
            else:
                preconditioner = None

            # Update densities
            rho_extended = self.mixer.mix(rho_extended, rho_new_extended, preconditioner)
            n_grid = rho_extended.shape[0] // 2
            rho_up, rho_dn = rho_extended[:n_grid], rho_extended[n_grid:]

            # Update density_data for next iteration using mixed density
            density_data_up = self.density_calculator.create_density_data_from_mixed(
                rho_mixed        = rho_up,
                orbitals         = occ_orbitals_up,
                compute_gradient = self.xc_requirements.needs_gradient,
                compute_tau      = self.xc_requirements.needs_tau,
                rho_nlcc         = self.rho_nlcc / 2.0 if self.rho_nlcc is not None else None
            )
            density_data_dn = self.density_calculator.create_density_data_from_mixed(
                rho_mixed        = rho_dn,
                orbitals         = occ_orbitals_dn,
                compute_gradient = self.xc_requirements.needs_gradient,
                compute_tau      = self.xc_requirements.needs_tau,
                rho_nlcc         = self.rho_nlcc / 2.0 if self.rho_nlcc is not None else None
            )
        
        # Print convergence footer
        if verbose:
            self.inner_convergence_checker.print_footer(converged, iteration + 1, prefix="")
        
        if not converged:
            print(INNER_SCF_DID_NOT_CONVERGE_WARNING.format(max_iter))


        # Update properties with or without MLXC
        occ_eigenvalues_up, occ_orbitals_up, occ_eigenvalues_dn, occ_orbitals_dn, v_xc_ml_up, v_xc_ml_dn = \
            self._update_properties_with_spin(
                rho_up                     = rho_up,
                rho_dn                     = rho_dn,
                v_x_oep_up                 = v_x_oep_up,
                v_x_oep_dn                 = v_x_oep_dn,
                v_c_oep_up                 = v_c_oep_up,
                v_c_oep_dn                 = v_c_oep_dn,
                density_data_up            = density_data_up,
                density_data_dn            = density_data_dn,
                H_hf_exchange_dict_by_l_up = H_hf_exchange_dict_by_l_up,
                H_hf_exchange_dict_by_l_dn = H_hf_exchange_dict_by_l_dn,
            )


        # Create final density_data from converged orbitals, do not include NLCC
        final_density_data_up : DensityData = \
            self.density_calculator.create_density_data_from_orbitals(
                orbitals         = occ_orbitals_up,
                compute_gradient = self.xc_requirements.needs_gradient,
                compute_tau      = self.xc_requirements.needs_tau,
                rho_nlcc         = None
            )
        final_density_data_dn : DensityData = \
            self.density_calculator.create_density_data_from_orbitals(
                orbitals         = occ_orbitals_dn,
                compute_gradient = self.xc_requirements.needs_gradient,
                compute_tau      = self.xc_requirements.needs_tau,
                rho_nlcc         = None
            )
        
        # TODO: Change this part, still problematic for spin-polarized case
        # Construct full eigenvalues/orbitals/l-terms at the *final SCF state*.
        # This is a single-shot full-spectrum diagonalization at the converged density.
        if False:
            if self.switches.use_oep or save_full_spectrum:
                full_eigen_energies, full_orbitals, full_l_terms = \
                    self._compute_full_orbitals_and_eigenvalues(
                        rho             = final_density_data.rho,
                        orbitals        = occ_orbitals,
                        v_x_oep         = v_x_oep,
                        v_c_oep         = v_c_oep,
                        v_xc_ml         = v_xc_ml,
                        switches        = self.switches,
                        xc_requirements = self.xc_requirements,
                        xc_calculator   = self.xc_calculator
                    )
        
        full_eigen_energies = None
        full_orbitals       = None
        full_l_terms        = None


        result = SCFResult(
            eigen_energies_up   = occ_eigenvalues_up,
            eigen_energies_dn   = occ_eigenvalues_dn,
            orbitals_up         = occ_orbitals_up,
            orbitals_dn         = occ_orbitals_dn,
            density_data_up     = final_density_data_up,
            density_data_dn     = final_density_data_dn,
            converged           = converged,
            iterations          = iteration + 1,
            rho_residual        = residual,
            full_eigen_energies = full_eigen_energies,
            full_orbitals       = full_orbitals,
            full_l_terms        = full_l_terms,
            intermediate_info   = intermediate_info,
        )

        return result


    def _outer_loop_spin_polarized(
        self,
        rho_initial        : np.ndarray,
        settings           : SCFSettings,
        orbitals_initial   : Optional[np.ndarray] = None,
        intermediate_info  : Optional[IntermediateInfo] = None,
        save_full_spectrum : bool = False,
    ) -> SCFResult:
        """
        Outer SCF loop: for orbital-dependent functionals
        
        Used for:
        - Hartree-Fock exchange (requires orbitals)
        - OEP methods (requires full spectrum)
        - RPA correlation (requires response functions)
        
        Parameters
        ----------
        rho_initial : np.ndarray
            Initial density
        settings : SCFSettings
            SCF settings
        orbitals_initial : np.ndarray, optional
            Initial orbitals guess for debugging
            Shape: (n_grid, n_orbitals)
            If provided, will be used as initial orbitals instead of solving eigenvalue problem
        intermediate_info : IntermediateInfo, optional
            Intermediate information for debugging and analysis
        save_full_spectrum : bool, optional
            If True, save full spectrum information from each iteration.
            Default is False.
        
        Returns
        -------
        result : SCFResult
            Converged SCF state from outer loop
        """
        raise NotImplementedError("Spin-polarized calculation is not supported yet.")


        
        # type check for required fields
        assert isinstance(rho_initial, np.ndarray), \
            RHO_INITIAL_TYPE_ERROR.format(type(rho_initial))
        assert isinstance(settings, SCFSettings), \
            SETTINGS_TYPE_ERROR.format(type(settings))
        if orbitals_initial is not None:
            assert isinstance(orbitals_initial, np.ndarray), \
            ORBITALS_INITIAL_TYPE_ERROR.format(type(orbitals_initial))
        assert isinstance(save_full_spectrum, bool), \
            SAVE_FULL_SPECTRUM_TYPE_ERROR.format(type(save_full_spectrum))

        # initialize variables
        max_outer_iter = settings.outer_max_iter
        verbose        = settings.verbose
        
        rho            = rho_initial.copy()
        orbitals       = orbitals_initial

        # Compute HF exchange matrices from initial orbitals if needed
        if self.switches.use_hf_exchange and (orbitals_initial is not None):
            H_hf_exchange_dict_by_l = self._compute_hf_exchange_matrices_dict(orbitals_initial)
        else:
            H_hf_exchange_dict_by_l = self._get_zero_hf_exchange_matrices_dict()

        # Reset outer convergence checker
        self.outer_convergence_checker.reset()
        
        # Initialize intermediate_info current_outer_iteration if needed
        if intermediate_info is not None:
            intermediate_info.current_outer_iteration = 0

        # Compute default full orbitals and eigenvalues if use_oep is True
        if self.switches.use_oep:
            full_eigen_energies, full_orbitals, full_l_terms = \
                self._compute_default_full_orbitals_and_eigenvalues(
                    rho           = rho_initial,
                    orbitals      = orbitals_initial,
                    xc_functional = 'GGA_PBE'
                )

        # Track outer convergence status
        outer_converged = False
        
        for outer_iter in range(max_outer_iter):
            # Update current outer iteration in intermediate_info
            if intermediate_info is not None:
                intermediate_info.current_outer_iteration = outer_iter + 1
            
            if verbose:
                print(f"Outer iteration {outer_iter + 1}")

            # Compute OEP potentials from initial orbitals if use_oep is True
            if self.switches.use_oep:
                v_x_oep, v_c_oep = self.oep_calculator.compute_oep_potentials(
                    full_eigen_energies = full_eigen_energies,
                    full_orbitals       = full_orbitals,
                    full_l_terms        = full_l_terms,
                    enable_parallelization = self.enable_parallelization,
                )
            else:
                v_x_oep = None
                v_c_oep = None
            
            # Run inner SCF with fixed HF exchange
            inner_result : SCFResult = self._inner_loop(
                rho_initial             = rho,
                settings                = settings,
                H_hf_exchange_dict_by_l = H_hf_exchange_dict_by_l,
                v_x_oep                 = v_x_oep,
                v_c_oep                 = v_c_oep,
                intermediate_info       = intermediate_info,
                save_full_spectrum      = save_full_spectrum,
            )
            
            # update rho and orbitals
            rho_new  = inner_result.density_data.rho
            orbitals = inner_result.orbitals

            # update HF exchange dictionary
            if self.switches.use_hf_exchange:
                H_hf_exchange_dict_by_l = self._compute_hf_exchange_matrices_dict(orbitals) 

            # Check outer loop convergence
            outer_converged, outer_residual = self.outer_convergence_checker.check(
                rho, rho_new, outer_iter + 1,
                print_status = verbose
            )
            
            # Save intermediate information if requested
            if intermediate_info is not None:
                intermediate_info.add_outer_iteration(
                    outer_iteration    = outer_iter + 1,
                    outer_rho_residual = outer_residual,
                    inner_result       = inner_result,
                )
            
            if outer_converged:
                break
            
            # Update for next outer iteration
            rho = rho_new
            if self.switches.use_oep:
                # if OEP is used, also update full orbitals and eigenvalues from inner result
                full_eigen_energies = inner_result.full_eigen_energies
                full_orbitals       = inner_result.full_orbitals
                full_l_terms        = inner_result.full_l_terms
        

        # Update outer loop info in result
        outer_iterations = outer_iter + 1
        # outer_converged is already set correctly from the convergence check above
        
        # Print outer loop footer if verbose is True
        if verbose:
            self.outer_convergence_checker.print_footer(outer_converged, outer_iterations)
        
        outer_result = SCFResult(
            eigen_energies      = inner_result.eigen_energies,
            orbitals            = inner_result.orbitals,
            density_data        = inner_result.density_data,
            converged           = outer_converged,
            iterations          = outer_iterations,
            rho_residual        = outer_residual,
            full_eigen_energies = inner_result.full_eigen_energies,
            full_orbitals       = inner_result.full_orbitals,
            full_l_terms        = inner_result.full_l_terms,
            intermediate_info   = intermediate_info,
        )

        return outer_result





    def _compute_full_orbitals_and_eigenvalues(
        self,
        rho             : np.ndarray,
        switches        : SwitchesFlags,
        xc_requirements : FunctionalRequirements,
        xc_calculator   : Optional[XCEvaluator] = None,
        orbitals        : Optional[np.ndarray]  = None,
        v_x_oep         : Optional[np.ndarray]  = None,
        v_c_oep         : Optional[np.ndarray]  = None,
        v_xc_ml         : Optional[np.ndarray]  = None,
        symmetrize      : bool                  = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute full (occupied + virtual) spectrum at a given density.

        This is a *single-shot* diagonalization for each angular momentum channel \(l\):
        given a density (and optional orbitals for meta-GGA ingredients), build the
        corresponding Kohn–Sham Hamiltonian and solve for its full eigen-spectrum.
                
        Fixed: external potential, HF exchange (if any)
        No SCF iteration is performed here.
        
        Parameters
        ----------
        rho : np.ndarray
            Density used to build Hartree/XC potentials and hence the Hamiltonian.
        switches : SwitchesFlags
            Switches flags
        xc_requirements : FunctionalRequirements
            Functional requirements
        xc_calculator : Optional[XCEvaluator]
            XC calculator, optional
        orbitals : np.ndarray, optional
            Orbitals used only to provide meta-GGA ingredients (e.g. \(\tau\)) when needed.
            If not provided, those ingredients are omitted/disabled in this helper.
            Shape: (n_grid, n_orbitals).
        v_x_oep : np.ndarray, optional
            OEP exchange potential
        v_c_oep : np.ndarray, optional
            OEP correlation potential
        
        Returns
        -------
        full_eigen_energies : np.ndarray
            Full eigenvalues (occupied followed by virtual)
        full_orbitals : np.ndarray
            Full orbitals interpolated to quadrature grid
        full_l_terms : np.ndarray
            Angular momentum index for each entry in full_eigen_energies
        """
        # type check for required fields
        assert isinstance(rho, np.ndarray), \
            RHO_TYPE_ERROR.format(type(rho))
        assert isinstance(switches, SwitchesFlags), \
            SWITCHES_TYPE_ERROR.format(type(switches))
        assert isinstance(xc_requirements, FunctionalRequirements), \
            XC_REQUIREMENTS_TYPE_ERROR.format(type(xc_requirements))

        # type check for optional fields        
        if xc_calculator is not None:
            assert isinstance(xc_calculator, XCEvaluator), \
            XC_CALCULATOR_TYPE_ERROR.format(type(xc_calculator))        
        if orbitals is not None:
            assert isinstance(orbitals, np.ndarray), \
            ORBITALS_TYPE_ERROR.format(type(orbitals))
        if v_x_oep is not None:
            assert isinstance(v_x_oep, np.ndarray), \
            V_X_OEP_TYPE_ERROR.format(type(v_x_oep))
        if v_c_oep is not None:
            assert isinstance(v_c_oep, np.ndarray), \
            V_C_OEP_TYPE_ERROR.format(type(v_c_oep))

        # initialize variables
        rho = self.density_calculator.normalize_density(rho.copy())
        density_data = self.density_calculator.create_density_data_from_mixed(
            rho_mixed        = rho,
            orbitals         = orbitals,
            compute_gradient = xc_requirements.needs_gradient,
            compute_tau      = xc_requirements.needs_tau and (orbitals is not None),
            rho_nlcc         = self.rho_nlcc
        )
            
        # ===== Step 1: Compute potentials =====
        # Hartree potential
        if switches.use_hartree:
            v_hartree = self.poisson_solver.solve_hartree(rho)
        else:
            v_hartree = np.zeros_like(rho)

        # XC potential
        # For pseudopotentials: use rho_total = rho_valence + rho_nlcc
        # For all-electron: rho_nlcc is zero, so rho_total = rho_valence
        v_x = np.zeros_like(rho)  # Default: no exchange potential
        v_c = np.zeros_like(rho)  # Default: no correlation potential
        de_xc_dtau = None         # Default: no meta-GGA derivative term

        if xc_calculator is not None:
            # Compute XC using new interface: DensityData → XCPotentialData
            xc_potential_data = xc_calculator.compute_xc(density_data)
            v_x = xc_potential_data.v_x
            v_c = xc_potential_data.v_c
            de_xc_dtau = xc_potential_data.de_xc_dtau
        
        # ===== Step 2: Build and solve for each l channel =====
        occ_eigenvalues_list    : List[np.ndarray] = []
        occ_eigenvectors_list   : List[np.ndarray] = []
        unocc_eigenvalues_list  : List[np.ndarray] = []  # needed only for OEP
        unocc_eigenvectors_list : List[np.ndarray] = []  # needed only for OEP


        if self.angular_momentum_cutoff is not None:
            unique_l_values = list(range(self.angular_momentum_cutoff + 1))
        else:
            unique_l_values = self.occupation_info.unique_l_values
        

        for l in unique_l_values:
            # Build Hamiltonian for this l
            H_l = self.hamiltonian_builder.build_for_l_channel(
                l                = l,
                v_hartree        = v_hartree,
                v_x              = v_x,
                v_c              = v_c,
                switches         = switches,
                v_x_oep          = v_x_oep,
                v_c_oep          = v_c_oep,
                v_xc_ml          = v_xc_ml,
                de_xc_dtau       = de_xc_dtau,
                symmetrize       = symmetrize,
                exclude_boundary = True,
            )

            # Number of states to solve for this l
            n_occ_states = self.occupation_info.n_states_for_l(l)

            # Solve eigenvalue problem
            if symmetrize:
                full_eigenvalues_l, full_eigenvectors_l = self.eigensolver.solve_full(H_l)
            else:
                S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                full_eigenvalues_l, full_eigenvectors_l = self.eigensolver.solve_generalized_full(H_l, S)
            
            # Append eigenvalues and eigenvectors to lists
            occ_eigenvalues_list.append(full_eigenvalues_l[:n_occ_states])
            occ_eigenvectors_list.append(full_eigenvectors_l[:, :n_occ_states])
            unocc_eigenvalues_list.append(full_eigenvalues_l[n_occ_states:])
            unocc_eigenvectors_list.append(full_eigenvectors_l[:, n_occ_states:])
        

        return self._construct_full_eigenvalues_and_orbitals_and_l_terms(
            occ_eigenvalues_list    = occ_eigenvalues_list,
            occ_eigenvectors_list   = occ_eigenvectors_list,
            unocc_eigenvalues_list  = unocc_eigenvalues_list,
            unocc_eigenvectors_list = unocc_eigenvectors_list,
            symmetrize              = symmetrize,
            angular_momentum_cutoff = self.angular_momentum_cutoff
        )


    def _compute_default_full_orbitals_and_eigenvalues(
        self, 
        rho           : np.ndarray,
        orbitals      : np.ndarray,
        xc_functional : str = 'GGA_PBE'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute full (occupied + virtual) spectrum at a given density/orbitals using
        a specified XC functional.

        This is a convenience wrapper around `_compute_full_orbitals_and_eigenvalues`.
        
        Parameters
        ----------
        rho : np.ndarray
            Density used to build potentials and the Hamiltonian.
        orbitals : np.ndarray
            Orbitals used only to provide meta-GGA ingredients (e.g. \(\tau\)) when needed.
        xc_functional : str
            XC functional, default: 'GGA_PBE'

        Returns
        -------
        full_eigen_energies : np.ndarray
            Full eigenvalues (occupied followed by virtual)
        full_orbitals : np.ndarray
            Full orbitals interpolated to quadrature grid
        full_l_terms : np.ndarray
            Angular momentum index for each entry in full_eigen_energies
        """
        # type check for required fields
        assert isinstance(rho, np.ndarray), \
            RHO_TYPE_ERROR.format(type(rho))
        assert isinstance(orbitals, np.ndarray), \
            ORBITALS_TYPE_ERROR.format(type(orbitals))
        assert isinstance(xc_functional, str), \
            XC_FUNCTIONAL_TYPE_ERROR.format(type(xc_functional))
        
        _switches        : SwitchesFlags          = SwitchesFlags(xc_functional)
        _xc_requirements : FunctionalRequirements = get_functional_requirements(xc_functional)
        _xc_calculator   : XCEvaluator            = \
            create_xc_evaluator(
                functional_name   = xc_functional,
                derivative_matrix = self.density_calculator.derivative_matrix,
                r_quad            = self.density_calculator.quadrature_nodes
            )

        return self._compute_full_orbitals_and_eigenvalues(
            rho              = rho,
            orbitals         = orbitals,
            switches         = _switches,
            xc_requirements  = _xc_requirements,
            xc_calculator    = _xc_calculator
        )

    
    
    def _check_occ_and_unocc_eigenvalues_and_eigenvectors_lists(
        self,
        occ_eigenvalues_list    : List[np.ndarray],
        occ_eigenvectors_list   : List[np.ndarray],
        unocc_eigenvalues_list  : List[np.ndarray],
        unocc_eigenvectors_list : List[np.ndarray],
        angular_momentum_cutoff : Optional[int] = None,
    ) -> None:
        """
        Check occ and unocc eigenvalues and eigenvectors lists.
        """
        # type check for required fields
        assert isinstance(occ_eigenvalues_list, list), \
            OCC_EIGENVALUES_LIST_NOT_LIST_ERROR.format(type(occ_eigenvalues_list))
        assert isinstance(occ_eigenvectors_list, list), \
            OCC_EIGENVECTORS_LIST_NOT_LIST_ERROR.format(type(occ_eigenvectors_list))
        assert isinstance(unocc_eigenvalues_list, list), \
            UNOCC_EIGENVALUES_LIST_NOT_LIST_ERROR.format(type(unocc_eigenvalues_list))
        assert isinstance(unocc_eigenvectors_list, list), \
            UNOCC_EIGENVECTORS_LIST_NOT_LIST_ERROR.format(type(unocc_eigenvectors_list))
        if angular_momentum_cutoff is not None:
            assert isinstance(angular_momentum_cutoff, int), \
            ANGULAR_MOMENTUM_CUTOFF_TYPE_ERROR.format(type(angular_momentum_cutoff))
        
        # length checks
        unique_l_values_number = len(self.occupation_info.unique_l_values) if angular_momentum_cutoff is None else angular_momentum_cutoff + 1
        assert len(occ_eigenvalues_list) == unique_l_values_number, \
            OCC_EIGENVALUES_LIST_LENGTH_ERROR.format(len(occ_eigenvalues_list))
        assert len(occ_eigenvectors_list) == unique_l_values_number, \
            OCC_EIGENVECTORS_LIST_LENGTH_ERROR.format(len(occ_eigenvectors_list))
        assert len(unocc_eigenvalues_list) == unique_l_values_number, \
            UNOCC_EIGENVALUES_LIST_LENGTH_ERROR.format(len(unocc_eigenvalues_list))
        assert len(unocc_eigenvectors_list) == unique_l_values_number, \
            UNOCC_EIGENVECTORS_LIST_LENGTH_ERROR.format(len(unocc_eigenvectors_list))
        
        # Dimention and shape checks
        n_physical_nodes = self.hamiltonian_builder.ops_builder.physical_nodes.shape[0]
        n_interior_nodes = n_physical_nodes - 2

        for i in range(unique_l_values_number):
            occ_eigenvalues    = occ_eigenvalues_list[i]
            occ_eigenvectors   = occ_eigenvectors_list[i]
            unocc_eigenvalues  = unocc_eigenvalues_list[i]
            unocc_eigenvectors = unocc_eigenvectors_list[i]
            
            # check dimention
            assert occ_eigenvalues.ndim == 1, \
                OCC_EIGENVALUES_LIST_NDIM_ERROR.format(occ_eigenvalues.ndim)
            assert occ_eigenvectors.ndim == 2, \
                OCC_EIGENVECTORS_LIST_NDIM_ERROR.format(occ_eigenvectors.ndim)
            assert unocc_eigenvalues.ndim == 1, \
                UNOCC_EIGENVALUES_LIST_NDIM_ERROR.format(unocc_eigenvalues.ndim)
            assert unocc_eigenvectors.ndim == 2, \
                UNOCC_EIGENVECTORS_LIST_NDIM_ERROR.format(unocc_eigenvectors.ndim)
            
            # occupied shape check
            assert occ_eigenvalues.shape[0] == occ_eigenvectors.shape[1], \
                OCC_EIGENVALUES_AND_EIGENVECTORS_LIST_SHAPE_MISMATCH_ERROR.format(occ_eigenvalues.shape[0], occ_eigenvectors.shape[1])
            assert occ_eigenvectors.shape[0] == n_interior_nodes, \
                OCC_EIGENVECTORS_LIST_SHAPE_ERROR.format(occ_eigenvectors.shape[0], n_interior_nodes)
            
            # unoccupied shape check
            assert unocc_eigenvalues.shape[0] == unocc_eigenvectors.shape[1], \
                UNOCC_EIGENVALUES_AND_EIGENVECTORS_LIST_SHAPE_MISMATCH_ERROR.format(unocc_eigenvalues.shape[0], unocc_eigenvectors.shape[1])
            assert unocc_eigenvectors.shape[0] == n_interior_nodes, \
                UNOCC_EIGENVECTORS_LIST_SHAPE_ERROR.format(unocc_eigenvectors.shape[0], n_interior_nodes)



    def _construct_full_eigenvalues_and_eigenvectors_and_l_terms(
        self,
        occ_eigenvalues_list    : List[np.ndarray],
        occ_eigenvectors_list   : List[np.ndarray],
        unocc_eigenvalues_list  : List[np.ndarray],
        unocc_eigenvectors_list : List[np.ndarray],
        angular_momentum_cutoff : Optional[int] = None,
        eigenvectors_pad_width  : Optional[int] = 0, # number of points to pad on each side of the eigenvectors
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct full eigenvalues, eigenvectors and l terms for OEP, only needed for OEP and at final readout
        """
        # type check for required fields
        self._check_occ_and_unocc_eigenvalues_and_eigenvectors_lists(
            occ_eigenvalues_list    = occ_eigenvalues_list,
            occ_eigenvectors_list   = occ_eigenvectors_list,
            unocc_eigenvalues_list  = unocc_eigenvalues_list,
            unocc_eigenvectors_list = unocc_eigenvectors_list,
            angular_momentum_cutoff = angular_momentum_cutoff
        )

        unique_l_values_number = len(self.occupation_info.unique_l_values) if angular_momentum_cutoff is None else angular_momentum_cutoff + 1


        # Reorder eigenstates to match occupation list order
        occ_eigenvalues, occ_eigenvectors = self._reorder_eigenstates_by_occupation(
            occ_eigenvalues_list, occ_eigenvectors_list
        )
        
        full_eigenvalues  = np.concatenate([occ_eigenvalues,  *unocc_eigenvalues_list],  axis = 0)
        full_eigenvectors = np.concatenate([occ_eigenvectors, *unocc_eigenvectors_list], axis = 1)


        # Pad eigenvectors if needed
        if eigenvectors_pad_width is not None and eigenvectors_pad_width > 0:
            full_eigenvectors = np.pad(full_eigenvectors,((eigenvectors_pad_width, eigenvectors_pad_width),(0,0)))


        # Compute angular momentum index for each entry in full_eigen_energies
        unocc_l_terms = np.concatenate([
            np.full(vals.shape[0], l, dtype=int)
            for l, vals in zip(range(unique_l_values_number), unocc_eigenvalues_list)
            if vals.size > 0
        ]) if len(unocc_eigenvalues_list) > 0 else np.empty(0, dtype=int)

        full_l_terms = np.concatenate([self.occupation_info.occ_l, unocc_l_terms])

        return full_eigenvalues, full_eigenvectors, full_l_terms


    def _construct_full_eigenvalues_and_orbitals_and_l_terms(
        self,
        occ_eigenvalues_list    : List[np.ndarray],
        occ_eigenvectors_list   : List[np.ndarray],
        unocc_eigenvalues_list  : List[np.ndarray],
        unocc_eigenvectors_list : List[np.ndarray],
        symmetrize              : bool = False,
        angular_momentum_cutoff : Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct full eigenvalues, orbitals and l terms for OEP, only needed for OEP and at final readout
        """
        # No type checking needed here, already included in the following function.
        full_eigenvalues, full_eigenvectors, full_l_terms = \
            self._construct_full_eigenvalues_and_eigenvectors_and_l_terms(
                occ_eigenvalues_list    = occ_eigenvalues_list,
                occ_eigenvectors_list   = occ_eigenvectors_list,
                unocc_eigenvalues_list  = unocc_eigenvalues_list,
                unocc_eigenvectors_list = unocc_eigenvectors_list,
                angular_momentum_cutoff = angular_momentum_cutoff,
                eigenvectors_pad_width  = 0, # No need to pad eigenvectors here, because potentially, they will be padded in the next step.
            )

        # Interpolate eigenvectors to quadrature points, also symmetrize the eigenvectors
        full_orbitals = self.hamiltonian_builder.interpolate_eigenvectors_to_quadrature(
            eigenvectors = full_eigenvectors,
            symmetrize   = symmetrize,
            pad_width    = 1,
        )

        return full_eigenvalues, full_orbitals, full_l_terms



    def _construct_input_features_for_ml_model(
        self, 
        features_list : List[str], 
        density_data  : DensityData,
    ) -> np.ndarray:
        """
        Construct input features for ML model.
        """
        rho_ml = np.asarray(density_data.rho, dtype=float).reshape(-1)
        r_grid = np.asarray(self.density_calculator.quadrature_nodes, dtype=float).reshape(-1)
        if r_grid.shape != rho_ml.shape:
            raise ValueError(ML_FEATURES_QUADRATURE_RHO_LENGTH_MISMATCH_ERROR.format(r_grid.shape, rho_ml.shape))
        if self.ml_smooth_rho_for_features:
            rho_ml = DataProcessor.smooth_rho_with_loglinear_asymptote(
                r_grid, rho_ml, 7.0, 10.0, 13.0, rho_floor=1e-30,
            )
        features = []
        for feature in features_list:
            if feature == "rho":
                features.append(rho_ml)
            elif feature == "grad_rho":
                grad_rho = self.density_calculator.compute_density_gradient(rho_ml)
                features.append(grad_rho)
            elif feature == "grad_rho_norm":
                grad_rho = self.density_calculator.compute_density_gradient(rho_ml)
                grad_rho_norm = np.abs(grad_rho)
                features.append(grad_rho_norm)
            elif feature == "lap_rho":
                lap_rho = self.density_calculator.compute_density_laplacian(rho_ml)
                features.append(lap_rho)
            elif feature == "grad_rho_reduced":
                # Reduced density gradient s(r) = |∇ρ| / (2 k_F ρ^(4/3)), k_F = (3π²)^(1/3)
                grad_rho = self.density_calculator.compute_density_gradient(rho_ml)
                kf = (3 * np.pi**2) ** (1.0 / 3.0)
                grad_rho_reduced = np.abs(grad_rho) / (2.0 * kf * (rho_ml ** (4.0 / 3.0)))
                features.append(grad_rho_reduced)
            elif feature == "lap_rho_reduced":
                # Reduced Laplacian q(r) = ∇²ρ / (4 k_F² ρ^(5/3)), k_F = (3π²)^(1/3); same FEM Laplacian as SCF
                lap_rho = self.density_calculator.compute_density_laplacian(rho_ml)
                kf = (3 * np.pi**2) ** (1.0 / 3.0)
                lap_rho_reduced = lap_rho / (4.0 * (kf ** 2) * (rho_ml ** (5.0 / 3.0)))
                features.append(lap_rho_reduced)
            elif feature == "hartree":
                hartree = self.poisson_solver.solve_hartree(rho_ml)
                features.append(hartree)
            elif feature == "lda_xc_potential":
                from ..xc.lda import lda_exchange_potential_generic, lda_correlation_potential_generic
                lda_exchange     = lda_exchange_potential_generic(rho_ml)
                lda_correlation  = lda_correlation_potential_generic(rho_ml)
                lda_xc_potential = lda_exchange + lda_correlation
                features.append(lda_xc_potential)
            else:
                raise ValueError(INVALID_FEATURES_IN_ML_MODEL_ERROR.format(feature))
        
        return np.column_stack(features)


    def _update_properties(
        self,
        rho          : np.ndarray,
        v_x_oep      : np.ndarray,
        v_c_oep      : np.ndarray,
        density_data : DensityData,
        symmetrize   : bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update properties with ML XC.

        Parameters
        ----------
        rho : np.ndarray
            Density
        v_x_oep : np.ndarray
            OEP exchange potential
        v_c_oep : np.ndarray
            OEP correlation potential
        density_data : DensityData
            Density data

        Returns
        -------
        occ_eigenvalues : np.ndarray
            Occupied eigenvalues
        occ_orbitals : np.ndarray
            Occupied orbitals
        v_xc_ml : np.ndarray
            ML XC potential
        """

        # Machine Learning XC energy correction
        v_xc_ml = None
        
        if self.ml_xc_calculator is not None:
            # Construct input features for ML model
            input_features = self._construct_input_features_for_ml_model(
                features_list = self.ml_xc_calculator.features_list,
                density_data = density_data
            )

            v_xc_ml = self.ml_xc_calculator.predict_vxc(
                features          = input_features,
                derivative_matrix = self.hamiltonian_builder.ops_builder.derivative_matrix,
                quadrature_nodes  = self.hamiltonian_builder.ops_builder.quadrature_nodes,
            )
    
        # ===== Step 1: Compute potentials =====
        # Hartree potential
        if self.switches.use_hartree:
            v_hartree = self.poisson_solver.solve_hartree(rho)
        else:
            v_hartree = np.zeros_like(rho)

        # XC potential
        # For pseudopotentials: use rho_total = rho_valence + rho_nlcc
        # For all-electron: rho_nlcc is zero, so rho_total = rho_valence
        v_x = np.zeros_like(rho)  # Default: no exchange potential
        v_c = np.zeros_like(rho)  # Default: no correlation potential
        de_xc_dtau = None         # Default: no meta-GGA derivative term

        if self.xc_calculator is not None:
            # Compute XC using new interface: DensityData → XCPotentialData
            xc_potential_data = self.xc_calculator.compute_xc(density_data)
            v_x = xc_potential_data.v_x
            v_c = xc_potential_data.v_c
            de_xc_dtau = xc_potential_data.de_xc_dtau
        
        # ===== Step 2: Build and solve for each l channel =====
        occ_eigenvalues_list    : List[np.ndarray] = []
        occ_eigenvectors_list   : List[np.ndarray] = []

        for l in self.occupation_info.unique_l_values:
            # Build Hamiltonian for this l
            H_l = self.hamiltonian_builder.build_for_l_channel(
                l                = l,
                v_hartree        = v_hartree,
                v_x              = v_x,
                v_c              = v_c,
                switches         = self.switches,
                v_x_oep          = v_x_oep,
                v_c_oep          = v_c_oep,
                v_xc_ml          = v_xc_ml,
                de_xc_dtau       = de_xc_dtau,
                symmetrize       = symmetrize,
                exclude_boundary = True,
            )
            # Number of states to solve for this l
            n_states = self.occupation_info.n_states_for_l(l)

            # Solve eigenvalue problem
            if symmetrize:
                occ_eigvals_l, occ_eigvecs_l = self.eigensolver.solve_lowest(H_l, n_states)
            else:
                S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                occ_eigvals_l, occ_eigvecs_l = self.eigensolver.solve_generalized_lowest(H_l, S, n_states)
            
            # Store occupied eigenvalues and eigenvectors
            occ_eigenvalues_list.append(occ_eigvals_l)
            occ_eigenvectors_list.append(occ_eigvecs_l)
        
        # Reorder eigenstates to match occupation list order
        occ_eigenvalues, occ_eigenvectors = self._reorder_eigenstates_by_occupation(
            occ_eigenvalues_list, occ_eigenvectors_list
        )
        
        # Interpolate eigenvectors to quadrature points, also symmetrize the eigenvectors
        occ_orbitals = self.hamiltonian_builder.interpolate_eigenvectors_to_quadrature(
            eigenvectors = occ_eigenvectors,
            symmetrize   = symmetrize,
            pad_width    = 1,
        )
        
        return occ_eigenvalues, occ_orbitals, v_xc_ml


    def _update_properties_with_spin(
        self,
        rho_up                     : np.ndarray,
        rho_dn                     : np.ndarray,
        v_x_oep_up                 : Optional[np.ndarray],
        v_x_oep_dn                 : Optional[np.ndarray],
        v_c_oep_up                 : Optional[np.ndarray],
        v_c_oep_dn                 : Optional[np.ndarray],
        density_data_up            : DensityData,
        density_data_dn            : DensityData,
        H_hf_exchange_dict_by_l_up : Optional[Dict[int, np.ndarray]] = None,
        H_hf_exchange_dict_by_l_dn : Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Update properties with ML XC for collinear spin-polarized KS.

        Analogous to :meth:`_update_properties`, but uses spin-resolved
        ``density_data_up`` / ``density_data_dn``, Hartree from total valence
        density ``rho_up + rho_dn``, and separate Hamiltonian builds per spin
        channel.

        Parameters
        ----------
        rho_up, rho_dn : np.ndarray
            Spin-channel densities on the quadrature grid.
        v_x_oep_up, v_x_oep_dn : np.ndarray, optional
            OEP exchange potentials per spin channel.
        v_c_oep_up, v_c_oep_dn : np.ndarray, optional
            OEP correlation potentials per spin channel.
        density_data_up, density_data_dn : DensityData
            Spin-channel density data for XC (and ML features).
        H_hf_exchange_dict_by_l_up, H_hf_exchange_dict_by_l_dn : dict, optional
            Per-:math:`l` Hartree--Fock exchange matrices for each spin channel,
            same layout as in :meth:`_inner_loop_spin_polarized`. Set on the
            builder immediately before the corresponding :math:`l` loops.

        Returns
        -------
        occ_eigenvalues_up : np.ndarray
            Occupied eigenvalues (spin-up), occupation order.
        occ_orbitals_up : np.ndarray
            Occupied orbitals (spin-up) on quadrature points.
        occ_eigenvalues_dn : np.ndarray
            Occupied eigenvalues (spin-down), occupation order.
        occ_orbitals_dn : np.ndarray
            Occupied orbitals (spin-down) on quadrature points.
        v_xc_ml_up : np.ndarray, optional
            ML XC correction on spin-up channel, or ``None`` if no ML calculator.
        v_xc_ml_dn : np.ndarray, optional
            ML XC correction on spin-down channel, or ``None`` if no ML calculator.
        """
        v_xc_ml_up: Optional[np.ndarray] = None
        v_xc_ml_dn: Optional[np.ndarray] = None

        if self.ml_xc_calculator is not None:
            input_features_up = self._construct_input_features_for_ml_model(
                features_list = self.ml_xc_calculator.features_list,
                density_data    = density_data_up,
            )
            input_features_dn = self._construct_input_features_for_ml_model(
                features_list = self.ml_xc_calculator.features_list,
                density_data    = density_data_dn,
            )
            v_xc_ml_up, v_xc_ml_dn = self.ml_xc_calculator.predict_vxc_with_spin(
                features_up       = input_features_up,
                features_dn       = input_features_dn,
                derivative_matrix = self.hamiltonian_builder.ops_builder.derivative_matrix,
                quadrature_nodes  = self.hamiltonian_builder.ops_builder.quadrature_nodes,
            )

        rho = rho_up + rho_dn
        if self.switches.use_hartree:
            v_hartree = self.poisson_solver.solve_hartree(rho)
        else:
            v_hartree = np.zeros_like(rho)

        v_x_up = np.zeros_like(rho_up)
        v_x_dn = np.zeros_like(rho_dn)
        v_c_up = np.zeros_like(rho_up)
        v_c_dn = np.zeros_like(rho_dn)
        de_xc_dtau_up: Optional[np.ndarray] = None
        de_xc_dtau_dn: Optional[np.ndarray] = None

        if self.xc_calculator is not None:
            xc_potential_data_up, xc_potential_data_dn = self.xc_calculator.compute_xc_with_spin(
                density_data_up, density_data_dn
            )
            v_x_up, v_x_dn = xc_potential_data_up.v_x, xc_potential_data_dn.v_x
            v_c_up, v_c_dn = xc_potential_data_up.v_c, xc_potential_data_dn.v_c
            de_xc_dtau_up = xc_potential_data_up.de_xc_dtau
            de_xc_dtau_dn = xc_potential_data_dn.de_xc_dtau

        occ_eigenvalues_up_list: List[np.ndarray] = []
        occ_eigenvectors_up_list: List[np.ndarray] = []

        occ_eigenvalues_dn_list: List[np.ndarray] = []
        occ_eigenvectors_dn_list: List[np.ndarray] = []

        if H_hf_exchange_dict_by_l_up is not None:
            self.hamiltonian_builder.set_hf_exchange_matrices(H_hf_exchange_dict_by_l_up)

        use_symmetrize = (self.switches.xc_functional != "Schrodinger")

        for l in self.occupation_info.unique_l_values:
            H_l_up = self.hamiltonian_builder.build_for_l_channel(
                l                = l,
                v_hartree        = v_hartree,
                v_x              = v_x_up,
                v_c              = v_c_up,
                switches         = self.switches,
                v_x_oep          = v_x_oep_up,
                v_c_oep          = v_c_oep_up,
                v_xc_ml          = v_xc_ml_up,
                de_xc_dtau       = de_xc_dtau_up,
                symmetrize       = use_symmetrize,
                exclude_boundary = True,
            )
            n_states = self.occupation_info.n_states_for_l(l)

            if self.switches.use_oep:
                if use_symmetrize:
                    full_eigvals_l_up, full_eigvecs_l_up = self.eigensolver.solve_full(H_l_up)
                else:
                    S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                    full_eigvals_l_up, full_eigvecs_l_up = self.eigensolver.solve_generalized_full(H_l_up, S)
                occ_eigvals_l_up = full_eigvals_l_up[:n_states]
                occ_eigvecs_l_up = full_eigvecs_l_up[:, :n_states]
            else:
                if use_symmetrize:
                    occ_eigvals_l_up, occ_eigvecs_l_up = self.eigensolver.solve_lowest(H_l_up, n_states)
                else:
                    S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                    occ_eigvals_l_up, occ_eigvecs_l_up = self.eigensolver.solve_generalized_lowest(H_l_up, S, n_states)

            occ_eigenvalues_up_list.append(occ_eigvals_l_up)
            occ_eigenvectors_up_list.append(occ_eigvecs_l_up)

        if H_hf_exchange_dict_by_l_dn is not None:
            self.hamiltonian_builder.set_hf_exchange_matrices(H_hf_exchange_dict_by_l_dn)

        for l in self.occupation_info.unique_l_values:
            H_l_dn = self.hamiltonian_builder.build_for_l_channel(
                l                = l,
                v_hartree        = v_hartree,
                v_x              = v_x_dn,
                v_c              = v_c_dn,
                switches         = self.switches,
                v_x_oep          = v_x_oep_dn,
                v_c_oep          = v_c_oep_dn,
                v_xc_ml          = v_xc_ml_dn,
                de_xc_dtau       = de_xc_dtau_dn,
                symmetrize       = use_symmetrize,
                exclude_boundary = True,
            )
            n_states = self.occupation_info.n_states_for_l(l)

            if self.switches.use_oep:
                if use_symmetrize:
                    full_eigvals_l_dn, full_eigvecs_l_dn = self.eigensolver.solve_full(H_l_dn)
                else:
                    S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                    full_eigvals_l_dn, full_eigvecs_l_dn = self.eigensolver.solve_generalized_full(H_l_dn, S)
                occ_eigvals_l_dn = full_eigvals_l_dn[:n_states]
                occ_eigvecs_l_dn = full_eigvecs_l_dn[:, :n_states]
            else:
                if use_symmetrize:
                    occ_eigvals_l_dn, occ_eigvecs_l_dn = self.eigensolver.solve_lowest(H_l_dn, n_states)
                else:
                    S = self.hamiltonian_builder.ops_builder.get_S(exclude_boundary=True)
                    occ_eigvals_l_dn, occ_eigvecs_l_dn = self.eigensolver.solve_generalized_lowest(H_l_dn, S, n_states)

            occ_eigenvalues_dn_list.append(occ_eigvals_l_dn)
            occ_eigenvectors_dn_list.append(occ_eigvecs_l_dn)

        occ_eigenvalues_up, occ_eigenvectors_up = self._reorder_eigenstates_by_occupation(
            occ_eigenvalues_up_list, occ_eigenvectors_up_list
        )
        occ_eigenvalues_dn, occ_eigenvectors_dn = self._reorder_eigenstates_by_occupation(
            occ_eigenvalues_dn_list, occ_eigenvectors_dn_list
        )

        occ_orbitals_up = self.hamiltonian_builder.interpolate_eigenvectors_to_quadrature(
            eigenvectors = occ_eigenvectors_up,
            symmetrize   = use_symmetrize,
            pad_width    = 1,
        )
        occ_orbitals_dn = self.hamiltonian_builder.interpolate_eigenvectors_to_quadrature(
            eigenvectors = occ_eigenvectors_dn,
            symmetrize   = use_symmetrize,
            pad_width    = 1,
        )

        return (
            occ_eigenvalues_up,
            occ_orbitals_up,
            occ_eigenvalues_dn,
            occ_orbitals_dn,
            v_xc_ml_up,
            v_xc_ml_dn,
        )
