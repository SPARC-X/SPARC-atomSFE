"""
Optimized Effective Potential (OEP) exchange and correlation potentials.

OEPCalculator builds the static response kernel chi_0 and the exchange driving term
from a converged Kohn-Sham spectrum, then solves the OEP equation for the local
exchange potential.  With RPA correlation enabled it also assembles the correlation
driving term and returns the correlation potential as the difference between the
exchange-correlation and exchange solutions.

Both the kernel and the exchange driving term contract the same orbital Green's
function, so they are produced in one loop over occupied orbitals.

The RPA correlation driving term comes from RPACorrelation and arrives with all its
factors already applied.
"""

from __future__ import annotations

import scipy
from scipy.linalg import LinAlgWarning
import numpy as np

import warnings
from typing import Tuple, List, Optional

from .hf import HartreeFockExchange
from .rpa import RPACorrelation, ValidRadialCoulombKernelType
from ..utils.occupation_states import OccupationInfo
from ..mesh.operators import RadialOperatorsBuilder

# Pin v_x for r >= this radius (Bohr). Solver default for oep_boundary_radius.
DEFAULT_EXX_TAIL_R_C = 9.0

# Error messages
USE_RPA_CORRELATION_NOT_BOOL_ERROR = \
    "Parameter use_rpa_correlation must be a bool, get type {} instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_NONE_ERROR = \
    "Parameter frequency_quadrature_point_number must be not None, get None instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR = \
    "Parameter frequency_quadrature_point_number must be an integer, get type {} instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "Parameter frequency_quadrature_point_number must be greater than 0, get {} instead."
OPS_BUILDER_NOT_RADIAL_OPERATORS_BUILDER_ERROR = \
    "Parameter ops_builder must be a RadialOperatorsBuilder instance, get type {} instead."
OPS_BUILDER_OEP_NOT_RADIAL_OPERATORS_BUILDER_ERROR = \
    "Parameter ops_builder_oep must be a RadialOperatorsBuilder instance, get type {} instead."
OPS_BUILDERS_NOT_CONSISTENT_AT_QUADRATURE_NODES_ERROR = \
    "Parameter ops_builder.quadrature_nodes must be equal to ops_builder_oep.quadrature_nodes, please check the grid data and the operators builder."
OPS_BUILDERS_NOT_CONSISTENT_AT_QUADRATURE_WEIGHTS_ERROR = \
    "Parameter ops_builder.quadrature_weights must be equal to ops_builder_oep.quadrature_weights, please check the grid data and the operators builder."
OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR = \
    "Parameter occupation_info must be a OccupationInfo instance, get type {} instead."
FULL_EIGEN_ENERGIES_NOT_NUMPY_ARRAY_ERROR = \
    "Parameter full_eigen_energies must be a numpy array, get type {} instead."
FULL_ORBITALS_NOT_NUMPY_ARRAY_ERROR = \
    "Parameter full_orbitals must be a numpy array, get type {} instead."
FULL_L_TERMS_NOT_NUMPY_ARRAY_ERROR = \
    "Parameter full_l_terms must be a numpy array, get type {} instead."
FULL_EIGEN_ENERGIES_NOT_1D_ARRAY_ERROR = \
    "Parameter full_eigen_energies must be a 1D array, get ndim={}."
FULL_ORBITALS_NOT_2D_ARRAY_ERROR = \
    "Parameter full_orbitals must be a 2D array, get ndim={}."
FULL_L_TERMS_NOT_1D_ARRAY_ERROR = \
    "Parameter full_l_terms must be a 1D array, get ndim={}."
FULL_EIGEN_ENERGIES_AND_ORBITALS_SHAPE_ERROR = \
    "Parameter full_eigen_energies.shape[0] must equal full_orbitals.shape[1], get {} and {} instead."
FULL_EIGEN_ENERGIES_AND_L_TERMS_SHAPE_ERROR = \
    "Parameter full_eigen_energies.shape[0] must equal full_l_terms.shape[0], get {} and {} instead."
FULL_ORBITALS_AND_L_TERMS_SHAPE_ERROR = \
    "Parameter full_orbitals.shape[1] must equal full_l_terms.shape[0], get {} and {} instead."
FULL_L_TERMS_NON_NEGATIVE_ERROR = \
    "Parameter full_l_terms must be non-negative, get {} instead of all non-negative values."
INVALID_EX_TAG_ERROR = \
    "Parameter 'ex_tag' must be 'exchange', 'correlation', or 'exchange_correlation', get {} instead."
R_NODES_AND_COEFFICIENT_DIMENSION_MISMATCH_ERROR = \
    "Parameter 'r_nodes' length must equal 'coefficient' length, get {} and {} instead."
ANGULAR_MOMENTUM_CUTOFF_NOT_INTEGER_ERROR = \
    "Parameter angular_momentum_cutoff must be a non-negative integer, get {} instead."
ANGULAR_MOMENTUM_CUTOFF_NOT_CONSISTENT_WITH_SPECTRUM_ERROR = \
    "Parameter angular_momentum_cutoff is {}, but the supplied spectrum spans l up to {}. The spectrum must be rebuilt whenever the cutoff changes."
OEP_BOUNDARY_RADIUS_NOT_FLOAT_ERROR = \
    "Parameter oep_boundary_radius must be a float, get type {} instead."
OEP_BOUNDARY_RADIUS_NOT_GREATER_THAN_0_ERROR = \
    "Parameter oep_boundary_radius must be greater than 0, get {} instead."
NO_OCCUPIED_FRONTIER_SUBSHELL_ERROR = \
    "Cannot select a frontier radial subshell: every occupation is non-positive."

# WARNING Messages
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_NONE_WHEN_RPA_CORRELATION_IS_NOT_USED_WARNING = \
    "WARNING: parameter 'frequency_quadrature_point_number' is not None when RPA correlation is not used, so it will be ignored"



class OEPCalculator(HartreeFockExchange, RPACorrelation):

    """Prepare and build OEP exchange/correlation potentials from eigenstates."""

    def __init__(
        self,
        ops_builder                       : RadialOperatorsBuilder,
        ops_builder_dense                 : RadialOperatorsBuilder,
        ops_builder_oep                   : RadialOperatorsBuilder,
        occupation_info                   : OccupationInfo,
        use_rpa_correlation               : bool,
        frequency_quadrature_point_number : Optional[int] = None,  # parameters for RPA correlation potential
        angular_momentum_cutoff           : Optional[int] = None,  # highest l channel present in the supplied spectrum
        radial_coulomb_kernel_apply       : ValidRadialCoulombKernelType = "differential_equation",
        oep_boundary_radius               : Optional[float] = None,
    ):

        """
        Parameters
        ----------
        ops_builder : RadialOperatorsBuilder
            Operators builder for the standard grid
        ops_builder_oep : RadialOperatorsBuilder
            Operators builder for the OEP grid
        occupation_info : OccupationInfo
            Occupation information
        use_rpa_correlation : bool
            Whether to use RPA correlation potential
            If True, use RPA correlation potential, otherwise return zero correlation potential
        radial_coulomb_kernel_apply : 'differential_equation' (default) or 'direct_integration'
            Forwarded to RPACorrelation.  'differential_equation' solves the radial Poisson
            equation in the FE basis; 'direct_integration' uses the analytic multipole
            kernel, which needs a much higher radial quadrature order, growing with Z.
            Same labels as ExchangeMethod in hf.py.
        oep_boundary_radius : float, optional
            Radius (Bohr) beyond which the OEP potential is replaced by
            ``-f_h/r``. ``f_h`` is the occupation of the occupied subshell
            with the largest eigenvalue, divided by ``2(2l_h+1)``.
            Defaults to ``DEFAULT_EXX_TAIL_R_C`` (9 Bohr).
        """
        assert isinstance(ops_builder, RadialOperatorsBuilder), \
            OPS_BUILDER_NOT_RADIAL_OPERATORS_BUILDER_ERROR.format(type(ops_builder))
        assert isinstance(ops_builder_oep, RadialOperatorsBuilder), \
            OPS_BUILDER_OEP_NOT_RADIAL_OPERATORS_BUILDER_ERROR.format(type(ops_builder_oep))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        assert isinstance(use_rpa_correlation, bool), \
            USE_RPA_CORRELATION_NOT_BOOL_ERROR.format(type(use_rpa_correlation))

        # check if the two ops_builders are consistent at quadrature nodes
        self._check_ops_builder_consistency_at_quadrature_nodes(ops_builder, ops_builder_oep)

        # initialize the Hartree-Fock exchange class
        HartreeFockExchange.__init__(
            self,
            ops_builder       = ops_builder,
            ops_builder_dense = ops_builder_dense,
            occupation_info   = occupation_info,
        )

        self.ops_builder_oep = ops_builder_oep
        self.physical_nodes  = ops_builder.physical_nodes

        # Some dimension information
        self.n_quad     : int = len(self.quadrature_nodes)
        self.n_interior : int = len(self.physical_nodes) - 2

        # Occupation information
        self.occupations  : np.ndarray = self.occupation_info.occupations
        self.occ_l_values : np.ndarray = self.occupation_info.l_values
        self.occ_n_values : np.ndarray = self.occupation_info.n_values

        if oep_boundary_radius is None:
            oep_boundary_radius = DEFAULT_EXX_TAIL_R_C
        assert isinstance(oep_boundary_radius, (int, float, np.integer, np.floating)), \
            OEP_BOUNDARY_RADIUS_NOT_FLOAT_ERROR.format(type(oep_boundary_radius))
        self.oep_boundary_radius = float(oep_boundary_radius)
        assert self.oep_boundary_radius > 0.0, \
            OEP_BOUNDARY_RADIUS_NOT_GREATER_THAN_0_ERROR.format(self.oep_boundary_radius)

        # Ill_conditioned warning
        self.ill_conditioned_warning_caught_times_for_exchange : int = 0
        self.ill_conditioned_warning_caught_times_for_correlation : int = 0
        self.rcond_list_for_exchange : List[float] = []
        self.rcond_list_for_correlation : List[float] = []

        # Parameters for RPA correlation potential
        self.use_rpa_correlation = use_rpa_correlation

        if use_rpa_correlation:
            # check frequency quadrature point number
            assert frequency_quadrature_point_number is not None, \
                FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_NONE_ERROR.format(frequency_quadrature_point_number)
            assert isinstance(frequency_quadrature_point_number, int), \
                FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(frequency_quadrature_point_number))
            assert frequency_quadrature_point_number > 0, \
                FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(frequency_quadrature_point_number)

            # Not used to bound any loop -- the coupling range is derived from the
            # spectrum itself.  Kept only to cross-check, once the spectrum arrives,
            # that it really spans the channels the caller diagonalized.
            assert isinstance(angular_momentum_cutoff, int) and angular_momentum_cutoff >= 0, \
                ANGULAR_MOMENTUM_CUTOFF_NOT_INTEGER_ERROR.format(angular_momentum_cutoff)
            self.angular_momentum_cutoff = angular_momentum_cutoff

            # initialize the RPA correlation class
            RPACorrelation.__init__(
                self,
                ops_builder                       = ops_builder,
                occupation_info                   = occupation_info,
                frequency_quadrature_point_number = frequency_quadrature_point_number,
                radial_coulomb_kernel_apply       = radial_coulomb_kernel_apply,
            )
            # RPACorrelation.__init__ rebinds self.ops_builder; restore both builders so
            # the exchange paths are unaffected
            self.ops_builder       = ops_builder
            self.ops_builder_dense = ops_builder_dense
        else:
            if frequency_quadrature_point_number is not None:
                print(FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_NONE_WHEN_RPA_CORRELATION_IS_NOT_USED_WARNING)


    def _check_ops_builder_consistency_at_quadrature_nodes(
        self,
        ops_builder    : RadialOperatorsBuilder,
        ops_builder_oep: RadialOperatorsBuilder
    ) -> None:
        assert np.allclose(ops_builder.quadrature_nodes, ops_builder_oep.quadrature_nodes), \
            OPS_BUILDERS_NOT_CONSISTENT_AT_QUADRATURE_NODES_ERROR
        assert np.allclose(ops_builder.quadrature_weights, ops_builder_oep.quadrature_weights), \
            OPS_BUILDERS_NOT_CONSISTENT_AT_QUADRATURE_WEIGHTS_ERROR

        # skip checking for now, will implement other consistency checks later (if needed)
        # raise NotImplementedError("Other consistency checks will be implemented later.")


    def reset(self, ):
        """
        Reset the OEP calculator
        """
        # reset the warning caught time
        self.ill_conditioned_warning_caught_times_for_exchange = 0
        self.ill_conditioned_warning_caught_times_for_correlation = 0
        # Reset the rcond list for exchange and correlation
        self.rcond_list_for_exchange.clear()
        self.rcond_list_for_correlation.clear()


    def compute_oep_potentials(
        self,
        full_eigen_energies    : np.ndarray,
        full_orbitals          : np.ndarray,
        full_l_terms           : np.ndarray,
        enable_parallelization : Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute OEP potentials from full orbitals and eigenvalues.

        Parameters
        ----------
        full_eigen_energies : np.ndarray
            Full eigenvalues of the system, shape (n_total_orbitals,)
        full_orbitals : np.ndarray
            Full orbitals of the system, shape (n_grid, n_total_orbitals)
        full_l_terms : np.ndarray
            Specify the l value of each orbital, shape (n_total_orbitals,)
        enable_parallelization : bool
            Flag for parallelization of RPA calculations

        Returns
        -------
        v_x_oep : np.ndarray
            OEP exchange potential, shape (n_grid,)
        v_c_oep : np.ndarray
            OEP correlation potential, shape (n_grid,)
        """

        # Type check for required fields
        self._validate_full_spectrum_inputs(full_eigen_energies, full_orbitals, full_l_terms)

        # normalise: the signature defaults to None, but the RPA path requires a bool
        enable_parallelization = bool(enable_parallelization)

        # Get occupation information
        occ_orbitals = full_orbitals[:, :len(self.occ_l_values)]

        # get the global interpolation matrix
        global_interpolation_matrix = self.ops_builder_oep.global_interpolation_matrix


        ### =========================================== ###
        ###  Part 1: Compute OEP exchange potential     ###
        ### =========================================== ###

        # Compute exact exchange potentials
        exact_exchange_potentials = self.compute_exchange_potentials(occ_orbitals)

        # Compute OEP exchange kernel and the exchange driving term
        chi_0_kernel, exchange_driving_term = \
            self._compute_oep_kernel_and_exchange_driving_term(
                full_eigen_energies = full_eigen_energies,
                full_orbitals       = full_orbitals,
                full_l_terms        = full_l_terms,
                exchange_potentials = exact_exchange_potentials
            )


        # Convert chi_0_kernel to sparser grid,
        #   Note: this matrix is shared while computing the RPA correlation potential
        chi_0_kernel_sparser_grid = \
            self.convert_chi_0_kernel_to_sparser_grid(
                chi_0_kernel                = chi_0_kernel,
                quadrature_weights          = self.quadrature_weights,
                global_interpolation_matrix = global_interpolation_matrix,
            )


        # Convert exchange_driving_term to sparser grid
        exchange_driving_term_sparser_grid = \
            self.convert_driving_term_to_sparser_grid(
                driving_term                = exchange_driving_term,
                quadrature_weights          = self.quadrature_weights,
                global_interpolation_matrix = global_interpolation_matrix,
            )

        # solve for the OEP coefficient
        oep_coefficient = self.solve_oep_coefficients(
            chi_0_kernel = chi_0_kernel_sparser_grid,
            driving_term = exchange_driving_term_sparser_grid,
            ex_tag       = 'exchange'
        )


        # Apply -f_h/r boundary condition for r >= oep_boundary_radius
        r_oep_nodes         = self.ops_builder_oep.physical_nodes
        r_cutoff            = self.oep_boundary_radius
        tail_is_replaceable = bool(np.any(r_oep_nodes >= r_cutoff))
        occ_energies        = full_eigen_energies[:len(self.occ_l_values)]
        f_h                 = self._frontier_occupation_fraction(occ_energies)

        oep_coefficient = self._apply_minus_one_over_r_boundary_condition(
            coefficient = oep_coefficient,
            r_nodes     = r_oep_nodes,
            r_cutoff    = r_cutoff,
            prefactor   = f_h,
        )

        # compute the OEP exchange potential
        v_x_oep = global_interpolation_matrix @ oep_coefficient

        # Domain shorter than r_cutoff: no node was replaced above, so the additive
        # constant left free by chi_0 is still unfixed.  Pin it by shifting the potential
        # so the outermost quadrature point sits on -f_h/r.
        if not tail_is_replaceable:
            v_x_oep = v_x_oep - v_x_oep[-1] - f_h / self.quadrature_nodes[-1]


        ### =========================================== ###
        ###  Part 2: Compute OEP correlation potential  ###
        ### =========================================== ###

        # compute RPA correlation potential, if needed
        if not self.use_rpa_correlation:
            v_c_oep = np.zeros_like(v_x_oep)
        else:
            # the coupling range is derived from full_l_terms, so a stale spectrum would
            # silently shrink it
            assert int(np.max(full_l_terms)) == self.angular_momentum_cutoff, \
                ANGULAR_MOMENTUM_CUTOFF_NOT_CONSISTENT_WITH_SPECTRUM_ERROR.format(
                    self.angular_momentum_cutoff, int(np.max(full_l_terms)))

            # returns the assembled Q1c + Q2c with all factors applied; nothing further
            # is applied here
            rpa_correlation_driving_term = self.compute_rpa_correlation_driving_term(
                full_eigen_energies    = full_eigen_energies,
                full_orbitals          = full_orbitals,
                full_l_terms           = full_l_terms,
                enable_parallelization = enable_parallelization,
            )

            # Convert RPA correlation driving term to sparser grid
            rpa_correlation_driving_term_sparser_grid = \
                self.convert_driving_term_to_sparser_grid(
                    driving_term                = rpa_correlation_driving_term,
                    quadrature_weights          = self.quadrature_weights,
                    global_interpolation_matrix = global_interpolation_matrix
                )

            # Solve for the HF exchange + RPA correlation coefficient
            #    This is to ensure the correct long-range behavior of the OEP potential
            hf_exchange_plus_rpa_correlation_coefficient = self.solve_oep_coefficients(
                chi_0_kernel = chi_0_kernel_sparser_grid,
                driving_term = rpa_correlation_driving_term_sparser_grid + exchange_driving_term_sparser_grid,
                ex_tag       = 'exchange_correlation'
            )

            # Apply -f_h/r boundary condition for r >= oep_boundary_radius
            hf_exchange_plus_rpa_correlation_coefficient = self._apply_minus_one_over_r_boundary_condition(
                coefficient = hf_exchange_plus_rpa_correlation_coefficient,
                r_nodes     = r_oep_nodes,
                r_cutoff    = r_cutoff,
                prefactor   = f_h,
            )

            # Compute the HF exchange + RPA correlation potential
            v_xc_oep = global_interpolation_matrix @ hf_exchange_plus_rpa_correlation_coefficient

            # same fix as for the exchange potential; pinning both to -f_h/r at the
            # outermost point makes the correlation potential vanish there
            if not tail_is_replaceable:
                v_xc_oep = v_xc_oep - v_xc_oep[-1] - f_h / self.quadrature_nodes[-1]

            # Compute the RPA correlation potential
            v_c_oep = v_xc_oep - v_x_oep

        return v_x_oep, v_c_oep


    def compute_oep_energy_densities(
        self,
        full_eigen_energies    : np.ndarray,
        full_orbitals          : np.ndarray,
        full_l_terms           : np.ndarray,
        enable_parallelization : Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute OEP potentials from full orbitals and eigenvalues.

        Parameters
        ----------
        full_eigen_energies : np.ndarray
            Full eigenvalues of the system, shape (n_total_orbitals,)
        full_orbitals : np.ndarray
            Full orbitals of the system, shape (n_grid, n_total_orbitals)
        full_l_terms : np.ndarray
            Specify the l value of each orbital, shape (n_total_orbitals,)
        enable_parallelization : bool
            Flag for parallelization of RPA calculations

        Returns
        -------
        e_x_oep : np.ndarray
            OEP exchange energy density, shape (n_grid,)
        e_c_oep : np.ndarray
            OEP correlation energy density, shape (n_grid,)

        Note: compute_correlation_energy_density returns (density, energy); the energy is
        a free by-product of the same eigendecomposition and is discarded here.
        """

        # Type check for required fields
        self._validate_full_spectrum_inputs(full_eigen_energies, full_orbitals, full_l_terms)

        enable_parallelization = bool(enable_parallelization)

        # Get occupation information
        occ_orbitals = full_orbitals[:, :len(self.occ_l_values)]

        # Compute OEP exchange energy density
        e_x_oep = self.compute_exchange_energy_density(occ_orbitals)


        # Compute RPA correlation energy density, if needed
        if not self.use_rpa_correlation:
            e_c_oep = np.zeros_like(e_x_oep)
        else:
            # the coupling range is derived from full_l_terms, so a stale spectrum would
            # silently shrink it
            assert int(np.max(full_l_terms)) == self.angular_momentum_cutoff, \
                ANGULAR_MOMENTUM_CUTOFF_NOT_CONSISTENT_WITH_SPECTRUM_ERROR.format(
                    self.angular_momentum_cutoff, int(np.max(full_l_terms)))

            # returns (density, energy); the energy is discarded here
            e_c_oep, _rpa_correlation_energy = self.compute_correlation_energy_density(
                full_eigen_energies    = full_eigen_energies,
                full_orbitals          = full_orbitals,
                full_l_terms           = full_l_terms,
                enable_parallelization = enable_parallelization,
            )

        return e_x_oep, e_c_oep




    def solve_oep_coefficients(
        self,
        chi_0_kernel : np.ndarray,
        driving_term : np.ndarray,
        ex_tag       : str,
    ) -> np.ndarray:
        """
        Solve the OEP coefficients for exchange or correlation potential

        Parameters
        ----------
        chi_0_kernel : np.ndarray
            Chi_0 kernel, shape (n_quad, n_quad)
        driving_term : np.ndarray
            Driving term, shape (n_quad,)
        ex_tag : str
            Tag for the potential, must be 'exchange', 'correlation', or 'exchange_correlation'
            This parameter is used to record the warning caught times and rcond_list for exchange or
            correlation potential while solving the OEP coefficients.
            If 'exchange_correlation', both exchange and correlation counters will be updated.

        Returns
        -------
        oep_coefficient : np.ndarray
            OEP coefficients, shape (n_quad,)
        """
        assert ex_tag in ['exchange', 'correlation', 'exchange_correlation'], \
            INVALID_EX_TAG_ERROR.format(ex_tag)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", LinAlgWarning)
            oep_coefficient = scipy.linalg.solve(chi_0_kernel, driving_term)

        if caught:
            cond = np.linalg.cond(chi_0_kernel)
            rcond = 1.0 / cond if cond != 0 else np.inf
            if ex_tag == 'exchange':
                self.ill_conditioned_warning_caught_times_for_exchange += 1
                self.rcond_list_for_exchange.append(rcond)
            elif ex_tag == 'correlation':
                self.ill_conditioned_warning_caught_times_for_correlation += 1
                self.rcond_list_for_correlation.append(rcond)
            elif ex_tag == 'exchange_correlation':
                self.ill_conditioned_warning_caught_times_for_exchange += 1
                self.rcond_list_for_exchange.append(rcond)
                self.ill_conditioned_warning_caught_times_for_correlation += 1
                self.rcond_list_for_correlation.append(rcond)
            else:
                raise ValueError(INVALID_EX_TAG_ERROR.format(ex_tag))

        return oep_coefficient


    def _frontier_occupation_fraction(self, occ_energies: np.ndarray) -> float:
        """Occupation of the highest occupied subshell, divided by ``2(2l_h+1)``."""
        n_occ = min(int(self.occupations.size), int(np.asarray(occ_energies).size))
        g = np.asarray(self.occupations[:n_occ], dtype=float)
        occupied = g > 0.0
        assert bool(np.any(occupied)), NO_OCCUPIED_FRONTIER_SUBSHELL_ERROR
        energies = np.asarray(occ_energies[:n_occ], dtype=float)
        energies_masked = np.where(occupied, energies, -np.inf)
        h = int(np.argmax(energies_masked))
        g_h = float(self.occupations[h])
        l_h = int(self.occ_l_values[h])
        return g_h / (2.0 * (2 * l_h + 1))


    @staticmethod
    def _apply_minus_one_over_r_boundary_condition(
        coefficient : np.ndarray,
        r_nodes     : np.ndarray,
        r_cutoff    : float = 9.0,
        prefactor   : float = 1.0,
    ) -> np.ndarray:
        """
        Apply ``-prefactor/r`` boundary condition for r >= r_cutoff Bohr.

        The default ``prefactor`` is the frontier-subshell occupation fraction
        ``f_h``. Closed shells have ``f_h = 1``, which recovers ``-1/r``.

        Parameters
        ----------
        coefficient : np.ndarray
            OEP coefficients to be modified, shape (n_nodes,)
        r_nodes : np.ndarray
            Radial grid nodes, shape (n_nodes,)
        r_cutoff : float
            Cutoff radius in Bohr, default is 9.0
            For r >= r_cutoff, coefficients are set to ``-prefactor/r``
        prefactor : float
            Monopole coefficient ``f_h``. Defaults to 1.0.

        Returns
        -------
        coefficient : np.ndarray
            Modified coefficients with boundary condition applied
        """
        assert len(r_nodes) == len(coefficient), \
            R_NODES_AND_COEFFICIENT_DIMENSION_MISMATCH_ERROR.format(len(r_nodes), len(coefficient))

        r_geq_cutoff_indices = np.argwhere(r_nodes >= r_cutoff)[:, 0]
        if len(r_geq_cutoff_indices) > 0:
            coefficient[r_geq_cutoff_indices] = -prefactor / r_nodes[r_geq_cutoff_indices]
            if r_geq_cutoff_indices[0] > 0:
                coefficient[:r_geq_cutoff_indices[0]] = \
                    coefficient[:r_geq_cutoff_indices[0]] + \
                    (-prefactor / r_nodes[r_geq_cutoff_indices[0] - 1] - coefficient[r_geq_cutoff_indices[0] - 1])

        return coefficient


    @staticmethod
    def _validate_full_spectrum_inputs(
        full_eigen_energies : np.ndarray,
        full_orbitals       : np.ndarray,
        full_l_terms        : np.ndarray
    ) -> None:
        """
        Validate the inputs of full spectrum
        """
        assert isinstance(full_eigen_energies, np.ndarray), \
            FULL_EIGEN_ENERGIES_NOT_NUMPY_ARRAY_ERROR.format(type(full_eigen_energies))
        assert isinstance(full_orbitals, np.ndarray), \
            FULL_ORBITALS_NOT_NUMPY_ARRAY_ERROR.format(type(full_orbitals))
        assert isinstance(full_l_terms, np.ndarray), \
            FULL_L_TERMS_NOT_NUMPY_ARRAY_ERROR.format(type(full_l_terms))
        assert full_eigen_energies.ndim == 1, \
            FULL_EIGEN_ENERGIES_NOT_1D_ARRAY_ERROR.format(full_eigen_energies.ndim)
        assert full_orbitals.ndim == 2, \
            FULL_ORBITALS_NOT_2D_ARRAY_ERROR.format(full_orbitals.ndim)
        assert full_l_terms.ndim == 1, \
            FULL_L_TERMS_NOT_1D_ARRAY_ERROR.format(full_l_terms.ndim)
        assert full_eigen_energies.shape[0] == full_orbitals.shape[1], \
            FULL_EIGEN_ENERGIES_AND_ORBITALS_SHAPE_ERROR.format(full_eigen_energies.shape[0], full_orbitals.shape[1])
        assert full_eigen_energies.shape[0] == full_l_terms.shape[0], \
            FULL_EIGEN_ENERGIES_AND_L_TERMS_SHAPE_ERROR.format(full_eigen_energies.shape[0], full_l_terms.shape[0])
        assert full_orbitals.shape[1] == full_l_terms.shape[0], \
            FULL_ORBITALS_AND_L_TERMS_SHAPE_ERROR.format(full_orbitals.shape[1], full_l_terms.shape[0])
        assert np.all(full_l_terms >= 0), \
            FULL_L_TERMS_NON_NEGATIVE_ERROR.format(full_l_terms)


    @staticmethod
    def convert_chi_0_kernel_to_sparser_grid(
        chi_0_kernel                : np.ndarray,
        quadrature_weights          : np.ndarray,
        global_interpolation_matrix : np.ndarray,
    ) -> np.ndarray:
        """
        Convert chi_0_kernel and driving_term to sparser grid
        """
        # convert chi_0_kernel and exchange_driving_term to sparser grid
        chi_0_kernel_sparser_grid = np.einsum('i,ij,il,lk,l->jk',
            quadrature_weights,
            global_interpolation_matrix,
            chi_0_kernel,
            global_interpolation_matrix,
            quadrature_weights,
            optimize=True
        )

        return chi_0_kernel_sparser_grid


    @staticmethod
    def convert_driving_term_to_sparser_grid(
        driving_term                : np.ndarray,
        quadrature_weights          : np.ndarray,
        global_interpolation_matrix : np.ndarray,
    ) -> np.ndarray:
        """
        Convert driving term to sparser grid
        """
        driving_term_sparser_grid = np.einsum('i, ij, i->j',
            quadrature_weights,
            global_interpolation_matrix,
            driving_term,
            optimize=True
        )
        return driving_term_sparser_grid


    def _compute_oep_kernel_and_exchange_driving_term(
        self,
        full_eigen_energies : np.ndarray,
        full_orbitals       : np.ndarray,
        full_l_terms        : np.ndarray,
        exchange_potentials : np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the OEP static response kernel chi_0 and the exchange driving term.

        Both contract the same orbital Green's function G_nl, so they are built in one
        loop over occupied orbitals.

        Note: the partner sum inside G_nl runs over the WHOLE l channel, occupied states
        included.  Occupied-occupied pairs cancel only when the two carry equal
        occupation, so dropping them would be wrong for open shells and for fractional
        occupations.
        """
        # get occupied orbitals
        occ_orbitals = full_orbitals[:, :len(self.occ_l_values)]

        # get l channel indices for all orbitals
        l_max = np.max(self.occ_l_values)
        l_channel_orbital_indices = np.zeros((l_max + 1, self.n_interior), dtype=np.int32)
        for l in range(l_max + 1):
            l_channel_orbital_indices[l, :] = np.argwhere(full_l_terms == l)[:,0]

        # compute chi_0_kernel and the exchange driving term
        chi_0_kernel          = np.zeros((self.n_quad, self.n_quad))
        exchange_driving_term = np.zeros(self.n_quad)

        # running 0-based position of each occupied orbital inside its own l channel.
        # Valid because occupied states come first in full_l_terms in the same relative
        # order as occ_l_values, so the k-th occupied state of a channel sits at channel
        # position k.  Derived from position rather than from the principal quantum
        # number, which would be off by the number of removed core states under a
        # pseudopotential.
        n_indices_table = np.zeros(l_max + 1, dtype=np.int32)

        for idx in range(len(self.occ_l_values)):
            # get l and n index
            l_value = int(self.occ_l_values[idx])
            n_value = int(n_indices_table[l_value])
            n_indices_table[l_value] += 1

            # all orbitals of this l channel, occupied partners included
            all_orbitals_in_l_channel = full_orbitals[:, l_channel_orbital_indices[l_value, :]]

            # 1/(eps_i - eps_j) within this l channel, zero on the diagonal.  The identity
            # is added before the reciprocal so the diagonal never divides by zero.
            l_channel_eigenvalues = full_eigen_energies[l_channel_orbital_indices[l_value, :]]
            n_in_channel = len(l_channel_eigenvalues)
            diff_eigenvalues = l_channel_eigenvalues.reshape(-1, 1) - l_channel_eigenvalues.reshape(1, -1) + np.eye(n_in_channel)
            one_over_diff_eigenvalues = 1 / diff_eigenvalues
            one_over_diff_eigenvalues[np.arange(n_in_channel), np.arange(n_in_channel)] = 0
            # green function block; the j == n_value self-term is already zero
            _exchange_green_block = np.einsum('ji,ki,i->jk',
                all_orbitals_in_l_channel,
                all_orbitals_in_l_channel,
                one_over_diff_eigenvalues[n_value, :],
                optimize=True
            )

            # get the orbital and corresponding exchange potential inside this for loop
            orbital            = occ_orbitals[:, idx]
            exchange_potential = exchange_potentials[idx]


            # # update chi_0_kernel
            chi_0_kernel += 2 * np.einsum('k,kj,j->kj',
                orbital,
                _exchange_green_block,
                orbital,
                optimize=True
            ) * self.occupations[idx]

            # update exchange driving term
            exchange_driving_term += 2 * np.einsum('k,kl,l->k',
                orbital,
                _exchange_green_block,
                exchange_potential,
                optimize=True
            ) * self.occupations[idx]

        return chi_0_kernel, exchange_driving_term
