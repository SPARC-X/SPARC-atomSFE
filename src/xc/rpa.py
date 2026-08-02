from __future__ import annotations


import warnings

import numpy as np
from typing import Any, Tuple, List, Dict, Literal

from .hf import CoulombCouplingCalculator
from ..mesh.builder import Quadrature1D
from ..mesh.operators import RadialOperatorsBuilder
from ..utils.occupation_states import OccupationInfo

from contextlib import nullcontext

try:
    # Optional dependency: used to limit BLAS/OpenMP threads during parallel sections
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None  # type: ignore

# Error messages
OPS_BUILDER_NOT_RADIAL_OPERATORS_BUILDER_ERROR = \
    "Parameter 'ops_builder' must be a 'RadialOperatorsBuilder' instance, get type '{}' instead."
OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR = \
    "Parameter 'occupation_info' must be a 'OccupationInfo' instance, get type '{}' instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR = \
    "Parameter 'frequency_quadrature_point_number' must be an integer, get type {} instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "Parameter 'frequency_quadrature_point_number' must be greater than 0, get {} instead."
ANGULAR_MOMENTUM_CUTOFF_NOT_INTEGER_ERROR = \
    "Parameter `angular_momentum_cutoff` must be an integer, get type {} instead."
ANGULAR_MOMENTUM_CUTOFF_NEGATIVE_ERROR = \
    "Parameter `angular_momentum_cutoff` must be non-negative, get {} instead."
FREQUENCY_NOT_FLOAT_ERROR = \
    "Parameter `frequency` must be a float or a scaler numpy array, get type {} instead."
GRID_TYPE_NOT_VALID_ERROR = \
    "Parameter `grid_type` must be one of 'inverse_linear' or 'rational', get {} instead."

ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_ERROR_MESSAGE = \
    "Parameter `angular_momentum_cutoff` must be not None, get None instead."
OCCUPATION_INFO_L_TERMS_NOT_CONSISTENT_WITH_OCCUPATION_INFO_ERROR = \
    "Occupied l terms are not consistent with the occupation information, please check your inputs, get {} instead of {}."
PARENT_CLASS_RPACORRELATION_NOT_INITIALIZED_ERROR = \
    "Parent class `RPACorrelation` is not initialized, please initialize it first."

RPA_DRIVING_TERM_OLD_DEPRECATED_ERROR = (
    "`_compute_rpa_correlation_driving_term_for_single_frequency_old` is deprecated "
    "and no longer used. It is the previous fully-vectorized (high-memory) "
    "implementation kept only for reference; use "
    "`_compute_rpa_correlation_driving_term_for_single_frequency` instead."
)

RPA_CORRELATION_ENERGY_OLD_DEPRECATED_ERROR = (
    "`_compute_correlation_energy_for_single_frequency_old` is deprecated "
    "and no longer used. It is the previous fully-vectorized (high-memory) "
    "implementation kept only for reference; use "
    "`_compute_correlation_energy_for_single_frequency` instead."
)


L_OCC_MAX_NOT_INTEGER_ERROR = \
    "Parameter `l_occ_max` must be an integer, get type {} instead."
L_UNOCC_MAX_NOT_INTEGER_ERROR = \
    "Parameter `l_unocc_max` must be an integer, get type {} instead."
L_COUPLE_MAX_NOT_INTEGER_ERROR = \
    "Parameter `l_couple_max` must be an integer, get type {} instead."
ENABLE_PARALLELIZATION_NOT_BOOL_ERROR = \
    "Parameter `enable_parallelization` must be a bool, get type {} instead."

ValidGridType = Literal["inverse_linear", "rational"]


class RPACorrelation:
    """
    Compute RPA correlation energy from eigenstates.
    """
    def __init__(
        self, 
        ops_builder                       : 'RadialOperatorsBuilder',
        occupation_info                   : 'OccupationInfo',
        frequency_quadrature_point_number : int,
        angular_momentum_cutoff           : int
    ):
        """
        Parameters
        ----------
        ops_builder                       : instance of RadialOperatorsBuilder
            RadialOperatorsBuilder instance
        occupation_info                   : instance of OccupationInfo
            Occupation information
        frequency_quadrature_point_number : int
            Number of frequency quadrature points
        angular_momentum_cutoff           : int
            Maximum angular momentum quantum number to include
        """
        assert isinstance(ops_builder, RadialOperatorsBuilder), \
            OPS_BUILDER_NOT_RADIAL_OPERATORS_BUILDER_ERROR.format(type(ops_builder))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        assert isinstance(frequency_quadrature_point_number, int), \
            FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(frequency_quadrature_point_number))
        assert frequency_quadrature_point_number > 0, \
            FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(frequency_quadrature_point_number)
        assert isinstance(angular_momentum_cutoff, int), \
            ANGULAR_MOMENTUM_CUTOFF_NOT_INTEGER_ERROR.format(type(angular_momentum_cutoff))
        assert angular_momentum_cutoff >= 0, \
            ANGULAR_MOMENTUM_CUTOFF_NEGATIVE_ERROR.format(angular_momentum_cutoff)


        # Extract quadrature data from ops_builder
        self.n_quad             = len(ops_builder.quadrature_nodes)
        self.n_grid             = self.n_quad
        self.quadrature_nodes   = ops_builder.quadrature_nodes
        self.quadrature_weights = ops_builder.quadrature_weights

        # initialize the frequency grid and weights
        self.frequency_quadrature_point_number = frequency_quadrature_point_number
        self.frequency_grid, self.frequency_weights = \
            self._initialize_frequency_grid_and_weights(frequency_quadrature_point_number, "inverse_linear")

        # occupation information
        self.occupation_info : OccupationInfo = occupation_info
        self.occupations  : np.ndarray = occupation_info.occupations
        self.occ_l_values : np.ndarray = occupation_info.l_values
        self.occ_n_values : np.ndarray = occupation_info.n_values

        # angular momentum cutoff
        self.angular_momentum_cutoff = angular_momentum_cutoff


    @classmethod
    def _initialize_frequency_grid_and_weights(cls, n: int, grid_type: ValidGridType) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize frequency grid and weights for RPA correlation energy calculations.
        
        Parameters
        ----------
        n : int
            Number of frequency quadrature points
        grid_type : ValidGridType
            Type of frequency grid

        Returns
        -------
        frequency_grid : np.ndarray
            Frequency grid
        frequency_weights : np.ndarray
            Frequency weights
        """
        assert grid_type in ["inverse_linear", "rational"], \
            GRID_TYPE_NOT_VALID_ERROR.format(grid_type)
        if grid_type == "inverse_linear":
            return cls._initialize_frequency_grid_and_weights_inverse_linear(n)
        else:
            return cls._initialize_frequency_grid_and_weights_rational(n)


    @classmethod
    def _initialize_frequency_grid_and_weights_inverse_linear(cls, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize frequency grid and weights using inverse-linear mapping.
        
        Maps Gauss-Legendre nodes from [0, 1] to [0, ∞) via:
            ω(ξ) = 1/ξ - 1,  with Jacobian: w_ω = w_ξ / (1-ξ)²
        
        Parameters
        ----------
        n : int
            Number of frequency quadrature points
        
        Returns
        -------
        frequency_grid : np.ndarray
            Frequency grid nodes on [0, ∞), shape (n,)
        frequency_weights : np.ndarray
            Corresponding quadrature weights, shape (n,)
        """
        # Get Gauss-Legendre nodes on [0, 1] and transform to [0, ∞)
        reference_nodes, reference_weights = Quadrature1D.gauss_legendre_on_interval(n, 0.0, 1.0)
        nodes   = np.flip(1.0 / reference_nodes - 1.0)  # ω = 1/ξ - 1, flipped for ascending order
        weights = reference_weights / (1 - reference_nodes)**2  # Jacobian factor

        return nodes, weights


    @staticmethod
    def _initialize_frequency_grid_and_weights_rational(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize frequency grid and weights using rational mapping.
        
        Maps Gauss-Legendre nodes from [-1, 1] to [0, ∞) via:
            ω(ξ) = α * (1 + ξ) / (1 - ξ),  with Jacobian: w_ω = w_ξ * 2α / (1 - ξ)²
        
        Reference:
            https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.134.016402/scrpa4_SM.pdf
        
        Parameters
        ----------
        n : int
            Number of frequency quadrature points
        
        Returns
        -------
        frequency_grid : np.ndarray
            Frequency grid nodes on [0, ∞), shape (n,)
        frequency_weights : np.ndarray
            Corresponding quadrature weights, shape (n,)
        """
        frequency_scale = 2.5  # Compression parameter α

        # Get Gauss-Legendre nodes on [-1, 1] and transform to [0, ∞)
        reference_nodes, reference_weights = Quadrature1D.gauss_legendre(n)
        nodes   = frequency_scale * (1 + reference_nodes) / (1 - reference_nodes)  # ω = α(1+ξ)/(1-ξ)
        weights = reference_weights * 2 * frequency_scale / (1 - reference_nodes)**2  # Jacobian factor

        return nodes, weights


    @staticmethod
    def _compute_rpa_correlation_driving_term_for_single_frequency(
        frequency               : float,
        angular_momentum_cutoff : int,
        occupation_info         : OccupationInfo,
        full_eigen_energies     : np.ndarray, 
        full_orbitals           : np.ndarray, 
        full_l_terms            : np.ndarray,
        wigner_symbols_squared  : np.ndarray,
        radial_kernels_dict     : Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RPA correlation driving term (Q1c and Q2c terms) at a given frequency.
        
        This function computes the two components of the RPA correlation driving term
        used in the OEP (Optimized Effective Potential) method:
        - Q1c term: First-order correlation term involving self-energy matrix elements
        - Q2c term: Second-order correlation term involving the Dyson-solved response function
        
        The computation is performed for a single imaginary frequency and includes
        contributions from all angular momentum coupling channels (l_couple).
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - full_q1c_term: Q1c term, shape (n_quad,)
                - full_q2c_term: Q2c term, shape (n_quad,)
        """
        try:
            frequency = float(frequency)
        except ValueError:
            raise ValueError(FREQUENCY_NOT_FLOAT_ERROR.format(type(frequency)))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        
        # get occupation information
        occupations  = occupation_info.occupations  # shape : (occ_orbitals_num,)
        occ_l_values = occupation_info.l_values     # shape : (occ_orbitals_num,)


        # get the number of occupied and unoccupied orbitals
        occ_orbitals_num   = len(occ_l_values)
        total_orbitals_num = len(full_eigen_energies)

        # get the number of quadrature and n_interior points
        n_quad     = radial_kernels_dict[0].shape[0]
        n_interior = len(np.argwhere(full_l_terms == 0)[:, 0])

        # get occupied and unoccupied orbitals and energies
        occ_orbitals   = full_orbitals[:, :occ_orbitals_num]     # shape: (n_grid, total_orbitals_num)
        occ_energies   = full_eigen_energies[:occ_orbitals_num]  # shape: (total_orbitals_num,)
        occ_l_terms    = full_l_terms[:occ_orbitals_num]         # shape: (total_orbitals_num,)
        unocc_orbitals = full_orbitals[:, occ_orbitals_num:]     # shape: (n_grid, unocc_orbitals_num)
        unocc_energies = full_eigen_energies[occ_orbitals_num:]  # shape: (unocc_orbitals_num,)
        unocc_l_terms  = full_l_terms[occ_orbitals_num:]         # shape: (total_orbitals_num,)

        assert np.all(occ_l_terms == occ_l_values), \
            OCCUPATION_INFO_L_TERMS_NOT_CONSISTENT_WITH_OCCUPATION_INFO_ERROR.format(occ_l_terms, occ_l_values)

        ### ================================================ ###
        ###  Part 1: Compute the RPA correlation prefactors  ###
        ### ================================================ ###

        # Occupation / angular factors entering (f_{nl} - f_{n'l'}) and C_{l';ll'}
        #   ov: f_{nl} * (2l' + 1) for occupied nl and virtual n'l'  (f_{n'l'}=0)
        #   shape: (occ_num, unocc_num)
        deg_factors_ov = occupations[:, np.newaxis] * (2 * unocc_l_terms + 1)[np.newaxis, :]
        #   oo: f_{nl}*(2l'+1) - f_{n'l'}*(2l+1) for occupied–occupied pairs
        #   shape: (occ_num, occ_num)
        deg_factors_oo = occupations[:, np.newaxis] * (2 * occ_l_terms + 1)[np.newaxis, :] - \
                         occupations[np.newaxis, :] * (2 * occ_l_terms + 1)[:, np.newaxis]

        # Eigenvalue differences λ_{nl} - λ_{n'l'}
        #   shape: (occ_num, unocc_num)  — occupied nl vs virtual n'l'
        delta_eps_ov = occ_energies[:, np.newaxis] - unocc_energies[np.newaxis, :]
        #   shape: (occ_num, occ_num)    — occupied nl vs occupied n'l'
        delta_eps_oo = occ_energies[:, np.newaxis] - occ_energies[np.newaxis, :]

        # Frequency factor D_{nl,n'l'}(iω) = (λ_{nl} - λ_{n'l'}) / [ω² + (λ_{nl} - λ_{n'l'})²]
        #   shape: (occ_num, unocc_num)
        lorentzian_factors_ov = delta_eps_ov / (delta_eps_ov ** 2 + frequency ** 2)
        #   shape: (occ_num, occ_num)
        lorentzian_factors_oo = delta_eps_oo / (delta_eps_oo ** 2 + frequency ** 2)

        # Frequency derivative of D for Q2c: (Δλ² - ω²) / (Δλ² + ω²)²
        #   arises from ∂χ̃_{0,l''}/∂(iω)
        #   shape: (occ_num, unocc_num)
        frequency_derivative_factors_ov = (delta_eps_ov ** 2 - frequency ** 2) / (delta_eps_ov ** 2 + frequency ** 2) ** 2
        #   shape: (occ_num, occ_num)
        frequency_derivative_factors_oo = (delta_eps_oo ** 2 - frequency ** 2) / (delta_eps_oo ** 2 + frequency ** 2) ** 2


        # Combine occupation/angular and frequency factors (Wigner 3j / C_{l';ll'} applied later)
        #   shape: (occ_num, unocc_num)
        prefactors_q1c_ov = deg_factors_ov * lorentzian_factors_ov
        prefactors_q2c_ov = deg_factors_ov * frequency_derivative_factors_ov
        #   shape: (occ_num, occ_num)
        prefactors_q1c_oo = deg_factors_oo * lorentzian_factors_oo
        prefactors_q2c_oo = deg_factors_oo * frequency_derivative_factors_oo


        # Self-energy prefactors ~ (f_{nl}-f_{n'l'}) * D_{nl,n'l'}(iω)  (without Wigner yet)
        #   shape: (occ_num, unocc_num)
        prefactors_self_energy_ov = deg_factors_ov * delta_eps_ov / (delta_eps_ov ** 2 + frequency ** 2)
        #   shape: (occ_num, occ_num)
        prefactors_self_energy_oo = deg_factors_oo * delta_eps_oo / (delta_eps_oo ** 2 + frequency ** 2)
        #   shape: (occ_num, total_num)  — oo then ov, matching orbital axis of full_orbitals
        prefactors_self_energy_all = np.concatenate(
            [prefactors_self_energy_oo, prefactors_self_energy_ov], axis=1
        )


        ### ================================================== ###
        ###  Part 2: Compute the nonzero Wigner symbols        ###
        ### ================================================== ###

        l_couple_min   = np.min(np.abs(occ_l_values[:, np.newaxis] - full_l_terms[np.newaxis, :])).astype(np.int32)
        l_couple_max   = np.max(       occ_l_values[:, np.newaxis] + full_l_terms[np.newaxis, :] ).astype(np.int32)
        l_couple_range = np.arange(l_couple_min, l_couple_max + 1)


        # Use advanced indexing with broadcasting - one operation instead of triple loop
        #   shape: (occ_num, unocc_num, l_couple_num)
        wigner_symbols_squared_ov = wigner_symbols_squared[
            occ_l_terms   .astype(np.int32)[:, np.newaxis, np.newaxis],  # shape: (occ_orbitals_num, 1, 1)
            unocc_l_terms .astype(np.int32)[np.newaxis, :, np.newaxis],  # shape: (1, unocc_num, 1)
            l_couple_range.astype(np.int32)[np.newaxis, np.newaxis, :],  # shape: (1, 1, l_couple_num)
        ]
        #   shape: (occ_num, occ_num, l_couple_num)
        wigner_symbols_squared_oo = wigner_symbols_squared[
            occ_l_terms   .astype(np.int32)[:, np.newaxis, np.newaxis],  # shape: (occ_num, 1, 1)
            occ_l_terms   .astype(np.int32)[np.newaxis, :, np.newaxis],  # shape: (1, occ_num, 1)
            l_couple_range.astype(np.int32)[np.newaxis, np.newaxis, :],  # shape: (1, 1, l_couple_num)
        ]
        #   shape: (occ_num, total_num, l_couple_num)
        wigner_symbols_squared_all = np.concatenate(
            [wigner_symbols_squared_oo, wigner_symbols_squared_ov], axis=1
        )

        # Active l'' channels: at least one nonzero Wigner symbol in ov or oo.
        active_l_couple_idx_list: List[int] = []
        for l_couple_idx in range(len(l_couple_range)):
            if (
                np.any(wigner_symbols_squared_ov[:, :, l_couple_idx] != 0)
                or np.any(wigner_symbols_squared_oo[:, :, l_couple_idx] != 0)
            ):
                active_l_couple_idx_list.append(l_couple_idx)

        ### ================================================== ###
        ###  Part 3: Solving for the RPA source term           ###
        ### ================================================== ###

        # Notice that the RPA source term all contain 
        # 1. The summation over coupled l index, which is l'' in our CPC paper.
        # 2. The integration over freqency quadrature points, which is ω in our CPC paper 
        #     (which is already decoupled in this function, since this function only takes in a single frequency of float type).
        # Therefore, only need to loop over all the coupled l index here, with non-zero Wigner symbols.
        #     In principle, this part of computation can be parallelized, but for now, we only loop over the coupled l index here.
        #     Note that we already support the parallelization over frequency, parallelization over l index will spawn too many sub-processes.

        #   shape: (total_orbitals_num, n_quad)
        full_self_energy_potential = np.zeros((total_orbitals_num, n_quad))
        #   shape: (n_quad,)
        full_q1c_term = np.zeros(n_quad)
        full_q2c_term = np.zeros(n_quad)

        for l_couple_idx in active_l_couple_idx_list:

            active_l_couple = l_couple_range[l_couple_idx]
            # Nonzero Wigner pairs for this l'': shape (n_nonzero, 2) -> [occ_i, partner_j]
            active_wigner_symbols_indices_ov = np.argwhere(
                wigner_symbols_squared_ov[:, :, l_couple_idx] != 0
            )
            active_wigner_symbols_indices_oo = np.argwhere(
                wigner_symbols_squared_oo[:, :, l_couple_idx] != 0
            )
            occ_valid_indices_ov   = active_wigner_symbols_indices_ov[:, 0]
            unocc_valid_indices_ov = active_wigner_symbols_indices_ov[:, 1]
            occ_valid_indices_oo   = active_wigner_symbols_indices_oo[:, 0]
            occ_partner_indices_oo = active_wigner_symbols_indices_oo[:, 1]
            del active_wigner_symbols_indices_ov, active_wigner_symbols_indices_oo

            # Step 1: Get radial kernel (Coulomb kernel for angular momentum channel L)
            #   shape: (n_quad, n_quad)
            #   R^{(L)}(r_i, r_j) = (1/(2L+1)) * (r_<^L / r_>^{L+1}) * w_i * w_j
            #   where:
            #     - r_< = min(r_i, r_j), r_> = max(r_i, r_j)
            #     - w_i, w_j: quadrature weights at radial points r_i, r_j
            #     - L = active_l_couple: angular momentum coupling channel
            #   This is the radial projection of the Coulomb interaction in channel L
            radial_kernel = radial_kernels_dict[active_l_couple] * (2 * active_l_couple + 1)

            # Step 2:Compute rpa_response_kernel
            #     shape: (n_quad, n_quad)
            # Since this function is supposed to be run with multiple threads, we choose to **disable** 
            #     the vectorization over (occ, unocc) index, in exchange for the storage efficiency.

            #   rpa_response_kernel_ov: occ–virt part of χ̃_{0,l''}(r,r';iω)
            #       χ̃_{0,l''}(r,r';iω)
            #         = 2 Σ_{nl, n'l'} (f_{nl} - f_{n'l'}) C_{l';ll'} D_{nl,n'l'}(iω)
            #             × R̃_{nl}(r) R̃_{n'l'}(r) R̃_{nl}(r') R̃_{n'l'}(r')
            #       with
            #         C_{l';ll'} = [(2l+1)(2l'+1)/(2l''+1)] * (l l' l''; 0 0 0)^2
            #         D_{nl,n'l'}(iω) = (λ_{nl} - λ_{n'l'}) / [ω² + (λ_{nl} - λ_{n'l'})²]
            #       Here nl runs over occupied orbitals, n'l' over virtual (f_{n'l'}=0);
            #       the leading 2 is spin degeneracy. Angular factor (2l''+1) is restored
            #       when forming the full rpa_response_kernel later.

            # Memory-friendly accumulation (OEP-style):
            #   χ₀ += Σ (pref * Wigner) * Φ ⊗ Φ
            #   with Φ_{nl,n'l'}(r) = φ_{nl}(r) φ_{n'l'}(r) built only for one occupied nl at a time.
            #   Equivalent to the old einsum over all pairs, but without storing (n_pairs, n_grid) tables.
            #   Accumulate ov then oo into one buffer: χ̃_{0,l''} = (χ̃^{ov} + χ̃^{oo}) / (2l''+1).

            #   shape: (n_quad, n_quad)
            rpa_response_kernel = np.zeros((n_quad, n_quad))

            # --- occ–virt (spin degeneracy ×2 applied after this loop) ---
            orbital_pair_product = constants = None
            for occ_index in np.unique(occ_valid_indices_ov):
                unocc_partners = unocc_valid_indices_ov[occ_valid_indices_ov == occ_index]
                #   shape: (n_quad, n_partners)  -- Φ(r) for fixed occupied nl and all allowed virtual n'l'
                orbital_pair_product = (
                    occ_orbitals[:, occ_index][:, np.newaxis]
                    * unocc_orbitals[:, unocc_partners]
                )
                #   shape: (n_partners,)
                constants = (
                    prefactors_q1c_ov[occ_index, unocc_partners]
                    * wigner_symbols_squared_ov[occ_index, unocc_partners, l_couple_idx]
                )
                #   (Φ * c) @ Φ.T  ==  Σ_i c_i Φ_i ⊗ Φ_i
                rpa_response_kernel += (orbital_pair_product * constants) @ orbital_pair_product.T
            del orbital_pair_product, constants
            rpa_response_kernel *= 2

            # --- occ–occ (fractional occupations) ---
            orbital_pair_product = constants = None
            for occ_index in np.unique(occ_valid_indices_oo):
                occ_partners = occ_partner_indices_oo[occ_valid_indices_oo == occ_index]
                #   shape: (n_quad, n_partners)
                orbital_pair_product = (
                    occ_orbitals[:, occ_index][:, np.newaxis]
                    * occ_orbitals[:, occ_partners]
                )
                #   shape: (n_partners,)
                constants = (
                    prefactors_q1c_oo[occ_index, occ_partners]
                    * wigner_symbols_squared_oo[occ_index, occ_partners, l_couple_idx]
                )
                rpa_response_kernel += (orbital_pair_product * constants) @ orbital_pair_product.T
            del orbital_pair_product, constants

            # Restore the 1/(2l''+1) factor omitted from C_{l';ll'} in the loops above.
            rpa_response_kernel /= (2 * active_l_couple + 1)

            # Screened interaction (correlation part) from the Dyson equation:
            #   W_{l''}(iω) = (I - ν̃ χ̃_{0,l''})^{-1} ν̃ - ν̃
            #   dyson_solved_response = np.linalg.solve(np.eye(n_quad) - radial_kernel @ rpa_response_kernel, radial_kernel) - radial_kernel
            eye_minus_nu_chi = np.eye(n_quad) - radial_kernel @ rpa_response_kernel
            del rpa_response_kernel
            dyson_solved_response = (
                np.linalg.solve(eye_minus_nu_chi, radial_kernel) - radial_kernel
            )
            del eye_minus_nu_chi

            # --------------------------------------------------------------
            # Self-energy / Q2c (memory-friendly, OEP-style loops)
            # --------------------------------------------------------------
            # Screened orbital contraction for each channel:
            #   Σ̃ contribution ~ Σ_{n'l'} (f_{nl}-f_{n'l'}) C_{l';ll'} D_{nl,n'l'}(iω)
            #                    × R̃_{n'l'}(r) ∫ R̃_{nl} R̃_{n'l'} W_{l''} dr''
            #   shape: (total_orbitals_num, n_quad)
            _self_energy_potential = np.zeros((total_orbitals_num, n_quad))

            # Occupied nl rows: loop occupied; partners = all spectrum with Wigner ≠ 0
            phi = temp = pref_all = None
            for occ_index in range(occ_orbitals_num):
                nonzero_full_indices = np.argwhere(
                    wigner_symbols_squared_all[occ_index, :, l_couple_idx] != 0
                )[:, 0]
                if len(nonzero_full_indices) == 0:
                    continue
                #   shape: (n_partners,)
                pref_all = (
                    prefactors_self_energy_all[occ_index, nonzero_full_indices]
                    * wigner_symbols_squared_all[occ_index, nonzero_full_indices, l_couple_idx]
                )
                #   shape: (n_partners, n_quad)  — Φ_{nl,n'l'}(r) = R̃_{nl}(r) R̃_{n'l'}(r)
                phi = (
                    occ_orbitals[:, occ_index][:, np.newaxis]
                    * full_orbitals[:, nonzero_full_indices]
                ).T
                #   shape: (n_partners, n_quad)  — Φ @ W_{l''}
                temp = phi @ dyson_solved_response
                #   accumulate V̂_c R̃_{nl} piece on occupied channel
                _self_energy_potential[occ_index, :] += np.einsum(
                    'ik,i,ik->k',
                    temp,
                    pref_all,
                    full_orbitals[:, nonzero_full_indices].T,
                    optimize=True,
                )
            del phi, temp, pref_all

            # Virtual n'l' rows (open-shell / fractional): accumulate onto unoccupied channels
            #   This vanishes for closed-shell atoms with integer occupations on all virt.
            pref_ov_L = (
                prefactors_self_energy_ov
                * wigner_symbols_squared_ov[:, :, l_couple_idx]
            )
            nonzero_unocc_indices = np.argwhere(~np.all(pref_ov_L == 0, axis=0))[:, 0]
            if len(nonzero_unocc_indices) > 0:
                phi = temp = None
                for occ_index in range(occ_orbitals_num):
                    pref_j = pref_ov_L[occ_index, nonzero_unocc_indices]
                    if not np.any(pref_j):
                        continue
                    phi = (
                        occ_orbitals[:, occ_index][:, np.newaxis]
                        * unocc_orbitals[:, nonzero_unocc_indices]
                    ).T
                    temp = phi @ dyson_solved_response
                    _self_energy_potential[occ_orbitals_num + nonzero_unocc_indices, :] += (
                        pref_j[:, np.newaxis]
                        * occ_orbitals[:, occ_index][np.newaxis, :]
                        * temp
                    )
                del phi, temp
            del pref_ov_L

            # Q2c: δE_c^{RPA}/δλ piece × δλ/δV_s ~ R̃², from D^{(2)}_{nl,n'l'}(iω)
            #   Q2c^{ov}: Σ (f_{nl}-f_{n'l'}) C D^{(2)} (R̃_{nl}² - R̃_{n'l'}²) Σ̃_{nl,n'l'}
            #   Q2c^{oo}: Σ C^{oo} D^{(2)} R̃_{nl}² Σ̃_{nl,n'l'}   (fractional occ.)
            #   shape: (n_quad,)
            _q2c_term = np.zeros(n_quad)

            # --- occ–virt ---
            phi = sigma_tilde = constants = squared_diff = None
            for occ_index in np.unique(occ_valid_indices_ov):
                partners = unocc_valid_indices_ov[occ_valid_indices_ov == occ_index]
                mask = (
                    (deg_factors_ov[occ_index, partners] != 0)
                    & (delta_eps_ov[occ_index, partners] != 0)
                )
                partners = partners[mask]
                if len(partners) == 0:
                    continue
                #   shape: (n_quad, n_partners)
                phi = (
                    occ_orbitals[:, occ_index][:, np.newaxis]
                    * unocc_orbitals[:, partners]
                )
                #   Σ̃_{nl,n'l'} = ∬ Φ(r') Φ(r'') W_{l''}(r'',r')  — pair scalars
                sigma_tilde = np.einsum(
                    'li,pi,pl->i', phi, phi, dyson_solved_response, optimize=True
                )
                constants = (
                    prefactors_q2c_ov[occ_index, partners]
                    * wigner_symbols_squared_ov[occ_index, partners, l_couple_idx]
                )
                squared_diff = (
                    occ_orbitals[:, occ_index][:, np.newaxis] ** 2
                    - unocc_orbitals[:, partners] ** 2
                )
                _q2c_term += squared_diff @ (constants * sigma_tilde)
            del phi, sigma_tilde, constants, squared_diff

            # --- occ–occ (fractional occupations) ---
            phi = sigma_tilde = constants = None
            for occ_index in range(occ_orbitals_num):
                partners = np.argwhere(
                    (deg_factors_oo[occ_index] != 0) & (delta_eps_oo[occ_index] != 0)
                )[:, 0]
                if len(partners) == 0:
                    continue
                phi = (
                    occ_orbitals[:, occ_index][:, np.newaxis]
                    * occ_orbitals[:, partners]
                ).T  # (n_partners, n_quad)
                sigma_tilde = np.einsum(
                    'il,ip,pl->i', phi, phi, dyson_solved_response, optimize=True
                )
                constants = (
                    prefactors_q2c_oo[occ_index, partners]
                    * wigner_symbols_squared_oo[occ_index, partners, l_couple_idx]
                )
                _q2c_term += (occ_orbitals[:, occ_index] ** 2) * np.dot(
                    constants, sigma_tilde
                )
            del phi, sigma_tilde, constants

            full_q2c_term += _q2c_term
            full_self_energy_potential += _self_energy_potential

            del _self_energy_potential, _q2c_term, dyson_solved_response, radial_kernel

        # --------------------------------------------------------------
        # Q1c after summing Σ over all l'' channels
        #   Involves same-l orbital Green:
        #     δR̃_{nl}/δV_s ~ - Σ_{n'≠n} R̃_{n'l}(r) R̃_{n'l}(r') / (λ_{n'l}-λ_{nl}) R̃_{nl}
        #   contracted with the accumulated self-energy potential.
        # --------------------------------------------------------------
        for l_value in range(angular_momentum_cutoff + 1):
            l_indices = np.argwhere(full_l_terms == l_value)[:, 0]
            total_orbitals_in_l_channel = full_orbitals[:, l_indices]
            self_energy_in_l_channel = full_self_energy_potential[l_indices, :]
            eigenvalues_in_l_channel = full_eigen_energies[l_indices]

            diff_eigenvalues = (
                eigenvalues_in_l_channel.reshape(-1, 1)
                - eigenvalues_in_l_channel.reshape(1, -1)
            )
            threshold = 1e-12
            one_over_diff_eigenvalues = np.divide(
                1.0,
                diff_eigenvalues,
                out=np.zeros_like(diff_eigenvalues),
                where=np.abs(diff_eigenvalues) > threshold,
            )

            q1c_term_in_l_channel = np.einsum(
                'ki,ik->k',
                total_orbitals_in_l_channel,
                np.einsum(
                    'ij,kj,ij->ik',
                    one_over_diff_eigenvalues,
                    total_orbitals_in_l_channel,
                    np.einsum(
                        'ix,xj->ij',
                        self_energy_in_l_channel,
                        total_orbitals_in_l_channel,
                        optimize=True,
                    ),
                    optimize=True,
                ),
                optimize=True,
            )
            # Minus: OEP driving-term sign convention
            full_q1c_term -= q1c_term_in_l_channel

        assert full_q1c_term.shape == (n_quad,)
        assert full_q2c_term.shape == (n_quad,)

        return full_q1c_term, full_q2c_term



    @staticmethod
    def _compute_rpa_correlation_driving_term_for_single_frequency_old(
        frequency               : float,
        angular_momentum_cutoff : int,
        occupation_info         : OccupationInfo,
        full_eigen_energies     : np.ndarray, 
        full_orbitals           : np.ndarray, 
        full_l_terms            : np.ndarray,
        wigner_symbols_squared  : np.ndarray,
        radial_kernels_dict     : Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RPA correlation driving term (Q1c and Q2c terms) at a given frequency.
        
        This function computes the two components of the RPA correlation driving term
        used in the OEP (Optimized Effective Potential) method:
        - Q1c term: First-order correlation term involving self-energy matrix elements
        - Q2c term: Second-order correlation term involving the Dyson-solved response function
        
        The computation is performed for a single imaginary frequency and includes
        contributions from all angular momentum coupling channels (l_couple).
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - full_q1c_term: Q1c term, shape (n_quad,)
                - full_q2c_term: Q2c term, shape (n_quad,)
        """
        raise RuntimeError(
            RPA_DRIVING_TERM_OLD_DEPRECATED_ERROR
        )

        try:
            frequency = float(frequency)
        except ValueError:
            raise ValueError(FREQUENCY_NOT_FLOAT_ERROR.format(type(frequency)))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        
        # get occupation information
        occupations  = occupation_info.occupations
        occ_l_values = occupation_info.l_values


        # get the number of occupied and unoccupied orbitals
        occ_orbitals_num   = len(occ_l_values)
        total_orbitals_num = len(full_eigen_energies)

        # get the number of quadrature and n_interior points
        n_quad     = radial_kernels_dict[0].shape[0]
        n_interior = len(np.argwhere(full_l_terms == 0)[:, 0])

        # get occupied and unoccupied orbitals and energies
        occ_orbitals   = full_orbitals[:, :occ_orbitals_num]     # shape: (n_grid, total_orbitals_num)
        occ_energies   = full_eigen_energies[:occ_orbitals_num]  # shape: (total_orbitals_num,)
        occ_l_terms    = full_l_terms[:occ_orbitals_num]         # shape: (total_orbitals_num,)
        unocc_orbitals = full_orbitals[:, occ_orbitals_num:]     # shape: (n_grid, unocc_orbitals_num)
        unocc_energies = full_eigen_energies[occ_orbitals_num:]  # shape: (unocc_orbitals_num,)
        unocc_l_terms  = full_l_terms[occ_orbitals_num:]         # shape: (total_orbitals_num,)

        assert np.all(occ_l_terms == occ_l_values), \
            OCCUPATION_INFO_L_TERMS_NOT_CONSISTENT_WITH_OCCUPATION_INFO_ERROR.format(occ_l_terms, occ_l_values)

        ### ================================================ ###
        ###  Part 1: Compute the RPA correlation prefactors  ###
        ### ================================================ ###

        # Angular degeneracy factors f_p * (2l_q + 1)
        #   shape: (occ_num, unocc_num), where 'o' for 'occupied', 'v' for 'virtual'
        deg_factors_ov = occupations[:, np.newaxis] * (2 * unocc_l_terms + 1)[np.newaxis, :]
        #   shape: (occ_num, occ_num), where 'o' for 'occupied'
        deg_factors_oo = occupations[:, np.newaxis] * (2 * occ_l_terms + 1)[np.newaxis, :] - \
                         occupations[np.newaxis, :] * (2 * occ_l_terms + 1)[:, np.newaxis]

        # Energy differences Δε_{pq} = ε_p - ε_q
        #   shape: (occ_num, unocc_num), where 'o' for 'occupied', 'v' for 'virtual'
        delta_eps_ov = occ_energies[:, np.newaxis] - unocc_energies[np.newaxis, :]
        #   shape: (occ_num, occ_num), where 'o' for 'occupied'
        delta_eps_oo = occ_energies[:, np.newaxis] - occ_energies[np.newaxis, :]


        # Filter out zero entries (valid (p,q) pairs)
        #     shape: (n_valid_pairs_ov, )
        occ_idx_ov, unocc_idx_ov = np.argwhere((deg_factors_ov != 0) & (delta_eps_ov != 0)).T
        deg_factors_valid_ov = deg_factors_ov[occ_idx_ov, unocc_idx_ov]
        delta_eps_valid_ov   = delta_eps_ov[occ_idx_ov, unocc_idx_ov]
        #     shape: (n_valid_pairs_oo, )
        occ_idx_x, occ_idx_y = np.argwhere((deg_factors_oo != 0) & (delta_eps_oo != 0)).T
        deg_factors_valid_oo = deg_factors_oo[occ_idx_x, occ_idx_y]
        delta_eps_valid_oo   = delta_eps_oo[occ_idx_x, occ_idx_y]
        

        # Compute Lorentzian frequency response: Δε / (Δε² + ω²)
        #   shape: (n_valid_pairs_ov, )
        lorentzian_factors_ov = delta_eps_valid_ov / (delta_eps_valid_ov ** 2 + frequency ** 2)
        #   shape: (n_valid_pairs_oo, )
        lorentzian_factors_oo = delta_eps_valid_oo / (delta_eps_valid_oo ** 2 + frequency ** 2)

        # Compute frequency derivative factors for Q2c: (Δε² - ω²) / (Δε² + ω²)²
        #   This arises from ∂χ₀,L/∂(iω), used in the Q2c driving term
        #   shape: (n_valid_pairs_ov, )
        frequency_derivative_factors_ov = (delta_eps_valid_ov ** 2 - frequency ** 2) / (delta_eps_valid_ov ** 2 + frequency ** 2) ** 2
        #   shape: (n_valid_pairs_oo, )
        frequency_derivative_factors_oo = (delta_eps_valid_oo ** 2 - frequency ** 2) / (delta_eps_valid_oo ** 2 + frequency ** 2) ** 2


        # Combine angular and frequency factors (without Wigner 3j symbol yet)
        #   shape: (n_valid_pairs_ov, )
        prefactors_q1c_ov = deg_factors_valid_ov * lorentzian_factors_ov
        prefactors_q2c_ov = deg_factors_valid_ov * frequency_derivative_factors_ov
        #   shape: (n_valid_pairs_oo, )
        prefactors_q1c_oo = deg_factors_valid_oo * lorentzian_factors_oo
        prefactors_q2c_oo = deg_factors_valid_oo * frequency_derivative_factors_oo


        #   shape: (occ_num, unocc_num)
        prefactors_self_energy_ov = deg_factors_ov * delta_eps_ov / (delta_eps_ov ** 2 + frequency ** 2)
        #   shape: (occ_num, occ_num)
        prefactors_self_energy_oo = deg_factors_oo * delta_eps_oo / (delta_eps_oo ** 2 + frequency ** 2)
        #   shape: (occ_num, total_num)
        prefactors_self_energy_all = np.concatenate([prefactors_self_energy_oo, prefactors_self_energy_ov], axis=1)


        ### ================================================== ###
        ###  Part 2: Compute the nonzero Wigner symbols        ###
        ### ================================================== ###

        l_couple_min   = np.min(np.abs(occ_l_values[:, np.newaxis] - full_l_terms[np.newaxis, :])).astype(np.int32)
        l_couple_max   = np.max(       occ_l_values[:, np.newaxis] + full_l_terms[np.newaxis, :] ).astype(np.int32)
        l_couple_range = np.arange(l_couple_min, l_couple_max + 1)


        # Use advanced indexing with broadcasting - one operation instead of triple loop
        #   shape: (occ_orbitals_num, unocc_orbitals_num, l_couple_num)
        wigner_symbols_squared_ov = wigner_symbols_squared[
            occ_l_terms   .astype(np.int32)[:, np.newaxis, np.newaxis],  # shape: (occ_orbitals_num, 1, 1)
            unocc_l_terms .astype(np.int32)[np.newaxis, :, np.newaxis],  # shape: (1, unocc_orbitals_num, 1)
            l_couple_range.astype(np.int32)[np.newaxis, np.newaxis, :],  # shape: (1, 1, l_couple_num)
        ]
        #   shape: (occ_orbitals_num, occ_orbitals_num, l_couple_num)
        wigner_symbols_squared_oo = wigner_symbols_squared[
            occ_l_terms   .astype(np.int32)[:, np.newaxis, np.newaxis],  # shape: (occ_orbitals_num, 1, 1)
            occ_l_terms   .astype(np.int32)[np.newaxis, :, np.newaxis],  # shape: (1, occ_orbitals_num, 1)
            l_couple_range.astype(np.int32)[np.newaxis, np.newaxis, :],  # shape: (1, 1, l_couple_num)
        ]
        #   shape: (occ_orbitals_num, total_orbitals_num, l_couple_num)
        wigner_symbols_squared_all = np.concatenate([wigner_symbols_squared_oo, wigner_symbols_squared_ov], axis=1)


        # Compute only the nonzero Wigner symbols for each l_couple channel
        # The selection rule can reduce the number of Wigner symbols to compute
        #   shape: (n_valid_pairs_ov, l_couple_num)
        wigner_symbols_squared_valid_ov = wigner_symbols_squared_ov[occ_idx_ov, unocc_idx_ov, :]
        #   shape: (n_valid_pairs_oo, l_couple_num)
        wigner_symbols_squared_valid_oo = wigner_symbols_squared_oo[occ_idx_x, occ_idx_y, :]
        

        # Compute only the nonzero Wigner symbols for each l_couple channel
        active_l_couple_idx_list           : List[int]        = []
        active_wigner_symbols_indices_list : List[np.ndarray] = []

        for l_couple_idx, l_couple in enumerate(l_couple_range):
            non_zero_wigner_symbols_indices_ov = np.argwhere(wigner_symbols_squared_valid_ov[:, l_couple_idx] != 0)[:, 0]

            # Collect active l_couple and corresponding nonzero Wigner symbols indices
            if len(non_zero_wigner_symbols_indices_ov) != 0:
                active_l_couple_idx_list.append(l_couple_idx)
                active_wigner_symbols_indices_list.append(non_zero_wigner_symbols_indices_ov)


        ### ================================================== ###
        ###  Part 3: Compute orbital products                  ###
        ### ================================================== ###

        # orbital outer product: φ_p(r) ⊗ φ_q(r) for all (p,q) pairs
        #   shape: (occ_num, unocc_num, n_grid)
        # orbital_product_outer_ov = np.einsum('li,lj->ijl',
        #     occ_orbitals,
        #     unocc_orbitals,
        #     optimize = True,
        # )
        #   shape: (occ_num, occ_num, n_grid)
        orbital_product_outer_oo = np.einsum('li,lj->ijl',
            occ_orbitals,
            occ_orbitals,
            optimize = True,
        )
        #   shape: (occ_num, total_num, n_grid)
        # orbital_product_outer_all = np.concatenate([orbital_product_outer_oo, orbital_product_outer_ov], axis=1)
        orbital_product_outer_all = np.einsum('li,lj->ijl',
            occ_orbitals,
            full_orbitals,
            optimize = True,
        )

        # Orbital squared difference: φ_p²(r) - φ_q²(r) for valid (p,q) pairs
        #   shape: (n_grid, n_valid_pairs_ov)
        orbital_squared_diff_ov = occ_orbitals[:, occ_idx_ov] ** 2 - unocc_orbitals[:, unocc_idx_ov] ** 2
        
        # Orbital pair product: Φ_{pq}(r) = φ_p(r)φ_q(r) for valid (p,q) pairs
        #   shape: (n_grid, n_valid_pairs_ov)
        orbital_pair_product_ov = occ_orbitals[:, occ_idx_ov] * unocc_orbitals[:, unocc_idx_ov]


        # initialize the full self-energy potential 
        #   shape: (total_orbitals_num, n_quad)
        full_self_energy_potential = np.zeros((total_orbitals_num, n_quad))


        ### ================================================== ###
        ###  Part 4: Compute RPA correlation driving term      ###
        ### ================================================== ###

        # initialize the q1c and q2c terms
        #   shape: (n_quad,)
        full_q1c_term = np.zeros(n_quad)
        full_q2c_term = np.zeros(n_quad)


        # TODO: parallelize this loop
        # Compute RPA correlation driving term for each l_couple channel
        for active_l_couple_idx, active_wigner_symbols_indices in \
            zip(active_l_couple_idx_list, active_wigner_symbols_indices_list):

            active_l_couple = l_couple_range[active_l_couple_idx]

            # Get radial kernel (Coulomb kernel for angular momentum channel L)
            #   shape: (n_quad, n_quad)
            #   R^{(L)}(r_i, r_j) = (1/(2L+1)) * (r_<^L / r_>^{L+1}) * w_i * w_j
            #   where:
            #     - r_< = min(r_i, r_j), r_> = max(r_i, r_j)
            #     - w_i, w_j: quadrature weights at radial points r_i, r_j
            #     - L = active_l_couple: angular momentum coupling channel
            #   This is the radial projection of the Coulomb interaction in channel L
            radial_kernel = radial_kernels_dict[active_l_couple] * (2 * active_l_couple + 1)
            
            # Compute rpa_response_kernel
            #   shape: (n_quad, n_quad)
            #   rpa_response_kernel_ov: the response kernel for occ-unocc pairs
            #       χ₀,L^(occ-virt)(r, r'; iω) = 2 * Σ_{p∈occ, q∈virt} [f_p(2l_q+1) * Δε_{pq} / (Δε_{pq}² + ω²)] * W_{pq}^{(L)} * Φ_{pq}(r) * Φ_{pq}(r')
            #       where:
            #         - f_p: occupation number of orbital p
            #         - Δε_{pq} = ε_p - ε_q: energy difference
            #         - W_{pq}^{(L)}: Wigner 3j symbol squared
            #         - Φ_{pq}(r) = φ_p(r)φ_q(r): orbital pair product
            #       The factor 2 comes from spin degeneracy
            rpa_response_kernel_ov = \
                2 * np.einsum(
                    'ij,ik->kj',
                    np.einsum('ji,i->ij',
                        orbital_pair_product_ov[:, active_wigner_symbols_indices],
                        prefactors_q1c_ov[active_wigner_symbols_indices],
                        optimize=True,
                    ),
                    np.einsum('i,ki->ik',
                        wigner_symbols_squared_valid_ov[active_wigner_symbols_indices, active_l_couple],
                        orbital_pair_product_ov[:, active_wigner_symbols_indices],
                        optimize=True,
                    ),
                    optimize=True,
                ) # Reference (Implementation): https://stackoverflow.com/questions/17437523/python-fast-way-to-sum-outer-products
            
            #   rpa_response_kernel_oo: the response kernel for occ-occ pairs (for fractional occupations)
            #       χ₀,L^(occ-occ)(r, r'; iω) = Σ_{p,p'∈occ} [C_{pp'}^{occ-occ} * Δε_{pp'} / (Δε_{pp'}² + ω²)] * W_{pp'}^{(L)} * Φ_{pp'}(r) * Φ_{pp'}(r')
            #       where:
            #         - C_{pp'}^{occ-occ} = f_p(2l_{p'}+1) - f_{p'}(2l_p+1): occ-occ coupling constant
            #         - Δε_{pp'} = ε_p - ε_{p'}: energy difference
            #         - W_{pp'}^{(L)}: Wigner 3j symbol squared
            #         - Φ_{pp'}(r) = φ_p(r)φ_{p'}(r): orbital pair product
            #       Note: No factor of 2 here (already included in C_{pp'}^{occ-occ})
            rpa_response_kernel_oo = \
                np.einsum(
                    'il, ip, i->pl',
                    orbital_product_outer_oo[occ_idx_x, occ_idx_y, :],  # shape: (n_valid_pairs_oo, n_grid)
                    orbital_product_outer_oo[occ_idx_x, occ_idx_y, :],  # shape: (n_valid_pairs_oo, n_grid)
                    prefactors_q1c_oo * wigner_symbols_squared_valid_oo[:, active_l_couple],  # shape: (n_valid_pairs_oo, )
                    optimize=True,
                )
                
            # Combine occ-unocc and occ-occ contributions
            #   Normalize by (2L+1) to account for angular momentum degeneracy
            #   χ₀,L(r, r'; iω) = [χ₀,L^(occ-virt)(r, r'; iω) + χ₀,L^(occ-occ)(r, r'; iω)] / (2L+1)
            #   The factor (2L+1) comes from the angular integration over m_L = -L, ..., L
            rpa_response_kernel = (rpa_response_kernel_ov + rpa_response_kernel_oo) / (2 * active_l_couple + 1)

            # Compute dyson solved response
            #   dyson_solved_response: χ_L(iω) = χ_{0,L}(iω) + χ_{0,L}(iω) v_L χ_L(iω)
            #   Therefore, dyson_solved_response = χ_L(iω) - χ_{0,L}(iω) = (I - χ_{0,L}(iω) v_L)^{-1} χ_{0,L}(iω)
            #   shape: (n_quad, n_quad)
            dyson_solved_response = np.linalg.solve(np.eye(n_quad) - radial_kernel @ rpa_response_kernel, radial_kernel) - radial_kernel

            
            # Compute self-energy potential term
            #   self_energy_potential: Σ_{pq}(ω) = ∫ dr' [χ_L(r,r';iω) - χ_{0,L}(r,r';iω)] Φ_{pq}(r')
            #   -> Includes both occ-unocc and occ-occ contributions (for fractional occupations)
            #   shape: (total_orbitals_num, n_quad)
            _self_energy_potential = np.zeros((total_orbitals_num, n_quad))

            #   occ-occ part: first (occ_orbitals_num, n_quad) out of full (total_orbitals_num, n_quad)
            for occ_l_index in range(occ_orbitals_num):
                # Get nonzero full indices for this occupied orbital
                #   shape: (full_nonzero_num, )
                _nonzero_full_indices = np.argwhere(wigner_symbols_squared_all[occ_l_index, :, active_l_couple] != 0)[:, 0]
                if len(_nonzero_full_indices) > 0:
                    _prefactor_all = prefactors_self_energy_all[occ_l_index, _nonzero_full_indices] * wigner_symbols_squared_all[occ_l_index, _nonzero_full_indices, active_l_couple]
                    _self_energy_potential[occ_l_index, :] += \
                        np.einsum('i,ki,il,lk->k',
                            _prefactor_all,                                                    # (full_nonzero_num, )
                            full_orbitals[:, _nonzero_full_indices],                           # (n_quad, full_nonzero_num, )
                            orbital_product_outer_all[occ_l_index, _nonzero_full_indices, :],  # (full_nonzero_num, n_quad)
                            dyson_solved_response,                                             # (n_quad, n_quad)
                            optimize=True,
                        )
            
            #   occ-unocc part: last (unocc_orbitals_num, n_quad) out of full (total_orbitals_num, n_quad)
            #   This term will not contribute to the self-energy potential if the atoms are closed-shell
            #       shape: (occ_num, unocc_num)
            _prefactor_ov = prefactors_self_energy_ov * wigner_symbols_squared_ov[:, :, active_l_couple]
            #       shape: (unocc_nonzero_num, )
            _nonzero_unocc_indices = np.argwhere(~np.all(_prefactor_ov == 0, axis=0))[:, 0]

            if len(_nonzero_unocc_indices) > 0:
                _self_energy_potential[occ_orbitals_num:, :][_nonzero_unocc_indices, :] = \
                    np.einsum(
                        'ji,kj,jil,lk->ik',
                        _prefactor_ov[:,_nonzero_unocc_indices],                # (occ_num, unocc_nonzero_num)
                        occ_orbitals,                                           # (n_quad, occ_num)
                        # orbital_product_outer_ov[:, _nonzero_unocc_indices, :], # (unocc_nonzero_num, n_quad)
                        orbital_product_outer_all[:, len(occ_l_values):, :][:, _nonzero_unocc_indices, :], # (unocc_nonzero_num, n_quad)
                        dyson_solved_response,                                  # (n_quad, n_quad)
                        optimize=True,
                    )
            
            # Compute q2c term (occ-unocc part)
            #   Q_{2c}^{occ-virt}(r_i) = (1/π) Σ_L (2L+1) Σ_ω w_ω Σ_{p∈occ} Σ_{q∈virt}
            #       f_p(2l_q+1) * [(Δε_{pq})² - ω²] / [(Δε_{pq})² + ω²]² * W_{pq}^{(L)} * [φ_p²(r_i) - φ_q²(r_i)] * Σ̃^{(L)}_{pq}(ω)
            #   where:
            #     - [φ_p²(r_i) - φ_q²(r_i)]: orbital_squared_diff_ov (orbital squared difference)
            #     - f_p(2l_q+1) * [(Δε_{pq})² - ω²] / [(Δε_{pq})² + ω²]²: prefactors_q2c_ov (frequency derivative factor)
            #     - W_{pq}^{(L)}: Wigner 3j symbol squared
            #     - Σ̃^{(L)}_{pq}(ω): screened self-energy from Dyson equation (inner einsum computes this)
            #   Inner einsum: compute Σ̃^{(L)}_{pq}(ω) = ∫ dr' [χ_L(r,r';iω) - χ_{0,L}(r,r';iω)] Φ_{pq}(r')
            #     'li,pi,pl->i': sum over radial grid points to get self-energy contribution for each (p,q) pair
            #   Outer einsum: sum over all (p,q) pairs to get Q2c at each radial grid point
            #     'ki,i,i->k': orbital_squared_diff * prefactor*Wigner * self_energy -> Q2c(r_k)
            _q2c_term_ov = np.einsum('ki, i, i->k',
                #     shape: (n_quad, n_valid_pairs): [φ_p²(r_k) - φ_q²(r_k)]
                orbital_squared_diff_ov[:, active_wigner_symbols_indices],
                #     shape: (n_valid_pairs,): prefactor * Wigner
                prefactors_q2c_ov[active_wigner_symbols_indices] * wigner_symbols_squared_valid_ov[active_wigner_symbols_indices, active_l_couple],
                np.einsum('li,pi,pl->i',
                    orbital_pair_product_ov[:, active_wigner_symbols_indices], # (n_quad, n_valid_pairs): Φ_{pq}(r_l)
                    orbital_pair_product_ov[:, active_wigner_symbols_indices], # (n_quad, n_valid_pairs): Φ_{pq}(r_p)
                    dyson_solved_response,                                     # (n_quad, n_quad): [χ_L - χ_{0,L}]
                    optimize=True,
                ),                                                             # -> (n_valid_pairs,): Σ̃^{(L)}_{pq}(ω)
                optimize=True,
            )
            
            # Compute q2c term (occ-occ part, for fractional occupations)
            #   Q_{2c}^{occ-occ}(r_i) = (1/2π) Σ_L (2L+1) Σ_ω w_ω Σ_{p,p'∈occ}
            #       C_{pp'}^{occ-occ} * [(Δε_{pp'})² - ω²] / [(Δε_{pp'})² + ω²]² * W_{pp'}^{(L)} * φ_p²(r_i) * Σ̃^{(L)}_{pp'}(ω)
            #   where:
            #     - C_{pp'}^{occ-occ} = f_p(2l_{p'}+1) - f_{p'}(2l_p+1): occ-occ coupling constant
            #     - φ_p²(r_i): occ_orbitals[:, occ_idx_x] ** 2 (orbital squared, only for first orbital in pair)
            #     - C_{pp'}^{occ-occ} * [(Δε_{pp'})² - ω²] / [(Δε_{pp'})² + ω²]²: prefactors_q2c_oo (frequency derivative factor)
            #     - W_{pp'}^{(L)}: Wigner 3j symbol squared
            #     - Σ̃^{(L)}_{pp'}(ω): screened self-energy from Dyson equation (inner einsum computes this)
            #   Inner einsum: compute Σ̃^{(L)}_{pp'}(ω) = ∫ dr' [χ_L(r,r';iω) - χ_{0,L}(r,r';iω)] Φ_{pp'}(r')
            #     'il,ip,pl->i': sum over radial grid points to get self-energy contribution for each (p,p') pair
            #   Outer einsum: sum over all (p,p') pairs to get Q2c at each radial grid point
            #     'ki,i,i->k': orbital_squared * prefactor*Wigner * self_energy -> Q2c(r_k)
            _q2c_term_oo = np.einsum('ki,i,i->k',
                occ_orbitals[:, occ_idx_x] ** 2,                                                       # (n_quad, n_valid_pairs_oo): φ_p²(r_k)
                prefactors_q2c_oo * wigner_symbols_squared_oo[occ_idx_x, occ_idx_y, active_l_couple],  # (n_valid_pairs_oo,): prefactor * Wigner
                np.einsum('il,ip,pl->i',
                    orbital_product_outer_oo[occ_idx_x, occ_idx_y, :],  # (n_valid_pairs_oo, n_quad): Φ_{pp'}(r_l)
                    orbital_product_outer_oo[occ_idx_x, occ_idx_y, :],  # (n_valid_pairs_oo, n_quad): Φ_{pp'}(r_p)
                    dyson_solved_response,                              # (n_quad, n_quad): [χ_L - χ_{0,L}]
                    optimize=True,
                ),                                                      # -> (n_valid_pairs_oo,): Σ̃^{(L)}_{pp'}(ω)
                optimize=True,
            )
            
            # Add occ-occ contribution to q2c_term
            _q2c_term = _q2c_term_ov + _q2c_term_oo

            # Update the full q2c term and self-energy potential
            full_q2c_term += _q2c_term
            full_self_energy_potential += _self_energy_potential
                        

        # Compute q1c term
        #   Q_{1c}(r_i) = 4 * Σ_l Σ_{i,j} φ_i(r_i) * [1/(ε_i - ε_j)] * φ_j(r_i) * Σ_{ij}(r_i)
        #   where:
        #     - Σ_{ij}(r_i) = ∫ dr' φ_i(r') Σ_c(r', r_i) φ_j(r'): self-energy matrix element in orbital basis
        #     - The factor 4 comes from spin and angular degeneracy
        #     - The sum is over all l channels (angular momentum channels)
        #   Note: The minus sign in "full_q1c_term -= q1c_term_in_l_channel" comes from the sign convention
        #         in the OEP equation (the driving term has opposite sign to the potential)
        for l_value in range(angular_momentum_cutoff + 1):

            # Get all orbitals in this l_value channel
            #   Only orbitals with the same angular momentum l can couple in Q1c calculation
            l_indices = np.argwhere(full_l_terms == l_value)[:, 0]
            total_orbitals_in_l_channel = full_orbitals[:, l_indices]               # shape: (n_quad, n_orbitals_in_l)
            self_energy_in_l_channel    = full_self_energy_potential[l_indices, :]  # shape: (n_orbitals_in_l, n_quad)
            eigenvalues_in_l_channel    = full_eigen_energies[l_indices]            # shape: (n_orbitals_in_l,)
            
            # Compute 1/(ε_i - ε_j) for all orbital pairs in this l channel
            #   shape: (n_orbitals_in_l, n_orbitals_in_l)
            #   Only compute where |ε_i - ε_j| > threshold, set to 0 otherwise (handles diagonal and degenerate cases)
            diff_eigenvalues = eigenvalues_in_l_channel.reshape(-1, 1) - eigenvalues_in_l_channel.reshape(1, -1)
            threshold = 1e-12
            one_over_diff_eigenvalues = np.divide(1.0, diff_eigenvalues, 
                                                  out=np.zeros_like(diff_eigenvalues),
                                                  where=np.abs(diff_eigenvalues) > threshold)

            # Compute q1c term for this l_value channel
            #    Three-level einsum structure:
            #    1. Inner einsum ('ix,xj->ij'): Compute self-energy matrix elements 
            #        Σ_{ij}(r_i) = ∫ dr' φ_i(r') Σ_c(r', r_i) φ_j(r') = Σ_x φ_i(r_x) Σ_c(i, r_x) φ_j(r_x)
            #    2. Middle einsum ('ij,kj,ij->ik'): Compute Σ_j [1/(ε_i - ε_j)] * φ_j(r_k) * Σ_{ij}(r_k)
            #    3. Outer einsum ('ki,ik->k'): Compute Σ_i φ_i(r_k) * [middle result] -> Q1c(r_k)
            q1c_term_in_l_channel = \
                np.einsum('ki,ik->k',
                    total_orbitals_in_l_channel,         # (n_quad, n_orbitals_in_l): φ_i(r_k)
                    np.einsum('ij,kj,ij->ik',
                        one_over_diff_eigenvalues,       # (n_orbitals_in_l, n_orbitals_in_l): 1/(ε_i - ε_j)
                        total_orbitals_in_l_channel,     # (n_quad, n_orbitals_in_l): φ_j(r_k)
                        np.einsum('ix,xj->ij',
                            self_energy_in_l_channel,    # (n_orbitals_in_l, n_quad): Σ_c(i, r_x)
                            total_orbitals_in_l_channel, # (n_quad, n_orbitals_in_l): φ_j(r_x)
                            optimize=True,
                        ),         # -> (n_orbitals_in_l, n_orbitals_in_l): Σ_{ij}
                    optimize=True,
                ),  # -> (n_quad,): Q1c(r_k) for this l channel
                optimize=True,
            )

            # Accumulate contribution from this l channel
            #   Note: Minus sign comes from OEP equation sign convention
            full_q1c_term -= q1c_term_in_l_channel

        assert full_q1c_term.shape == (n_quad,)
        assert full_q2c_term.shape == (n_quad,)

        return full_q1c_term, full_q2c_term




    @staticmethod
    def _compute_rpa_wigner_symbols_squared(
        l_occ_max   : int,
        l_unocc_max : int,
    ) -> np.ndarray:
        """
        Compute RPA Wigner symbols squared array.

        Parameters
        ----------
        l_occ_max : int
            Maximum angular momentum quantum number for occupied orbitals
        l_unocc_max : int
            Maximum angular momentum quantum number for unoccupied orbitals

        Returns
        -------
        wigner_symbols_squared : np.ndarray
            Wigner symbols squared array
            shape: (l_occ_max + 1, l_unocc_max + 1, l_couple_max + 1), where l_couple_max = l_occ_max + l_unocc_max
        """
        try:
            l_occ_max = int(l_occ_max)
        except ValueError:
            raise ValueError(L_OCC_MAX_NOT_INTEGER_ERROR.format(type(l_occ_max)))
        try:
            l_unocc_max = int(l_unocc_max)
        except ValueError:
            raise ValueError(L_UNOCC_MAX_NOT_INTEGER_ERROR.format(type(l_unocc_max)))
        assert l_occ_max >= 0 and l_unocc_max >= 0, \
            "All angular momentum quantum numbers must be non-negative"

        # Compute the maximum angular momentum quantum number for the coupled system
        l_couple_max = l_occ_max + l_unocc_max

        # Initialize Wigner symbols squared array
        wigner_symbols_squared = np.zeros((l_occ_max + 1, l_unocc_max + 1, l_couple_max + 1))

        # Compute Wigner symbols squared for all (l_occ, l_unocc, l_couple) combinations
        for l_occ in range(l_occ_max + 1):
            for l_unocc in range(l_unocc_max + 1):
                for l_couple in range(l_couple_max + 1):
                    wigner_symbols_squared[l_occ, l_unocc, l_couple] = \
                        CoulombCouplingCalculator.wigner_3j_000(l_occ, l_unocc, l_couple) ** 2

        return wigner_symbols_squared

        
    @staticmethod
    def _compute_correlation_energy_for_single_frequency(
        frequency              : float,
        occupation_info        : OccupationInfo,
        full_eigen_energies    : np.ndarray, 
        full_orbitals          : np.ndarray, 
        full_l_terms           : np.ndarray,
        wigner_symbols_squared : np.ndarray,
        radial_kernels_dict    : Dict[int, np.ndarray],
    ) -> float:
        """
        Compute the single-frequency integrand of the RPA correlation energy.

        At imaginary frequency iω this returns the sum over angular channels
            Σ_{l''} (2l''+1) Tr[ χ̃_{0,l''} ν_{l''} + log(I - χ̃_{0,l''} ν_{l''}) ]
        (the outer 1/(2π) and ∫ dω weight are applied in compute_correlation_energy).
        """
        try:
            frequency = float(frequency)
        except ValueError:
            raise ValueError(FREQUENCY_NOT_FLOAT_ERROR.format(type(frequency)))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        
        # get occupation information
        occupations  = occupation_info.occupations
        occ_l_values = occupation_info.l_values


        # get the number of occupied and unoccupied orbitals
        occ_orbitals_num = len(occ_l_values)

        # get the number of quadrature points
        n_quad = radial_kernels_dict[0].shape[0]

        # get occupied and unoccupied orbitals and energies
        occ_orbitals   = full_orbitals[:, :occ_orbitals_num]     # shape: (n_grid, occ_num)
        occ_energies   = full_eigen_energies[:occ_orbitals_num]  # shape: (occ_num,)
        occ_l_terms    = full_l_terms[:occ_orbitals_num]         # shape: (occ_num,)
        unocc_orbitals = full_orbitals[:, occ_orbitals_num:]     # shape: (n_grid, unocc_num)
        unocc_energies = full_eigen_energies[occ_orbitals_num:]  # shape: (unocc_num,)
        unocc_l_terms  = full_l_terms[occ_orbitals_num:]         # shape: (unocc_num,)

        assert np.all(occ_l_terms == occ_l_values), \
            OCCUPATION_INFO_L_TERMS_NOT_CONSISTENT_WITH_OCCUPATION_INFO_ERROR.format(occ_l_terms, occ_l_values)

        ### ================================================ ###
        ###  Part 1: Compute the RPA correlation prefactors  ###
        ### ================================================ ###

        # Occupation / angular factors entering (f_{nl} - f_{n'l'}) and C_{l';ll'}
        #   ov: f_{nl} * (2l' + 1) for occupied nl and virtual n'l'  (f_{n'l'}=0)
        #   shape: (occ_num, unocc_num)
        deg_factors_ov = occupations[:, np.newaxis] * (2 * unocc_l_terms + 1)[np.newaxis, :]
        #   oo: f_{nl}*(2l'+1) - f_{n'l'}*(2l+1) for occupied–occupied pairs
        #   shape: (occ_num, occ_num)
        deg_factors_oo = occupations[:, np.newaxis] * (2 * occ_l_terms + 1)[np.newaxis, :] - \
                         occupations[np.newaxis, :] * (2 * occ_l_terms + 1)[:, np.newaxis]

        # Eigenvalue differences λ_{nl} - λ_{n'l'}
        #   shape: (occ_num, unocc_num)
        delta_eps_ov = occ_energies[:, np.newaxis] - unocc_energies[np.newaxis, :]
        #   shape: (occ_num, occ_num)
        delta_eps_oo = occ_energies[:, np.newaxis] - occ_energies[np.newaxis, :]

        # Frequency factor D_{nl,n'l'}(iω) = (λ_{nl} - λ_{n'l'}) / [ω² + (λ_{nl} - λ_{n'l'})²]
        #   shape: (occ_num, unocc_num)
        lorentzian_factors_ov = delta_eps_ov / (delta_eps_ov ** 2 + frequency ** 2)
        #   shape: (occ_num, occ_num)
        lorentzian_factors_oo = delta_eps_oo / (delta_eps_oo ** 2 + frequency ** 2)

        # Combine occupation/angular and frequency factors (Wigner / C_{l';ll'} later)
        #   shape: (occ_num, unocc_num)
        prefactors_ov = deg_factors_ov * lorentzian_factors_ov
        #   shape: (occ_num, occ_num)
        prefactors_oo = deg_factors_oo * lorentzian_factors_oo

        ### ================================================== ###
        ###  Part 2: Compute the nonzero Wigner symbols        ###
        ### ================================================== ###

        l_couple_min   = np.min(np.abs(occ_l_values[:, np.newaxis] - full_l_terms[np.newaxis, :])).astype(np.int32)
        l_couple_max   = np.max(       occ_l_values[:, np.newaxis] + full_l_terms[np.newaxis, :] ).astype(np.int32)
        l_couple_range = np.arange(l_couple_min, l_couple_max + 1)

        #   shape: (occ_num, unocc_num, l_couple_num)
        wigner_symbols_squared_ov = wigner_symbols_squared[
            occ_l_terms   .astype(np.int32)[:, np.newaxis, np.newaxis],
            unocc_l_terms .astype(np.int32)[np.newaxis, :, np.newaxis],
            l_couple_range.astype(np.int32)[np.newaxis, np.newaxis, :],
        ]
        #   shape: (occ_num, occ_num, l_couple_num)
        wigner_symbols_squared_oo = wigner_symbols_squared[
            occ_l_terms   .astype(np.int32)[:, np.newaxis, np.newaxis],
            occ_l_terms   .astype(np.int32)[np.newaxis, :, np.newaxis],
            l_couple_range.astype(np.int32)[np.newaxis, np.newaxis, :],
        ]

        # Active l'' channels: at least one nonzero Wigner symbol in ov or oo.
        active_l_couple_idx_list: List[int] = []
        for l_couple_idx in range(len(l_couple_range)):
            if (
                np.any(wigner_symbols_squared_ov[:, :, l_couple_idx] != 0)
                or np.any(wigner_symbols_squared_oo[:, :, l_couple_idx] != 0)
            ):
                active_l_couple_idx_list.append(l_couple_idx)

        ### ================================================== ###
        ###  Part 3: RPA correlation energy integrand          ###
        ### ================================================== ###
        # E_c^{RPA} = (1/2π) Σ_{l''} (2l''+1) ∫ Tr[ χ̃_{0,l''} ν_{l''}
        #              + log(I - χ̃_{0,l''} ν_{l''}) ] dω
        # This function accumulates the frequency integrand (no 1/2π, no ω weight).

        full_correlation_energy_at_single_frequency = 0.0

        for l_couple_idx in active_l_couple_idx_list:

            active_l_couple = l_couple_range[l_couple_idx]
            # Nonzero Wigner pairs for this l'': shape (n_nonzero, 2) -> [occ_i, partner_j]
            active_wigner_symbols_indices_ov = np.argwhere(
                wigner_symbols_squared_ov[:, :, l_couple_idx] != 0
            )
            active_wigner_symbols_indices_oo = np.argwhere(
                wigner_symbols_squared_oo[:, :, l_couple_idx] != 0
            )
            occ_valid_indices_ov   = active_wigner_symbols_indices_ov[:, 0]
            unocc_valid_indices_ov = active_wigner_symbols_indices_ov[:, 1]
            occ_valid_indices_oo   = active_wigner_symbols_indices_oo[:, 0]
            occ_partner_indices_oo = active_wigner_symbols_indices_oo[:, 1]
            del active_wigner_symbols_indices_ov, active_wigner_symbols_indices_oo

            # Coulomb kernel ν_{l''} (stored radial_kernels already include 1/(2l''+1);
            # multiply back so radial_kernel ≡ ν̃ used with χ̃ as in the driving-term path)
            #   shape: (n_quad, n_quad)
            radial_kernel = radial_kernels_dict[active_l_couple] * (2 * active_l_couple + 1)

            # Accumulate ov then oo into one buffer: χ̃_{0,l''} = (χ̃^{ov} + χ̃^{oo}) / (2l''+1)
            #   shape: (n_quad, n_quad)
            rpa_response_kernel = np.zeros((n_quad, n_quad))

            # --- occ–virt (spin degeneracy ×2 after this loop) ---
            orbital_pair_product = constants = None
            for occ_index in np.unique(occ_valid_indices_ov):
                unocc_partners = unocc_valid_indices_ov[occ_valid_indices_ov == occ_index]
                orbital_pair_product = (
                    occ_orbitals[:, occ_index][:, np.newaxis]
                    * unocc_orbitals[:, unocc_partners]
                )
                constants = (
                    prefactors_ov[occ_index, unocc_partners]
                    * wigner_symbols_squared_ov[occ_index, unocc_partners, l_couple_idx]
                )
                rpa_response_kernel += (orbital_pair_product * constants) @ orbital_pair_product.T
            del orbital_pair_product, constants
            rpa_response_kernel *= 2

            # --- occ–occ (fractional occupations) ---
            orbital_pair_product = constants = None
            for occ_index in np.unique(occ_valid_indices_oo):
                occ_partners = occ_partner_indices_oo[occ_valid_indices_oo == occ_index]
                orbital_pair_product = (
                    occ_orbitals[:, occ_index][:, np.newaxis]
                    * occ_orbitals[:, occ_partners]
                )
                constants = (
                    prefactors_oo[occ_index, occ_partners]
                    * wigner_symbols_squared_oo[occ_index, occ_partners, l_couple_idx]
                )
                rpa_response_kernel += (orbital_pair_product * constants) @ orbital_pair_product.T
            del orbital_pair_product, constants

            rpa_response_kernel /= (2 * active_l_couple + 1)

            # Channel contribution: (2l''+1) [ log det(I - ν̃ χ̃) + Tr(ν̃ χ̃) ]
            #   energy_contrib = (2 * active_l_couple + 1) * (np.log(np.linalg.det(np.eye(n_quad) - radial_kernel @ rpa_response_kernel)) + np.trace(radial_kernel @ rpa_response_kernel))
            nu_chi = radial_kernel @ rpa_response_kernel
            trace_nu_chi = np.trace(nu_chi)
            del rpa_response_kernel
            eye_minus_nu_chi = np.eye(n_quad) - nu_chi
            del nu_chi
            energy_contrib = (2 * active_l_couple + 1) * (
                np.log(np.linalg.det(eye_minus_nu_chi)) + trace_nu_chi
            )
            del eye_minus_nu_chi

            full_correlation_energy_at_single_frequency += energy_contrib
            del radial_kernel, energy_contrib, trace_nu_chi

        return full_correlation_energy_at_single_frequency


    @staticmethod
    def _matrix_log_I_minus_nu_chi(
        chi: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build νχ and log(I - νχ) via the symmetrized product A = ν^{1/2} χ ν^{1/2}.

        Strategy (numerically stabler than eigendecomposing the non-symmetric νχ):
          1. ν^{1/2}, ν^{-1/2} from eigh(ν)  (ν is SPD Coulomb kernel)
          2. A = ν^{1/2} χ ν^{1/2}           (symmetric; eig(A) = eig(νχ))
          3. log(I - νχ) = ν^{1/2} log(I - A) ν^{-1/2}
          4. νχ = ν @ χ

        Returns
        -------
        nu_chi : np.ndarray
            νχ, shape (n, n)
        log_I_minus_nu_chi : np.ndarray
            log(I - νχ), shape (n, n), possibly complex
        """
        evals_nu, evecs_nu = np.linalg.eigh(nu)
        evals_nu = np.maximum(evals_nu, 0.0)
        sqrt_nu = np.sqrt(evals_nu)
        inv_sqrt_nu = np.zeros_like(sqrt_nu)
        positive = sqrt_nu > 1e-14
        inv_sqrt_nu[positive] = 1.0 / sqrt_nu[positive]

        nu_half = (evecs_nu * sqrt_nu) @ evecs_nu.T
        nu_inv_half = (evecs_nu * inv_sqrt_nu) @ evecs_nu.T

        A = nu_half @ chi @ nu_half
        evals_A, evecs_A = np.linalg.eigh(A)

        # Principal logarithm of (I - A) in the eigenbasis of A.
        one_minus = (1.0 - evals_A).astype(np.complex128)
        log_vals = np.log(one_minus)
        log_I_minus_A = (evecs_A.astype(np.complex128) * log_vals) @ evecs_A.T

        log_I_minus_nu_chi = nu_half.astype(np.complex128) @ log_I_minus_A @ nu_inv_half.astype(np.complex128)
        nu_chi = nu @ chi
        return nu_chi, log_I_minus_nu_chi


    @staticmethod
    def _compute_correlation_energy_density_for_single_frequency(
        frequency              : float,
        occupation_info        : OccupationInfo,
        full_eigen_energies    : np.ndarray,
        full_orbitals          : np.ndarray,
        full_l_terms           : np.ndarray,
        wigner_symbols_squared : np.ndarray,
        radial_kernels_dict    : Dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Single-frequency integrand of the RPA correlation energy density.

        Same χ̃_{0,l''} construction as `_compute_correlation_energy_for_single_frequency`,
        but accumulates the diagonal of
            (2l''+1) [ ν̃ χ̃ + log(I - ν̃ χ̃) ]
        instead of its Trace / log-det.  The outer 1/(2π), ω weight, and
        division by 4π r² w are applied in `compute_correlation_energy_density`.

        Returns
        -------
        np.ndarray
            Complex array of shape (n_quad,); imaginary parts from branch cuts /
            roundoff are cleaned by the caller.
        """
        try:
            frequency = float(frequency)
        except ValueError:
            raise ValueError(FREQUENCY_NOT_FLOAT_ERROR.format(type(frequency)))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))

        occupations  = occupation_info.occupations
        occ_l_values = occupation_info.l_values
        occ_orbitals_num = len(occ_l_values)
        n_quad = radial_kernels_dict[0].shape[0]

        occ_orbitals   = full_orbitals[:, :occ_orbitals_num]
        occ_energies   = full_eigen_energies[:occ_orbitals_num]
        occ_l_terms    = full_l_terms[:occ_orbitals_num]
        unocc_orbitals = full_orbitals[:, occ_orbitals_num:]
        unocc_energies = full_eigen_energies[occ_orbitals_num:]
        unocc_l_terms  = full_l_terms[occ_orbitals_num:]

        assert np.all(occ_l_terms == occ_l_values), \
            OCCUPATION_INFO_L_TERMS_NOT_CONSISTENT_WITH_OCCUPATION_INFO_ERROR.format(occ_l_terms, occ_l_values)

        deg_factors_ov = occupations[:, np.newaxis] * (2 * unocc_l_terms + 1)[np.newaxis, :]
        deg_factors_oo = occupations[:, np.newaxis] * (2 * occ_l_terms + 1)[np.newaxis, :] - \
                         occupations[np.newaxis, :] * (2 * occ_l_terms + 1)[:, np.newaxis]

        delta_eps_ov = occ_energies[:, np.newaxis] - unocc_energies[np.newaxis, :]
        delta_eps_oo = occ_energies[:, np.newaxis] - occ_energies[np.newaxis, :]

        lorentzian_factors_ov = delta_eps_ov / (delta_eps_ov ** 2 + frequency ** 2)
        lorentzian_factors_oo = delta_eps_oo / (delta_eps_oo ** 2 + frequency ** 2)

        prefactors_ov = deg_factors_ov * lorentzian_factors_ov
        prefactors_oo = deg_factors_oo * lorentzian_factors_oo

        l_couple_min   = np.min(np.abs(occ_l_values[:, np.newaxis] - full_l_terms[np.newaxis, :])).astype(np.int32)
        l_couple_max   = np.max(       occ_l_values[:, np.newaxis] + full_l_terms[np.newaxis, :] ).astype(np.int32)
        l_couple_range = np.arange(l_couple_min, l_couple_max + 1)

        wigner_symbols_squared_ov = wigner_symbols_squared[
            occ_l_terms   .astype(np.int32)[:, np.newaxis, np.newaxis],
            unocc_l_terms .astype(np.int32)[np.newaxis, :, np.newaxis],
            l_couple_range.astype(np.int32)[np.newaxis, np.newaxis, :],
        ]
        wigner_symbols_squared_oo = wigner_symbols_squared[
            occ_l_terms   .astype(np.int32)[:, np.newaxis, np.newaxis],
            occ_l_terms   .astype(np.int32)[np.newaxis, :, np.newaxis],
            l_couple_range.astype(np.int32)[np.newaxis, np.newaxis, :],
        ]

        active_l_couple_idx_list: List[int] = []
        for l_couple_idx in range(len(l_couple_range)):
            if (
                np.any(wigner_symbols_squared_ov[:, :, l_couple_idx] != 0)
                or np.any(wigner_symbols_squared_oo[:, :, l_couple_idx] != 0)
            ):
                active_l_couple_idx_list.append(l_couple_idx)

        # Accumulator for diag{ Σ_{l''} (2l''+1) [ν̃χ̃ + log(I - ν̃χ̃)] }
        density_at_frequency = np.zeros(n_quad, dtype=np.complex128)

        for l_couple_idx in active_l_couple_idx_list:

            active_l_couple = l_couple_range[l_couple_idx]
            active_wigner_symbols_indices_ov = np.argwhere(
                wigner_symbols_squared_ov[:, :, l_couple_idx] != 0
            )
            active_wigner_symbols_indices_oo = np.argwhere(
                wigner_symbols_squared_oo[:, :, l_couple_idx] != 0
            )
            occ_valid_indices_ov   = active_wigner_symbols_indices_ov[:, 0]
            unocc_valid_indices_ov = active_wigner_symbols_indices_ov[:, 1]
            occ_valid_indices_oo   = active_wigner_symbols_indices_oo[:, 0]
            occ_partner_indices_oo = active_wigner_symbols_indices_oo[:, 1]
            del active_wigner_symbols_indices_ov, active_wigner_symbols_indices_oo

            radial_kernel = radial_kernels_dict[active_l_couple] * (2 * active_l_couple + 1)

            rpa_response_kernel = np.zeros((n_quad, n_quad))

            orbital_pair_product = constants = None
            for occ_index in np.unique(occ_valid_indices_ov):
                unocc_partners = unocc_valid_indices_ov[occ_valid_indices_ov == occ_index]
                orbital_pair_product = (
                    occ_orbitals[:, occ_index][:, np.newaxis]
                    * unocc_orbitals[:, unocc_partners]
                )
                constants = (
                    prefactors_ov[occ_index, unocc_partners]
                    * wigner_symbols_squared_ov[occ_index, unocc_partners, l_couple_idx]
                )
                rpa_response_kernel += (orbital_pair_product * constants) @ orbital_pair_product.T
            del orbital_pair_product, constants
            rpa_response_kernel *= 2

            orbital_pair_product = constants = None
            for occ_index in np.unique(occ_valid_indices_oo):
                occ_partners = occ_partner_indices_oo[occ_valid_indices_oo == occ_index]
                orbital_pair_product = (
                    occ_orbitals[:, occ_index][:, np.newaxis]
                    * occ_orbitals[:, occ_partners]
                )
                constants = (
                    prefactors_oo[occ_index, occ_partners]
                    * wigner_symbols_squared_oo[occ_index, occ_partners, l_couple_idx]
                )
                rpa_response_kernel += (orbital_pair_product * constants) @ orbital_pair_product.T
            del orbital_pair_product, constants

            rpa_response_kernel /= (2 * active_l_couple + 1)

            # Energy uses Tr / log det; here take the matrix diagonal of the same operator.
            #   density_contrib = (2l''+1) * diag( ν̃χ̃ + log(I - ν̃χ̃) )
            # with log(I - ν̃χ̃) from A = ν̃^{1/2} χ̃ ν̃^{1/2} (see _matrix_log_I_minus_nu_chi).
            nu_chi, log_I_minus_nu_chi = RPACorrelation._matrix_log_I_minus_nu_chi(
                rpa_response_kernel, radial_kernel
            )
            del rpa_response_kernel
            channel_matrix = nu_chi.astype(np.complex128) + log_I_minus_nu_chi
            del nu_chi, log_I_minus_nu_chi
            density_at_frequency += (2 * active_l_couple + 1) * np.diag(channel_matrix)
            del channel_matrix, radial_kernel

        return density_at_frequency


    @staticmethod
    def _compute_correlation_energy_for_single_frequency_old(
        frequency              : float,
        occupation_info        : OccupationInfo,
        full_eigen_energies    : np.ndarray, 
        full_orbitals          : np.ndarray, 
        full_l_terms           : np.ndarray,
        wigner_symbols_squared : np.ndarray,
        radial_kernels_dict    : Dict[int, np.ndarray],
    ) -> float:
        """
        Compute RPA correlation driving term for at given frequency.
        """
        raise RuntimeError(
            RPA_CORRELATION_ENERGY_OLD_DEPRECATED_ERROR
        )

        try:
            frequency = float(frequency)
        except ValueError:
            raise ValueError(FREQUENCY_NOT_FLOAT_ERROR.format(type(frequency)))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        
        # get occupation information
        occupations  = occupation_info.occupations
        occ_l_values = occupation_info.l_values


        # get the number of occupied and unoccupied orbitals
        occ_orbitals_num = len(occ_l_values)

        # get the number of quadrature and n_interior points
        n_quad = radial_kernels_dict[0].shape[0]

        # get occupied and unoccupied orbitals and energies
        occ_orbitals   = full_orbitals[:, :occ_orbitals_num]     # shape: (n_grid, total_orbitals_num)
        occ_energies   = full_eigen_energies[:occ_orbitals_num]  # shape: (total_orbitals_num,)
        occ_l_terms    = full_l_terms[:occ_orbitals_num]         # shape: (total_orbitals_num,)
        unocc_orbitals = full_orbitals[:, occ_orbitals_num:]     # shape: (n_grid, unocc_orbitals_num)
        unocc_energies = full_eigen_energies[occ_orbitals_num:]  # shape: (unocc_orbitals_num,)
        unocc_l_terms  = full_l_terms[occ_orbitals_num:]         # shape: (total_orbitals_num,)

        assert np.all(occ_l_terms == occ_l_values), \
            OCCUPATION_INFO_L_TERMS_NOT_CONSISTENT_WITH_OCCUPATION_INFO_ERROR.format(occ_l_terms, occ_l_values)

        ### ================================================ ###
        ###  Part 1: Compute the RPA correlation prefactors  ###
        ### ================================================ ###

        # Angular degeneracy factors f_p * (2l_q + 1)
        #   shape: (occ_num, unocc_num), where 'o' for 'occupied', 'v' for 'virtual'
        deg_factors_ov = occupations[:, np.newaxis] * (2 * unocc_l_terms + 1)[np.newaxis, :]
        #   shape: (occ_num, occ_num), where 'o' for 'occupied'
        deg_factors_oo = occupations[:, np.newaxis] * (2 * occ_l_terms + 1)[np.newaxis, :] - \
                         occupations[np.newaxis, :] * (2 * occ_l_terms + 1)[:, np.newaxis]

        # Energy differences Δε_{pq} = ε_p - ε_q
        #   shape: (occ_num, unocc_num), where 'o' for 'occupied', 'v' for 'virtual'
        delta_eps_ov = occ_energies[:, np.newaxis] - unocc_energies[np.newaxis, :]
        #   shape: (occ_num, occ_num), where 'o' for 'occupied'
        delta_eps_oo = occ_energies[:, np.newaxis] - occ_energies[np.newaxis, :]


        # Filter out zero entries (valid (p,q) pairs)
        #     shape: (n_valid_pairs_ov, )
        occ_idx_ov, unocc_idx_ov = np.argwhere((deg_factors_ov != 0) & (delta_eps_ov != 0)).T
        deg_factors_valid_ov = deg_factors_ov[occ_idx_ov, unocc_idx_ov]
        delta_eps_valid_ov   = delta_eps_ov[occ_idx_ov, unocc_idx_ov]
        #     shape: (n_valid_pairs_oo, )
        occ_idx_x, occ_idx_y = np.argwhere((deg_factors_oo != 0) & (delta_eps_oo != 0)).T
        deg_factors_valid_oo = deg_factors_oo[occ_idx_x, occ_idx_y]
        delta_eps_valid_oo   = delta_eps_oo[occ_idx_x, occ_idx_y]
        
        
        # Compute Lorentzian frequency response: Δε / (Δε² + ω²)
        #   shape: (n_valid_pairs_ov, )
        lorentzian_factors_ov = delta_eps_valid_ov / (delta_eps_valid_ov ** 2 + frequency ** 2)
        #   shape: (n_valid_pairs_oo, )
        lorentzian_factors_oo = delta_eps_valid_oo / (delta_eps_valid_oo ** 2 + frequency ** 2)

        # Combine angular and frequency factors (without Wigner 3j symbol yet)
        #   shape: (n_valid_pairs_ov, )
        prefactors_q1c_ov = deg_factors_valid_ov * lorentzian_factors_ov
        #   shape: (n_valid_pairs_oo, )
        prefactors_q1c_oo = deg_factors_valid_oo * lorentzian_factors_oo

        ### ================================================== ###
        ###  Part 2: Compute the nonzero Wigner symbols        ###
        ### ================================================== ###

        l_couple_min   = np.min(np.abs(occ_l_values[:, np.newaxis] - full_l_terms[np.newaxis, :])).astype(np.int32)
        l_couple_max   = np.max(       occ_l_values[:, np.newaxis] + full_l_terms[np.newaxis, :] ).astype(np.int32)
        l_couple_range = np.arange(l_couple_min, l_couple_max + 1)


        # Use advanced indexing with broadcasting - one operation instead of triple loop
        #   shape: (occ_orbitals_num, unocc_orbitals_num, l_couple_num)
        wigner_symbols_squared_ov = wigner_symbols_squared[
            occ_l_terms   .astype(np.int32)[:, np.newaxis, np.newaxis],  # shape: (occ_orbitals_num, 1, 1)
            unocc_l_terms .astype(np.int32)[np.newaxis, :, np.newaxis],  # shape: (1, unocc_orbitals_num, 1)
            l_couple_range.astype(np.int32)[np.newaxis, np.newaxis, :],  # shape: (1, 1, l_couple_num)
        ]
        #   shape: (occ_orbitals_num, occ_orbitals_num, l_couple_num)
        wigner_symbols_squared_oo = wigner_symbols_squared[
            occ_l_terms   .astype(np.int32)[:, np.newaxis, np.newaxis],  # shape: (occ_orbitals_num, 1, 1)
            occ_l_terms   .astype(np.int32)[np.newaxis, :, np.newaxis],  # shape: (1, occ_orbitals_num, 1)
            l_couple_range.astype(np.int32)[np.newaxis, np.newaxis, :],  # shape: (1, 1, l_couple_num)
        ]

        # Compute only the nonzero Wigner symbols for each l_couple channel
        # The selection rule can reduce the number of Wigner symbols to compute
        #   shape: (n_valid_pairs_ov, l_couple_num)
        wigner_symbols_squared_valid_ov = wigner_symbols_squared_ov[occ_idx_ov, unocc_idx_ov, :]
        #   shape: (n_valid_pairs_oo, l_couple_num)
        wigner_symbols_squared_valid_oo = wigner_symbols_squared_oo[occ_idx_x, occ_idx_y, :]
        

        # Compute only the nonzero Wigner symbols for each l_couple channel
        active_l_couple_idx_list           : List[int]        = []
        active_wigner_symbols_indices_list : List[np.ndarray] = []

        for l_couple_idx, l_couple in enumerate(l_couple_range):
            non_zero_wigner_symbols_indices_ov = np.argwhere(wigner_symbols_squared_valid_ov[:, l_couple_idx] != 0)[:, 0]

            # Collect active l_couple and corresponding nonzero Wigner symbols indices
            if len(non_zero_wigner_symbols_indices_ov) != 0:
                active_l_couple_idx_list.append(l_couple_idx)
                active_wigner_symbols_indices_list.append(non_zero_wigner_symbols_indices_ov)


        ### ================================================== ###
        ###  Part 3: Compute the RPA correlation energy        ###
        ### ================================================== ###

        # orbital outer product: φ_p(r) ⊗ φ_q(r) for all (p,q) pairs
        #   shape: (occ_num, occ_num, n_grid)
        orbital_product_outer_oo = np.einsum('li,lj->ijl',
            occ_orbitals,
            occ_orbitals,
            optimize = True,
        )
        
        # Orbital pair product: Φ_{pq}(r) = φ_p(r)φ_q(r) for valid (p,q) pairs
        #   shape: (n_grid, n_valid_pairs_ov)
        orbital_pair_product_ov = occ_orbitals[:, occ_idx_ov] * unocc_orbitals[:, unocc_idx_ov]


        # initialize the q1c and q2c terms
        #   shape: (n_quad,)
        full_correlation_energy_at_single_frequency = 0.0

        # Compute RPA correlation driving term for each l_couple channel
        for active_l_couple_idx, active_wigner_symbols_indices in \
            zip(active_l_couple_idx_list, active_wigner_symbols_indices_list):

            active_l_couple = l_couple_range[active_l_couple_idx]

            # Get radial kernel (Coulomb kernel for angular momentum channel L)
            #   shape: (n_quad, n_quad)
            radial_kernel = radial_kernels_dict[active_l_couple] * (2 * active_l_couple + 1)
            
            # Compute rpa_response_kernel
            #   shape: (n_quad, n_quad)
            rpa_response_kernel_ov = \
                2 * np.einsum(
                    'ij,ik->kj',
                    np.einsum('ji,i->ij',
                        orbital_pair_product_ov[:, active_wigner_symbols_indices],
                        prefactors_q1c_ov[active_wigner_symbols_indices],
                        optimize=True,
                    ),
                    np.einsum('i,ki->ik',
                        wigner_symbols_squared_valid_ov[active_wigner_symbols_indices, active_l_couple],
                        orbital_pair_product_ov[:, active_wigner_symbols_indices],
                        optimize=True,
                    ),
                    optimize=True,
                ) # Reference (Implementation): https://stackoverflow.com/questions/17437523/python-fast-way-to-sum-outer-products
            
            #   rpa_response_kernel_oo: the response kernel for occ-occ pairs (for fractional occupations)
            rpa_response_kernel_oo = \
                np.einsum(
                    'il, ip, i->pl',
                    orbital_product_outer_oo[occ_idx_x, occ_idx_y, :],  # shape: (n_valid_pairs_oo, n_grid)
                    orbital_product_outer_oo[occ_idx_x, occ_idx_y, :],  # shape: (n_valid_pairs_oo, n_grid)
                    prefactors_q1c_oo * wigner_symbols_squared_valid_oo[:, active_l_couple],  # shape: (n_valid_pairs_oo, )
                    optimize=True,
                )
                
            # Combine occ-unocc and occ-occ contributions
            #   χ₀,L(r, r'; iω) = [χ₀,L^(occ-virt)(r, r'; iω) + χ₀,L^(occ-occ)(r, r'; iω)] / (2L+1)
            rpa_response_kernel = (rpa_response_kernel_ov + rpa_response_kernel_oo) / (2 * active_l_couple + 1)


            # Compute dyson solved response
            #   shape: (n_quad, n_quad)
            full_correlation_energy_at_single_frequency += \
                (2 * active_l_couple + 1) * (np.log(np.linalg.det(np.eye(n_quad) - radial_kernel @ rpa_response_kernel)) + np.trace(radial_kernel @ rpa_response_kernel))

        return full_correlation_energy_at_single_frequency




    def compute_correlation_energy(
        self, 
        full_eigen_energies : np.ndarray, 
        full_orbitals       : np.ndarray, 
        full_l_terms        : np.ndarray,
        enable_parallelization: bool = False,
    ) -> float:
        """
        Compute RPA correlation energy from eigenstates.
        """
        assert hasattr(self, 'frequency_grid') and hasattr(self, 'frequency_weights'), \
            PARENT_CLASS_RPACORRELATION_NOT_INITIALIZED_ERROR
        assert isinstance(enable_parallelization, bool), \
            ENABLE_PARALLELIZATION_NOT_BOOL_ERROR.format(type(enable_parallelization))

        self._validate_full_spectrum_inputs(full_eigen_energies, full_orbitals, full_l_terms)

        l_occ_max    = np.max(self.occ_l_values)
        l_unocc_max  = np.max(full_l_terms)
        l_couple_max = l_occ_max + l_unocc_max

        # Compute RPA Wigner symbols squared array
        wigner_symbols_squared = self._compute_rpa_wigner_symbols_squared(
            l_occ_max    = np.max(self.occ_l_values),
            l_unocc_max  = np.max(full_l_terms),
        )

        # Compute RPA radial kernels dictionary
        radial_kernels_dict = {}
        for l_couple in range(l_couple_max + 1):
            radial_kernels_dict[l_couple] = CoulombCouplingCalculator.radial_kernel(
                l         = l_couple,
                r_nodes   = self.quadrature_nodes,
                r_weights = self.quadrature_weights,
            )

        # Compute RPA correlation energy at each frequency and sum them up
        correlation_energy = 0.0

        if not enable_parallelization:
            for frequency, frequency_weight in zip(self.frequency_grid, self.frequency_weights):
                correlation_energy_at_single_frequency = self._compute_correlation_energy_for_single_frequency(
                    frequency               = frequency,
                    occupation_info         = self.occupation_info,
                    full_eigen_energies     = full_eigen_energies,
                    full_orbitals           = full_orbitals,
                    full_l_terms            = full_l_terms,
                    wigner_symbols_squared  = wigner_symbols_squared,
                    radial_kernels_dict     = radial_kernels_dict,
                )
                
                correlation_energy += correlation_energy_at_single_frequency * frequency_weight
        else:
            import multiprocessing as mp
            from concurrent.futures import ThreadPoolExecutor

            def _single_frequency_task(args):
                idx, (frequency, frequency_weight) = args
                correlation_energy_at_single_frequency = self._compute_correlation_energy_for_single_frequency(
                    frequency               = frequency,
                    occupation_info         = self.occupation_info,
                    full_eigen_energies     = full_eigen_energies,
                    full_orbitals           = full_orbitals,   
                    full_l_terms            = full_l_terms,
                    wigner_symbols_squared  = wigner_symbols_squared,
                    radial_kernels_dict     = radial_kernels_dict,   
                )
                return (
                    idx,
                    correlation_energy_at_single_frequency * frequency_weight,
                )

            n_workers = min(max(1, mp.cpu_count()), len(self.frequency_grid))

            # limit BLAS/OpenMP threads during parallel sections
            if threadpool_limits is not None:
                blas_ctx = threadpool_limits(limits=1)
            else:
                blas_ctx = nullcontext()


            with blas_ctx, ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = executor.map(
                    _single_frequency_task,
                    enumerate(zip(self.frequency_grid, self.frequency_weights))
                )
                for _, correlation_energy_single_weighted in results:
                    correlation_energy += correlation_energy_single_weighted


        return correlation_energy / (2 * np.pi)



    def compute_correlation_energy_density(
        self, 
        full_eigen_energies : np.ndarray, 
        full_orbitals       : np.ndarray, 
        full_l_terms        : np.ndarray,
        enable_parallelization: bool = False,
    ) -> np.ndarray:
        """
        Compute RPA correlation energy density from eigenstates.

        The total RPA correlation energy is
            E_c^{RPA} = (1/2π) Σ_{l''} (2l''+1)
                        ∫ Tr[ χ̃_{0,l''}(iω) ν_{l''}
                              + log(I - χ̃_{0,l''}(iω) ν_{l''}) ] dω.

        The spatial energy density e_c(r) is defined from the *diagonal* of the
        same matrix integrand (instead of the Trace), then converted to a 3D
        density by dividing by 4π r² w (same convention as HF exchange energy
        density): E_c = ∫ e_c(r) 4π r² dr, with no extra ρ factor.

        Returns
        -------
        np.ndarray
            Real RPA correlation energy density at quadrature nodes, shape (n_grid,).
        """
        assert hasattr(self, 'frequency_grid') and hasattr(self, 'frequency_weights'), \
            PARENT_CLASS_RPACORRELATION_NOT_INITIALIZED_ERROR
        assert isinstance(enable_parallelization, bool), \
            ENABLE_PARALLELIZATION_NOT_BOOL_ERROR.format(type(enable_parallelization))

        self._validate_full_spectrum_inputs(full_eigen_energies, full_orbitals, full_l_terms)

        l_occ_max    = np.max(self.occ_l_values)
        l_unocc_max  = np.max(full_l_terms)
        l_couple_max = l_occ_max + l_unocc_max

        wigner_symbols_squared = self._compute_rpa_wigner_symbols_squared(
            l_occ_max    = np.max(self.occ_l_values),
            l_unocc_max  = np.max(full_l_terms),
        )

        radial_kernels_dict = {}
        for l_couple in range(l_couple_max + 1):
            radial_kernels_dict[l_couple] = CoulombCouplingCalculator.radial_kernel(
                l         = l_couple,
                r_nodes   = self.quadrature_nodes,
                r_weights = self.quadrature_weights,
            )

        correlation_energy_density = np.zeros(self.n_quad, dtype=np.complex128)

        if not enable_parallelization:
            for frequency, frequency_weight in zip(self.frequency_grid, self.frequency_weights):
                density_at_frequency = self._compute_correlation_energy_density_for_single_frequency(
                    frequency               = frequency,
                    occupation_info         = self.occupation_info,
                    full_eigen_energies     = full_eigen_energies,
                    full_orbitals           = full_orbitals,
                    full_l_terms            = full_l_terms,
                    wigner_symbols_squared  = wigner_symbols_squared,
                    radial_kernels_dict     = radial_kernels_dict,
                )
                correlation_energy_density += density_at_frequency * frequency_weight
        else:
            import multiprocessing as mp
            from concurrent.futures import ThreadPoolExecutor

            def _single_frequency_task(args):
                idx, (frequency, frequency_weight) = args
                density_at_frequency = self._compute_correlation_energy_density_for_single_frequency(
                    frequency               = frequency,
                    occupation_info         = self.occupation_info,
                    full_eigen_energies     = full_eigen_energies,
                    full_orbitals           = full_orbitals,
                    full_l_terms            = full_l_terms,
                    wigner_symbols_squared  = wigner_symbols_squared,
                    radial_kernels_dict     = radial_kernels_dict,
                )
                return idx, density_at_frequency * frequency_weight

            n_workers = min(max(1, mp.cpu_count()), len(self.frequency_grid))

            if threadpool_limits is not None:
                blas_ctx = threadpool_limits(limits=1)
            else:
                blas_ctx = nullcontext()

            with blas_ctx, ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = executor.map(
                    _single_frequency_task,
                    enumerate(zip(self.frequency_grid, self.frequency_weights))
                )
                for _, density_weighted in results:
                    correlation_energy_density += density_weighted

        # Frequency integral prefactor (same as total energy)
        correlation_energy_density /= (2 * np.pi)

        # Convert Trace-basis diagonal → 3D energy density (HF convention):
        #   E = Σ_i e_i * 4π r_i² w_i
        correlation_energy_density /= (
            4 * np.pi * self.quadrature_nodes**2 * self.quadrature_weights
        )

        # Discard tiny imaginary parts from log branch / roundoff; warn if large.
        imag_abs = np.abs(np.imag(correlation_energy_density))
        imag_max = float(np.max(imag_abs)) if imag_abs.size else 0.0
        real_scale = float(np.max(np.abs(np.real(correlation_energy_density)))) if imag_abs.size else 0.0
        imag_tol = max(1e-8, 1e-10 * max(real_scale, 1.0))
        if imag_max > imag_tol:
            warnings.warn(
                "RPA correlation energy density has a non-negligible imaginary part "
                f"(max |Im| = {imag_max:.3e}); returning the real part.",
                RuntimeWarning,
                stacklevel=2,
            )

        return np.real(correlation_energy_density)

