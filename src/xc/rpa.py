"""
Random Phase Approximation (RPA) correlation from a Kohn-Sham spectrum.

Computes
  * compute_correlation_energy            -> E_c
  * compute_correlation_energy_density    -> e_c(r), and E_c as a by-product
  * compute_rpa_correlation_driving_term  -> Q1c + Q2c, the OEP correlation right hand side

STRUCTURE -- two nested loops.  L (angular channel) is OUTER, so the radial Coulomb
kernel, which depends on L alone, is built once per channel and reused by every omega.
omega (imaginary frequency) is INNER.

    L = 0      [ w0 | w1 | w2 | ... | wN ]   <-- these run in parallel
    L = 1      [ w0 | w1 | w2 | ... | wN ]
      ...                                        one thread pool spans all L
    L = Lmax   [ w0 | w1 | w2 | ... | wN ]

WHY omega AND NOT L is the parallel axis:
  * every omega costs the same, so the workers finish together -- good load balance
  * the L channels do not: the number of contributing orbital pairs falls off with L,
    so splitting there would leave workers idle waiting for the cheap channels


HOW IT IS PARALLELISED -- concurrent.futures.ThreadPoolExecutor, one omega per worker,
one worker per available core (capped at the number of frequencies).  Threads rather
than processes because the per-omega work is nearly all BLAS, which releases the GIL,
and because the large read-only arrays are then shared rather than copied.

BLAS itself is held at one thread per worker, so all the cores go into the frequency
loop.  Giving BLAS threads too was measured slower: threadpool_limits is process-wide,
so they form a pool shared by every worker rather than a per-worker allocation, and the
per-omega matrices are too small to pay for the extra thread teams.  A genuine
workers x threads core count would need the task level to be processes (MPI or
ProcessPoolExecutor) rather than threads.

Pass enable_parallelization=False to run serially.
"""

from __future__ import annotations

import os
import numpy as np
import scipy.linalg
from typing import Tuple, Dict, Literal

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


# Cores this process may actually use.  Affinity, not cpu_count(): under a SLURM
# cgroup cpu_count() reports the whole node.  RPA_N_WORKERS overrides it.
AVAILABLE_CORES = int(os.environ.get("RPA_N_WORKERS") or 0) or (
    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
    else (os.cpu_count() or 1))

# Error messages
OPS_BUILDER_NOT_RADIAL_OPERATORS_BUILDER_ERROR = \
    "Parameter 'ops_builder' must be a 'RadialOperatorsBuilder' instance, get type '{}' instead."
OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR = \
    "Parameter 'occupation_info' must be a 'OccupationInfo' instance, get type '{}' instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR = \
    "Parameter 'frequency_quadrature_point_number' must be an integer, get type {} instead."
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "Parameter 'frequency_quadrature_point_number' must be greater than 0, get {} instead."
GRID_TYPE_NOT_VALID_ERROR = \
    "Parameter `grid_type` must be one of 'sinh' or 'algebraic', get {} instead."
BASE_RULE_NOT_VALID_ERROR = \
    "Parameter `base_rule` must be one of 'midpoint', 'trapezoid', 'clenshaw_curtis' " \
    "or 'gauss_legendre', get {} instead."
NO_CONTRIBUTING_TRANSITION_ERROR = \
    "No contributing transition in the spectrum: every pair is either degenerate or " \
    "shares an occupation per spatial orbital."
FREQUENCY_GRID_NOT_BUILT_ERROR = \
    "The 'sinh' frequency grid is built from the Kohn-Sham spectrum and does not exist " \
    "until one of the compute_* methods is called. Do not read frequency_grid before then."

PARENT_CLASS_RPACORRELATION_NOT_INITIALIZED_ERROR = \
    "Parent class `RPACorrelation` is not initialized, please initialize it first."

L_OCC_MAX_NOT_INTEGER_ERROR = \
    "Parameter `l_occ_max` must be an integer, get type {} instead."
L_UNOCC_MAX_NOT_INTEGER_ERROR = \
    "Parameter `l_unocc_max` must be an integer, get type {} instead."
ENABLE_PARALLELIZATION_NOT_BOOL_ERROR = \
    "Parameter `enable_parallelization` must be a bool, get type {} instead."

# NEW
CONSTANTS_CALLER_NOT_VALID_ERROR = \
    "Parameter `caller` must be one of 'energy' or 'potential', get {} instead."
RADIAL_COULOMB_KERNEL_APPLY_NOT_VALID_ERROR = \
    "Parameter `radial_coulomb_kernel_apply` must be one of 'differential_equation' or 'direct_integration', get {} instead."

ValidGridType                = Literal["sinh", "algebraic"]
ValidBaseRule                = Literal["midpoint", "trapezoid", "clenshaw_curtis",
                                       "gauss_legendre"]
ValidRadialCoulombKernelType = Literal["differential_equation", "direct_integration"]
ValidConstantsCaller         = Literal["energy", "potential"]


class RPACorrelation:
    """
    Compute the RPA correlation energy and the RPA correlation driving term from
    eigenstates.
    """

    def __init__(
        self,
        ops_builder                       : 'RadialOperatorsBuilder',
        occupation_info                   : 'OccupationInfo',
        frequency_quadrature_point_number : int,
        radial_coulomb_kernel_apply       : ValidRadialCoulombKernelType = "differential_equation",
        frequency_grid_type               : ValidGridType = "sinh",
        frequency_base_rule               : ValidBaseRule = "midpoint",
        omega_ceiling                     : float = 1.0e8,
        algebraic_scale                   : float = 1.0,
    ):
        """
        Parameters
        ----------
        ops_builder                       : RadialOperatorsBuilder
        occupation_info                   : OccupationInfo
        frequency_quadrature_point_number : int
        radial_coulomb_kernel_apply       : 'differential_equation' (default) or 'direct_integration'
            'differential_equation' solves the radial Poisson equation in the FE basis and
            is converged at radial quadrature order q same as polynomial order.
            'direct_integration' uses the analytic multipole kernel,
            which needs significantly higher q (and increasing with atomic number)
            for the same accuracy.
            Same labels as ExchangeMethod in hf.py.
        frequency_grid_type               : 'sinh' (default) or 'algebraic'
            'sinh' places the poles of chi_0 on a strip of constant height for every
            transition energy, so the node count is essentially independent of Z
            (~20 from Ar to Au for energy and ~50 for eigenvalues, against 200-400
            for 'algebraic').  Its scale constant is the smallest transition energy,
            so it is built from the spectrum at each compute_* call rather than here.
            'algebraic' is the previous default and is spectrum-independent.
        frequency_base_rule               : quadrature rule underlying the 'sinh' map
            'midpoint' (default), 'trapezoid', 'clenshaw_curtis' or 'gauss_legendre'.
            The transplanted integrand is analytic in a strip, where the equispaced
            rules converge as exp(-pi^2 n / y_max); the clustered ones are optimal for
            a Bernstein ellipse instead.  Ignored by 'algebraic', which is Gauss-Legendre
            by construction.
        omega_ceiling                     : float
            Upper reach of the 'sinh' grid, in Hartree.  An ABSOLUTE ceiling, NOT
            Delta_max: the top of the finite-element basis (~3.7e8 Ha) is a basis
            artifact carrying negligible spectral weight.  Ignored by 'algebraic'.
        algebraic_scale                   : float
            The c in omega = c (1 + xi)/(1 - xi).  c = 1 (default) is the former
            'inverse_linear', c = 2.5 the former 'rational'.  Ignored by 'sinh'.
        """
        assert isinstance(ops_builder, RadialOperatorsBuilder), \
            OPS_BUILDER_NOT_RADIAL_OPERATORS_BUILDER_ERROR.format(type(ops_builder))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        assert isinstance(frequency_quadrature_point_number, int), \
            FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(frequency_quadrature_point_number))
        assert frequency_quadrature_point_number > 0, \
            FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(frequency_quadrature_point_number)
        if radial_coulomb_kernel_apply not in ("differential_equation", "direct_integration"):
            raise ValueError(RADIAL_COULOMB_KERNEL_APPLY_NOT_VALID_ERROR.format(radial_coulomb_kernel_apply))
        if frequency_base_rule not in ("midpoint", "trapezoid", "clenshaw_curtis",
                                       "gauss_legendre"):
            raise ValueError(BASE_RULE_NOT_VALID_ERROR.format(frequency_base_rule))

        # Quadrature data
        self.n_quad             = len(ops_builder.quadrature_nodes)
        self.quadrature_nodes   = ops_builder.quadrature_nodes
        self.quadrature_weights = ops_builder.quadrature_weights

        # This is the operative builder only when the class is instantiated on its own;
        # when it is mixed into a calculator that also carries a dense (Hartree-basis)
        # builder, the radial Poisson  operator is assembled on that one instead
        # --see _radial_poisson_operator.
        self.ops_builder = ops_builder

        # Frequency grid.  'sinh' takes its scale constant from the Kohn-Sham spectrum,
        # which does not exist yet, so it is left unbuilt and rebuilt at the top of each
        # compute_* call -- one grid per SCF iteration, always matched to the current
        # gap.  'algebraic' needs only n and is built now.
        self.frequency_quadrature_point_number = frequency_quadrature_point_number
        self.frequency_grid_type               = frequency_grid_type
        self.frequency_base_rule               = frequency_base_rule
        self.omega_ceiling                     = omega_ceiling
        self.algebraic_scale                   = algebraic_scale

        if frequency_grid_type == "sinh":
            self.frequency_grid    = None
            self.frequency_weights = None
        else:
            self.frequency_grid, self.frequency_weights = \
                self._initialize_frequency_grid_and_weights(
                    frequency_quadrature_point_number, frequency_grid_type,
                    algebraic_scale=algebraic_scale)

        # Occupation information.  The container itself is kept alongside the three
        # unpacked arrays so the class is usable standalone, without relying on a
        # co-inherited calculator to have set it.
        self.occupation_info : OccupationInfo = occupation_info
        self.occupations     : np.ndarray     = occupation_info.occupations
        self.occ_l_values    : np.ndarray     = occupation_info.l_values
        self.occ_n_values    : np.ndarray     = occupation_info.n_values

        self.radial_coulomb_kernel_apply = radial_coulomb_kernel_apply

    # =================================================================================
    #  Frequency grid for the imaginary-axis integral
    # =================================================================================

    @classmethod
    def _initialize_frequency_grid_and_weights(
            cls, n: int, grid_type: ValidGridType,
            full_eigen_energies: np.ndarray = None,
            occupations: np.ndarray = None,
            occ_l_values: np.ndarray = None,
            omega_max: float = None,
            base_rule: ValidBaseRule = "midpoint",
            algebraic_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the frequency grid and weights.

        full_eigen_energies / occupations / occ_l_values / omega_max / base_rule are
        used by 'sinh' only; algebraic_scale by 'algebraic' only.
        """
        assert grid_type in ["sinh", "algebraic"], \
            GRID_TYPE_NOT_VALID_ERROR.format(grid_type)
        if grid_type == "sinh":
            return cls._initialize_frequency_grid_and_weights_sinh(
                n, full_eigen_energies, occupations, occ_l_values, omega_max,
                base_rule)
        return cls._initialize_frequency_grid_and_weights_algebraic(n, algebraic_scale)

    @staticmethod
    def _base_rule_nodes_and_weights(n: int, upper: float,
                                     rule: ValidBaseRule) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quadrature on [0, upper] for one of the four base rules.

        'midpoint' and 'trapezoid' are equispaced, open and closed respectively;
        'clenshaw_curtis' and 'gauss_legendre' cluster at the ends, again closed and
        open.  Only the closed rules carry a node at 0.
        Clenshaw-Curtis weights from the cosine series (Trefethen, Spectral Methods in
        MATLAB, clencurt.m).
        """
        if rule not in ("midpoint", "trapezoid", "clenshaw_curtis", "gauss_legendre"):
            raise ValueError(BASE_RULE_NOT_VALID_ERROR.format(rule))

        if rule == "midpoint":
            return (upper * (2.0 * np.arange(n) + 1.0) / (2.0 * n),
                    np.full(n, upper / n))

        if rule == "gauss_legendre":
            return Quadrature1D.gauss_legendre_on_interval(n, 0.0, upper)

        if n == 1:                                       # degenerate ladder point
            return np.array([0.0]), np.array([upper])

        if rule == "trapezoid":
            weights = np.full(n, upper / (n - 1))
            weights[0]  *= 0.5
            weights[-1] *= 0.5
            return np.linspace(0.0, upper, n), weights

        '''The code below forms the clenshaw-curtis omega points in [0, inf)'''
        order = n - 1
        theta = np.pi * np.arange(n) / order
        weights = np.empty(n)
        interior = theta[1:-1]
        partial = np.ones(n - 2)
        if order % 2 == 0:
            weights[0] = weights[-1] = 1.0 / (order * order - 1.0)
            for k in range(1, order // 2):
                partial -= 2.0 * np.cos(2.0 * k * interior) / (4.0 * k * k - 1.0)
            partial -= np.cos(order * interior) / (order * order - 1.0)
        else:
            weights[0] = weights[-1] = 1.0 / (order * order)
            for k in range(1, (order + 1) // 2):
                partial -= 2.0 * np.cos(2.0 * k * interior) / (4.0 * k * k - 1.0)
        weights[1:-1] = 2.0 * partial / order
        # theta ascends, so cos(theta) descends; reverse for ascending nodes
        return (0.5 * upper * (1.0 + np.cos(theta)[::-1]),
                0.5 * upper * weights[::-1])

    @classmethod
    def _initialize_frequency_grid_and_weights_sinh(
            cls, n: int, full_eigen_energies: np.ndarray, occupations: np.ndarray,
            occ_l_values: np.ndarray, omega_max: float,
            base_rule: ValidBaseRule = "midpoint") -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Sinh mapping, omega = c sinh(y) on y in [0, y_max]:

            c = Delta_min,   y_max = arcsinh(omega_max / c),   domega/dy = c cosh(y)

        chi_0's poles land at Im(y) = pi/2 for EVERY Delta, a strip of constant height,
        so the node count is set by y_max and is only logarithmic in the spectrum --
        ~20 nodes from Ar to Au against 200-400 for 'algebraic'.
        Reference: Johnston & Elliott, Int. J. Numer. Meth. Engng 62 (2005) 564.

        c must not EXCEED any contributing Delta: c <= Delta puts its pole at Im(y) =
        pi/2 exactly, c > Delta at arcsin(Delta/c) < pi/2, i.e. off the strip and toward
        the contour.  Erring small is nearly free, since c enters only through the
        arcsinh.  Delta_min therefore runs over every pair with a non-zero occupation
        difference -- occupied-occupied included, which is any open shell or fractional
        filling -- matching exactly the pairs _build_the_constants gives a non-zero
        A_pq.

        omega_max is an ABSOLUTE ceiling, not Delta_max: the top of the finite-element
        basis (~3.7e8 Ha) is a basis artifact carrying negligible spectral weight.

        The closed base rules evaluate omega = 0, where chi_0's denominator
        Delta^2 + omega^2 degenerates on the p == q diagonal; the numerator vanishes
        there too and both call sites zero those entries.
        """
        eigenvalues     = np.asarray(full_eigen_energies, dtype=float)
        occupations     = np.asarray(occupations, dtype=float)
        occ_l_values    = np.asarray(occ_l_values, dtype=float)
        occupied_number = occ_l_values.size

        occupation_per_orbital     = occupations / (2 * occ_l_values + 1)
        all_occupation_per_orbital = np.zeros(eigenvalues.size)
        all_occupation_per_orbital[:occupied_number] = occupation_per_orbital

        delta_eps   = np.abs(eigenvalues[:occupied_number][:, np.newaxis]
                             - eigenvalues[np.newaxis, :])
        contributes = (occupation_per_orbital[:, np.newaxis]
                       != all_occupation_per_orbital[np.newaxis, :]) \
                      & (delta_eps > 0.0) & np.isfinite(delta_eps)
        assert contributes.any(), NO_CONTRIBUTING_TRANSITION_ERROR
        delta_min = float(delta_eps[contributes].min())

        y_max   = float(np.arcsinh(omega_max / delta_min))
        y, w    = cls._base_rule_nodes_and_weights(n, y_max, base_rule)
        return delta_min * np.sinh(y), w * delta_min * np.cosh(y)

    @staticmethod
    def _initialize_frequency_grid_and_weights_algebraic(
            n: int, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gauss-Legendre nodes xi on [-1, 1] mapped to (0, inf):

            omega(xi) = c (1 + xi)/(1 - xi),   w_omega = w_xi 2c/(1 - xi)^2

        Spectrum-independent.  c = 1 is the former 'inverse_linear', c = 2.5 the former
        'rational'.  Nodes are interior, so omega > 0 strictly -- nothing divides by
        zero in Delta_eps^2 + omega^2 even on the p == q diagonal.
        Reference:
        https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.134.016402/scrpa4_SM.pdf
        """
        reference_nodes, reference_weights = Quadrature1D.gauss_legendre(n)
        nodes   = scale * (1 + reference_nodes) / (1 - reference_nodes)
        weights = reference_weights * 2 * scale / (1 - reference_nodes) ** 2
        return nodes, weights

    # =================================================================================
    #  Angular coupling coefficients -- shared by all three entry points
    # =================================================================================

    @staticmethod
    def _compute_rpa_wigner_symbols_squared(l_occ_max: int, l_unocc_max: int) -> np.ndarray:
        """
        Squared Wigner 3j symbols (l_occ l_unocc l_couple; 0 0 0)^2 -- the angular part
        of the Coulomb coupling.

        Returns
        -------
        (l_occ_max+1, l_unocc_max+1, l_couple_max+1), l_couple_max = l_occ_max + l_unocc_max
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

        l_couple_max = l_occ_max + l_unocc_max
        wigner_symbols_squared = np.zeros((l_occ_max + 1, l_unocc_max + 1, l_couple_max + 1))
        for l_occ in range(l_occ_max + 1):
            for l_unocc in range(l_unocc_max + 1):
                for l_couple in range(l_couple_max + 1):
                    wigner_symbols_squared[l_occ, l_unocc, l_couple] = \
                        CoulombCouplingCalculator.wigner_3j_000(l_occ, l_unocc, l_couple) ** 2
        return wigner_symbols_squared

    # =================================================================================
    #  SHARED building blocks -- used by the energy path AND the driving-term path
    # =================================================================================

    @staticmethod
    def _build_the_constants(occupations, occ_l_values, full_l_terms, full_eigen_energies,
                             caller: ValidConstantsCaller):
        """
        Occupation / degeneracy prefactors for every (occupied, partner) pair, as one
        (occ_num, total_num) array covering occupied and virtual partners alike:

            A_pq = ( f_p/(2l_p+1) - f_q/(2l_q+1) ) (2l_p+1)(2l_q+1)
                 = f_p (2l_q+1) - f_q (2l_p+1)

        which reduces to f_p (2l_q+1) on the virtual columns (f_q = 0).  Correct for
        fractional occupations; occupied-occupied pairs cancel when f_p == f_q.

        caller : 'energy'     only the constants required for chi0 -> q1c/q2c constants returned as None
                 'potential'  also the two unscaled forms

        Returns
        -------
        delta_eps_squared : (occ_num, total_num)  (eps_p - eps_q)^2
        occ_all_constants : A_pq * dEps, virtual columns x2   -> chi_0
        occ_q1c_constants : A_pq * dEps                       -> Q1c
        occ_q2c_constants : A_pq alone                        -> Q2c
        """
        if caller not in ("energy", "potential"):
            raise ValueError(CONSTANTS_CALLER_NOT_VALID_ERROR.format(caller))
        build_q_constants = (caller == "potential")

        occ_num = len(occ_l_values)
        occ_div = occupations / (2 * occ_l_values + 1)

        all_occ_div           = np.zeros(len(full_l_terms))
        all_occ_div[:occ_num] = occ_div

        occ_all_constants = np.zeros((occ_num, len(full_l_terms)))
        occ_all_constants[:, :occ_num] = \
            (occ_div[:, np.newaxis] - occ_div[np.newaxis, :]) * \
            ((2 * occ_l_values + 1)[:, np.newaxis] * (2 * occ_l_values + 1)[np.newaxis, :])
        occ_all_constants[:, occ_num:] = \
            (occ_div[:, np.newaxis] - all_occ_div[occ_num:][np.newaxis, :]) * \
            ((2 * occ_l_values + 1)[:, np.newaxis] * (2 * full_l_terms[occ_num:] + 1)[np.newaxis, :])

        occ_q2c_constants = occ_all_constants.copy() if build_q_constants else None

        delta_eps         = full_eigen_energies[:occ_num][:, np.newaxis] \
                            - full_eigen_energies[np.newaxis, :]
        delta_eps_squared = delta_eps ** 2

        occ_all_constants *= delta_eps

        occ_q1c_constants = occ_all_constants.copy() if build_q_constants else None

        # symmetry factor on the occ-virt block: the virt-occ block is counted here too
        occ_all_constants[:, occ_num:] *= 2

        return delta_eps_squared, occ_all_constants, occ_q1c_constants, occ_q2c_constants

    @staticmethod
    def _build_channel_indices(full_l_terms) -> Dict[int, np.ndarray]:
        """
        Map {l_channel: indices of the orbitals in that channel}, built once above both
        loops so the loops never re-scan full_l_terms.

        full_l_terms is l-major and contiguous, so each entry is a contiguous range.
        Index arrays only -- a few kB.
        """
        return {int(l): np.argwhere(full_l_terms == l)[:, 0]
                for l in np.unique(full_l_terms).astype(np.int32)}

    @staticmethod
    def _one_over_diff_eigenvalues(l_channel, channel_indices, full_eigen_energies) -> np.ndarray:
        """
        1/(eps_i - eps_j) between the states of one l channel, with a zero diagonal --
        the orbital Green's function denominator.

        Note: the zero diagonal excludes the i == j self-term.  Degenerate off-diagonal
        pairs are not guarded, but radial states of a given l are non-degenerate.
        """
        eps  = full_eigen_energies[channel_indices[l_channel]]
        diff = eps[:, np.newaxis] - eps[np.newaxis, :]
        diag = np.arange(len(eps))
        diff[diag, diag] = 1.0
        np.reciprocal(diff, out=diff)
        diff[diag, diag] = 0.0
        return diff

    @staticmethod
    def _build_rpa_response_kernel(frequency, active_l_couple, occ_orbitals, full_orbitals,
                                   occ_l_values, occ_all_constants, delta_eps_squared,
                                   wigner_symbols_squared, channel_indices, n_quad):
        """
        Independent-particle response chi_0,L(r, r'; i omega) for one channel and one
        frequency, (n_quad, n_quad). Used in computing RPA energy, energy density, and RHS terms

        Accumulated one (occupied orbital, l channel) block at a time: the pair product
        is built for that block, contracted in, and rebound next iteration.  Peak
        temporary is (n_quad, n_states_in_channel); no three-index tensor is formed.
        Blocks with a zero Wigner symbol are skipped.

        """
        rpa_response_kernel = np.zeros((n_quad, n_quad))
        for occ_index in range(len(occ_l_values)):
            l_occ = int(occ_l_values[occ_index])
            for l_channel, state_indices in channel_indices.items():
                wigner = wigner_symbols_squared[l_occ, l_channel, active_l_couple]
                if wigner == 0.0:
                    continue
                # The p == q self-pair has delta_eps = 0 and a vanishing numerator, so it
                # contributes nothing; the denominator is neutralised before the divide
                # because the closed base rules evaluate omega = 0, where 0/0 -> nan.
                degenerate   = delta_eps_squared[occ_index, state_indices] == 0
                denominators = delta_eps_squared[occ_index, state_indices] + frequency ** 2
                denominators[degenerate] = 1.0
                constants = occ_all_constants[occ_index, state_indices] / denominators
                constants = constants * wigner
                constants[degenerate] = 0.0
                orbital_pair_product = full_orbitals[:, state_indices] * occ_orbitals[:, occ_index][:, np.newaxis]
                rpa_response_kernel += (orbital_pair_product * constants) @ orbital_pair_product.T
        rpa_response_kernel[:,:] /= (2 * active_l_couple + 1)
        return rpa_response_kernel  

    # =================================================================================
    #  Radial Coulomb kernel nu^(L): differential_equation (default) or direct_integration.
    #  BOTH return the BARE v_L, so no (2L+1) is applied at any call site.
    # =================================================================================

    @staticmethod
    def _precompute_integral_radial_coulomb_kernel_terms(quadrature_nodes, quadrature_weights):
        """
        The two L-INDEPENDENT matrices behind the integral kernel:
        
            r_term1 = w_i w_j / r_>            r_term2 = r_< / r_>
        
        so the kernel at any L costs one power and one product,
        
            (r_term2 ** L) * r_term1 = r_<^L / r_>^(L+1) w_i w_j
        """
        r_i     = quadrature_nodes[:, np.newaxis]
        r_j     = quadrature_nodes[np.newaxis, :]
        w_outer = quadrature_weights[:, np.newaxis] * quadrature_weights[np.newaxis, :]

        r_greater = np.maximum(r_i, r_j)
        r_term1   = w_outer / r_greater
        r_term2   = np.minimum(r_i, r_j) / r_greater
        return r_term1, r_term2

    def _precompute_radial_coulomb_kernel_terms(self) -> tuple:
        """
        The L-INDEPENDENT part of nu^(L) for the active construction, built once above
        the l_couple loop (so to prevent redundant computations)

            'direct_integration'    (r_term1, r_term2)
            'differential_equation' (weighted_interp,)  interpolation matrix scaled by w / r
        """
        if self.radial_coulomb_kernel_apply == "direct_integration":
            return self._precompute_integral_radial_coulomb_kernel_terms(self.quadrature_nodes,
                                                        self.quadrature_weights)
        # The interpolation matrix must come from the same basis as the operator it is
        # later solved against -- see the note in _radial_poisson_operator.
        ops_builder = getattr(self, "ops_builder_dense", self.ops_builder)
        return (ops_builder.get_global_interpolation_matrix()[:, 1:] *
                (self.quadrature_weights / self.quadrature_nodes)[:, np.newaxis],)

    @staticmethod
    def _build_radial_coulomb_kernel_integral(r_term1, r_term2, l_coupling: int) -> np.ndarray:
        """
        Radial Coulomb kernel in the analytic multipole form, from the precomputed
        L-independent terms.
        Returns the bare v_L.  Converges slowly in the radial quadrature:
        """
        if l_coupling == 0:
            return r_term1.copy()
        return (r_term2 ** l_coupling) * r_term1

    def _radial_poisson_operator(self, l_coupling: int) -> np.ndarray:
        """
        Radial Poisson operator for channel L -- the inverse Green's function the
        differential kernel solves against:
        """
        L           = int(l_coupling)
        ops_builder = getattr(self, "ops_builder_dense", self.ops_builder)
        laplacian   = ops_builder.get_laplacian()
        h_r_inv_sq  = ops_builder.get_H_r_inv_sq()
        r_max       = ops_builder.physical_nodes[-1]

        operator = -laplacian[1:, 1:] + h_r_inv_sq[1:, 1:] * (L * (L + 1))
        operator[-1, -1] += L / r_max
        return operator

    def _build_radial_coulomb_kernel_differential(self, l_coupling: int,
                                          weighted_interp: np.ndarray) -> np.ndarray:
        """
        Radial Coulomb kernel by solving the radial Poisson equation in the
        finite-element basis. The radial coulomb kernel is then projected onto the quadrature points
        in this function using the dense projection basis.
        
        A solve is used rather than an explicit inverse -- the Green's function is
        applied to one matrix, so an inverse costs more and conditions worse.
        """
        L        = int(l_coupling)
        operator = self._radial_poisson_operator(L)
        return (2 * L + 1) * (
            weighted_interp @ scipy.linalg.solve(operator, weighted_interp.T,
                                                check_finite=False,
                                                 overwrite_a=True)
        )

    def _build_radial_coulomb_kernel(self, l_coupling: int, coulomb_kernel_terms: tuple) -> np.ndarray:
        """
        Dispatch on self.radial_coulomb_kernel_apply.  Called once per l_couple, INSIDE the
        l_couple loop. Avoid storing it so as to save memory. For heavy elements, the l_coupling can 
        grow large, and storing it can consume too much memory unnecessarily. Computing it is cheap.
        """
        if self.radial_coulomb_kernel_apply == "direct_integration":
            return self._build_radial_coulomb_kernel_integral(coulomb_kernel_terms[0], coulomb_kernel_terms[1],
                                                      l_coupling)
        else:
            return self._build_radial_coulomb_kernel_differential(l_coupling, coulomb_kernel_terms[0])

    # =================================================================================
    #  Coulomb FACTOR G, with nu^(L) = G G^T -- used by the energy path
    # =================================================================================

    @staticmethod
    def _symmetric_inverse_sqrt(matrix: np.ndarray, rcond: float = 1e-14) -> np.ndarray:
        """
        M^(-1/2) for symmetric positive semi-definite M, via eigendecomposition.

        Directions with eigenvalue <= rcond * max are PROJECTED OUT rather than
        inverted -- inverting them would amplify roundoff without adding information,
        since they carry no weight in nu.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        cutoff = rcond * max(float(eigenvalues.max()), 0.0)
        inverse_sqrt = np.zeros_like(eigenvalues)
        keep = eigenvalues > cutoff
        inverse_sqrt[keep] = 1.0 / np.sqrt(eigenvalues[keep])
        return (eigenvectors * inverse_sqrt) @ eigenvectors.T

    @staticmethod
    def _symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
        """
        M^(1/2) for symmetric positive semi-definite M.  Eigenvalues are clamped at
        zero first: they are PSD in exact arithmetic, so any negative one is roundoff.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        np.clip(eigenvalues, 0.0, None, out=eigenvalues)
        return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T

    def _build_coulomb_factor(self, l_coupling: int, coulomb_kernel_terms: tuple) -> np.ndarray:
        """
        G with nu^(L) = G G^T, built once per channel and reused by every frequency.

        WHY A FACTOR AND NOT THE KERNEL.  The energy needs the eigenvalues of
        nu.chi_0, and nu.chi_0 is not symmetric (a product of two symmetric matrices
        is not).  But G^T chi_0 G IS symmetric and carries the same nonzero spectrum,
        because eig(AB)\\{0} = eig(BA)\\{0} with A = G, B = G^T chi_0.  That buys a
        symmetric eigenproblem (eigvalsh, real eigenvalues by construction) instead of
        a nonsymmetric one, at ~2x the cost of the slogdet it replaces.

        WHY THIS FACTORISATION.  The differential kernel is already built as
        nu = (2L+1) W M^(-1) W^T, with W = weighted_interp and M the radial Poisson
        operator.  M is SPD, so M = R^T R gives

            nu = (2L+1) W R^(-1) R^(-T) W^T = G G^T,    G = sqrt(2L+1) W R^(-1)

        and nu itself is never square-rooted.  That matters: cond(nu) ~ 8e20 at the
        default mesh, against cond(M) ~ 5e7-2e8 -- thirteen orders better -- because
        nu inherits both M's spectrum and W's rank deficiency (rank(nu) <= n_basis,
        and n_quad >= n_basis always, since the solver requires
        quadrature_point_number >= 2*polynomial_order+1).  Factoring the small
        well-conditioned operator instead of the large ill-conditioned kernel is the
        whole point.

        Returns (n_quad, n_basis) for the differential kernel; (n_quad, n_quad) for
        the integral kernel, which has no such structure and falls back to nu^(1/2).
        """
        l_coupling = int(l_coupling)
        scale = np.sqrt(2 * l_coupling + 1)

        # The integral kernel is an explicit multipole sum with no W M^-1 W^T form,
        # so there is nothing to factor cheaply -- take the square root of nu itself.
        if self.radial_coulomb_kernel_apply != "differential_equation":
            return self._symmetric_sqrt(
                self._build_radial_coulomb_kernel(l_coupling, coulomb_kernel_terms))

        weighted_interp = coulomb_kernel_terms[0]
        operator = self._radial_poisson_operator(l_coupling)
        operator = 0.5 * (operator + operator.T)   # SPD to ~1e-17; kill the asymmetry

        try:
            # M = R^T R, then G^T = sqrt(2L+1) R^(-T) W^T by triangular solve.
            R = scipy.linalg.cholesky(operator, lower=False)
            factor = scale * scipy.linalg.solve_triangular(
                R, weighted_interp.T, trans='T', lower=False).T
            if np.all(np.isfinite(factor)):
                return factor
        except (np.linalg.LinAlgError, ValueError, scipy.linalg.LinAlgError):
            pass

        # FALLBACK, never raises.  Cholesky rejects a matrix with any negative
        # eigenvalue, and M can pick one up at the roundoff level for a nearly
        # singular channel.  The eigendecomposition route tolerates that: it clamps
        # instead of failing, and gives an equally valid factor since
        # nu = (2L+1) W M^(-1/2) M^(-1/2) W^T = G G^T.  G differs from the Cholesky
        # one by an orthogonal factor on the right, which leaves eig(G^T chi_0 G)
        # unchanged -- so the two paths are interchangeable, not merely similar.
        return scale * (weighted_interp @ self._symmetric_inverse_sqrt(operator))

    # =================================================================================
    #  ENERGY PATH
    # =================================================================================

    def _compute_correlation_energy_per_L_omega(
        self, frequency, active_l_couple, coulomb_factor, occ_orbitals, full_orbitals,
        occ_l_values, occ_all_constants, delta_eps_squared, wigner_symbols_squared,
        channel_indices, n_quad,
    ) -> float:
        """
        ONE (l_couple, frequency) term -- the whole unit dispatched to the thread pool:

            (1/2pi) * (2L+1) * ( ln det(I - nu chi_0)  +  Tr(nu chi_0) )

        The frequency weight is not applied -- the quadrature sum is done once at the end
        of compute_correlation_energy.

        EVALUATED THROUGH EIGENVALUES, NOT slogdet.  With nu = G G^T (see
        _build_coulomb_factor), S = G^T chi_0 G is symmetric and carries the same
        nonzero spectrum as nu.chi_0, so

            ln det(I - nu chi_0) + Tr(nu chi_0) = sum_k [ ln(1 - lam_k) + lam_k ]

        summed PER EIGENVALUE.  The zero eigenvalues G drops contribute
        ln(1-0) + 0 = 0, so nothing is lost by working in the smaller space.

        WHY NOT slogdet + Tr, WHICH IS THE SAME QUANTITY.  Both terms are O(omega^-2)
        and cancel to O(omega^-4).  Once |lam| < sqrt(eps) the determinant rounds to
        exactly 1, slogdet returns 0, and the answer is destroyed -- measured as a hard
        noise floor of ~1e-13 for omega above ~3e4, which the sinh Jacobian (~omega)
        then amplifies into a quadrature error growing LINEARLY with omega_ceiling.
        That was the entire reason omega_ceiling needed tuning.  Per-eigenvalue the
        cancellation never happens: each term is -lam^2/2 - lam^3/3 - ...

        log1p is applied to REAL input deliberately.  numpy has no complex log1p --
        it evaluates log(1+z), which loses exactly as much as slogdet.  chi_0 and nu
        are symmetric, so S is symmetric and eigvalsh returns real eigenvalues
        directly; there is no imaginary part to discard.
        """
        rpa_response_kernel = self._build_rpa_response_kernel(
            frequency, active_l_couple, occ_orbitals, full_orbitals, occ_l_values,
            occ_all_constants, delta_eps_squared, wigner_symbols_squared,
            channel_indices, n_quad,
        )

        screened = coulomb_factor.T @ rpa_response_kernel @ coulomb_factor
        screened = 0.5 * (screened + screened.T)      # discard the roundoff asymmetry
        eigenvalues = np.linalg.eigvalsh(screened)

        return (1 / (2 * np.pi)) * (2 * active_l_couple + 1) * \
            float(np.sum(np.log1p(-eigenvalues) + eigenvalues))

    def compute_correlation_energy(
        self,
        full_eigen_energies    : np.ndarray,
        full_orbitals          : np.ndarray,
        full_l_terms           : np.ndarray,
        enable_parallelization : bool = False,
    ) -> float:
        """
        Compute the RPA correlation energy from the full Kohn-Sham spectrum.

        l_couple outer, frequency inner, so the radial Coulomb kernel is built once per
        channel.  Per-(L, omega) contributions are stored, summed over L, and only then
        contracted with the frequency weights, which keeps the integrand available for
        convergence checks against the frequency grid.
        """
        # 'sinh' only, and rebuilt on EVERY call: its scale constant c = Delta_min comes
        # from the spectrum, so in a self-consistent run the grid follows the gap as the
        # orbitals relax.  The other maps depend on n alone, were built in __init__, and
        # are deliberately left untouched here -- rebuilding them would be pure waste,
        # and would also overwrite a grid a caller had assigned by hand.
        if self.frequency_grid_type == "sinh":
            self.frequency_grid, self.frequency_weights = \
                self._initialize_frequency_grid_and_weights(
                    self.frequency_quadrature_point_number, self.frequency_grid_type,
                    full_eigen_energies, self.occupations, self.occ_l_values,
                    self.omega_ceiling, self.frequency_base_rule,
                    self.algebraic_scale)
        assert self.frequency_grid is not None and self.frequency_weights is not None, \
            FREQUENCY_GRID_NOT_BUILT_ERROR
        assert isinstance(enable_parallelization, bool), \
            ENABLE_PARALLELIZATION_NOT_BOOL_ERROR.format(type(enable_parallelization))
        if hasattr(self, '_validate_full_spectrum_inputs'):
            # supplied by the calculator this class is mixed into, when there is one
            self._validate_full_spectrum_inputs(full_eigen_energies, full_orbitals, full_l_terms)

        occ_num      = len(self.occ_l_values)
        occ_orbitals = full_orbitals[:, :occ_num]
        n_quad       = self.n_quad

        l_occ_max    = int(np.max(self.occ_l_values))
        l_unocc_max  = int(np.max(full_l_terms))
        l_couple_max = l_occ_max + l_unocc_max

        wigner_symbols_squared = self._compute_rpa_wigner_symbols_squared(
            l_occ_max=l_occ_max, l_unocc_max=l_unocc_max)

        # ---- frequency- AND l_couple-independent: built ONCE, outside both loops ----
        delta_eps_squared, occ_all_constants, _, _ = self._build_the_constants(
            self.occupations, self.occ_l_values, full_l_terms, full_eigen_energies,
            caller="energy")
        channel_indices = self._build_channel_indices(full_l_terms)
        coulomb_kernel_terms    = self._precompute_radial_coulomb_kernel_terms()

        correlation_energy_per_L_omega = np.zeros((l_couple_max + 1, len(self.frequency_grid)))

        if not enable_parallelization:
            for active_l_couple in range(l_couple_max + 1):                    # <-- OUTER
                coulomb_factor = self._build_coulomb_factor(active_l_couple, coulomb_kernel_terms)
                for index, frequency in enumerate(self.frequency_grid):        # <-- INNER
                    correlation_energy_per_L_omega[active_l_couple, index] = \
                        self._compute_correlation_energy_per_L_omega(
                            frequency, active_l_couple, coulomb_factor, occ_orbitals,
                            full_orbitals, self.occ_l_values, occ_all_constants,
                            delta_eps_squared, wigner_symbols_squared, channel_indices,
                            n_quad)
        else:
            from concurrent.futures import ThreadPoolExecutor

            # one worker per core over the frequency loop, single-threaded BLAS
            n_workers = min(AVAILABLE_CORES, len(self.frequency_grid))
            blas_ctx  = threadpool_limits(limits=1) \
                        if threadpool_limits is not None else nullcontext()

            # one pool for the whole l_couple loop, not one per channel
            with blas_ctx, ThreadPoolExecutor(max_workers=n_workers) as executor:
                for active_l_couple in range(l_couple_max + 1):                # <-- OUTER
                    coulomb_factor = self._build_coulomb_factor(active_l_couple,
                                                              coulomb_kernel_terms)
                    results = executor.map(                                    # <-- INNER
                        lambda frequency, l=active_l_couple, k=coulomb_factor:
                            self._compute_correlation_energy_per_L_omega(
                                frequency, l, k, occ_orbitals, full_orbitals,
                                self.occ_l_values, occ_all_constants, delta_eps_squared,
                                wigner_symbols_squared, channel_indices, n_quad),
                        self.frequency_grid,
                    )
                    correlation_energy_per_L_omega[active_l_couple, :] = list(results)
                    
        correlation_energy_integrand = np.sum(correlation_energy_per_L_omega, axis = 0)
        
        correlation_energy = np.sum(self.frequency_weights * correlation_energy_integrand)
        
        return correlation_energy


    def _compute_correlation_energy_density_per_L_omega(
        self, frequency, frequency_weight, active_l_couple, coulomb_factor,
        occ_orbitals, full_orbitals, occ_l_values, occ_all_constants,
        delta_eps_squared, wigner_symbols_squared, channel_indices, n_quad,
    ) -> Tuple[np.ndarray, float]:
        """
        One (l_couple, frequency) term of the energy density, and of the energy.

        Both are the same matrix function.  With f(x) = x + ln(1 - x),

            density = diag f(nu.chi_0),        energy = Tr f(nu.chi_0)

        so the density is the diagonal of the matrix whose trace is the energy, and one
        eigendecomposition still serves both.

        EVALUATED VIA THE FACTOR, NOT THE KERNEL.  With nu = G G^T (see
        _build_coulomb_factor) and S = G^T chi_0 G, the identity
        f(AB) = A g(BA) B, valid for analytic f with f(0) = 0 and g(x) = f(x)/x, gives

            f(nu.chi_0) = G g(S) G^T chi_0,    g(S) = V diag(g(lam)) V^T,  S = V lam V^T

        Three consequences, all improvements over forming nu.chi_0 directly:

          * S is SYMMETRIC, so eigh replaces the nonsymmetric eig -- cheaper, and the
            eigenvalues are real by construction rather than by discarding an imaginary
            part that LAPACK produced as roundoff.
          * V is ORTHOGONAL, so V^T replaces np.linalg.inv(eigenvectors).  Inverting a
            nonsymmetric eigenvector matrix is expensive and ill-conditioned when
            eigenvalues cluster -- and here most of the spectrum sits near zero.
          * NO CANCELLATION.  diag(nu.chi_0) + diag(ln(I - nu.chi_0)) subtracted two
            O(omega^-2) quantities to leave O(omega^-4), so once |lam| < sqrt(eps) the
            result was destroyed exactly as slogdet was in the energy path.  Measured
            against the energy path, the old form drifted 1.7e-9 at omega_ceiling = 1e6
            and 7.4e-6 at 1e9.

        The energy below is now the SAME expression compute_correlation_energy uses, so
        the two entry points agree by construction rather than coincidentally.

        Returns
        -------
        density_contribution : (n_quad,)  frequency weight already applied
        energy_contribution  : float      frequency weight NOT applied
        """
        rpa_response_kernel = self._build_rpa_response_kernel(
            frequency, active_l_couple, occ_orbitals, full_orbitals, occ_l_values,
            occ_all_constants, delta_eps_squared, wigner_symbols_squared,
            channel_indices, n_quad,
        )

        screened = coulomb_factor.T @ rpa_response_kernel @ coulomb_factor
        eigenvalues, eigenvectors = np.linalg.eigh(screened)

        # g(lam) = [ln(1-lam) + lam] / lam, the exact closed form -- no series, no
        # threshold.  lam = 0 is a REMOVABLE singularity with the exact limit g(0) = 0,
        # since f(lam) = -lam^2/2 + O(lam^3); the mask is a 0/0 guard, not an
        # approximation.  Exact zeros do occur: S is (n_basis x n_basis) while chi_0's
        # rank is bounded by the contributing orbital pairs in the channel, so the high-L
        # channels are rank deficient.
        #
        # Unlike the slogdet failure this replaces, the quotient cannot be destroyed by
        # cancellation.  ln(1-lam) + lam carries absolute error ~eps*|lam|, so g carries
        # absolute error ~eps whatever lam is, while g(lam) -> -lam/2 -> 0 alongside it;
        # the result is then contracted against a chi_0 that is vanishing there too.
        # lam <= 0 throughout (nu is PSD, chi_0 at imaginary frequency is NSD), so
        # ln(1-lam) never sees a non-positive argument.
        g_values = np.zeros_like(eigenvalues)
        nonzero = eigenvalues != 0.0
        lam_nonzero = eigenvalues[nonzero]
        g_values[nonzero] = (np.log1p(-lam_nonzero) + lam_nonzero) / lam_nonzero

        # diag f(nu.chi_0) = diag([G g(S) G^T] chi_0), a row sum of the Hadamard product
        # since chi_0 is symmetric.  The (n_quad, n_quad) product is formed once and
        # contracted immediately, so no matrix function is retained.
        g_transformed = coulomb_factor @ ((eigenvectors * g_values) @ eigenvectors.T) \
            @ coulomb_factor.T
        diagonal_f = np.sum(g_transformed * rpa_response_kernel, axis=1)

        density_contribution = (1 / (2 * np.pi)) * frequency_weight * \
            (2 * active_l_couple + 1) * diagonal_f

        energy_contribution = (1 / (2 * np.pi)) * (2 * active_l_couple + 1) * \
            float(np.sum(np.log1p(-eigenvalues) + eigenvalues))

        return density_contribution, energy_contribution

    def compute_correlation_energy_density(
        self,
        full_eigen_energies    : np.ndarray,
        full_orbitals          : np.ndarray,
        full_l_terms           : np.ndarray,
        enable_parallelization : bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute the RPA correlation energy density, and the correlation energy as a
        by-product of the same eigendecomposition.
        
        Returns
        -------
        correlation_energy_density : (n_quad,) energy per unit volume, so that
            E_c = sum_i e_i 4 pi r_i^2 w_i -- the exact-exchange convention
        correlation_energy : float
        """
        # 'sinh' only, and rebuilt on EVERY call: its scale constant c = Delta_min comes
        # from the spectrum, so in a self-consistent run the grid follows the gap as the
        # orbitals relax.  The other maps depend on n alone, were built in __init__, and
        # are deliberately left untouched here -- rebuilding them would be pure waste,
        # and would also overwrite a grid a caller had assigned by hand.
        if self.frequency_grid_type == "sinh":
            self.frequency_grid, self.frequency_weights = \
                self._initialize_frequency_grid_and_weights(
                    self.frequency_quadrature_point_number, self.frequency_grid_type,
                    full_eigen_energies, self.occupations, self.occ_l_values,
                    self.omega_ceiling, self.frequency_base_rule,
                    self.algebraic_scale)
        assert self.frequency_grid is not None and self.frequency_weights is not None, \
            FREQUENCY_GRID_NOT_BUILT_ERROR
        assert isinstance(enable_parallelization, bool), \
            ENABLE_PARALLELIZATION_NOT_BOOL_ERROR.format(type(enable_parallelization))
        if hasattr(self, '_validate_full_spectrum_inputs'):
            self._validate_full_spectrum_inputs(full_eigen_energies, full_orbitals, full_l_terms)

        occ_num      = len(self.occ_l_values)
        occ_orbitals = full_orbitals[:, :occ_num]
        n_quad       = self.n_quad

        l_occ_max    = int(np.max(self.occ_l_values))
        l_unocc_max  = int(np.max(full_l_terms))
        l_couple_max = l_occ_max + l_unocc_max

        wigner_symbols_squared = self._compute_rpa_wigner_symbols_squared(
            l_occ_max=l_occ_max, l_unocc_max=l_unocc_max)

        # ---- frequency- AND l_couple-independent: built ONCE, outside both loops ----
        delta_eps_squared, occ_all_constants, _, _ = self._build_the_constants(
            self.occupations, self.occ_l_values, full_l_terms, full_eigen_energies,
            caller="energy")
        channel_indices      = self._build_channel_indices(full_l_terms)
        coulomb_kernel_terms = self._precompute_radial_coulomb_kernel_terms()

        correlation_energy_density     = np.zeros(n_quad)
        correlation_energy_per_L_omega = np.zeros((l_couple_max + 1, len(self.frequency_grid)))

        frequency_pairs = list(zip(self.frequency_grid, self.frequency_weights))

        if not enable_parallelization:
            for active_l_couple in range(l_couple_max + 1):                    # <-- OUTER
                coulomb_factor = self._build_coulomb_factor(active_l_couple,
                                                            coulomb_kernel_terms)
                for index, (frequency, frequency_weight) in enumerate(frequency_pairs):  # <-- INNER
                    density_contribution, energy_contribution = \
                        self._compute_correlation_energy_density_per_L_omega(
                            frequency, frequency_weight, active_l_couple,
                            coulomb_factor, occ_orbitals,
                            full_orbitals, self.occ_l_values, occ_all_constants,
                            delta_eps_squared, wigner_symbols_squared, channel_indices,
                            n_quad)
                    correlation_energy_density += density_contribution
                    correlation_energy_per_L_omega[active_l_couple, index] = energy_contribution
        else:
            from concurrent.futures import ThreadPoolExecutor

            # one worker per core over the frequency loop -- as in the energy path
            n_workers = min(AVAILABLE_CORES, len(frequency_pairs))
            blas_ctx  = threadpool_limits(limits=1) \
                        if threadpool_limits is not None else nullcontext()

            # one pool for the whole l_couple loop, not one per channel
            with blas_ctx, ThreadPoolExecutor(max_workers=n_workers) as executor:
                for active_l_couple in range(l_couple_max + 1):                # <-- OUTER
                    coulomb_factor = self._build_coulomb_factor(active_l_couple,
                                                                coulomb_kernel_terms)
                    # bind the current channel and its factor into each task
                    results = executor.map(                                    # <-- INNER
                        lambda pair, l=active_l_couple, k=coulomb_factor:
                            self._compute_correlation_energy_density_per_L_omega(
                                pair[0], pair[1], l, k, occ_orbitals, full_orbitals,
                                self.occ_l_values, occ_all_constants, delta_eps_squared,
                                wigner_symbols_squared, channel_indices, n_quad),
                        frequency_pairs,
                    )
                    # accumulated on the main thread; += inside the workers would race
                    for index, (density_contribution, energy_contribution) in enumerate(results):
                        correlation_energy_density += density_contribution
                        correlation_energy_per_L_omega[active_l_couple, index] = energy_contribution

        correlation_energy_integrand = np.sum(correlation_energy_per_L_omega, axis = 0)

        correlation_energy = np.sum(self.frequency_weights * correlation_energy_integrand)

        # matrix diagonal -> energy per unit volume: E = sum_i e_i 4 pi r_i^2 w_i
        correlation_energy_density /= (
            4 * np.pi * self.quadrature_nodes ** 2 * self.quadrature_weights
        )

        return correlation_energy_density, correlation_energy

    # =================================================================================
    #  OEP DRIVING-TERM PATH  (Q1c / Q2c)
    # =================================================================================

    def _driving_term_per_L_omega_frequency(
        self, frequency, frequency_weight, active_l_couple, radial_coulomb_kernel,
        occ_orbitals, occ_orbitals_squared, full_orbitals, full_eigen_energies,
        occ_l_values, occ_all_constants, occ_q1c_constants, occ_q2c_constants,
        delta_eps_squared, wigner_symbols_squared, channel_indices,
        n_occ_in_channel, n_quad, diag_indices,
    ):
        """
        One (l_couple, frequency) term of the OEP correlation driving term.  Builds
        chi_0, solves for the correlation part of the screened interaction, and
        contracts it against orbital pair products.

        Returns
        -------
        sigma_1_term : (occ_num, n_quad)  self-energy on the occupied rows; the caller
            contracts it with the orbital Green's function once at the end
        sigma_2_term : (n_quad,)  virtual-row contribution to Q1c, contracted here so no
            (total_num, n_quad) array is needed
        second_term  : (n_quad,)  Q2c

        All three already carry the frequency weight and 1/2pi.  Unlike the energy path
        the weight cannot be deferred, since the caller accumulates over both loops.
        """
        occ_num      = len(occ_l_values)
        sigma_1_term = np.zeros((occ_num, n_quad))
        sigma_2_term = np.zeros(n_quad)
        second_term  = np.zeros(n_quad)

        # ---- chi_0, one block at a time (shared with the energy path) ----
        rpa_response_kernel = self._build_rpa_response_kernel(
            frequency, active_l_couple, occ_orbitals, full_orbitals, occ_l_values,
            occ_all_constants, delta_eps_squared, wigner_symbols_squared,
            channel_indices, n_quad,
        )

        # I - nu.chi_0.  *= -1, not /= -(2L+1): chi_0 already carries its 1/(2L+1).
        # The identity is added in place, avoiding a temporary.
        nu_chi0 = radial_coulomb_kernel @ rpa_response_kernel
        nu_chi0 *= -1
        nu_chi0[diag_indices, diag_indices] += 1.0

        # correlation part of the screened interaction; overwrite_a is safe here
        screened_coulomb_correlation = scipy.linalg.solve(nu_chi0, radial_coulomb_kernel,
                                                overwrite_a=True,
                                                check_finite=False) - radial_coulomb_kernel

        # ---- contract W_c into sigma_1 / sigma_2 / second, block by block ----
        for occ_index in range(occ_num):
            l_occ     = int(occ_l_values[occ_index])
            integral2 = 0.0
            for l_channel, state_indices in channel_indices.items():
                wigner = wigner_symbols_squared[l_occ, l_channel, active_l_couple]
                if wigner == 0.0:
                    continue

                # occupied states come first, so [n_occ_num:] selects the virtuals
                n_occ_num = n_occ_in_channel[l_channel]

                # Same p == q self-pair and same neutralised denominator as in
                # _build_rpa_response_kernel; here nothing downstream would overwrite a
                # nan, so it would reach Q1c and Q2c.
                degenerate   = delta_eps_squared[occ_index, state_indices] == 0
                denominators = delta_eps_squared[occ_index, state_indices] + frequency ** 2
                denominators[degenerate] = 1.0

                constants1_1 = occ_q1c_constants[occ_index, state_indices] * wigner
                constants1_1 = constants1_1 / denominators
                constants1_1[degenerate] = 0.0

                constants2_1 = occ_q2c_constants[occ_index, state_indices] * wigner
                constants2_1 = constants2_1 * \
                    (delta_eps_squared[occ_index, state_indices] - frequency ** 2) / \
                    denominators ** 2
                constants2_1[degenerate] = 0.0

                orbitals_in_channel  = full_orbitals[:, state_indices]
                orbital_pair_product = orbitals_in_channel * occ_orbitals[:, occ_index][:, np.newaxis]
                
                # screened interaction applied to the pair products; this ordering is
                # valid because the kernel is symmetric, and keeps the result contiguous
                temp_mat1 = screened_coulomb_correlation @ orbital_pair_product

                sigma_1_term[occ_index, :] += (temp_mat1 * orbitals_in_channel) @ constants1_1

                one_term     = orbital_pair_product.T @ temp_mat1     # (n_states, n_states)
                constants2_1 *= np.diagonal(one_term).copy()

                one_term = one_term[n_occ_num:, :]
                
                one_term *= self._one_over_diff_eigenvalues(
                    l_channel, channel_indices, full_eigen_energies)[n_occ_num:, :]
                one_term *= constants1_1[n_occ_num:, np.newaxis]
                one_term = orbitals_in_channel @ one_term.T       # (n_quad, n_unocc)

                sigma_2_term += np.einsum(
                    'ji, ji-> j', orbitals_in_channel[:, n_occ_num:], one_term,
                    optimize=False)

                integral2 += float(np.sum(constants2_1))
                constants2_1[n_occ_num:] *= -1.0
                unocc = orbitals_in_channel[:, n_occ_num:]
                second_term += (unocc**2) @ constants2_1[n_occ_num:]
                
            second_term += integral2 * occ_orbitals_squared[:, occ_index]

        # frequency weight and 1/2pi; the extra 2 on second_term is spin degeneracy
        sigma_1_term *= (-1.0 / (2 * np.pi)) * frequency_weight
        sigma_2_term *= (-1.0 / (2 * np.pi)) * frequency_weight
        second_term  *= (2.0 / (2 * np.pi)) * frequency_weight

        return sigma_1_term, sigma_2_term, second_term


    def _accumulate_q1c(self, sigma_1_accumulated, q1c_part, occ_orbitals, full_orbitals,
                        occ_l_values, full_eigen_energies, channel_indices):
        """
        Contract the accumulated occupied-row self-energy with the orbital Green's
        function to give Q1c.

        Runs once after both loops, on the (occ_num, n_quad) accumulator.  The
        virtual-row contribution arrives already contracted, in q1c_part, so the full
        self-energy over all orbitals is never formed.
        """
        q1c_term = np.zeros(self.n_quad)

        for l_channel in range(int(np.max(occ_l_values)) + 1):
            occ_rows = np.argwhere(occ_l_values == l_channel)[:, 0]
            orbitals_in_channel = full_orbitals[:, channel_indices[l_channel]]

            one_term = sigma_1_accumulated[occ_rows, :] @ orbitals_in_channel
            one_term *= self._one_over_diff_eigenvalues(
                l_channel, channel_indices, full_eigen_energies)[:len(occ_rows), :]
            one_term = one_term @ orbitals_in_channel.T

            q1c_term += np.einsum('ki, ik->k', occ_orbitals[:, occ_rows], one_term,
                                  optimize=True)

        q1c_term += q1c_part                 # virtual-row contribution
        q1c_term *= 4                        # spin degeneracy and OEP sign convention
        return q1c_term


    def compute_rpa_correlation_driving_term(
        self,
        full_eigen_energies    : np.ndarray,
        full_orbitals          : np.ndarray,
        full_l_terms           : np.ndarray,
        enable_parallelization : bool = False,
    ) -> np.ndarray:
        """
        Compute the RPA correlation driving term Q1c + Q2c for the OEP equation, summed
        over coupling channels and integrated over frequency.
        Returns
        -------
        (n_quad,) driving term on the radial quadrature grid.
        """
        # 'sinh' only, and rebuilt on EVERY call: its scale constant c = Delta_min comes
        # from the spectrum, so in a self-consistent run the grid follows the gap as the
        # orbitals relax.  The other maps depend on n alone, were built in __init__, and
        # are deliberately left untouched here -- rebuilding them would be pure waste,
        # and would also overwrite a grid a caller had assigned by hand.
        if self.frequency_grid_type == "sinh":
            self.frequency_grid, self.frequency_weights = \
                self._initialize_frequency_grid_and_weights(
                    self.frequency_quadrature_point_number, self.frequency_grid_type,
                    full_eigen_energies, self.occupations, self.occ_l_values,
                    self.omega_ceiling, self.frequency_base_rule,
                    self.algebraic_scale)
        assert self.frequency_grid is not None and self.frequency_weights is not None, \
            FREQUENCY_GRID_NOT_BUILT_ERROR
        assert isinstance(enable_parallelization, bool), \
            ENABLE_PARALLELIZATION_NOT_BOOL_ERROR.format(type(enable_parallelization))
        if hasattr(self, '_validate_full_spectrum_inputs'):
            self._validate_full_spectrum_inputs(full_eigen_energies, full_orbitals, full_l_terms)

        occ_l_values         = self.occ_l_values
        occ_num              = len(occ_l_values)
        n_quad               = self.n_quad
        occ_orbitals         = full_orbitals[:, :occ_num]
        occ_orbitals_squared = occ_orbitals ** 2

        l_occ_max    = int(np.max(occ_l_values))
        l_unocc_max  = int(np.max(full_l_terms))
        l_couple_max = l_occ_max + l_unocc_max

        wigner_symbols_squared = self._compute_rpa_wigner_symbols_squared(
            l_occ_max=l_occ_max, l_unocc_max=l_unocc_max)

        # ---- frequency- AND l_couple-independent: built ONCE ----
        (delta_eps_squared, occ_all_constants,
         occ_q1c_constants, occ_q2c_constants) = self._build_the_constants(
            self.occupations, occ_l_values, full_l_terms, full_eigen_energies,
            caller="potential")
        channel_indices  = self._build_channel_indices(full_l_terms)
        diag_indices     = np.arange(n_quad)
        n_occ_in_channel = {l: int(np.count_nonzero(occ_l_values == l))
                            for l in channel_indices}
        coulomb_kernel_terms     = self._precompute_radial_coulomb_kernel_terms()

        # accumulators over both loops: occupied-row self-energy, the virtual-row part of
        # Q1c, and Q2c
        sigma_1_accumulated = np.zeros((occ_num, n_quad))
        q1c_part            = np.zeros(n_quad)
        q2c_term            = np.zeros(n_quad)

        # both the frequency AND its weight are needed inside, so the pairs stay zipped
        frequency_pairs = list(zip(self.frequency_grid, self.frequency_weights))

        if not enable_parallelization:
            for active_l_couple in range(l_couple_max + 1):                    # <-- OUTER
                radial_coulomb_kernel = self._build_radial_coulomb_kernel(active_l_couple, coulomb_kernel_terms)
                for frequency, frequency_weight in frequency_pairs:            # <-- INNER
                    s1, s2, sec = self._driving_term_per_L_omega_frequency(
                        frequency, frequency_weight, active_l_couple, radial_coulomb_kernel,
                        occ_orbitals, occ_orbitals_squared, full_orbitals,
                        full_eigen_energies, occ_l_values, occ_all_constants,
                        occ_q1c_constants, occ_q2c_constants, delta_eps_squared,
                        wigner_symbols_squared, channel_indices, n_occ_in_channel,
                        n_quad, diag_indices)
                    sigma_1_accumulated += s1
                    q1c_part            += s2
                    q2c_term            += sec
        else:
            from concurrent.futures import ThreadPoolExecutor

            # one worker per core over the frequency loop -- as in the energy path
            n_workers = min(AVAILABLE_CORES, len(frequency_pairs))
            blas_ctx  = threadpool_limits(limits=1) \
                        if threadpool_limits is not None else nullcontext()

            # one pool for the whole l_couple loop, not one per channel
            with blas_ctx, ThreadPoolExecutor(max_workers=n_workers) as executor:
                for active_l_couple in range(l_couple_max + 1):                # <-- OUTER
                    radial_coulomb_kernel = self._build_radial_coulomb_kernel(active_l_couple,
                                                              coulomb_kernel_terms)
                    results = executor.map(                                    # <-- INNER
                        lambda pair, l=active_l_couple, k=radial_coulomb_kernel:
                            self._driving_term_per_L_omega_frequency(
                                pair[0], pair[1], l, k, occ_orbitals,
                                occ_orbitals_squared, full_orbitals, full_eigen_energies,
                                occ_l_values, occ_all_constants, occ_q1c_constants,
                                occ_q2c_constants, delta_eps_squared,
                                wigner_symbols_squared, channel_indices,
                                n_occ_in_channel, n_quad, diag_indices),
                        frequency_pairs,
                    )
                    for s1, s2, sec in results:
                        sigma_1_accumulated += s1
                        q1c_part            += s2
                        q2c_term            += sec

        q1c_term = self._accumulate_q1c(sigma_1_accumulated, q1c_part, occ_orbitals,
                                        full_orbitals, occ_l_values, full_eigen_energies,
                                        channel_indices)

        assert q1c_term.shape == (n_quad,)
        assert q2c_term.shape == (n_quad,)
        # no further /(2 pi): both accumulators already carry it
        return q1c_term + q2c_term
