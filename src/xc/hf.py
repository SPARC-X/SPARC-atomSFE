"""
Hartree-Fock Exchange Calculation

Implements exact (Hartree-Fock) exchange for hybrid functionals.
This is used for orbital-dependent functionals like PBE0, B3LYP, etc.

Reference implementation: datagen/tools/HF_EX.py
"""

from __future__ import annotations
import numpy as np
import scipy
from typing import Dict, Optional, Tuple, List, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.occupation_states import OccupationInfo
    from ..mesh.operators import RadialOperatorsBuilder

# Type aliases
ExchangeMethod = Literal["direct_integration", "differential_equation"]

# Error messages
FACTORIAL_N_MUST_BE_NON_NEGATIVE_INTEGER_ERROR = \
    "parameter 'n' must be a non-negative integer, get {} instead."

L_VALUES_MUST_BE_INTEGERS_ERROR = \
    "parameter 'l_values' in class OccupationInfo must be integers, get type {} instead."

ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR = \
    "parameter 'orbitals' must be a numpy array, get type {} instead."
ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR = \
    "parameter 'orbitals' must be a 2D numpy array, get dimension {} instead."
ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR = \
    "parameter 'orbitals' must have n_grid rows, get {} rows instead of {} rows."
ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR = \
    "parameter 'orbitals' must have n_orbitals columns, get {} columns instead of {} columns."

EXCHANGE_POTENTIAL_OUTPUT_SHAPE_ERROR = \
    "exchange potential must have shape (n_orbitals, n_grid), get shape {} instead."
EXCHANGE_ENERGY_DENSITY_OUTPUT_SHAPE_ERROR = \
    "exchange energy density must have shape (n_grid,), get shape {} instead."

INVALID_EXCHANGE_METHOD_ERROR = \
    "parameter 'method' must be either 'direct_integration' or 'differential_equation', get '{}' instead."

def factorial(n: int) -> int:
    """
    Compute factorial n! = n * (n-1) * ... * 2 * 1
    
    For n = 0, returns 1.
    
    Uses lookup table for common values to avoid repeated computation.
    """
    assert n >= 0 and isinstance(n, int), \
        FACTORIAL_N_MUST_BE_NON_NEGATIVE_INTEGER_ERROR.format(n)

    if n == 0: return 1
    elif n == 1: return 1
    elif n == 2: return 2
    elif n == 3: return 6
    elif n == 4: return 24
    elif n == 5: return 120
    elif n == 6: return 720
    elif n == 7: return 5040
    elif n == 8: return 40320
    else:
        # Use iterative approach to avoid recursion depth issues
        result = 40320
        for i in range(9, n + 1):
            result *= i
        return result



class CoulombCouplingCalculator:

    @staticmethod
    def radial_kernel(l: int, r_nodes: np.ndarray, r_weights: np.ndarray) -> np.ndarray:
        """
        Compute kernel K^(l) with entries:
            K_ij^(l) = [ r_<^l / r_>^(l+1) ] * (w_i w_j) / (2l + 1),
        where r_< = min(r_i, r_j), r_> = max(r_i, r_j).

        This term represents the radial part of the spherical harmonic expansion of the Coulomb interaction.
        """
        r_min = np.minimum(r_nodes, r_nodes.reshape(-1, 1))
        r_max = np.maximum(r_nodes, r_nodes.reshape(-1, 1))
        
        return ((r_min / r_max)**l / r_max) * (r_weights * r_weights.reshape(-1, 1)) / (2*l + 1)


    @staticmethod
    def wigner_3j_000(l1: int, l2: int, L: int) -> float:
        """
        Wigner 3j symbol (l1 l2 L; 0 0 0) with built-in selection rules.
        """
        assert isinstance(l1, int), "l1 must be an integer, get type {} instead".format(type(l1))
        assert isinstance(l2, int), "l2 must be an integer, get type {} instead".format(type(l2))
        assert isinstance(L, int) , "L must be an integer, get type {} instead".format(type(L))
        J = l1 + l2 + L
        # parity: l1 + l2 + L must be even
        if (J & 1) == 1:
            return 0.0
        # triangle inequalities
        if l1 < abs(l2 - L) or l1 > l2 + L:
            return 0.0
        if l2 < abs(l1 - L) or l2 > l1 + L:
            return 0.0
        if L  < abs(l1 - l2) or L  > l1 + l2:
            return 0.0

        g = J // 2
        W = (-1)**g
        W *= np.sqrt(
            factorial(J - 2*l1) * factorial(J - 2*l2) * factorial(J - 2*L)
            / factorial(J + 1)
        )
        W *= factorial(g) / (factorial(g - l1) * factorial(g - l2) * factorial(g - L))
        return float(W)



class HartreeFockExchange:
    """
    Hartree-Fock Exchange Calculator
    
    Computes exact (Hartree-Fock) exchange for hybrid functionals.
    This is used for orbital-dependent functionals like PBE0, B3LYP, etc.
    """
    
    def __init__(
        self,
        ops_builder       : 'RadialOperatorsBuilder',
        ops_builder_dense : 'RadialOperatorsBuilder',
        occupation_info   : 'OccupationInfo'
    ):
        """
        Initialize HF exchange calculator.
        
        Parameters
        ----------
        ops_builder : RadialOperatorsBuilder
            RadialOperatorsBuilder instance containing quadrature data
        ops_builder_dense : RadialOperatorsBuilder
            RadialOperatorsBuilder instance containing quadrature data
        occupation_info : OccupationInfo
            Occupation information containing l_values and occupations
        """
        self.ops_builder       = ops_builder
        self.ops_builder_dense = ops_builder_dense
        self.occupation_info   = occupation_info
        
        # Extract quadrature data from ops_builder
        self.quadrature_nodes   = ops_builder.quadrature_nodes
        self.quadrature_weights = ops_builder.quadrature_weights
        self.n_grid = len(self.quadrature_nodes)
        
        # Extract occupation data
        self.l_values    = occupation_info.l_values
        self.occupations = occupation_info.occupations
        self.n_orbitals  = len(self.l_values)

        assert self.l_values.dtype == int, \
            L_VALUES_MUST_BE_INTEGERS_ERROR.format(self.l_values.dtype)
        
        # Cache for radial Green's function (computed once, reused for all l_value)
        self._radial_green_function_cache = None


    def _compute_exchange_matrix(
        self,
        l_value  : int,
        orbitals : np.ndarray,
        method   : ExchangeMethod = "differential_equation",
    ) -> np.ndarray:
        """
        Compute HF exchange matrix using specified method.
        
        Two methods are available:
        1. "direct_integration": Directly computes the radial kernel K^(L) and performs
           integration on quadrature nodes.
        2. "differential_equation": Solves the radial Poisson equation to obtain the Green's
           function G^(L) and computes the exchange matrix as φ_i^T · G^(L) · φ_i.
        
        Both methods compute the same exchange interaction:
            H_x = -Σ_i f_i Σ_L (2L+1) |W_{l,l_i,L}|² · ∫∫ φ_i(r) φ_i(r') K^(L)(r,r') dr dr'
        
        Parameters
        ----------
        l_value : int
            Angular momentum channel for which to compute exchange matrix
        orbitals : np.ndarray
            Orbital wavefunctions at quadrature nodes, shape (n_grid, n_orbitals)
        method : ExchangeMethod, optional
            Method to use: "direct_integration" or "differential_equation" (default)
            
        Returns
        -------
        np.ndarray
            HF exchange matrix on physical grid, shape (n_physical, n_physical)
        """
        if method == "direct_integration":
            return self._compute_exchange_matrix_direct_integration(l_value, orbitals)
        elif method == "differential_equation":
            return self._compute_exchange_matrix_differential_equation(l_value, orbitals)
        else:
            raise ValueError(INVALID_EXCHANGE_METHOD_ERROR.format(method))


    def _initialize_radial_green_function(self, max_l_coupling: int) -> Dict[int, np.ndarray]:
        """
        Initialize radial Green's function G^(L) for L from 0 to max_l_coupling.
        
        This function is called once and the result is cached to avoid recomputation.
        
        Parameters
        ----------
        max_l_coupling : int
            Maximum angular momentum coupling value L
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping L to radial Green's function G^(L)
        """
        laplacian = self.ops_builder_dense.laplacian
        H_r_inv_sq = self.ops_builder_dense.get_H_r_inv_sq()
        physical_nodes = self.ops_builder_dense.physical_nodes
        R = physical_nodes[-1]
        
        radial_green_function = {}
        for L in range(0, max_l_coupling + 1):
            operator = - laplacian[1:, 1:] + H_r_inv_sq[1:, 1:] * (L * (L + 1))
            # Apply Robin boundary condition: u'(R) + (L+1)/R · u(R) = 0
            operator[-1, -1] += L / R
            radial_green_function[L] = -scipy.linalg.inv(operator)
        
        return radial_green_function


    def _compute_exchange_matrix_differential_equation(
        self, 
        l_value  : int,
        orbitals : np.ndarray
    ) -> np.ndarray:
        """
        Compute HF exchange matrix using differential equation method.
        
        This method solves the radial Poisson equation to obtain the Green's function G^(L),
        which satisfies: L_L G^(L)(r, r') = δ(r - r'), where L_L = -d²/dr² + L(L+1)/r².
        
        Boundary conditions:
        The Green's function on a finite interval [0, R] implicitly contains boundary
        conditions. For physical correctness, the external region (r > R) must satisfy
        the decaying solution u(r) ∝ r^{-L-1} (Coulomb potential must not diverge).
        This gives a Robin boundary condition at r = R:
            u'(R) + (L+1)/R · u(R) = 0
        
        This enforces infinite decay on a finite domain by using the Dirichlet-to-Neumann
        map of the external analytical solution.
        
        Parameters
        ----------
        l_value : int
            Angular momentum channel for which to compute exchange matrix
        orbitals : np.ndarray
            Orbital wavefunctions at quadrature nodes, shape (n_grid, n_orbitals)
            
        Returns
        -------
        np.ndarray
            HF exchange matrix on physical grid, shape (n_physical, n_physical)
        """
        
        # Get necessary matrices from ops_builder
        interpolation_matrix       = self.ops_builder.global_interpolation_matrix
        interpolation_matrix_dense = self.ops_builder_dense.global_interpolation_matrix
        
        # Determine angular momentum coupling range: L from 0 to max(l + l_i)
        l_max            = np.max(l_value + self.l_values)
        l_coupling_range = np.arange(0, int(l_max) + 1)
        n_physical       = interpolation_matrix.shape[1]
        
        H_hf_exchange_matrix = np.zeros((n_physical, n_physical), dtype=float)
        
        # Get or initialize radial Green's function (cached for efficiency)
        if self._radial_green_function_cache is None:
            # Compute maximum L needed across all possible l_value combinations
            max_l_coupling = int(2 * np.max(self.l_values))
            self._radial_green_function_cache = self._initialize_radial_green_function(max_l_coupling)
        
        radial_green_function = self._radial_green_function_cache
        
        # Construct orbital matrix representation on physical grid
        # Each orbital is represented as a matrix on the physical grid
        orbital_matrix_on_grid = np.einsum(
            'i, ij, il, ik -> ljk',
            self.quadrature_weights / self.quadrature_nodes, # (n_quad, )
            interpolation_matrix_dense,                      # (n_quad, n_physical_dense)
            orbitals,                                        # (n_quad, n_orbitals)
            interpolation_matrix,                            # (n_quad, n_physical)
            optimize=True,
        )[:, 1:, 1:-1]  # (n_orbitals, n_physical_dense-1, n_physical-2)


        # Compute Wigner 3j symbols squared for all (l_value, l_i, L) combinations
        wigner_3j_squared = np.array([
            CoulombCouplingCalculator.wigner_3j_000(int(l_value), int(self.l_values[i]), int(l_coupling_range[j]))**2 
            for i in range(len(self.l_values)) 
            for j in range(len(l_coupling_range))
        ]).reshape(len(self.l_values), len(l_coupling_range))
        
        # Find non-zero Wigner terms (only compute for non-zero combinations).
        # Empty subshells (the candidates of occupation_rule='aufbau', set to exactly 0) are skipped:
        # their prefactor is 0, so the result is unchanged and the expensive products are saved.
        occupied_orbitals = np.asarray(self.occupations, dtype=float) != 0.0
        nonzero_wigner_indices = np.argwhere((wigner_3j_squared != 0) & occupied_orbitals[:, np.newaxis])

        for idx in range(nonzero_wigner_indices.shape[0]):
            orbital_idx = nonzero_wigner_indices[idx, 0]  # Orbital index
            l_coupling_idx = nonzero_wigner_indices[idx, 1]  # Angular momentum coupling index
            L = int(l_coupling_range[l_coupling_idx])  # Angular momentum coupling value
            
            # Compute: orbital^T @ (G^(L) @ orbital) -> (n_physical-2, n_physical-2)
            exchange_contribution = orbital_matrix_on_grid[orbital_idx,:,:].T @ (
                radial_green_function[L] @ orbital_matrix_on_grid[orbital_idx,:,:]
            )
            
            # Prefactor: (2L+1) * occupation * Wigner^2
            prefactor = (2*L + 1) * (0.5*self.occupations[orbital_idx]) * wigner_3j_squared[orbital_idx, l_coupling_idx]
            
            # Add contribution to exchange matrix (map to interior nodes)
            H_hf_exchange_matrix[1:-1, 1:-1] += prefactor * exchange_contribution
            

        return H_hf_exchange_matrix


    def _compute_exchange_matrix_direct_integration(
        self, 
        l_value  : int,
        orbitals : np.ndarray
    ) -> np.ndarray:
        """
        Compute HF exchange matrix using direct integration method.
        
        This method directly evaluates the exchange integral by expanding the Coulomb
        interaction 1/|r-r'| in spherical harmonics. The radial kernel is computed
        explicitly as K^(L)(r,r') = (1/(2L+1)) * (r_<^L / r_>^{L+1}) * w_i * w_j,
        where r_< = min(r,r') and r_> = max(r,r').
        
        The analytical form of the kernel automatically incorporates the correct
        boundary conditions (infinite decay at r → ∞), avoiding the need to solve
        differential equations or implement boundary conditions explicitly.
        """
        # interpolation_matrix: (n_quad, n_physical)
        interpolation_matrix = self.ops_builder.global_interpolation_matrix
        
        # Determine angular momentum coupling range
        # L can range from |l - l_i| to l + l_i (triangle inequality for Wigner 3j)
        l_min = np.min(np.abs(l_value - self.l_values))
        l_max = np.max(l_value + self.l_values)
        l_coupling = np.arange(int(l_min), int(l_max) + 1)

        # Compute exchange matrix on physical grid
        n_physical = interpolation_matrix.shape[1]  # (n_quad, n_physical)
        H_hf_exchange_matrix = np.zeros((n_physical, n_physical), dtype=float)


        for l_prime in l_coupling:
            # Angular part: Wigner 3j symbols for angular momentum coupling
            # W_{l,l_i,L} = (l l_i L; 0 0 0) encodes the angular part of the
            # spherical harmonic expansion of the Coulomb interaction
            w3j_values = np.array([
                CoulombCouplingCalculator.wigner_3j_000(int(l_value), int(lj), int(l_prime)) for lj in self.l_values
            ], dtype=float)
            
            # Radial part: compute the radial coupling kernel K^(L)
            # K^(L)_ij = (1/(2L+1)) * (r_<^L / r_>^{L+1}) * w_i * w_j
            # This is the radial projection of 1/|r-r'| in channel L
            radial_kernel = CoulombCouplingCalculator.radial_kernel(
                int(l_prime), self.quadrature_nodes, self.quadrature_weights
            )

            # Compute exchange matrix contribution at quadrature nodes
            # This evaluates: Σ_i f_i |W_{l,l_i,L}|² · φ_i(r) φ_i(r') · K^(L)(r,r')
            # where f_i = 0.5 * (n_↑ + n_↓) is the occupation number
            H_hf_exchange_matrix_l_contribution_at_quadrature_nodes = np.einsum(
                'jn,ln,n->jl',
                orbitals,                                     # (n_grid, n_orbitals)
                orbitals,                                     # (n_grid, n_orbitals)
                (0.5 * self.occupations) * (w3j_values ** 2), # (n_orbitals, )
                optimize=True,
            ) * radial_kernel

            # Interpolate from quadrature nodes to physical grid and accumulate
            # The factor (2L+1) comes from the spherical harmonic expansion
            H_hf_exchange_matrix += (2 * l_prime + 1) * \
                np.einsum(
                    'ij,lk,il->jk',
                    interpolation_matrix,                                    # (n_quad, n_physical)
                    interpolation_matrix,                                    # (n_quad, n_physical)
                    H_hf_exchange_matrix_l_contribution_at_quadrature_nodes, # (n_quad, n_quad)
                    optimize=True,
                )
        
        # Exchange interaction has negative sign (attractive)
        return - H_hf_exchange_matrix


    def compute_exchange_potentials(
        self,
        orbitals,
        method : ExchangeMethod = "differential_equation",
    ):
        """
        Compute Hartree-Fock exchange potentials for all angular momentum channels.

        This function is useful for the OEP calculation. Here, the input orbitals should only contain the occupied orbitals.

        The two methods differ only in how the radial Coulomb kernel v^(L) is built:
        1. "direct_integration": the analytic multipole kernel r_<^L / r_>^(L+1), which
           converges slowly in the radial quadrature order (and more slowly with Z).
        2. "differential_equation": solves the radial Poisson equation in the dense FE
           basis, converged at radial quadrature order comparable to the polynomial order.

        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
        method : ExchangeMethod, optional
            Method to use: "direct_integration" or "differential_equation" (default)

        Returns
        -------
        np.ndarray
            Hartree-Fock exchange potential for all angular momentum channels
            Shape: (len(l_values), n_grid)
        """
        if method not in ("direct_integration", "differential_equation"):
            raise ValueError(INVALID_EXCHANGE_METHOD_ERROR.format(method))
        # Check Type and shape
        assert isinstance(orbitals, np.ndarray), \
            ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR.format(type(orbitals))
        assert orbitals.ndim == 2, \
            ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR.format(orbitals.ndim)
        assert orbitals.shape[0] == self.n_grid, \
            ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR.format(self.n_grid, orbitals.shape[0])
        assert orbitals.shape[1] == self.n_orbitals, \
            ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR.format(self.n_orbitals, orbitals.shape[1])

        # Compute HF exchange matrices for all l channels
        l_coupling = np.arange(0, 2 * np.max(self.l_values) + 1)

        # L-independent part of the differential kernel, built once above the loop.  The
        # interpolation matrix must come from the same basis as the operator it is solved
        # against, hence the dense builder on both.
        if method == "differential_equation":
            weighted_interp = self.ops_builder_dense.global_interpolation_matrix[:, 1:] * \
                              (self.quadrature_weights / self.quadrature_nodes)[:, np.newaxis]
            laplacian  = self.ops_builder_dense.laplacian
            h_r_inv_sq = self.ops_builder_dense.get_H_r_inv_sq()
            r_max      = self.ops_builder_dense.physical_nodes[-1]

        # Compute exchange potential for each l channel
        exchange_potential_l_contribution_list : List[np.ndarray] = []
        for l_prime in l_coupling:
            _wigner_term = np.array([
                CoulombCouplingCalculator.wigner_3j_000(int(l1), int(l2), int(l_prime))**2 for l1 in self.l_values for l2 in self.l_values
            ], dtype=float).reshape(len(self.l_values), len(self.l_values))

            # bare v^(L) on the quadrature grid, weights included; both routes return the
            # same object, so the (2L+1) below is applied identically
            if method == "direct_integration":
                _radial_kernel = CoulombCouplingCalculator.radial_kernel(
                    l_prime, self.quadrature_nodes, self.quadrature_weights)
            else:
                # Robin condition u'(R) + (L+1)/R u(R) = 0 enforces the r^{-L-1} decay
                _operator = - laplacian[1:, 1:] + h_r_inv_sq[1:, 1:] * (l_prime * (l_prime + 1))
                _operator[-1, -1] += l_prime / r_max
                # weighted_interp carries the quadrature weights, so the resolvent is
                # already normalised -- no Wronskian factor here
                _radial_kernel = weighted_interp @ scipy.linalg.solve(
                    _operator, weighted_interp.T, check_finite=False, overwrite_a=True)

            _exchange_potential_l_contribution = -0.5 * np.einsum(
                'ki,ji,ikj->kj',
                _wigner_term * self.occupations,
                orbitals,
                np.einsum('li,lk,jl->ikj',
                    orbitals,
                    orbitals,
                    _radial_kernel * (2 * l_prime + 1),
                ),
                optimize=True,
            )

            exchange_potential_l_contribution_list.append(_exchange_potential_l_contribution)

        exchange_potential = np.sum(exchange_potential_l_contribution_list, axis=0)
        assert exchange_potential.shape == (self.n_orbitals, self.n_grid), \
            EXCHANGE_POTENTIAL_OUTPUT_SHAPE_ERROR.format(exchange_potential.shape, self.n_orbitals, self.n_grid)
        
        return exchange_potential


    def compute_exchange_matrices_dict(
        self,
        orbitals: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Compute Hartree-Fock exchange matrices for all l channels.
        
        This method calculates HF exchange matrices for each angular momentum
        channel separately and returns them as a dictionary.
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping l values to HF exchange matrices
            Keys are unique l values from occupation_info
            Values are HF exchange matrices of shape (n_physical, n_physical)
        """
        # Check Type and shape
        assert isinstance(orbitals, np.ndarray), \
            ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR.format(type(orbitals))
        assert orbitals.ndim == 2, \
            ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR.format(orbitals.ndim)
        assert orbitals.shape[0] == self.n_grid, \
            ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR.format(orbitals.shape[0])
        assert orbitals.shape[1] == self.n_orbitals, \
            ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR.format(orbitals.shape[1])
        

        # Compute HF exchange matrices for all l channels
        H_hf_exchange_matrices_dict : Dict[int, np.ndarray] = {}
        for l_value in self.occupation_info.unique_l_values:
            H_hf_exchange_matrix = self._compute_exchange_matrix(l_value, orbitals)
            H_hf_exchange_matrices_dict[l_value] = H_hf_exchange_matrix
        
        return H_hf_exchange_matrices_dict


    def compute_exchange_energy(
        self,
        orbitals: np.ndarray,
        method: ExchangeMethod = "differential_equation",
    ) -> float:
        """
        Compute Hartree-Fock exchange energy using specified method.
        
        Two methods are available:
        1. "direct_integration": Directly computes the exchange energy by expanding the
           Coulomb interaction 1/|r-r'| in spherical harmonics and performing integration
           on quadrature nodes using the radial kernel K^(L).
        2. "differential_equation": Solves the radial Poisson equation to obtain the Green's
           function G^(L) and computes the exchange energy using the Green's function method.
        
        Both methods compute the same exchange energy:
            E_x = -0.25 * Σ_L (2L+1) Σ_{i,j} f_i f_j |W_{l_i,l_j,L}|² · ∫∫ φ_i(r) φ_j(r) φ_i(r') φ_j(r') K^(L)(r,r') dr dr'
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
        method : ExchangeMethod, optional
            Method to use: "direct_integration" or "differential_equation" (default)
            
        Returns
        -------
        float
            Total Hartree-Fock exchange energy (scalar)
        """
        if method == "direct_integration":
            return self._compute_exchange_energy_direct_integration(orbitals)
        elif method == "differential_equation":
            return self._compute_exchange_energy_differential_equation(orbitals)
        else:
            raise ValueError(INVALID_EXCHANGE_METHOD_ERROR.format(method))


    def _compute_exchange_energy_direct_integration(
        self,
        orbitals: np.ndarray
    ) -> float:
        """
        Compute Hartree-Fock exchange energy using direct integration method.
        
        This method directly evaluates the exchange integral by expanding the Coulomb
        interaction 1/|r-r'| in spherical harmonics. The radial kernel is computed
        explicitly as K^(L)(r,r') = (1/(2L+1)) * (r_<^L / r_>^{L+1}) * w_i * w_j,
        where r_< = min(r,r') and r_> = max(r,r').
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
            
        Returns
        -------
        float
            Total Hartree-Fock exchange energy (scalar)
        """
        # Check Type and shape
        assert isinstance(orbitals, np.ndarray), \
            ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR.format(type(orbitals))
        assert orbitals.ndim == 2, \
            ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR.format(orbitals.ndim)
        assert orbitals.shape[0] == self.n_grid, \
            ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR.format(orbitals.shape[0])
        assert orbitals.shape[1] == self.n_orbitals, \
            ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR.format(orbitals.shape[1])
        
        # Extract occupation data
        l_values    = self.l_values    # Angular momentum quantum numbers
        occupations = self.occupations # Occupation numbers
        
        # Initialize total exchange energy
        E_HF = 0.0
        
        # Loop over all possible l values for coupling
        max_l = np.max(l_values)
        for l_coupling in range(0, 2 * max_l + 1):
            
            # Create Wigner 3j symbol matrix for this l coupling
            wigner_matrix = np.zeros((len(l_values), len(l_values)))
            for i1 in range(len(l_values)):
                for i2 in range(len(l_values)):
                    wigner_matrix[i1, i2] = CoulombCouplingCalculator.wigner_3j_000(int(l_values[i1]), int(l_values[i2]), int(l_coupling))**2
            
            # Create occupation matrix
            occ_matrix = occupations * occupations.reshape(-1, 1)
            
            # Compute radial kernel for this l coupling
            r_kernel = CoulombCouplingCalculator.radial_kernel(l_coupling, self.quadrature_nodes, self.quadrature_weights)
            
            # Compute exchange energy contribution for this l coupling
            # This is the complex einsum from the reference code:
            # 'ij,li,ki,kj,lj,kl->'
            # where:
            # - ij: occupation * wigner matrix
            # - li,ki: orbitals (first two indices)
            # - kj,lj: orbitals (second two indices) 
            # - kl: radial kernel
            exchange_contribution = -0.25 * (2 * l_coupling + 1) * np.einsum(
                'ij,li,ki,kj,lj,kl->',
                occ_matrix * wigner_matrix,
                orbitals,  # li
                orbitals,  # ki  
                orbitals,  # kj
                orbitals,  # lj
                r_kernel,  # kl
                optimize=True
            )
            
            E_HF += exchange_contribution
        
        return E_HF


    def _compute_exchange_energy_differential_equation(
        self,
        orbitals: np.ndarray
    ) -> float:
        """
        Compute HF exchange energy using differential equation method.
        
        Uses Green's function G^(L) from radial Poisson equation: L_L G^(L) = δ(r-r'),
        where L_L = -d²/dr² + L(L+1)/r².
        
        Exchange energy formula:
            E_x = -0.25 * Σ_L (2L+1) Σ_{i,j} f_i f_j |W_{l_i,l_j,L}|² · (φ_i φ_j)^T @ G^(L) @ (φ_i φ_j)
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals at quadrature points, shape (n_grid, n_orbitals)
            
        Returns
        -------
        float
            Total HF exchange energy
        """
        # Check Type and shape
        assert isinstance(orbitals, np.ndarray), \
            ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR.format(type(orbitals))
        assert orbitals.ndim == 2, \
            ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR.format(orbitals.ndim)
        assert orbitals.shape[0] == self.n_grid, \
            ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR.format(orbitals.shape[0])
        assert orbitals.shape[1] == self.n_orbitals, \
            ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR.format(orbitals.shape[1])
        
        # Extract occupation data
        l_values    = self.l_values    # Angular momentum quantum numbers
        occupations = self.occupations # Occupation numbers
        
        # Get necessary matrices from ops_builder
        interpolation_matrix       = self.ops_builder.global_interpolation_matrix
        interpolation_matrix_dense = self.ops_builder_dense.global_interpolation_matrix
        
        # Initialize/cache radial Green's function G^(L)
        if self._radial_green_function_cache is None:
            max_l_coupling = int(2 * np.max(self.l_values))
            self._radial_green_function_cache = self._initialize_radial_green_function(max_l_coupling)
        
        radial_green_function = self._radial_green_function_cache
        
        # Precompute: f_i f_j / 4 and Wigner 3j symbols |W_{l_i,l_j,L}|²
        occupations_matrix = 0.25 * occupations[:, np.newaxis] * occupations[np.newaxis, :]
        
        # Compute Wigner 3j symbol matrix for all (l_value, l_i, L) combinations
        l_coupling_range = np.arange(0, int(2 * np.max(l_values) + 1))
        wigner_3j_squared = np.array([
            CoulombCouplingCalculator.wigner_3j_000(int(l_values[i]), int(l_values[j]), int(l_coupling_range[k]))**2 
            for i in range(len(l_values)) 
            for j in range(len(l_values))
            for k in range(len(l_coupling_range))
        ]).reshape(len(l_values), len(l_values), len(l_coupling_range))

        
        # Initialize total exchange energy
        E_HF = 0.0
        
        # Off-diagonal terms: i ≠ j (computed once, multiplied by 2 for symmetry)
        for i in range(len(l_values)):
            for j in range(i+1, len(l_values)):
                # (φ_i φ_j) interpolated on dense grid
                orbital_product_on_grid = (orbitals[:, i] * orbitals[:, j] * self.quadrature_weights / self.quadrature_nodes) @ interpolation_matrix_dense

                for k, l_couple in enumerate(l_coupling_range):
                    if wigner_3j_squared[i, j, k] != 0:
                        # Contribution: f_i f_j |W|² (2L+1) · (φ_i φ_j)^T @ G^(L) @ (φ_i φ_j)
                        E_HF += occupations_matrix[i, j] * wigner_3j_squared[i, j, k] * (2 * l_couple + 1) * \
                                (orbital_product_on_grid[1:] @ radial_green_function[l_couple] @ orbital_product_on_grid[1:])

        E_HF *= 2  # Account for symmetry: (i,j) and (j,i)

        # Diagonal terms: i = j
        for i in range(len(l_values)):
            orbital_product_on_grid = (orbitals[:, i] * orbitals[:, i] * self.quadrature_weights / self.quadrature_nodes) @ interpolation_matrix_dense
            for k, l_couple in enumerate(l_coupling_range):
                if wigner_3j_squared[i, i, k] != 0:
                    E_HF += occupations_matrix[i, i] * wigner_3j_squared[i, i, k] * (2 * l_couple + 1) * \
                            (orbital_product_on_grid[1:] @ radial_green_function[l_couple] @ orbital_product_on_grid[1:])

        return E_HF




    def compute_exchange_energy_density(
        self,
        orbitals: np.ndarray,
        method: ExchangeMethod = "direct_integration",
    ) -> np.ndarray:
        """
        Compute HF exchange energy density at quadrature points using specified method.
        
        Two methods are available:
        1. "direct_integration": Directly computes the exchange energy density by expanding the
           Coulomb interaction 1/|r-r'| in spherical harmonics and performing integration
           on quadrature nodes using the radial kernel K^(L).
        2. "differential_equation": Solves the radial Poisson equation to obtain the Green's
           function G^(L) and computes the exchange energy density using the Green's function method.
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
        method : ExchangeMethod, optional
            Method to use: "direct_integration" or "differential_equation" (default)
            
        Returns
        -------
        np.ndarray
            Hartree-Fock exchange energy density at quadrature points
            Shape: (n_grid,)
        """
        if method == "direct_integration":
            return self._compute_exchange_energy_density_direct_integration(orbitals)
        elif method == "differential_equation":
            return self._compute_exchange_energy_density_differential_equation(orbitals)
        else:
            raise ValueError(INVALID_EXCHANGE_METHOD_ERROR.format(method))


    def _compute_exchange_energy_density_direct_integration(
        self,
        orbitals: np.ndarray
    ) -> np.ndarray:
        """
        Compute HF exchange energy density using direct integration method.
        
        This method directly evaluates the exchange energy density by expanding the Coulomb
        interaction 1/|r-r'| in spherical harmonics. The radial kernel is computed
        explicitly as K^(L)(r,r') = (1/(2L+1)) * (r_<^L / r_>^{L+1}) * w_i * w_j.
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
            
        Returns
        -------
        np.ndarray
            HF exchange energy density at quadrature points, shape (n_grid,)
        """
        # Check Type and shape
        assert isinstance(orbitals, np.ndarray), \
            ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR.format(type(orbitals))
        assert orbitals.ndim == 2, \
            ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR.format(orbitals.ndim)
        assert orbitals.shape[0] == self.n_grid, \
            ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR.format(orbitals.shape[0])
        assert orbitals.shape[1] == self.n_orbitals, \
            ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR.format(orbitals.shape[1])
        
        # Extract occupation data
        l_values    = self.l_values    # Angular momentum quantum numbers
        occupations = self.occupations # Occupation numbers
        
        # Initialize exchange energy density
        exchange_energy_density = np.zeros(self.n_grid)
        
        # Loop over all possible l values for coupling
        max_l = np.max(l_values)
        for l_coupling in range(0, 2 * max_l + 1):
            
            # Create Wigner 3j symbol matrix for this l coupling
            wigner_matrix = np.zeros((len(l_values), len(l_values)))
            for i1 in range(len(l_values)):
                for i2 in range(len(l_values)):
                    wigner_matrix[i1, i2] = CoulombCouplingCalculator.wigner_3j_000(int(l_values[i1]), int(l_values[i2]), int(l_coupling))**2
            
            # Create occupation matrix
            occ_matrix = occupations * occupations.reshape(-1, 1)
            
            # Compute radial kernel for this l coupling
            r_kernel = CoulombCouplingCalculator.radial_kernel(l_coupling, self.quadrature_nodes, self.quadrature_weights)
            
            # Compute exchange energy density contribution for this l coupling
            # einsum: 'ij,li,ki,kj,lj,kl->l'
            # - ij: occupation * wigner matrix
            # - li,ki: orbitals (first two indices)
            # - kj,lj: orbitals (second two indices) 
            # - kl: radial kernel
            exchange_contribution = -0.25 * (2 * l_coupling + 1) * np.einsum(
                'ij,li,ki,kj,lj,kl->l',
                occ_matrix * wigner_matrix,
                orbitals,  # li
                orbitals,  # ki  
                orbitals,  # kj
                orbitals,  # lj
                r_kernel,  # kl
                optimize=True
            )
            
            exchange_energy_density += exchange_contribution
        
        # Remove the denominator of the quadrature weights
        exchange_energy_density /= (4 * np.pi * self.quadrature_nodes**2 * self.quadrature_weights)

        assert exchange_energy_density.shape == (self.n_grid,), \
            EXCHANGE_ENERGY_DENSITY_OUTPUT_SHAPE_ERROR.format(exchange_energy_density.shape, self.n_grid)

        return exchange_energy_density


    def _compute_exchange_energy_density_differential_equation(
        self,
        orbitals: np.ndarray
    ) -> np.ndarray:
        """
        Compute HF exchange energy density using differential equation method.
                
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
            
        Returns
        -------
        np.ndarray
            HF exchange energy density at quadrature points, shape (n_grid,)
        """
        # Check Type and shape
        assert isinstance(orbitals, np.ndarray), \
            ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR.format(type(orbitals))
        assert orbitals.ndim == 2, \
            ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR.format(orbitals.ndim)
        assert orbitals.shape[0] == self.n_grid, \
            ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR.format(orbitals.shape[0])
        assert orbitals.shape[1] == self.n_orbitals, \
            ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR.format(orbitals.shape[1])
        
        # Extract occupation data
        l_values    = self.l_values    # Angular momentum quantum numbers
        occupations = self.occupations # Occupation numbers
        
        # Get necessary matrices from ops_builder
        interpolation_matrix       = self.ops_builder.global_interpolation_matrix
        interpolation_matrix_dense = self.ops_builder_dense.global_interpolation_matrix
        
        # Initialize/cache radial Green's function G^(L)
        if self._radial_green_function_cache is None:
            max_l_coupling = int(2 * np.max(self.l_values))
            self._radial_green_function_cache = self._initialize_radial_green_function(max_l_coupling)
        
        radial_green_function = self._radial_green_function_cache
        
        # Precompute: f_i f_j / 4 and Wigner 3j symbols |W_{l_i,l_j,L}|²
        occupations_matrix = 0.25 * occupations[:, np.newaxis] * occupations[np.newaxis, :]
        
        # Compute Wigner 3j symbol matrix for all (l_value, l_i, L) combinations
        l_coupling_range = np.arange(0, int(2 * np.max(l_values) + 1))
        wigner_3j_squared = np.array([
            CoulombCouplingCalculator.wigner_3j_000(int(l_values[i]), int(l_values[j]), int(l_coupling_range[k]))**2 
            for i in range(len(l_values)) 
            for j in range(len(l_values))
            for k in range(len(l_coupling_range))
        ]).reshape(len(l_values), len(l_values), len(l_coupling_range))

        
        # Initialize total exchange energy
        exchange_energy_density_dense_grid = np.zeros(len(self.ops_builder_dense.physical_nodes))
        
        # Off-diagonal terms: i ≠ j (computed once, multiplied by 2 for symmetry)
        for i in range(len(l_values)):
            for j in range(i+1, len(l_values)):
                # (φ_i φ_j) interpolated on dense grid: shape (n_physical_dense,)
                orbital_product_on_dense_grid = (orbitals[:, i] * orbitals[:, j] * self.quadrature_weights / self.quadrature_nodes) @ interpolation_matrix_dense

                for k, l_couple in enumerate(l_coupling_range):
                    if wigner_3j_squared[i, j, k] != 0:
                        # Contribution: f_i f_j |W|² (2L+1) · (φ_i φ_j)^T @ G^(L) @ (φ_i φ_j)
                        exchange_energy_density_dense_grid[1:] += \
                            occupations_matrix[i, j] * wigner_3j_squared[i, j, k] * (2 * l_couple + 1) * \
                                np.einsum(
                                    'i, ij, j -> i',
                                    orbital_product_on_dense_grid[1:],
                                    radial_green_function[l_couple],
                                    orbital_product_on_dense_grid[1:],
                                    optimize=True
                                )

        exchange_energy_density_dense_grid *= 2  # Account for symmetry: (i,j) and (j,i)

        # Diagonal terms: i = j
        for i in range(len(l_values)):
            orbital_product_on_dense_grid = (orbitals[:, i] * orbitals[:, i] * self.quadrature_weights / self.quadrature_nodes) @ interpolation_matrix_dense
            for k, l_couple in enumerate(l_coupling_range):
                if wigner_3j_squared[i, i, k] != 0:
                    exchange_energy_density_dense_grid[1:] += \
                        occupations_matrix[i, i] * wigner_3j_squared[i, i, k] * (2 * l_couple + 1) * \
                            np.einsum(
                                'i, ij, j -> i',
                                orbital_product_on_dense_grid[1:],
                                radial_green_function[l_couple],
                                orbital_product_on_dense_grid[1:],
                                optimize=True
                            )
        
        # Interpolate from dense grid to quadrature points
        exchange_energy_density = self.ops_builder.evaluate_single_field_on_grid(
            given_grid = self.quadrature_nodes,
            field_values = exchange_energy_density_dense_grid,
        )

        # Remove the denominator of the quadrature weights
        exchange_energy_density /= (4 * np.pi * self.quadrature_nodes**2 * self.quadrature_weights)

        return exchange_energy_density