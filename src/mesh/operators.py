from __future__ import annotations
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple
from .builder import LagrangeShapeFunctions, Mesh1D, Quadrature1D
from dataclasses import dataclass

MATRIX_ASSEMBLY_DTYPE = np.float64

# Error messages
NUMBER_OF_FINITE_ELEMENTS_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'number_of_finite_elements' must be greater than 0, get {} instead."
FINITE_ELEMENT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'finite_element_number' must be greater than 0, get {} instead."
FINITE_ELEMENT_NUMBER_REQUIRED_ERROR = \
    "parameter 'finite_element_number' is required."
PHYSICAL_NODES_REQUIRED_ERROR = \
    "parameter 'physical_nodes' is required."
QUADRATURE_NODES_REQUIRED_ERROR = \
    "parameter 'quadrature_nodes' is required."
QUADRATURE_WEIGHTS_REQUIRED_ERROR = \
    "parameter 'quadrature_weights' is required."
NUMBER_OF_FINITE_ELEMENTS_DEPRECATED_WARNING = \
    "WARNING: parameter 'number_of_finite_elements' is now deprecated, use 'finite_element_number' instead."
PHYSICAL_NODES_NOT_1D_ARRAY_ERROR = \
    "parameter 'physical_nodes' must be a 1D array."
QUADRATURE_NODES_NOT_1D_ARRAY_ERROR = \
    "parameter 'quadrature_nodes' must be a 1D array."
QUADRATURE_WEIGHTS_NOT_1D_ARRAY_ERROR = \
    "parameter 'quadrature_weights' must be a 1D array."
QUADRATURE_NODES_AND_WEIGHTS_NOT_THE_SAME_LENGTH_ERROR = \
    "parameter 'quadrature_nodes' and quadrature_weights must have the same length."
Z_NUCLEAR_NOT_FLOAT_ERROR = \
    "parameter 'z_nuclear' must be a float, get {} instead."
ALL_ELECTRON_FLAG_NOT_PROVIDED_ERROR = \
    "parameter 'all_electron_flag' must be provided."
V_LOCAL_COMPONENT_PSP_NOT_NP_NDARRAY_ERROR = \
    "parameter 'v_local_component_psp' must be a numpy array, get {} instead."
V_LOCAL_COMPONENT_PSP_NOT_THE_SAME_SIZE_AS_QUADRATURE_NODES_ERROR = \
    "parameter 'v_local_component_psp' must have the same size as quadrature_nodes, get {} and {} instead."
POTENTIAL_VALUES_DO_NOT_MATCH_QUADRATURE_NODES_ERROR = \
    "parameter 'potential_values' shape {} does not match quadrature_nodes shape {}."
POTENTIAL_VALUES_DO_NOT_MATCH_NUMBER_OF_FINITE_ELEMENTS_ERROR = \
    "parameter 'potential_values' shape {} does not match number_of_finite_elements shape {}."
POTENTIAL_VALUES_DO_NOT_MATCH_QUADRATURE_NODE_NUMBER_ERROR = \
    "parameter 'potential_values' shape {} does not match quadrature_node_number shape {}."
POTENTIAL_VALUES_NDIM_ERROR = \
    "parameter 'potential_values' must be a 1D or 2D array, get dimension {} instead."

RHO_TYPE_ERROR_MESSAGE = \
    "parameter 'rho' must be a numpy array, get type {} instead."
RHO_NDIM_ERROR_MESSAGE = \
    "parameter 'rho' must be a 1D array, get dimension {} instead."
RHO_SHAPE_ERROR_MESSAGE = \
    "parameter 'rho' shape {} does not match quadrature_node_number shape {}."
GRAD_RHO_TYPE_ERROR_MESSAGE = \
    "parameter 'grad_rho' must be a numpy array, get type {} instead."
GRAD_RHO_NDIM_ERROR_MESSAGE = \
    "parameter 'grad_rho' must be a 1D array, get dimension {} instead."
GRAD_RHO_SHAPE_ERROR_MESSAGE = \
    "parameter 'grad_rho' shape {} does not match quadrature_node_number shape {}."

DE_XC_DTAU_SHAPE_ERROR_MESSAGE = \
    "parameter 'de_xc_dtau' shape {} does not match quadrature_node_number shape {}."
DE_XC_DTAU_NDIM_ERROR = \
    "parameter 'de_xc_dtau' must be 1D or 2D array, get {}D instead."

GIVEN_GRID_NOT_MONOTONICALLY_INCREASING_ERROR = \
    "The given grid must be monotonically increasing."
GIVEN_GRID_NOT_WITHIN_PHYSICAL_NODES_ERROR = \
    "The given grid must be within the physical nodes."
FIELD_VALUES_NDIM_ERROR_MESSAGE = \
    "parameter 'field_values' must be a 1D array, get dimension {} instead."
FIELD_VALUES_2D_NDIM_ERROR_MESSAGE = \
    "parameter 'field_values' must be a 2D array with shape (n_elem * n_quad, n_fields), get dimension {} instead."
FIELD_VALUES_SHAPE_ERROR_MESSAGE = \
    "parameter 'field_values' shape {} does not match number_of_finite_elements * quadrature_node_number shape {}."

ORBITAL_COEFFICIENTS_SHAPE_ERROR_MESSAGE = \
    "parameter 'field_values' (here the orbital coefficients) length {} does not match number of FE nodes excluding boundary nodes shape {}."


# Warning messages
V_LOCAL_COMPONENT_PSP_NOT_USED_IN_ALL_ELECTRON_CALCULATIONS_WARNING = \
    "WARNING: parameter 'v_local_component_psp' is not used in all-electron calculations"
Z_NUCLEAR_NOT_USED_IN_NON_ALL_ELECTRON_CALCULATIONS_WARNING = \
    "WARNING: parameter 'z_nuclear' is not used in non-all-electron calculations"


@dataclass(frozen=True)
class GridData:
    """
    Immutable grid information needed for XC calculations.
    
    Parameters
    ----------
    finite_element_number : int
        Number of finite elements
    physical_nodes : np.ndarray
        Physical FE nodes
    quadrature_nodes : np.ndarray
        Quadrature points
    quadrature_weights : np.ndarray
        Quadrature weights
    """
    finite_element_number : int
    physical_nodes        : np.ndarray
    quadrature_nodes      : np.ndarray
    quadrature_weights    : np.ndarray
    
    # Deprecated: for backward compatibility
    @property
    def number_of_finite_elements(self) -> int:
        """Deprecated: use finite_element_number instead."""
        return self.finite_element_number


    def __post_init__(self):
        # check if the input parameters are valid
        assert self.finite_element_number > 0, \
            FINITE_ELEMENT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(self.finite_element_number)
        assert self.physical_nodes.ndim == 1, \
            PHYSICAL_NODES_NOT_1D_ARRAY_ERROR
        assert self.quadrature_nodes.ndim == 1, \
            QUADRATURE_NODES_NOT_1D_ARRAY_ERROR
        assert self.quadrature_weights.ndim == 1, \
            QUADRATURE_WEIGHTS_NOT_1D_ARRAY_ERROR
        assert self.quadrature_nodes.shape[0] == self.quadrature_weights.shape[0], \
            QUADRATURE_NODES_AND_WEIGHTS_NOT_THE_SAME_LENGTH_ERROR
            


    @classmethod
    def from_basic(
        cls, 
        domain_size             : float,
        finite_element_number   : int,
        polynomial_order        : int,
        quadrature_point_number : int,
        mesh_type               : str,
        mesh_concentration      : float,
    ) -> 'GridData':

        # Generate Lobatto interpolation nodes on reference interval [-1, 1]
        interp_nodes_ref, _ = Quadrature1D.lobatto(polynomial_order)
        
        # Generate mesh boundaries
        mesh1d = Mesh1D(
            domain_size            = domain_size,
            finite_elements_number = finite_element_number,
            mesh_type              = mesh_type,
            clustering_param       = mesh_concentration,
        )
        boundaries_nodes, _ = mesh1d.generate_mesh_nodes_and_width()

        # Generate standard FE nodes
        global_nodes = Mesh1D.generate_fe_nodes(
            boundaries_nodes = boundaries_nodes,
            interp_nodes     = interp_nodes_ref
        )

        # Generate Gauss-Legendre quadrature nodes and weights
        quadrature_nodes_ref, quadrature_weights_ref = Quadrature1D.gauss_legendre(quadrature_point_number)

        # Map quadrature to physical elements
        quadrature_nodes, quadrature_weights = Mesh1D.map_quadrature_to_physical_elements(
            boundaries_nodes = boundaries_nodes,
            interp_nodes     = quadrature_nodes_ref,
            interp_weights   = quadrature_weights_ref,
            flatten          = True
        )

        return cls(
            finite_element_number = finite_element_number,
            physical_nodes        = global_nodes,
            quadrature_nodes      = quadrature_nodes,
            quadrature_weights    = quadrature_weights,
        )
        


class RadialOperatorsBuilder:
    """
    Assemble radial FE operators and interpolation matrices.

    Initialize from individual parameters, or use from_grid_data() for GridData.
    """

    def __init__(self, 
        finite_element_number     : Optional[int]        = None,
        physical_nodes            : Optional[np.ndarray] = None, 
        quadrature_nodes          : Optional[np.ndarray] = None,
        quadrature_weights        : Optional[np.ndarray] = None,
        verbose                   : bool = False,
        builder_label             : str = "",  # label for the builder, will be printed in the summary

        # Deprecated parameters
        number_of_finite_elements : Optional[int] = None,  # Deprecated: use finite_element_number instead
    ):
        """
        Initialize radial operators builder from individual parameters.
        
        For initialization from GridData, use RadialOperatorsBuilder.from_grid_data().
        
        Parameters
        ----------
        finite_element_number : int
            Number of finite elements
        physical_nodes : np.ndarray
            Physical FE nodes
        quadrature_nodes : np.ndarray
            Quadrature points
        quadrature_weights : np.ndarray
            Quadrature weights
        verbose : bool, optional
            If True, print initialization summary. Default: False
        builder_label : str, optional
            Label for the builder. Default: ""
        number_of_finite_elements : int, optional, deprecated
            Deprecated parameter. Use 'finite_element_number' instead.
        """

        # Handle deprecated parameter
        if number_of_finite_elements is not None:
            if finite_element_number is not None:
                raise ValueError("Cannot specify both 'finite_element_number' and deprecated 'number_of_finite_elements'. Use 'finite_element_number' only.")
            finite_element_number = number_of_finite_elements
            if verbose:
                print(NUMBER_OF_FINITE_ELEMENTS_DEPRECATED_WARNING)

        # Required parameters (must be at the beginning)
        assert physical_nodes is not None, PHYSICAL_NODES_REQUIRED_ERROR
        assert quadrature_nodes is not None, QUADRATURE_NODES_REQUIRED_ERROR
        assert quadrature_weights is not None, QUADRATURE_WEIGHTS_REQUIRED_ERROR
        assert finite_element_number is not None, FINITE_ELEMENT_NUMBER_REQUIRED_ERROR
        
        # dimension and shape checks
        assert physical_nodes.ndim == 1, PHYSICAL_NODES_NOT_1D_ARRAY_ERROR
        assert quadrature_nodes.ndim == 1, QUADRATURE_NODES_NOT_1D_ARRAY_ERROR
        assert quadrature_weights.ndim == 1, QUADRATURE_WEIGHTS_NOT_1D_ARRAY_ERROR
        assert quadrature_nodes.shape[0] == quadrature_weights.shape[0], \
            QUADRATURE_NODES_AND_WEIGHTS_NOT_THE_SAME_LENGTH_ERROR

        self.assembly_dtype        = MATRIX_ASSEMBLY_DTYPE
        self.finite_element_number = finite_element_number
        self.physical_nodes        = np.asarray(physical_nodes)
        self.quadrature_nodes      = np.asarray(quadrature_nodes)
        self.quadrature_weights    = np.asarray(quadrature_weights)
        self.verbose               = verbose
        
        # Reshape to element-wise structure
        self._reshape_grid_data()
        
        # Compute Lagrange basis functions
        self._compute_basis_functions()
        
        # Print summary if verbose
        if self.verbose:
            self._print_initialization_summary(builder_label)
    
    
    @classmethod
    def from_grid_data(cls, grid_data: GridData, verbose: bool = False, builder_label: str = "") -> 'RadialOperatorsBuilder':
        """
        Create RadialOperatorsBuilder from GridData.
        
        Convenience factory method for cleaner code.
        
        Parameters
        ----------
        grid_data : GridData
            Grid data object
        verbose : bool, optional
            If True, print initialization summary. Default: False
        builder_label : str, optional
            Label for the builder. Default: ""
        
        Returns
        -------
        RadialOperatorsBuilder
            Initialized operators builder
        
        Example
        -------
        >>> grid_data = GridData(...)
        >>> ops = RadialOperatorsBuilder.from_grid_data(grid_data, verbose=True)
        """
        return cls(
            finite_element_number = grid_data.finite_element_number,
            physical_nodes        = grid_data.physical_nodes,
            quadrature_nodes      = grid_data.quadrature_nodes,
            quadrature_weights    = grid_data.quadrature_weights,
            verbose               = verbose,
            builder_label         = builder_label,
        )


    def _reshape_grid_data(self):
        """Reshape 1D arrays to (n_elem, n_points) structure."""
        self.physical_nodes_reshaped = Mesh1D.fe_flat_to_block2d(
            self.physical_nodes, 
            self.finite_element_number, 
            endpoints_shared=True
        )
        self.quadrature_nodes_reshaped = Mesh1D.fe_flat_to_block2d(
            self.quadrature_nodes, 
            self.finite_element_number, 
            endpoints_shared=False
        )
        self.quadrature_weights_reshaped = Mesh1D.fe_flat_to_block2d(
            self.quadrature_weights, 
            self.finite_element_number, 
            endpoints_shared=False
        )
        
        self.physical_node_number = self.physical_nodes_reshaped.shape[1]
        self.quadrature_node_number = self.quadrature_nodes_reshaped.shape[1]
    
    
    def _compute_basis_functions(self):
        """Compute Lagrange basis functions and derivatives."""
        self.lagrange_basis, self.lagrange_basis_derivatives = \
            LagrangeShapeFunctions.lagrange_basis_and_derivatives(
                x_node=self.physical_nodes_reshaped,
                x_eval=self.quadrature_nodes_reshaped
            )
        # Shape: (n_elem, n_quad, n_basis)
    
    
    def _print_initialization_summary(self, label: str = ""):
        """Print initialization summary."""
        print("=" * 75)
        print("RadialOperatorsBuilder ({})".format(label).center(75))
        print("=" * 75)
        print(f"\t Number of elements            : {self.finite_element_number}")
        print(f"\t Physical nodes per element    : {self.physical_nodes_reshaped.shape[1]}")
        print(f"\t Quadrature points per element : {self.quadrature_node_number}")
        print(f"\t Total DOFs                    : {self.finite_element_number * self.physical_node_number + 1}")
        print(f"\t Total quadrature points       : {len(self.quadrature_nodes)}")
        print(f"\t Lagrange basis shape          : {self.lagrange_basis.shape}")
        print(f"\t Lagrange derivatives shape    : {self.lagrange_basis_derivatives.shape}")
        print()
    
    # Deprecated: for backward compatibility
    @property
    def number_of_finite_elements(self) -> int:
        """Deprecated: use finite_element_number instead."""
        return self.finite_element_number
    
    
    def get_H_kinetic(self) -> np.ndarray:
        """
        Kinetic energy matrix
        """
        # check if the kinetic energy matrix has been computed
        if hasattr(self, "_H_kinetic"):
            return self._H_kinetic
        
        # compute the kinetic energy matrix
        H_kinetic = - 0.5 * self.get_laplacian()
        
        # store the kinetic energy matrix
        self._H_kinetic = H_kinetic

        return H_kinetic


    def get_laplacian(self) -> np.ndarray:
        """
        Laplacian matrix
        """
        # check if the Laplacian matrix has been computed
        if hasattr(self, "_laplacian"):
            return self._laplacian
        
        # compute the Laplacian matrix
        laplacian_local = - np.einsum("emi,emk,em->eik", 
            self.lagrange_basis_derivatives, 
            self.lagrange_basis_derivatives, 
            self.quadrature_weights_reshaped, 
            optimize=True)

        # Assemble local matrices into global matrix
        laplacian = self._assemble_local_to_global_matrix(laplacian_local)
        
        # store the Laplacian matrix
        self._laplacian = laplacian
        
        return laplacian


    def build_potential_matrix(self, potential_values: np.ndarray) -> np.ndarray:
        """
        Construct Hamiltonian matrix from potential energy values at quadrature points.
        
        This is a general-purpose matrix builder that can be used for any potential:
        - Nuclear Coulomb potential: V(r) = -Z/r
        - Local pseudopotential: V_loc(r)
        - Hartree potential: V_H(r)
        - Exchange-correlation potential: V_xc(r)
        
        Theory
        ------
        Given potential V(r) sampled at quadrature points, compute:
            H_V[i,j] = ∫ φ_i(r) V(r) φ_j(r) dr
        
        Using finite element basis φ and Gauss quadrature.
        
        Parameters
        ----------
        potential_values : np.ndarray
            Potential energy values at quadrature points.
            Can be 1D (flat, length = n_elem * n_quad) or 
            2D (reshaped, shape = (n_elem, n_quad))
        
        Returns
        -------
        H_potential : np.ndarray
            Assembled global Hamiltonian matrix contribution from this potential.
            Shape: (n_global_dofs, n_global_dofs)
        
        Examples
        --------
        >>> # Nuclear Coulomb potential
        >>> V_nuc = -Z / ops.quadrature_nodes
        >>> H_nuc = ops.build_potential_matrix(V_nuc)
        
        >>> # XC potential
        >>> V_xc = compute_xc_potential(rho)
        >>> H_xc = ops.build_potential_matrix(V_xc)
        """
        # Reshape potential if needed
        if potential_values.ndim == 1:
            assert potential_values.size == self.quadrature_nodes_reshaped.size, \
                POTENTIAL_VALUES_DO_NOT_MATCH_QUADRATURE_NODES_ERROR.format(potential_values.size, self.quadrature_nodes_reshaped.size)
            potential_reshaped = potential_values.reshape(self.quadrature_nodes_reshaped.shape)
        elif potential_values.ndim == 2:
            assert potential_values.shape[0] == self.finite_element_number, \
                POTENTIAL_VALUES_DO_NOT_MATCH_NUMBER_OF_FINITE_ELEMENTS_ERROR.format(potential_values.shape[0], self.finite_element_number)
            assert potential_values.shape[1] == self.quadrature_node_number, \
                POTENTIAL_VALUES_DO_NOT_MATCH_QUADRATURE_NODE_NUMBER_ERROR.format(potential_values.shape[1], self.quadrature_node_number)
            potential_reshaped = potential_values
        else:
            raise ValueError(POTENTIAL_VALUES_NDIM_ERROR.format(potential_values.ndim))
        
        # Compute local element matrices: H^e[i,k] = ∫ φ_i V φ_k dr
        H_local = np.einsum("emi,emk,em,em->eik", 
            self.lagrange_basis,       # φ_i at quadrature points
            self.lagrange_basis,       # φ_k at quadrature points
            self.quadrature_weights_reshaped,  # quadrature weights
            potential_reshaped,        # V(r) at quadrature points
            optimize=True)
        
        # Assemble local matrices into global matrix
        H_potential = self._assemble_local_to_global_matrix(H_local)
        
        return H_potential


    def build_metagga_kinetic_density_matrix(self, de_xc_dtau: np.ndarray) -> np.ndarray:
        """
        Construct Hamiltonian matrix from meta-GGA kinetic density term at quadrature points.
        
        This implements the meta-GGA contribution to the Hamiltonian:
            H_metagga[i,j] = ∫ φ'_i * (0.5 * de_xc_dtau) * φ'_j dr
                          + ∫ φ_i * (0.5 * w/r * d(de_xc_dtau)/dr) * φ_j dr
        
        Theory
        ------
        For meta-GGA functionals, the kinetic energy density τ contributes additional
        terms to the Hamiltonian. The implementation follows:
            V3 = de_xc_dtau
            V3grad = d(V3)/dr (computed via derivative matrix)
            
            Term 1: ∫ φ'_i * (0.5 * V3 * w) * φ'_j dr
            Term 2: ∫ φ_i * (0.5 * w/r * V3grad) * φ_j dr
        
        Parameters
        ----------
        de_xc_dtau : np.ndarray
            Derivative of XC energy density w.r.t. kinetic energy density τ.
            Can be 1D (flat, length = n_elem * n_quad) or 
            2D (reshaped, shape = (n_elem, n_quad))
        
        Returns
        -------
        H_metagga : np.ndarray
            Assembled global Hamiltonian matrix contribution from meta-GGA kinetic density term.
            Shape: (n_global_dofs, n_global_dofs)
        """
        # Reshape de_xc_dtau if needed
        if de_xc_dtau.ndim == 1:
            assert de_xc_dtau.size == self.quadrature_nodes_reshaped.size, \
                DE_XC_DTAU_SHAPE_ERROR_MESSAGE.format(de_xc_dtau.size, self.quadrature_nodes_reshaped.size)
            de_xc_dtau_reshaped = de_xc_dtau.reshape(self.quadrature_nodes_reshaped.shape)
        elif de_xc_dtau.ndim == 2:
            assert de_xc_dtau.shape[0] == self.finite_element_number, \
                DE_XC_DTAU_SHAPE_ERROR_MESSAGE.format(de_xc_dtau.shape[0], self.finite_element_number)
            assert de_xc_dtau.shape[1] == self.quadrature_node_number, \
                DE_XC_DTAU_SHAPE_ERROR_MESSAGE.format(de_xc_dtau.shape[1], self.quadrature_node_number)
            de_xc_dtau_reshaped = de_xc_dtau
        else:
            raise ValueError(DE_XC_DTAU_NDIM_ERROR.format(de_xc_dtau.ndim))
        
        # Get derivative matrix for computing d(de_xc_dtau)/dr
        D = self.get_derivative_matrix_with_quadrature_basis()  # Shape: (n_elem, n_quad, n_quad)
        
        # Compute gradient: v3grad = D @ de_xc_dtau (element-wise)
        # For each element: v3grad[e, i] = sum_k D[e, i, k] * de_xc_dtau[e, k]
        de_xc_dtau_grad = np.einsum("eik,ek->ei", D, de_xc_dtau_reshaped)
        
        # Term 1: ∫ φ'_i * (0.5 * de_xc_dtau * w) * φ'_j dr
        # Using derivative basis functions
        H_term1_local = np.einsum("emi,emk,em,em->eik",
            self.lagrange_basis_derivatives,  # φ'_i
            self.lagrange_basis_derivatives,  # φ'_j
            self.quadrature_weights_reshaped,  # w
            0.5 * de_xc_dtau_reshaped,  # 0.5 * de_xc_dtau
            optimize=True)
        
        # Term 2: ∫ φ_i * (0.5 * w/r * d(de_xc_dtau)/dr) * φ_j dr
        # Using regular basis functions
        # Note: 0.5 * w / r * v3grad
        weight_over_r = 0.5 * self.quadrature_weights_reshaped / self.quadrature_nodes_reshaped
        H_term2_local = np.einsum("emi,emk,em,em->eik",
            self.lagrange_basis,  # φ_i
            self.lagrange_basis,  # φ_j
            weight_over_r,  # 0.5 * w / r
            de_xc_dtau_grad,  # d(de_xc_dtau)/dr
            optimize=True)
        
        # Combine both terms
        H_metagga_local = H_term1_local + H_term2_local
        
        # Assemble local matrices into global matrix
        H_metagga = self._assemble_local_to_global_matrix(H_metagga_local)
        
        return H_metagga




    def get_nuclear_coulomb_potential(self, z_nuclear: float) -> np.ndarray:
        """
        Compute nuclear Coulomb potential: V_nuc(r) = -Z/r
        
        Parameters
        ----------
        z_nuclear : float
            Nuclear charge (atomic number for all-electron, or effective charge)
        
        Returns
        -------
        V_nuclear : np.ndarray
            Nuclear Coulomb potential at quadrature points.
            Shape: (n_elem * n_quad,) flat array
        
        Notes
        -----
        For all-electron calculations, z_nuclear is the atomic number Z.
        The negative sign accounts for attractive electron-nucleus interaction.
        """
        try:
            z_nuclear = float(z_nuclear)
        except:
            raise ValueError(Z_NUCLEAR_NOT_FLOAT_ERROR.format(type(z_nuclear)))
        
        # V_nuc(r) = -Z/r at all quadrature points
        V_nuclear = -z_nuclear / self.quadrature_nodes
        
        return V_nuclear


    def get_H_r_inv_sq(self) -> np.ndarray:
        """
        Inverse square of the radial distance matrix
        """
        # check if the inverse square of the radial distance matrix has been computed
        if hasattr(self, "_H_r_inv_sq"):
            return self._H_r_inv_sq
    
        # compute the inverse square of the radial distance matrix
        H_r_inv_sq_local = np.einsum("emi,emk,em,em->eik", 
            self.lagrange_basis, 
            self.lagrange_basis, 
            self.quadrature_weights_reshaped, 
            1.0 / self.quadrature_nodes_reshaped**2, 
            optimize=True)
        
        # Assemble local matrices into global matrix
        H_r_inv_sq = self._assemble_local_to_global_matrix(H_r_inv_sq_local)
        
        # store the inverse square of the radial distance matrix
        self._H_r_inv_sq = H_r_inv_sq
        
        return H_r_inv_sq


    def get_S(self, exclude_boundary: bool = False) -> np.ndarray:
        """
        Overlap matrix S
        """
        # check if the overlap matrix has been computed
        if hasattr(self, "_S"):
            if exclude_boundary:
                return self._S[1:-1,1:-1]
            return self._S
        
        # compute the overlap matrix
        S_local = np.einsum("emi,emk,em->eik", 
            self.lagrange_basis, 
            self.lagrange_basis, 
            self.quadrature_weights_reshaped, 
            optimize=True)

        # Assemble local matrices into global matrix
        S = self._assemble_local_to_global_matrix(S_local)
        
        # store the overlap matrix
        self._S = S

        if exclude_boundary:
            return S[1:-1,1:-1]

        return S


    def get_S_inv_sqrt(self) -> np.ndarray:
        """
        Inverse square root of the overlap matrix: S^(-1/2)
        Computed via eigendecomposition: S^(-1/2) = V * Λ^(-1/2) * V^T
        """
        if hasattr(self, "_S_inv_sqrt"):
            return self._S_inv_sqrt
        
        S = self.get_S()
        try:
            eigvals, eigvecs = np.linalg.eigh(S, UPLO='L')
        except TypeError:
            S64 = np.asarray(S, dtype=np.float64)
            eigvals, eigvecs = np.linalg.eigh(S64, UPLO='L')
        S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        S_inv_sqrt = 0.5 * (S_inv_sqrt + S_inv_sqrt.T)  # Symmetrize
        
        self._S_inv_sqrt = S_inv_sqrt
        return S_inv_sqrt


    def get_derivative_matrix(self) -> np.ndarray:
        """
        Differentiation matrix for computing derivatives at quadrature points.
        
        .. deprecated:: 
            This method is deprecated and will be removed in a future version.
            Use :meth:`get_derivative_matrix_with_quadrature_basis` instead.
        
        This matrix D enables direct computation of derivatives from function values:
            f'(x_quad) = D @ f(x_quad)
        
        Theory:
        -------
        For a function f(x) = sum_j c_j φ_j(x) in the FE space,
        we have f' = (dL/dx) @ L⁺ @ f, where:
        - L[i,j] = φ_j(x_quad_i) : basis functions at quadrature points
        - L⁺ : pseudoinverse of L (maps function values -> coefficients)
        - dL/dx[i,j] = φ'_j(x_quad_i) : basis derivatives at quadrature points
        
        Returns
        -------
        np.ndarray
            Shape: (n_elements, n_quad_points, n_quad_points)
            D[e, i, k] maps f(x_k) → f'(x_i) within element e
        """
        warnings.warn(
            "'get_derivative_matrix()' is deprecated and will be removed in a future version. "
            "Use 'get_derivative_matrix_with_quadrature_basis()' instead. "
            "The new method uses quadrature points as basis nodes, which is more efficient "
            "and matches the reference implementation.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # ============================================================================
        # OLD IMPLEMENTATION (preserved for reference, not executed)
        # ============================================================================
        # The original implementation using pseudoinverse of Lagrange basis.
        # This approach computes D = (dL/dx) @ L⁺ where L⁺ is the pseudoinverse.
        # Kept here for reference in case needed for future use cases.
        #
        # if hasattr(self, "_derivative_matrix"):
        #     return self._derivative_matrix
        # 
        # # Compute pseudoinverse of Lagrange basis: L⁺ maps values → coefficients
        # # Shape: (n_elem, n_basis, n_quad) where n_basis = physical_node_number
        # n_elem = self.number_of_finite_elements
        # n_basis = self.physical_node_number
        # n_quad = self.quadrature_node_number
        # 
        # basis_pseudoinverse = np.zeros((n_elem, n_basis, n_quad))
        # for elem_idx in range(n_elem):
        #     # self.lagrange_basis[elem_idx] has shape (n_quad, n_basis)
        #     basis_pseudoinverse[elem_idx] = np.linalg.pinv(self.lagrange_basis[elem_idx])
        # 
        # # Differentiation matrix: D = (dL/dx) @ L⁺
        # # Maps function values at quadrature points to derivative values
        # derivative_matrix = np.matmul(self.lagrange_basis_derivatives, basis_pseudoinverse)
        # 
        # self._derivative_matrix = derivative_matrix
        # return derivative_matrix
        
        # For backward compatibility, delegate to the new method
        # This ensures the function still works but uses the improved implementation
        return self.get_derivative_matrix_with_quadrature_basis()



    def get_derivative_matrix_with_quadrature_basis(self) -> np.ndarray:
        """
        Compute derivative matrix where basis function nodes = quadrature points.
        
        This method computes a special derivative matrix where the basis function nodes
        are set to be the same as quadrature points. This matches the reference code:
            D = lagrange_polynomial(r_quad1, q, r_quad1, q, S)[1]
        
        In this case:
        - n_quad == n_basis (quadrature points = basis nodes)
        - D[i, j] directly gives derivative of basis function j at quadrature point i
        - No need for inverse or pseudoinverse!
        
        The derivative matrix D satisfies:
            f'(x_quad) = D @ f(x_quad)
        
        where:
        - f(x_quad): function values at quadrature points, shape (n_elem, n_quad)
        - f'(x_quad): derivative values at quadrature points, shape (n_elem, n_quad)
        - D: derivative matrix, shape (n_elem, n_quad, n_quad)
        
        Returns
        -------
        np.ndarray
            Shape: (n_elements, n_quad_points, n_quad_points)
            D[e, i, k] maps f(x_k) → f'(x_i) within element e
        """
        if hasattr(self, "_derivative_matrix_quad_basis"):
            return self._derivative_matrix_quad_basis
        
        # Compute Lagrange basis and derivatives where nodes = quadrature points
        _, lagrange_basis_derivatives_quad = \
            LagrangeShapeFunctions.lagrange_basis_and_derivatives(
                x_node=self.quadrature_nodes_reshaped,  # Basis nodes = quadrature points
                x_eval=self.quadrature_nodes_reshaped,   # Evaluate at quadrature points
            )
        
        derivative_matrix = lagrange_basis_derivatives_quad  # (n_elem, n_quad, n_quad)
        
        self._derivative_matrix_with_quadrature_basis = derivative_matrix

        return derivative_matrix


    def get_derivative_matrix_with_physical_basis(self) -> np.ndarray:
        """
        Compute derivative matrix where basis function nodes = physical nodes.
        """
        if hasattr(self, "_derivative_matrix_with_physical_basis"):
            return self._derivative_matrix_with_physical_basis
        
        _, lagrange_basis_derivatives_physical = \
            LagrangeShapeFunctions.lagrange_basis_and_derivatives(
                x_node=self.physical_nodes_reshaped,
                x_eval=self.physical_nodes_reshaped,
            )
        
        # Store the derivative matrix with physical basis
        self._derivative_matrix_with_physical_basis = lagrange_basis_derivatives_physical
        return lagrange_basis_derivatives_physical



    def get_global_interpolation_matrix(self) -> np.ndarray:
        """
        Global interpolation matrix for evaluating functions at quadrature points.
        
        This matrix maps global nodal coefficients to function values at all quadrature points:
            f(x_quad) = interpolation_matrix @ f_nodal
        
        Theory
        ------
        For a function f(x) represented by nodal values on the global FE grid,
        this matrix evaluates f at all quadrature points across all elements.
        
        Unlike _assemble_local_to_global_matrix (which accumulates contributions),
        this constructs a rectangular interpolation matrix by:
        1. Placing local Lagrange basis matrices in block-diagonal form
        2. Merging columns corresponding to shared boundary nodes
        3. Removing duplicate columns
        
        Returns
        -------
        np.ndarray
            Shape: (n_elements * n_quad_points, n_global_nodes)
            where n_global_nodes = n_elements * polynomial_order + 1
            
            Example: n_elem=17, p=31, n_quad=95 → shape (1615, 528)
        
        Usage
        -----
        Essential for Hartree-Fock exchange (HF, PBE0) where orbital overlaps
        must be computed at quadrature points:
            psi_i(x_quad) = interp_matrix @ psi_i_nodal
        
        Not needed for LDA, GGA, or meta-GGA functionals.
        """
        if hasattr(self, "_global_interpolation_matrix"):
            return self._global_interpolation_matrix
        
        n_elem   = self.finite_element_number
        n_quad   = self.quadrature_node_number
        n_local  = self.physical_node_number
        n_global = n_elem * (self.physical_node_number - 1) + 1
        
        # Method: Build block-diagonal then remove duplicate columns
        # Alternative could use fancy indexing, but this is clearer
        
        # Reshape lagrange_basis from (n_elem, n_quad, n_local) to 2D blocks
        # Result shape: (n_elem * n_quad, n_elem * n_local)
        interp_matrix = np.zeros((n_elem * n_quad, n_elem * n_local))
        
        for elem_idx in range(n_elem):
            row_slice = slice(elem_idx * n_quad, (elem_idx + 1) * n_quad)
            col_slice = slice(elem_idx * n_local, (elem_idx + 1) * n_local)
            interp_matrix[row_slice, col_slice] = self.lagrange_basis[elem_idx]
        
        # Shared boundary nodes: right edge of each element (except last)
        # These correspond to duplicate columns that must be merged
        shared_cols = np.array([n_local * (i + 1) - 1 for i in range(n_elem - 1)])
        
        # Merge: add right neighbor column to left neighbor column
        interp_matrix[:, shared_cols] += interp_matrix[:, shared_cols + 1]
        
        # Remove duplicate columns
        interp_matrix = np.delete(interp_matrix, shared_cols + 1, axis=1)
        
        assert interp_matrix.shape == (n_elem * n_quad, n_global), \
            f"Shape mismatch: expected {(n_elem * n_quad, n_global)}, get {interp_matrix.shape}."
        
        self._global_interpolation_matrix = interp_matrix
        return interp_matrix


    def get_lagrange_basis_pseudoinverse(self) -> np.ndarray:
        """
        Compute the pseudoinverse of the Lagrange basis for each finite element.
        """
        if hasattr(self, "_lagrange_basis_pseudoinverse"):
            return self._lagrange_basis_pseudoinverse
        
        lagrange_basis_pseudoinverse = np.zeros(
            (self.finite_element_number, self.physical_node_number, self.quadrature_node_number)
        )
        for elem_idx in range(self.finite_element_number):
            lagrange_basis_pseudoinverse[elem_idx, :, :] = np.linalg.pinv(self.lagrange_basis[elem_idx])

        self._lagrange_basis_pseudoinverse = lagrange_basis_pseudoinverse

        return lagrange_basis_pseudoinverse



    def assemble_poisson_rhs_vector_no_bc(self, rho: np.ndarray) -> np.ndarray:
        """
        Assemble the right-hand side vector for the Poisson equation.
        
        Computes the RHS vector for solving d²u/dr² = -4πrρ(r) where u = rV.
        The weak form involves integrating -4πrρ(r) against test functions.
        
        Parameters
        ----------
        rho : np.ndarray
            Electron density at quadrature points
            Shape: (n_elements * n_quad_points,)
        
        Returns
        -------
        rhs_vector : np.ndarray
            Assembled global RHS vector
            Shape: (n_global_dofs,)
            Note: Caller must set boundary values:
                rhs_vector[0] = left_bc
                rhs_vector[-1] = right_bc
        
        Notes
        -----
        - The local RHS contribution for each element is:
                RHS_local[i] = ∫ φ_i(r) * (-4πrρ(r)) dr
            These are assembled into a global vector, with contributions from
            adjacent elements added at shared boundary nodes.
        """
        # Type and shape validation
        assert isinstance(rho, np.ndarray), \
            RHO_TYPE_ERROR_MESSAGE.format(type(rho))
        assert rho.ndim == 1, \
            RHO_NDIM_ERROR_MESSAGE.format(rho.ndim)
        assert rho.shape[0] == self.finite_element_number * self.quadrature_node_number, \
            RHO_SHAPE_ERROR_MESSAGE.format(rho.shape[0], self.finite_element_number * self.quadrature_node_number)
        # Reshape density to element-wise structure
        rho_reshaped = rho.reshape(self.finite_element_number, self.quadrature_node_number)
        
        # Compute source term at quadrature points: -4πrρ
        r_rho = self.quadrature_nodes_reshaped * rho_reshaped
        source = - 4.0 * np.pi * r_rho

        
        # Compute local RHS: ∫ φ_i(r) * source(r) dr
        rhs_vector_local = np.einsum(
            "emi,em,em->ei",
            self.lagrange_basis,              # φ_i at quadrature points
            self.quadrature_weights_reshaped, # quadrature weights
            source,                           # -4πrρ at quadrature points
            optimize=True
        )  # Shape: (n_elem, n_physical_nodes)

        # Assemble local vectors into global vector
        # Shared boundary nodes will have contributions from adjacent elements added
        rhs_vector = self._assemble_local_to_global_vector(rhs_vector_local)

        return rhs_vector




    def assemble_laplacian_rhs_vector_with_gradient_and_no_bc(self, grad_rho: np.ndarray) -> np.ndarray:
        """
        Assemble the right-hand side vector for the Laplacian weak form.
        
        Weak form (integration by parts on [0, R]):
            ∫₀^R φ_i ℓ r² dr = -∫₀^R φ_i' r² ρ' dr + [φ_i(r) r² ρ'(r)]₀^R
        where ℓ = ∇²ρ (Laplacian of density), φ_i are test functions, ρ' = dρ/dr.
        
        This method implements only the first term on the RHS (the integral with
        negative sign): -∫₀^R φ_i' r² ρ' dr. The boundary term is handled elsewhere.
        
        Parameters
        ----------
        grad_rho : np.ndarray
            Radial derivative of density ρ' = dρ/dr at quadrature points.
        """
        # Type and shape validation
        assert isinstance(grad_rho, np.ndarray), \
            GRAD_RHO_TYPE_ERROR_MESSAGE.format(type(grad_rho))
        assert grad_rho.ndim == 1, \
            GRAD_RHO_NDIM_ERROR_MESSAGE.format(grad_rho.ndim)
        assert grad_rho.shape[0] == self.finite_element_number * self.quadrature_node_number, \
            GRAD_RHO_SHAPE_ERROR_MESSAGE.format(grad_rho.shape[0], self.finite_element_number * self.quadrature_node_number)
        
        # Reshape gradient of density to element-wise structure
        grad_rho_reshaped = grad_rho.reshape(self.finite_element_number, self.quadrature_node_number)

        # Compute source term at quadrature points: - r² ρ'
        source = - self.quadrature_nodes_reshaped ** 2 * grad_rho_reshaped

        # Compute local RHS: ∫ φ_i(r) * source(r) dr
        rhs_vector_local = np.einsum(
            "emi,em,em->ei",
            self.lagrange_basis_derivatives,  # φ_i' at quadrature points
            self.quadrature_weights_reshaped, # quadrature weights
            source,                           # - r² ρ' at quadrature points
            optimize=True
        )  # Shape: (n_elem, n_physical_nodes)

        # Assemble local vectors into global vector
        # Shared boundary nodes will have contributions from adjacent elements added
        rhs_vector = self._assemble_local_to_global_vector(rhs_vector_local)

        return rhs_vector


    def assemble_laplacian_rhs_vector_with_rho_and_no_bc(self, rho: np.ndarray) -> np.ndarray:
        """
        Assemble the right-hand side vector for the Laplacian weaker form.

        Weaker form (integration by parts on [0, R]):
            ∫₀^R φ_i ℓ r² dr = ∫₀^R (r² φ_i')' ρ dr - [r² φ_i'(r) ρ(r)]₀^R + [r² φ_i(r) ρ'(r)]₀^R
        where ℓ = ∇²ρ (Laplacian of density), φ_i are test functions, ρ' = dρ/dr.

        Uses (r² φ_i')' = 2r φ_i' + r² φ_i'' at quadrature points.

        Parameters
        ----------
        rho : np.ndarray
            Electron density at quadrature points.
        grad_rho : np.ndarray
            Radial derivative of density ρ' = dρ/dr at quadrature points.
        """
        # check if the input parameters are valid
        assert isinstance(rho, np.ndarray), \
            RHO_TYPE_ERROR_MESSAGE.format(type(rho))
        assert rho.ndim == 1, \
            RHO_NDIM_ERROR_MESSAGE.format(rho.ndim)
        assert rho.shape[0] == self.finite_element_number * self.quadrature_node_number, \
            RHO_SHAPE_ERROR_MESSAGE.format(rho.shape[0], self.finite_element_number * self.quadrature_node_number)

        # Reshape density to element-wise structure
        rho_reshaped = rho.reshape(self.finite_element_number, self.quadrature_node_number)

        # Compute local RHS: ∫ φ_i(r) * source(r) dr
        rhs_vector_local = np.einsum(
            "em, eml, el, eli -> ei",
            self.quadrature_weights_reshaped * rho_reshaped,  # (n_elements, n_quad)
            self.derivative_matrix,                           # (n_elements, n_quad, n_quad)
            self.quadrature_nodes_reshaped ** 2,              # (n_elements, n_quad)
            self.lagrange_basis_derivatives,                  # (n_elements, n_quad, n_physical_nodes)
            optimize=True
        )

        rhs_vector = self._assemble_local_to_global_vector(rhs_vector_local)

        return rhs_vector


    def _assemble_local_to_global_vector(self, local_vector: np.ndarray) -> np.ndarray:
        """
        Assemble local element vectors into a global vector.
        
        This method takes local element-wise vectors (shape: [n_elements, n_dofs_per_elem])
        and assembles them into a single global vector by adding overlapping contributions 
        at shared nodes.
        
        Parameters
        ----------
        local_vector : np.ndarray
            Local element vectors with shape (n_elements, n_local_dofs)
            where n_local_dofs = polynomial_order + 1 = physical_node_number
            
        Returns
        -------
        np.ndarray
            Assembled global vector with shape (n_global_dofs,)
            where n_global_dofs = n_elements * polynomial_order + 1
            
        Notes
        -----
        The assembly process accounts for shared endpoints between adjacent elements,
        accumulating contributions from all elements that share a degree of freedom.
        
        For example, with 3 elements and polynomial order 2:
            Element 0: [a0, a1, a2]  → global indices [0, 1, 2]
            Element 1: [b0, b1, b2]  → global indices [2, 3, 4]  (b0 adds to global[2])
            Element 2: [c0, c1, c2]  → global indices [4, 5, 6]  (c0 adds to global[4])
        
        Global vector: [a0, a1, a2+b0, b1, b2+c0, c1, c2]
        """
        # Initialize global vector
        global_size = self.finite_element_number * (self.physical_node_number - 1) + 1
        global_vector = np.zeros(global_size, dtype=self.assembly_dtype)
        
        # Get assembly indices (mapping from local to global DOFs)
        indices_global = self._build_assembly_indices_vector()
        
        # Assemble: add local contributions to global vector
        # np.add.at handles accumulation at repeated indices automatically
        np.add.at(global_vector, indices_global, np.asarray(local_vector, dtype=self.assembly_dtype).reshape(-1))
        
        return global_vector


    def _assemble_local_to_global_matrix(self, local_matrix: np.ndarray) -> np.ndarray:
        """
        Assemble local element matrices into a global matrix.
        
        This method takes local element-wise matrices (shape: [n_elements, n_dofs_per_elem, n_dofs_per_elem])
        and assembles them into a single global matrix by adding overlapping contributions at shared nodes.
        
        Parameters
        ----------
        local_matrix : np.ndarray
            Local element matrices with shape (n_elements, n_local_dofs, n_local_dofs)
            where n_local_dofs = polynomial_order + 1
            
        Returns
        -------
        np.ndarray
            Assembled global matrix with shape (n_global_dofs, n_global_dofs)
            where n_global_dofs = n_elements * polynomial_order + 1
            
        Notes
        -----
        The assembly process accounts for shared endpoints between adjacent elements,
        accumulating contributions from all elements that share a degree of freedom.
        """
        # Initialize global matrix
        global_size = self.finite_element_number * (self.physical_node_number - 1) + 1
        global_matrix = np.zeros((global_size, global_size), dtype=self.assembly_dtype)
        
        # Get assembly indices (mapping from local to global DOFs)
        rows_global, cols_global = self._build_assembly_indices()
        
        # Assemble: add local contributions to global matrix
        np.add.at(
            global_matrix,
            (rows_global, cols_global),
            np.asarray(local_matrix, dtype=self.assembly_dtype).reshape(-1),
        )
        
        return global_matrix
    
    
    def _build_assembly_indices_vector(self):
        """
        Return local→global indices for assembling element vectors into
        a global vector (with shared endpoints).
        
        This is similar to _build_assembly_indices but for vectors (1D).

        Primary FE space:
            local dofs per element       : N_grid = physical_node_number
            global stride (overlap by 1) : stride = N_grid - 1
            global dofs                  : N_elem * stride + 1
        
        Returns
        -------
        indices_global : np.ndarray
            1D array of global indices, shape (n_elements * n_local_dofs,)
            Maps local_vector.reshape(-1) to positions in global_vector
        
        Example
        -------
        For 3 elements with 3 nodes each (polynomial order 2):
            Element 0: local indices [0, 1, 2] → global indices [0, 1, 2]
            Element 1: local indices [0, 1, 2] → global indices [2, 3, 4]
            Element 2: local indices [0, 1, 2] → global indices [4, 5, 6]
        
        Returns: [0, 1, 2, 2, 3, 4, 4, 5, 6]
                  ^^^^^^^  ^^^^^^^  ^^^^^^^
                  elem 0   elem 1   elem 2
        
        Note the repeated indices (2, 2) and (4, 4) at element boundaries.
        """
        N_elem = self.finite_element_number  # e for element
        N_grid = self.physical_node_number       # g for grid

        # compute the assembly indices
        stride = N_grid - 1
        elem_array = np.arange(N_elem)           # [0, 1, 2, ...]
        grid_array = np.arange(N_grid)           # [0, 1, 2, ..., N_grid-1]
        
        # Global index for each (element, local_node) pair
        # I_eg[elem, local_idx] = global_idx
        I_eg = elem_array[:, None] * stride + grid_array[None, :]   # (N_elem, N_grid)
        
        # Flatten to 1D array
        indices_global = I_eg.reshape(-1)
        
        return indices_global
    
    
    def _build_assembly_indices(self):
        """
        Return clean local→global row/col indices for assembling element blocks into
        a global matrix (with shared endpoints).

        Primary FE space:
            local dofs per element       : N_g    = physical_node_number - 1
            global stride (overlap by 1) : stride = N_g - 1
            global dofs                  : N_e * stride + 1
        """
        # ----- primary FE space -----
        N_elem = self.number_of_finite_elements # e for element
        N_grid = self.physical_node_number      # g for grid

        # compute the assembly indices
        stride = N_grid - 1
        elem_array = np.arange(N_elem)
        grid_array = np.arange(N_grid)
        I_eg = elem_array[:, None]*stride + grid_array[None, :]   # (N_elem, N_grid)

        rows_primary = np.repeat(I_eg[:, :, None], N_grid, axis=2).reshape(-1)
        cols_primary = np.repeat(I_eg[:, None, :], N_grid, axis=1).reshape(-1)

        return rows_primary, cols_primary


    def _build_basis_on_arbitrary_grid_dict(self, given_grid: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Evaluate Lagrange basis functions on arbitrary grid points, per element.

        Only elements that contain at least one point in ``given_grid`` appear in the
        returned dict (coarse uniform meshes may leave many elements out).

        Returns
        -------
        dict[int, np.ndarray]
            Keys are finite-element indices. Values have shape (1, n_points_in_element, n_basis).
        """

        # Internal helper: callers validate ``given_grid`` before calling.
        basis_on_arbitrary_grid_dict: Dict[int, np.ndarray] = {}
        for elem_idx in range(self.finite_element_number):
            idx_uniform = (
                (given_grid >= self.physical_nodes_reshaped[elem_idx:elem_idx + 1, 0])
                & (given_grid <= self.physical_nodes_reshaped[elem_idx:elem_idx + 1, -1])
            )
            if not np.any(idx_uniform):
                continue
            basis_func_in_current_element, _ = LagrangeShapeFunctions.lagrange_basis_and_derivatives(
                x_node = self.physical_nodes_reshaped[elem_idx:elem_idx + 1, :],
                x_eval = given_grid[idx_uniform],
            )
            basis_on_arbitrary_grid_dict[elem_idx] = basis_func_in_current_element
        return basis_on_arbitrary_grid_dict


    @property
    def H_kinetic(self) -> np.ndarray:
        return self.get_H_kinetic()

    @property
    def H_r_inv_sq(self) -> np.ndarray:
        return self.get_H_r_inv_sq()

    @property
    def S(self) -> np.ndarray:
        return self.get_S()

    @property
    def S_inv_sqrt(self) -> np.ndarray:
        return self.get_S_inv_sqrt()

    @property
    def laplacian(self) -> np.ndarray:
        return self.get_laplacian()

    @property
    def derivative_matrix(self) -> np.ndarray:
        # return self.get_derivative_matrix()
        # Now using the quadrature basis as the basis nodes
        return self.get_derivative_matrix_with_quadrature_basis()
        
    @property
    def derivative_matrix_with_quadrature_basis(self) -> np.ndarray:
        return self.get_derivative_matrix_with_quadrature_basis()

    @property
    def global_interpolation_matrix(self) -> np.ndarray:
        """
        Property accessor for global interpolation matrix.
        Used for HF/PBE0 calculations.
        """
        return self.get_global_interpolation_matrix()
    
    @property
    def lagrange_basis_pseudoinverse(self) -> np.ndarray:
        return self.get_lagrange_basis_pseudoinverse()



    @property
    def grid_data(self) -> GridData:
        """
        Property accessor for grid data.
        Used for response function calculation, etc.
        """
        return GridData(
            finite_element_number = self.finite_element_number,
            physical_nodes        = self.physical_nodes,
            quadrature_nodes      = self.quadrature_nodes,
            quadrature_weights    = self.quadrature_weights,
        )


    def evaluate_orbitals_on_arbitrary_grid(
        self,
        given_grid                   : np.ndarray,
        orbital_coefficients         : np.ndarray,
        basis_on_arbitrary_grid_dict : Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Interpolate FE nodal orbital coefficients onto an arbitrary grid.

        Use when SCF eigenvectors are stored as global FE node coefficients
        (``symmetrize=False`` path). For quadrature-point orbitals, use
        ``evaluate_quantites_on_arbitrary_grid`` instead.

        Parameters
        ----------
        given_grid : np.ndarray, shape (n_points,)
            Monotonically increasing points within the physical domain.
        orbital_coefficients : np.ndarray
            Shape ``(n_global_fe_dofs - 1, n_orbitals)`` before internal padding.
        basis_on_arbitrary_grid_dict : dict[int, np.ndarray], optional
            Precomputed basis from ``_build_basis_on_arbitrary_grid_dict``.

        Returns
        -------
        np.ndarray, shape (n_points, n_orbitals)
            Orbitals sampled on ``given_grid``.
        """
        # Validate input: grid must be monotonically increasing
        assert np.all(np.diff(given_grid) > 0.0), \
            GIVEN_GRID_NOT_MONOTONICALLY_INCREASING_ERROR

        # Validate input: grid must be within physical domain
        assert np.all(given_grid >= self.physical_nodes[0]) and \
               np.all(given_grid <= self.physical_nodes[-1]), \
            GIVEN_GRID_NOT_WITHIN_PHYSICAL_NODES_ERROR
            
        # Validate input: FE nodal coefficients (unpadded global dof count)
        assert orbital_coefficients.shape[0] == (self.finite_element_number * (self.physical_node_number - 1) - 1), \
            ORBITAL_COEFFICIENTS_SHAPE_ERROR_MESSAGE.format(orbital_coefficients.shape[0], (self.finite_element_number * (self.physical_node_number - 1) - 1))

        if basis_on_arbitrary_grid_dict is None:
            basis_on_arbitrary_grid_dict = self._build_basis_on_arbitrary_grid_dict(given_grid)


        orbital_coefficients = np.pad(orbital_coefficients, ((1, 1), (0, 0)))
        n_elem = self.finite_element_number
        n_basis = self.physical_node_number

        orbitals_arbitrary_grid = np.zeros((len(given_grid), orbital_coefficients.shape[1]))
        for elem_idx in range(n_elem):
            idx_FE = np.arange(elem_idx * (n_basis - 1), (elem_idx + 1) * (n_basis - 1) + 1)
            idx_uniform = (
                (given_grid >= self.physical_nodes_reshaped[elem_idx:elem_idx + 1, 0])
                & (given_grid <= self.physical_nodes_reshaped[elem_idx:elem_idx + 1, -1])
            )
            if not np.any(idx_uniform):
                continue
            orbitals_arbitrary_grid[idx_uniform, :] = (
                basis_on_arbitrary_grid_dict[elem_idx][0, :, :] @ orbital_coefficients[idx_FE, :]
            )
        return orbitals_arbitrary_grid


    def evaluate_quantites_on_arbitrary_grid(
        self,
        given_grid                   : np.ndarray,
        field_values                 : np.ndarray,
        basis_on_arbitrary_grid_dict : Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Evaluate quadrature-point fields on an arbitrary grid.

        For each finite element, quadrature values are mapped to nodal coefficients
        via the cached Lagrange-basis pseudoinverse, then interpolated to
        ``given_grid`` using precomputed basis values on that grid.

        Parameters
        ----------
        given_grid : np.ndarray, shape (n_points,)
            Monotonically increasing evaluation points within the physical domain.
        field_values : np.ndarray, shape (n_elem * n_quad, n_fields)
            Field values at global quadrature points. Use ``field[:, None]`` for a
            single field.
        basis_on_arbitrary_grid_dict : dict[int, np.ndarray], optional
            Precomputed Lagrange basis on ``given_grid``, keyed by element index.
            Each value has shape ``(1, n_points_in_element, n_basis)``.
            If ``None``, the basis is built internally for this call.
            Pass the dict from ``_build_basis_on_arbitrary_grid_dict`` to avoid
            rebuilding when evaluating multiple fields on the same grid.

        Returns
        -------
        np.ndarray, shape (n_points, n_fields)
            Field values interpolated onto ``given_grid``.
        """
        # Validate input: grid must be monotonically increasing
        assert np.all(np.diff(given_grid) > 0.0), \
            GIVEN_GRID_NOT_MONOTONICALLY_INCREASING_ERROR

        # Validate input: grid must be within physical domain
        assert np.all(given_grid >= self.physical_nodes[0]) and \
               np.all(given_grid <= self.physical_nodes[-1]), \
            GIVEN_GRID_NOT_WITHIN_PHYSICAL_NODES_ERROR
        
        # Validate input: field values must be a 1D or 2D array with shape (n_elem * n_quad, n_fields)
        if field_values.ndim == 1:
            field_values = field_values[:, None]
        assert field_values.ndim == 2, \
            FIELD_VALUES_2D_NDIM_ERROR_MESSAGE.format(field_values.ndim)
        assert field_values.shape[0] == self.finite_element_number * self.quadrature_node_number, \
            FIELD_VALUES_SHAPE_ERROR_MESSAGE.format(field_values.shape[0], self.finite_element_number * self.quadrature_node_number,)

        n_elem = self.finite_element_number
        n_quad = self.quadrature_node_number
        basis_pseudoinverse = self.lagrange_basis_pseudoinverse

        if basis_on_arbitrary_grid_dict is None:
            basis_on_arbitrary_grid_dict = self._build_basis_on_arbitrary_grid_dict(given_grid)

        quantities_arbitrary_grid = np.zeros((len(given_grid), field_values.shape[1]))
        for elem_idx in range(n_elem):
            idx_uniform = (
                (given_grid >= self.physical_nodes_reshaped[elem_idx:elem_idx + 1, 0])
                & (given_grid <= self.physical_nodes_reshaped[elem_idx:elem_idx + 1, -1])
            )
            if not np.any(idx_uniform):
                continue
            field_values_coeffs_from_pseudoinverse = (
                basis_pseudoinverse[elem_idx, :, :]
                @ field_values[elem_idx * n_quad:(elem_idx + 1) * n_quad, :]
            )
            quantities_arbitrary_grid[idx_uniform, :] = (
                basis_on_arbitrary_grid_dict[elem_idx] @ field_values_coeffs_from_pseudoinverse
            )

        return quantities_arbitrary_grid


    def evaluate_single_field_on_grid(
        self,
        given_grid                   : np.ndarray,
        field_values                 : np.ndarray,
        basis_on_arbitrary_grid_dict : Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Evaluate a single field on a given grid using Lagrange interpolation.
        
        This function takes a field represented by its values at quadrature
        points and evaluates it at arbitrary grid points using finite element
        Lagrange basis functions. For each grid point, the function:
        1. Identifies which finite element contains the point
        2. Converts quadrature point values to nodal coefficients (using pseudoinverse)
        3. Evaluates the Lagrange basis functions at that point
        4. Computes the orbital value as a linear combination of basis functions
        
        Parameters
        ----------
        given_grid : np.ndarray, shape (n_points,)
            Grid points where the field should be evaluated.
            Must be monotonically increasing and within the physical domain.
        field_values : np.ndarray, shape (n_elem * n_quad,)
            Field values at global quadrature points.
        basis_on_arbitrary_grid_dict : dict[int, np.ndarray], optional
            Precomputed basis from ``_build_basis_on_arbitrary_grid_dict``.
            Reuse when mapping several fields onto the same ``given_grid``.

        Returns
        -------
        field_on_grid : np.ndarray, shape (n_points,)
            Field values on ``given_grid``.

        Notes
        -----
        Delegates to ``evaluate_quantites_on_arbitrary_grid`` (vectorized per
        element). Input is quadrature-point data, not FE nodal coefficients.
        
        Example
        -------
        >>> # Given field values at quadrature points
        >>> field_values = np.array([...])  # shape: (n_elem * n_quad,)
        >>> # Evaluate on a uniform grid
        >>> uniform_grid = np.linspace(0, domain_size, 1000)
        >>> field_on_grid = ops_builder.evaluate_single_field_on_grid(
        ...     given_grid=uniform_grid,
        ...     field_values=field_values
        ... )
        >>> # field_on_grid.shape = (1000,)
        """
        # Validate input: field values must be a 1D array with shape (n_elem * n_quad,)
        assert field_values.ndim == 1, \
            FIELD_VALUES_NDIM_ERROR_MESSAGE.format(field_values.ndim)
        assert field_values.shape[0] == self.finite_element_number * self.quadrature_node_number, \
            FIELD_VALUES_SHAPE_ERROR_MESSAGE.format(field_values.shape[0], self.finite_element_number * self.quadrature_node_number,)
        
        return self.evaluate_quantites_on_arbitrary_grid(
            given_grid                   = given_grid,
            field_values                 = field_values,
            basis_on_arbitrary_grid_dict = basis_on_arbitrary_grid_dict,
        ).reshape(-1)



    @staticmethod
    def build_cubic_spline_derivative_matrix(
        r                : np.ndarray,
        left_derivative  : float = 0.0,
        right_derivative : float = 0.0,
    ) -> np.ndarray:
        """
        Build a 2D derivative matrix for a 1D cubic spline on ``r``.

        For grid values ``f`` on ``r``, ``D @ f`` gives ``df/dr`` at each grid point,
        where ``D[i, j] = ∂f/∂f_j`` evaluated at ``r_i``. The spline uses clamped
        boundary conditions ``f'(r_0) = left_derivative`` and
        ``f'(r_{n-1}) = right_derivative`` (default: zero slope at both ends).

        Parameters
        ----------
        r : np.ndarray
            Strictly increasing 1D radial grid, shape ``(n,)``.
        left_derivative, right_derivative : float
            First-derivative boundary values at the left and right endpoints.

        Returns
        -------
        np.ndarray
            Shape ``(n, n)``, ``float64``.
        """
        from scipy.interpolate import CubicSpline

        r = np.asarray(r, dtype=np.float64).reshape(-1)
        n = r.size
        if n < 2:
            raise ValueError(
                "build_cubic_spline_derivative_matrix requires at least 2 grid points, "
                f"got {n}."
            )
        if not np.all(np.diff(r) > 0.0):
            raise ValueError("Grid r must be strictly increasing.")

        bc_type = ((1, float(left_derivative)), (1, float(right_derivative)))
        D = np.zeros((n, n), dtype=np.float64)
        eye = np.eye(n, dtype=np.float64)
        for j in range(n):
            spline = CubicSpline(r, eye[:, j], bc_type=bc_type)
            D[:, j] = spline(r, nu=1)
        return D

