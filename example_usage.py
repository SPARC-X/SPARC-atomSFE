
from src.solver import AtomicDFTSolver

solver = AtomicDFTSolver(
    atomic_number             = 42,
    xc_functional             = 'HF',
    domain_size               = 40.0,
    finite_element_number     = 4,
    polynomial_order          = 20,
    quadrature_point_number   = 60,
    mesh_type                 = "exponential",
    mesh_concentration        = 101.0,
    scf_tolerance             = 1e-9,
    verbose                   = True, 
    all_electron_flag         = True,
    use_oep                   = False,
    use_preconditioner        = True,
)

results = solver.solve()

# print("\nSolver results:")
# print(f"\t total energy             = {results['energy']:.6f} (Ha)")
# print(f"\t density.shape            = {results['rho'].shape}")
# print(f"\t quadrature_nodes.shape   = {results['quadrature_nodes'].shape}")
# print(f"\t quadrature_weights.shape = {results['quadrature_weights'].shape}")