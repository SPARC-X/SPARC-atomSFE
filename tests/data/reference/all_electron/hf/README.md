# HF reference datasets

## `lehtola_closed_subshell_atoms_hf.json`

- Citation: Cinal, M. "Highly accurate numerical solution of Hartree--Fock equation with pseudospectral method for closed-shell atoms." Journal of Mathematical Chemistry 58(8), 1571--1600 (2020).
- Source: transcribed from **Table 2** (closed-shell / closed-subshell atoms, He to No).

### Per-species fields mapping

| Physical meaning | Mathematical expression | Variable name (JSON key) |
| --- | --- | --- |
| Atomic number | $Z$ | `Z` |
| Element symbol | N/A | `atom` |
| Total energy (stored as a positive magnitude) | $-E_{\mathrm{tot}}$ (a.u.) | `minus_E_tot_au` |
| Exchange energy (stored as a positive magnitude) | $-E_{\mathrm{x}}$ (a.u.) | `minus_E_x_au` |
| Scaled deviation of $q_{\mathrm{vir}}$ from 2 | $(q_{\mathrm{vir}}-2)\times 10^{14}$ | `q_vir_minus_2_times_1e14` |
| Scaled deviation of $q_{\mathrm{vir}}^{\mathrm{alt}}$ from 2 | $(q_{\mathrm{vir}}^{\mathrm{alt}}-2)\times 10^{14}$ | `q_vir_alt_minus_2_times_1e14` |
| HOMO eigenvalue (stored as a positive magnitude) | $-\epsilon_{\mathrm{HOMO}}$ (a.u.) | `minus_epsilon_homo_au` |
| HOMO orbital label | N/A | `homo_orbital` |

**Notes**

- The virial fields use $q_{\mathrm{vir}}=-E_{\mathrm{pot}}/E_{\mathrm{kin}}$ and $q_{\mathrm{vir}}^{\mathrm{alt}}=-E_{\mathrm{pot}}/E_{\mathrm{kin}}^{\mathrm{alt}}$.
- For nonrelativistic Coulomb systems, virial arguments give $q_{\mathrm{vir}}\approx 2$ and $q_{\mathrm{vir}}^{\mathrm{alt}}\approx 2$, so the stored virial fields are tiny deviations from 2 after multiplying by $10^{14}$.

## `lehtola_charged_atoms_hf.json`

- Citation: Lehtola, S. (2019), *International Journal of Quantum Chemistry* 119(19), e25945, https://doi.org/10.1002/qua.25945.
- Literature Gaussian column (Table 5, fourth data column in Lehtola): values correspond to the Gaussian-basis reference associated with Anderson, L. N.; Oviedo, M. B.; Wong, B. M., *J. Chem. Theory Comput.* **13**(4), 1656--1666 (2017), https://doi.org/10.1021/acs.jctc.7b00048 (Lehtola cites this as Ref. [56]).
- Source: transcribed from **Table 5** (HF total energies for a mixed set of neutral atoms and atomic ions: finite element vs Gaussian Erkale vs literature Gaussian; difference column).

### Spin and open-shell HF (important for comparisons)

Lehtola’s HELFEM paper formulates Hartree–Fock in a **spin-resolved** way (Roothaan / **Pople–Nesbet**):

$$
F_\sigma C_\sigma = S C_\sigma \epsilon_\sigma
$$

(their Eq. (17)), with

$$
F_\sigma = T + V^{\mathrm{nuc}} + J(P) + K(P_\sigma)
$$

(Eq. (26)),

$$
P = P_\alpha + P_\beta
$$

where $P_\sigma$ is built from occupied orbitals of spin $\sigma$ (Eq. (27)) and $P$ is the **total** Coulomb density (Eq. (28)). So **Coulomb uses the total density; exchange uses the spin-specific density**—this matters whenever $P_\alpha \neq P_\beta$.

The program supports **restricted closed-shell RHF**, **ROHF** (via constrained UHF), and **UHF**; SCF occupations follow **Aufbau** unless orbital symmetries are set explicitly, and the implementation uses **atomic \(m\)-symmetry** in the default diagonalization. Table 5 energies are **not** “spin-free HF”: closed-shell species collapse to the usual RHF picture, but **open-shell** entries must be compared using the **same** HF variant (RHF vs ROHF vs UHF) and occupation convention as in the original FEM / Erkale / Gaussian calculations. If your solver flags **spin-unpolarized** SCF while the reference row is **open-shell**, total energies can disagree by **much more** than numerical or CBS-truncation error.

### Per-species fields mapping

| Physical meaning | Mathematical expression | Variable name (JSON key) |
| --- | --- | --- |
| Species label | N/A | `species` |
| HF total energy, finite element | $E_{\mathrm{HF}}^{\mathrm{FEM}}$ (Hartree) | `E_fem_au` |
| HF total energy, Gaussian (Erkale) | $E_{\mathrm{HF}}^{\mathrm{Erkale}}$ (Hartree) | `E_erkale_au` |
| HF total energy, Gaussian (literature column, Lehtola Table 5) | $E_{\mathrm{HF}}^{\mathrm{lit.}}$ (Hartree) | `E_gaussian_au` |
| Energy difference (finite element minus Erkale) | $(E_{\mathrm{HF}}^{\mathrm{FEM}}-E_{\mathrm{HF}}^{\mathrm{Erkale}})\times 10^{6}$ ($\mu E_h$) | `delta_fem_minus_erkale_microEh` |

**Notes**

- HF means Hartree-Fock; $\mu E_h$ means microhartree ($10^{-6}$ Hartree).
- The fifth table column is reproduced as `delta_fem_minus_erkale_microEh` (finite element minus Erkale).
- For $\mathrm{S}^{+}$, the printed microhartree entry does not match the displayed Hartree columns to two decimal places in $\mu E_h$ (finite element minus Erkale is $-19.325~\mu E_h$ while the table lists $-19.33~\mu E_h$). The JSON keeps the table values as printed.
