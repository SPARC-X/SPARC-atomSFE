


import numpy as np
from typing import Optional, Tuple
from scipy.optimize import brentq
from scipy.special import expit


'''
Input: Atomic Number Z
Output: Occ matrix containing "n" quantum number list in first row, "l"
quantum number in the second row, the corresponding spin-up occupation 
in the third row and spin-down occupation in the fourth row.
'''


# Error messages 
Z_NOT_INT_ERROR = \
    "parameter 'Z' must be an integer, get type {} instead."
Z_NOT_IN_VALID_RANGE_ERROR = \
    "parameter 'Z' must be between 1 and 92 (1-92), get {} instead."
Z_NUCLEAR_NOT_INT_OR_FLOAT_ERROR = \
    "parameter 'z_nuclear' must be an integer or float, get type {} instead."
Z_NUCLEAR_NOT_GREATER_THAN_0_OR_LESS_THAN_93_ERROR = \
    "parameter 'z_nuclear' must be greater than 0 and less than 93 (1-92), get {} instead."

Z_VALENCE_NOT_INT_OR_FLOAT_ERROR = \
    "parameter 'z_valence' must be an integer or float, get type {} instead."
Z_VALENCE_NOT_GREATER_THAN_0_OR_LESS_THAN_93_ERROR = \
    "parameter 'z_valence' must be greater than 0 and less than 93 (1-92), get {} instead."
ALL_ELECTRON_FLAG_NOT_BOOL_ERROR = \
    "parameter 'all_electron_flag' must be a boolean, get type {} instead."    
N_ELECTRONS_NOT_INT_OR_FLOAT_ERROR = \
    "parameter 'n_electrons' must be an integer or float, get type {} instead."
N_ELECTRONS_NOT_GREATER_THAN_0_ERROR = \
    "parameter 'n_electrons' must be greater than 0, get {} instead."
N_ELECTRONS_NOT_LESS_THAN_OR_EQUAL_TO_92_ERROR = \
    "parameter 'n_electrons' must be less than or equal to 92, get {} instead."

CHARGE_SYSTEMS_NOT_SUPPORTED_FOR_PSEUDOPOTENTIAL_CALCULATION_ERROR = \
    "Charged systems are not supported with pseudopotentials. Use all-electron calculations for non-neutral systems."
Z_NUCLEAR_NOT_INTEGER_VALUED_FOR_PSEUDOPOTENTIAL_CALCULATION_ERROR = \
    "parameter 'z_nuclear' must be integer-valued for pseudopotential calculations, get {} instead."
TOTAL_OCCUPATION_NUMBERS_DO_NOT_MATCH_THE_NUMBER_OF_ELECTRONS_ERROR = \
    "Total occupation numbers do not match the number of electrons {} != {}, this should not happen."
OCC_ENERGIES_SIZE_MISMATCH_FOR_HOMO_FRACTION_ERROR = \
    "occ_energies has {} entries but the occupation list has {} subshells."
AUFBAU_SMEARING_NOT_NON_NEGATIVE_FLOAT_ERROR = \
    "parameter 'smearing' must be a non-negative float (Fermi-Dirac kT in Ha), get {} instead."
AUFBAU_L_MAX_NOT_NON_NEGATIVE_INT_ERROR = \
    "parameter 'l_max' must be a non-negative integer, get {} instead."
AUFBAU_N_CANDIDATES_NOT_POSITIVE_INT_ERROR = \
    "parameter 'n_candidates_per_channel' must be a positive integer, get {} instead."
AUFBAU_CAPACITY_TOO_SMALL_ERROR = \
    "aufbau candidate subshells hold {} electrons, not more than the {} electrons to place."
AUFBAU_SET_OCCUPATIONS_TABLE_RULE_ERROR = \
    "set_occupations needs occupation_rule 'aufbau' (float occupation arrays); this list uses '{}'."
AUFBAU_OCCUPATION_LIST_MISMATCH_ERROR = \
    "occupations to set have (n, l) = {} but this occupation list has (n, l) = {}."
AUFBAU_OCCUPATION_SUM_MISMATCH_ERROR = \
    "occupations to set hold {} electrons, but this occupation list holds {}."
AUFBAU_TOP_CANDIDATE_OCCUPIED_WARNING = \
    "WARNING: aufbau occupation puts {:.3e} electrons in {}, the highest candidate subshell of channel l={}; the candidate list may be too short."



HARDCORED_OCCUPATION_EXCEPTION_ORBITAL_INDEX_DICT = {
    # The key is the atomic number, the value is the index of the orbital 
    #   that should be changed when the number of electrons is fractional.
    24: -2,  # Cr, 3d
    29: -2,  # Cu, 3d
    41: -2,  # Nb, 4d
    44: -2,  # Ru, 4d
    46: -2,  # Pd, 4p
    58: -5,  # Ce, 4f
    59: -5,  # Pr, 4d
    64: -2,  # Gd, 5d
    65: -5,  # Tb, 4d
    78: -2,  # Pt, 5d
    91: -5,  # Pa, 5f
}



def get_neutral_occupation_states(Z : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get occupation states for given atomic number when the system is electrically neutral.

    Parameters
    ----------
    Z : int
        Atomic number

    Returns
    -------
    n_quantum : np.ndarray
        Principal quantum number n for each orbital
    l_quantum : np.ndarray
        Angular momentum quantum number l for each orbital
    s_quantum_up : np.ndarray
        Spin-up occupation for each orbital
    s_quantum_down : np.ndarray
        Spin-down occupation for each orbital
    """
    # Type checking
    assert isinstance(Z, int), \
        Z_NOT_INT_ERROR.format(type(Z))
  
    if Z == 1:
        n_quantum = np.array([1])
        l_quantum = np.array([0])
        s_quantum_up = np.array([1])
        s_quantum_down = np.array([0])
    elif Z == 2:
        n_quantum = np.array([1])
        l_quantum = np.array([0])
        s_quantum_up = np.array([1])
        s_quantum_down = np.array([1])
    elif Z == 3:
        n_quantum = np.array([1,2])
        l_quantum = np.array([0,0])
        s_quantum_up = np.array([1,1])
        s_quantum_down = np.array([1,0])
    elif Z == 4:
        n_quantum = np.array([1,2])
        l_quantum = np.array([0,0])
        s_quantum_up = np.array([1,1])
        s_quantum_down = np.array([1,1])
    elif Z == 5:
        n_quantum = np.array([1,2,2])
        l_quantum = np.array([0,0,1])
        s_quantum_up = np.array([1,1,1])
        s_quantum_down = np.array([1,1,0])
    elif Z == 6:
        n_quantum = np.array([1,2,2])
        l_quantum = np.array([0,0,1])
        s_quantum_up = np.array([1,1,2])
        s_quantum_down = np.array([1,1,0])
    elif Z == 7:
        n_quantum = np.array([1,2,2])
        l_quantum = np.array([0,0,1])
        s_quantum_up = np.array([1,1,3])
        s_quantum_down = np.array([1,1,0])
    elif Z == 8:
        n_quantum = np.array([1,2,2])
        l_quantum = np.array([0,0,1])
        s_quantum_up = np.array([1,1,3])
        s_quantum_down = np.array([1,1,1])
    elif Z == 9:
        n_quantum = np.array([1,2,2])
        l_quantum = np.array([0,0,1])
        s_quantum_up = np.array([1,1,3])
        s_quantum_down = np.array([1,1,2])
    elif Z == 10:
        n_quantum = np.array([1,2,2])
        l_quantum = np.array([0,0,1])
        s_quantum_up = np.array([1,1,3])
        s_quantum_down = np.array([1,1,3])
    elif Z == 11:
        n_quantum = np.array([1,2,2,3])
        l_quantum = np.array([0,0,1,0])
        s_quantum_up = np.array([1,1,3,1])
        s_quantum_down = np.array([1,1,3,0])
    elif Z == 12:
        n_quantum = np.array([1,2,2,3])
        l_quantum = np.array([0,0,1,0])
        s_quantum_up = np.array([1,1,3,1])
        s_quantum_down = np.array([1,1,3,1])
    elif Z == 13:
        n_quantum = np.array([1,2,2,3,3])
        l_quantum = np.array([0,0,1,0,1])
        s_quantum_up = np.array([1,1,3,1,1])
        s_quantum_down = np.array([1,1,3,1,0])
    elif Z == 14:
        n_quantum = np.array([1,2,2,3,3])
        l_quantum = np.array([0,0,1,0,1])
        s_quantum_up = np.array([1,1,3,1,2])
        s_quantum_down = np.array([1,1,3,1,0])
    elif Z == 15:
        n_quantum = np.array([1,2,2,3,3])
        l_quantum = np.array([0,0,1,0,1])
        s_quantum_up = np.array([1,1,3,1,3])
        s_quantum_down = np.array([1,1,3,1,0])
    elif Z == 16:
        n_quantum = np.array([1,2,2,3,3])
        l_quantum = np.array([0,0,1,0,1])
        s_quantum_up = np.array([1,1,3,1,3])
        s_quantum_down = np.array([1,1,3,1,1])
    elif Z == 17:
        n_quantum = np.array([1,2,2,3,3])
        l_quantum = np.array([0,0,1,0,1])
        s_quantum_up = np.array([1,1,3,1,3])
        s_quantum_down = np.array([1,1,3,1,2])
    elif Z == 18:
        n_quantum = np.array([1,2,2,3,3])
        l_quantum = np.array([0,0,1,0,1])
        s_quantum_up = np.array([1,1,3,1,3])
        s_quantum_down = np.array([1,1,3,1,3])
    elif Z == 19:
        n_quantum = np.array([1,2,2,3,3,4])
        l_quantum = np.array([0,0,1,0,1,0])
        s_quantum_up = np.array([1,1,3,1,3,1])
        s_quantum_down = np.array([1,1,3,1,3,0])
    elif Z == 20:
        n_quantum = np.array([1,2,2,3,3,4])
        l_quantum = np.array([0,0,1,0,1,0])
        s_quantum_up = np.array([1,1,3,1,3,1])
        s_quantum_down = np.array([1,1,3,1,3,1])
    elif Z == 21:
        n_quantum = np.array([1, 2, 2, 3, 3, 3, 4])
        l_quantum = np.array([0, 0, 1, 0, 1, 2, 0])
        s_quantum_up = np.array([1, 1, 3, 1, 3, 1, 1])
        s_quantum_down = np.array([1, 1, 3, 1, 3, 0, 1])
    elif Z == 22:
        n_quantum = np.array([1, 2, 2, 3, 3, 3, 4])
        l_quantum = np.array([0, 0, 1, 0, 1, 2, 0])
        s_quantum_up = np.array([1, 1, 3, 1, 3, 2, 1])
        s_quantum_down = np.array([1, 1, 3, 1, 3, 0, 1])
    elif Z == 23:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 0, 1 ])
    elif Z == 24:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 0, 0 ])
    elif Z == 25:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 0, 1 ])
    elif Z == 26:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 1, 1 ])
    elif Z == 27:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 2, 1 ])
    elif Z == 28:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 3, 1 ])
    elif Z == 29:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 0 ])
    elif Z == 30:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
    elif Z == 31:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 0 ])
    elif Z == 32:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 2 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 0 ])
    elif Z == 33:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 0 ])
    elif Z == 34:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 1 ])
    elif Z == 35:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 2 ])
    elif Z == 36:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3 ])
    elif Z == 37:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0 ])
    elif Z == 38:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 1 ])
    elif Z == 39:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0, 1 ])
    elif Z == 40:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 2, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0, 1 ])
    elif Z == 41:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 4, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0, 0 ])
    elif Z == 42:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0, 0 ])
    elif Z == 43:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0, 1 ])
    elif Z == 44:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 2, 0 ])
    elif Z == 45:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 3, 0 ])
    elif Z == 46:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5 ])
    elif Z == 47:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0 ])
    elif Z == 48:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
    elif Z == 49:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 0 ])
    elif Z == 50:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 2 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 0 ])
    elif Z == 51:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 0 ])
    elif Z == 52:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 1 ])
    elif Z == 53:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 2 ])
    elif Z == 54:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 ])
    elif Z == 55:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 0 ])
    elif Z == 56:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1 ])
    elif Z == 57:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 0, 1 ])
    elif Z == 58:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 1, 3, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 0, 1 ])
    elif Z == 59:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 3, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 ])
    elif Z == 60:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 4, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 ])
    elif Z == 61:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 5, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 ])
    elif Z == 62:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 6, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 ])
    elif Z == 63:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 ])
    elif Z == 64:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 0, 1 ])
    elif Z == 65:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 2, 1, 3, 1 ])
    elif Z == 66:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 3, 1, 3, 1 ])
    elif Z == 67:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 4, 1, 3, 1 ])
    elif Z == 68:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 5, 1, 3, 1 ])
    elif Z == 69:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 6, 1, 3, 1 ])
    elif Z == 70:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
    elif Z == 71:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 ])
    elif Z == 72:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 2, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 ])
    elif Z == 73:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 ])
    elif Z == 74:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 4, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 ])
    elif Z == 75:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 ])
    elif Z == 76:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1, 1 ])
    elif Z == 77:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 2, 1 ])
    elif Z == 78:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 4, 0 ])
    elif Z == 79:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 0 ])
    elif Z == 80:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
    elif Z == 81:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 0 ])
    elif Z == 82:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 2 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 0 ])
    elif Z == 83:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 0 ])
    elif Z == 84:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 1 ])
    elif Z == 85:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 2 ])
    elif Z == 86:
        n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
        l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
        s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 ])
    elif Z == 87:
        n_quantum      = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7 ])
        l_quantum      = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0 ])
        s_quantum_up   = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 0 ])
    elif Z == 88:
        n_quantum      = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7 ])
        l_quantum      = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0 ])
        s_quantum_up   = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1 ])
    elif Z == 89:
        n_quantum      = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7 ])
        l_quantum      = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up   = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 0, 1 ])
    elif Z == 90:
        n_quantum      = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7 ])
        l_quantum      = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0 ])
        s_quantum_up   = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 2, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 0, 1 ])
    elif Z == 91:
        n_quantum      = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7 ])
        l_quantum      = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up   = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 2, 1, 3, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 0, 1, 3, 0, 1 ])
    elif Z == 92:
        n_quantum      = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7 ])
        l_quantum      = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0 ])
        s_quantum_up   = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 3, 1, 3, 1, 1 ])
        s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 0, 1, 3, 0, 1 ])
    else:
        raise ValueError(Z_NOT_IN_VALID_RANGE_ERROR.format(Z))
    
    return n_quantum, l_quantum, s_quantum_up, s_quantum_down 



def get_fraction_occupation_states(n_electrons : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get occupation states for fractional number of electrons.

    Parameters
    ----------
    n_electrons : float
        Number of electrons in the system

    Returns
    -------
    n_quantum : np.ndarray[int]
        Principal quantum number n for each orbital
    l_quantum : np.ndarray[int]
        Angular momentum quantum number l for each orbital
    s_quantum_up : np.ndarray[float]
        Spin-up occupation for each orbital
    s_quantum_down : np.ndarray[float]
        Spin-down occupation for each orbital
    """

    # Type checking
    assert isinstance(n_electrons, int | float), \
        N_ELECTRONS_NOT_INT_OR_FLOAT_ERROR.format(type(n_electrons))
    assert n_electrons > 0, \
        N_ELECTRONS_NOT_GREATER_THAN_0_ERROR.format(n_electrons)
    assert n_electrons <= 92, \
        N_ELECTRONS_NOT_LESS_THAN_OR_EQUAL_TO_92_ERROR.format(n_electrons)

    # Use the smallest integer >= n_electrons as the reference configuration.
    ceil_electrons = int(np.ceil(n_electrons))
    fractional_delta = n_electrons - ceil_electrons  # <= 0 for fractional (removing charge)

    n_quantum, l_quantum, occ_spin_up, occ_spin_down = get_neutral_occupation_states(ceil_electrons)
    occ_spin_up   = occ_spin_up.astype(float)
    occ_spin_down = occ_spin_down.astype(float)

    if n_electrons.is_integer():
        assert fractional_delta == 0
    elif ceil_electrons not in HARDCORED_OCCUPATION_EXCEPTION_ORBITAL_INDEX_DICT:
        if ceil_electrons == 1:
            # Hydrogen-like case: only one orbital exists.
            occ_spin_up   = np.array([n_electrons], dtype=float)
            occ_spin_down = np.array([0.0], dtype=float)
        else:
            # Align Z-1 orbitals to the Z orbital list, then locate the changed orbital.
            (n_quantum_prev, l_quantum_prev, occ_spin_up_prev, occ_spin_down_prev) = get_neutral_occupation_states(ceil_electrons - 1)

            occ_spin_up_prev_aligned   = np.zeros_like(occ_spin_up, dtype=float)
            occ_spin_down_prev_aligned = np.zeros_like(occ_spin_down, dtype=float)

            prev_idx = 0
            curr_idx = 0
            # Two-pointer walk to align Z-1 orbitals into the Z orbital list.
            # When (n,l) matches, copy the previous occupations into the current index.
            while prev_idx < len(n_quantum_prev) and curr_idx < len(n_quantum):
                if (n_quantum_prev[prev_idx] == n_quantum[curr_idx]) and \
                   (l_quantum_prev[prev_idx] == l_quantum[curr_idx]):
                    occ_spin_up_prev_aligned[curr_idx] = occ_spin_up_prev[prev_idx]
                    occ_spin_down_prev_aligned[curr_idx] = occ_spin_down_prev[prev_idx]
                    prev_idx += 1
                    curr_idx += 1
                else:
                    # The current Z list has an extra (new) orbital not in Z-1.
                    # Leave its aligned occupation as zero and advance only curr_idx.
                    curr_idx += 1

            # Calculate the difference between the current and previous occupations.
            diff_up   = occ_spin_up - occ_spin_up_prev_aligned
            diff_down = occ_spin_down - occ_spin_down_prev_aligned

            # Find the indices of the orbitals that have changed.
            changed_up   = np.argwhere(diff_up != 0)[:, 0]
            changed_down = np.argwhere(diff_down != 0)[:, 0]

            if len(changed_up) > 0:
                occ_spin_up[changed_up] = occ_spin_up[changed_up] + fractional_delta
            elif len(changed_down) > 0:
                occ_spin_down[changed_down] = occ_spin_down[changed_down] + fractional_delta

            if len(changed_up) > 0 and len(changed_down) > 0:
                raise ValueError("Ambiguous fractional occupation change in both spins, this should not happen.")
    else:
        # Hardcoded exceptions for discontinuous occupation changes.
        assert ceil_electrons in HARDCORED_OCCUPATION_EXCEPTION_ORBITAL_INDEX_DICT, \
            "Invalid atomic number {} for fractional occupation states, this should not happen.".format(ceil_electrons)
        offset = HARDCORED_OCCUPATION_EXCEPTION_ORBITAL_INDEX_DICT[ceil_electrons]
        occ_spin_up[offset] = occ_spin_up[offset] + fractional_delta

    return n_quantum, l_quantum, occ_spin_up, occ_spin_down


class OccupationInfo:
    """
    Occupation information for atomic states.
    """
    z_valence                  : int | float  # Valence charge (for pseudopotential)
    z_nuclear                  : int | float  # True nuclear charge of the atom
    n_electrons                : int | float  # Number of electrons in the system, can be fractional
    all_electron_flag          : bool         # Whether to use all-electron or pseudopotential
    occ_n                      : np.ndarray   # Principal quantum number n for each orbital
    occ_l                      : np.ndarray   # Angular momentum quantum number l for each orbital
    occ_spin_up                : np.ndarray   # Spin-up occupation for each orbital
    occ_spin_down              : np.ndarray   # Spin-down occupation for each orbital
    occ_spin_up_plus_spin_down : np.ndarray   # Total occupation (spin-up + spin-down)
    occupation_rule            : str          # 'table' (fixed, default) or 'aufbau' (eigenvalue order)
    occupation_smearing        : Optional[float]  # Fermi-Dirac kT (Ha) for 'aufbau', None for 'table'

    # Fermi-Dirac tails below this many electrons per subshell are set to exactly zero, so
    # "occupied" (occupations > 0, e.g. the OEP frontier choice) keeps its meaning.
    AUFBAU_OCCUPATION_FLOOR = 1.0e-12


    def __init__(self, 
        z_nuclear         : int | float,                  # True nuclear charge (atomic number)
        z_valence         : int | float,                  # Valence charge (for pseudopotential Coulomb tail)
        all_electron_flag : bool,                         # Whether to use all-electron or pseudopotential
        n_electrons       : Optional[int | float] = None, # Number of electrons in the system, can be fractional
    ):
        """
        Initialize occupation information.
        
        Parameters
        ----------
        z_nuclear : int | float
            True nuclear charge of the atom (atomic number)
        z_valence : int | float
            Valence charge for pseudopotential calculations
        all_electron_flag : bool
            True for all-electron, False for pseudopotential
        n_electrons : int | float, optional
            Number of electrons in the system, can be fractional, by default, set to float(z_nuclear)
        """

        self.z_nuclear         : float = z_nuclear
        self.z_valence         : float = z_valence
        self.n_electrons       : float = n_electrons
        self.all_electron_flag : bool  = all_electron_flag
        self._set_and_check_initial_parameters()


        if all_electron_flag:

            n_quantum, l_quantum, s_quantum_up, s_quantum_down = get_fraction_occupation_states(self.n_electrons)
            self.occ_n = n_quantum
            self.occ_l = l_quantum
            self.occ_spin_up   = s_quantum_up
            self.occ_spin_down = s_quantum_down
        else:
            # For pseudopotential: check if z_nuclear is integer-valued and equal to n_electrons
            assert self.z_nuclear.is_integer(), \
                Z_NUCLEAR_NOT_INTEGER_VALUED_FOR_PSEUDOPOTENTIAL_CALCULATION_ERROR.format(self.z_nuclear)
            assert self.z_nuclear == self.n_electrons, \
                CHARGE_SYSTEMS_NOT_SUPPORTED_FOR_PSEUDOPOTENTIAL_CALCULATION_ERROR.format(self.n_electrons)

            n_quantum, l_quantum, s_quantum_up, s_quantum_down = get_neutral_occupation_states(int(self.z_nuclear))

            # select only valence electrons
            n_core_electrons = z_nuclear - z_valence
            orbital_occupation_numbers = s_quantum_up + s_quantum_down
            cumulative_occupation = np.cumsum(orbital_occupation_numbers)
            valence_orbitals_indices = np.where(cumulative_occupation > n_core_electrons)[0]
            self.occ_n = n_quantum[valence_orbitals_indices]
            self.occ_l = l_quantum[valence_orbitals_indices]
            self.occ_spin_up   = s_quantum_up[valence_orbitals_indices]
            self.occ_spin_down = s_quantum_down[valence_orbitals_indices]
            
        self.occ_spin_up_plus_spin_down = self.occ_spin_up + self.occ_spin_down

        # Check if the total occupation numbers match the number of electrons
        if self.all_electron_flag:
            assert np.sum(self.occ_spin_up_plus_spin_down) == self.n_electrons, \
                TOTAL_OCCUPATION_NUMBERS_DO_NOT_MATCH_THE_NUMBER_OF_ELECTRONS_ERROR.format(np.sum(self.occ_spin_up_plus_spin_down), self.n_electrons)
        else:
            assert np.sum(self.occ_spin_up_plus_spin_down) == self.z_valence, \
                TOTAL_OCCUPATION_NUMBERS_DO_NOT_MATCH_THE_NUMBER_OF_ELECTRONS_ERROR.format(np.sum(self.occ_spin_up_plus_spin_down), self.z_valence)

        # Occupation rule. 'table' (default) keeps the occupations above for the whole SCF.
        # 'aufbau' refills the subshells from their eigenvalues every SCF iteration, see
        # enable_aufbau_occupation and update_occupations_from_eigenvalues.
        self.occupation_rule     : str             = "table"
        self.occupation_smearing : Optional[float] = None
        self._aufbau_warned_channels : set         = set()



    def _set_and_check_initial_parameters(self):
        """
        Set and check initial parameters.
        """
        # Type checking and conversion
        # z_nuclear
        assert isinstance(self.z_nuclear, (int, float)), \
            Z_NUCLEAR_NOT_INT_OR_FLOAT_ERROR.format(type(self.z_nuclear))
        assert self.z_nuclear > 0 and self.z_nuclear < 93, \
            Z_NUCLEAR_NOT_GREATER_THAN_0_OR_LESS_THAN_93_ERROR.format(self.z_nuclear)
        self.z_nuclear = float(self.z_nuclear)

        # z_valence
        assert isinstance(self.z_valence, (int, float)), \
            Z_VALENCE_NOT_INT_OR_FLOAT_ERROR.format(type(self.z_valence))
        assert self.z_valence > 0 and self.z_valence < 93, \
            Z_VALENCE_NOT_GREATER_THAN_0_OR_LESS_THAN_93_ERROR.format(self.z_valence)
        self.z_valence = float(self.z_valence)

        # all electron flag
        assert isinstance(self.all_electron_flag, bool), \
            ALL_ELECTRON_FLAG_NOT_BOOL_ERROR.format(type(self.all_electron_flag))
        
        # n_electrons
        if self.n_electrons is None:
            self.n_electrons = self.z_nuclear
        else:
            assert isinstance(self.n_electrons, (int, float)), \
                N_ELECTRONS_NOT_INT_OR_FLOAT_ERROR.format(type(self.n_electrons))
            assert self.n_electrons > 0, \
                N_ELECTRONS_NOT_GREATER_THAN_0_ERROR.format(self.n_electrons)
            self.n_electrons = float(self.n_electrons)


    @property
    def n_free_electrons(self) -> float:
        """
        Total number of free electrons in the system.

        This is the sum of spin-up and spin-down occupations over the active
        orbital set:
        - All-electron calculations: sum over all-electron occupations.
        - Pseudopotential calculations: sum over valence occupations.
        """
        return np.sum(self.occ_spin_up_plus_spin_down)

    @property
    def n_free_electrons_up(self) -> float:
        """
        Number of spin-up free electrons in the system.

        This is the sum of spin-up occupations over the active orbital set:
        - All-electron calculations: sum over all-electron spin-up occupations.
        - Pseudopotential calculations: sum over valence spin-up occupations.
        """
        return np.sum(self.occ_spin_up)
    
    @property
    def n_free_electrons_dn(self) -> float:
        """
        Number of spin-down free electrons in the system.

        This is the sum of spin-down occupations over the active orbital set:
        - All-electron calculations: sum over all-electron spin-down occupations.
        - Pseudopotential calculations: sum over valence spin-down occupations.
        """
        return np.sum(self.occ_spin_down)


    @property
    def occupations(self) -> np.ndarray:
        """
        Total occupation numbers (spin-up + spin-down) for each orbital.
        Alias for occ_spin_up_plus_spin_down for cleaner API.
        """
        return self.occ_spin_up_plus_spin_down
    
    @property
    def occupations_up(self) -> np.ndarray:
        """
        Spin-up occupation numbers for each orbital.
        """
        return self.occ_spin_up
    
    @property
    def occupations_dn(self) -> np.ndarray:
        """
        Spin-down occupation numbers for each orbital.
        """
        return self.occ_spin_down




    @property
    def l_values(self) -> np.ndarray:
        """
        Angular momentum quantum numbers for each orbital.
        Alias for occ_l for cleaner API.
        """
        return self.occ_l
    
    @property
    def n_values(self) -> np.ndarray:
        """
        Principal quantum numbers for each orbital.
        Alias for occ_n for cleaner API.
        """
        return self.occ_n
    
    @property
    def unique_l_values(self) -> np.ndarray:
        """Get unique angular momentum quantum numbers present in occupied states."""
        return np.unique(self.occ_l)
    

    @property
    def n_states(self) -> int:
        """Get total number of occupied states."""
        return len(self.occ_n)


    def n_states_for_l(self, l: int) -> int:
        """
        Get number of occupied states for a given angular momentum quantum number.
        
        Parameters
        ----------
        l : int
            Angular momentum quantum number
        
        Returns
        -------
        n_states : int
            Number of states with this l value
        """
        return np.sum(self.occ_l == l)



    def enable_aufbau_occupation(
        self,
        smearing                 : float         = 1.0e-3,
        l_max                    : Optional[int] = None,
        n_candidates_per_channel : int           = 2,
    ) -> None:
        """
        Switch to eigenvalue-ordered (Aufbau) occupations.

        n_candidates_per_channel empty subshells are appended to every channel l = 0..l_max:
        the next n above the listed ones, or n = l+1, l+2, ... for a channel with no listed
        subshell. A level the table leaves empty (3d of Ti2+, 4f of La, 5f of Th) can then
        take electrons, and the last candidate of each channel stays empty unless the list
        is too short (see warn_if_aufbau_candidates_occupied). Every
        SCF iteration, update_occupations_from_eigenvalues refills all listed subshells from
        their eigenvalues, the way 3D codes fill bands (M-SPARC src/occupations.m):

            N_i = 2(2l_i+1) / (1 + exp((eps_i - mu)/kT)),   sum_i N_i = n_free_electrons,

        with kT = smearing (Ha). smearing = 0 fills strictly in eigenvalue order, the last
        subshell fractionally. The candidates come after the table entries, so each channel
        stays in ascending n and the per-l state mapping of the SCF driver is unchanged.

        Parameters
        ----------
        smearing : float
            Fermi-Dirac kT in Ha. Default 1e-3 (315.8 K, the M-SPARC isolated-atom setting).
        l_max : int, optional
            Highest channel that gets candidates. Default min(max listed l + 1, 3).
        n_candidates_per_channel : int
            Empty subshells appended per channel. Default 2.
        """
        assert isinstance(smearing, (int, float)) and smearing >= 0.0, \
            AUFBAU_SMEARING_NOT_NON_NEGATIVE_FLOAT_ERROR.format(smearing)
        if l_max is None:
            l_max = min(int(np.max(self.occ_l)) + 1, 3)
        assert isinstance(l_max, (int, np.integer)) and l_max >= 0, \
            AUFBAU_L_MAX_NOT_NON_NEGATIVE_INT_ERROR.format(l_max)
        assert isinstance(n_candidates_per_channel, (int, np.integer)) and n_candidates_per_channel >= 1, \
            AUFBAU_N_CANDIDATES_NOT_POSITIVE_INT_ERROR.format(n_candidates_per_channel)

        occ_n = [int(n) for n in self.occ_n]
        occ_l = [int(l) for l in self.occ_l]
        for l in range(int(l_max) + 1):
            n_in_channel = [n for n, l_i in zip(occ_n, occ_l) if l_i == l]
            n_first = max(n_in_channel) + 1 if n_in_channel else l + 1
            for k in range(int(n_candidates_per_channel)):
                occ_n.append(n_first + k)
                occ_l.append(l)
        n_added = len(occ_n) - len(self.occ_n)

        self.occ_n         = np.asarray(occ_n, dtype=np.asarray(self.occ_n).dtype)
        self.occ_l         = np.asarray(occ_l, dtype=np.asarray(self.occ_l).dtype)
        # Spin-unpolarized: only the sum matters, so the table's Hund split is dropped and
        # both spins carry half of each subshell.
        total = np.concatenate([np.asarray(self.occ_spin_up_plus_spin_down, dtype=float), np.zeros(n_added)])
        self.occ_spin_up   = 0.5 * total
        self.occ_spin_down = 0.5 * total
        self.occ_spin_up_plus_spin_down = self.occ_spin_up + self.occ_spin_down

        self.occupation_rule      = "aufbau"
        self.occupation_smearing  = float(smearing)
        self._aufbau_n_electrons  = float(np.sum(self.occ_spin_up_plus_spin_down))


    def update_occupations_from_eigenvalues(
        self,
        occ_energies : np.ndarray,
    ) -> None:
        """
        Refill the subshells from their eigenvalues (occupation_rule 'aufbau' only).

        occ_energies are the subshell eigenvalues in this occupation order (the SCF driver's
        occ_eigenvalues). The occupation arrays are updated IN PLACE: the density, response
        and OEP calculators hold references to them. No-op for occupation_rule 'table'.
        """
        if self.occupation_rule != "aufbau":
            return
        energies = np.asarray(occ_energies, dtype=float).reshape(-1)
        if energies.size != self.occupations.size:
            raise ValueError(
                OCC_ENERGIES_SIZE_MISMATCH_FOR_HOMO_FRACTION_ERROR.format(energies.size, self.occupations.size)
            )
        degeneracy = 2.0 * (2 * np.asarray(self.occ_l, dtype=float) + 1)
        total = self.fermi_dirac_subshell_occupations(
            energies    = energies,
            degeneracy  = degeneracy,
            n_electrons = self._aufbau_n_electrons,
            smearing    = self.occupation_smearing,
        )
        # drop the far Fermi-Dirac tails and restore the electron count exactly
        total[total < self.AUFBAU_OCCUPATION_FLOOR] = 0.0
        total *= self._aufbau_n_electrons / float(np.sum(total))
        self.occ_spin_up[:]                = 0.5 * total
        self.occ_spin_down[:]              = 0.5 * total
        self.occ_spin_up_plus_spin_down[:] = total



    def warn_if_aufbau_candidates_occupied(
        self,
        threshold : float = 1.0e-3,
    ) -> None:
        """
        Warn (once per channel) when the last listed subshell of a channel holds more than
        `threshold` electrons: the next level of that channel is then missing from the list.
        Called by the SCF driver after the inner loop, so early SCF transients do not warn.
        No-op for occupation_rule 'table'.
        """
        if self.occupation_rule != "aufbau":
            return
        for l in np.unique(self.occ_l):
            top = int(np.nonzero(self.occ_l == l)[0][-1])
            if self.occupations[top] > threshold and int(l) not in self._aufbau_warned_channels:
                print(AUFBAU_TOP_CANDIDATE_OCCUPIED_WARNING.format(
                    self.occupations[top], "{}{}".format(int(self.occ_n[top]), "spdfghik"[int(l)]), int(l)))
                self._aufbau_warned_channels.add(int(l))


    def occupation_table(self) -> np.ndarray:
        """(n_subshells, 3) array of n, l and the total subshell occupation, in list order."""
        return np.column_stack([
            np.asarray(self.occ_n, dtype=float),
            np.asarray(self.occ_l, dtype=float),
            np.asarray(self.occupations, dtype=float),
        ])


    def set_occupations(
        self,
        occ_n       : np.ndarray,
        occ_l       : np.ndarray,
        occupations : np.ndarray,
    ) -> None:
        """
        Overwrite the subshell occupations IN PLACE with a saved set (e.g. occupations.txt).

        The (n, l) list must equal this one: build the solver with the same occupation_rule
        first, so that 'aufbau' has appended the same candidate subshells. Used to replay a
        saved aufbau state in a forward pass or at an intermediate SCF iteration, where the
        occupations must not be recomputed.
        """
        if self.occupation_rule != "aufbau":
            raise ValueError(AUFBAU_SET_OCCUPATIONS_TABLE_RULE_ERROR.format(self.occupation_rule))
        occ_n       = np.asarray(occ_n, dtype=float).reshape(-1)
        occ_l       = np.asarray(occ_l, dtype=float).reshape(-1)
        occupations = np.asarray(occupations, dtype=float).reshape(-1)
        mine = list(zip(np.asarray(self.occ_n, dtype=int).tolist(), np.asarray(self.occ_l, dtype=int).tolist()))
        given = list(zip(np.rint(occ_n).astype(int).tolist(), np.rint(occ_l).astype(int).tolist()))
        if given != mine:
            raise ValueError(AUFBAU_OCCUPATION_LIST_MISMATCH_ERROR.format(given, mine))
        if not np.isclose(float(np.sum(occupations)), float(np.sum(self.occ_spin_up_plus_spin_down)), rtol=0.0, atol=1e-8):
            raise ValueError(AUFBAU_OCCUPATION_SUM_MISMATCH_ERROR.format(
                float(np.sum(occupations)), float(np.sum(self.occ_spin_up_plus_spin_down))))
        self.occ_spin_up[:]                = 0.5 * occupations
        self.occ_spin_down[:]              = 0.5 * occupations
        self.occ_spin_up_plus_spin_down[:] = occupations


    def smearing_entropy_term(self) -> float:
        """
        -T S of the Fermi-Dirac subshell occupations (Ha). Added to the total energy it gives
        the free energy that M-SPARC reports. Zero for occupation_rule 'table' or smearing 0.
        """
        if self.occupation_rule != "aufbau" or not self.occupation_smearing:
            return 0.0
        degeneracy = 2.0 * (2 * np.asarray(self.occ_l, dtype=float) + 1)
        f = np.clip(self.occupations / degeneracy, 1.0e-300, 1.0 - 1.0e-16)
        s = f * np.log(f) + (1.0 - f) * np.log1p(-f)
        return float(self.occupation_smearing * np.sum(degeneracy * s))


    @staticmethod
    def fermi_dirac_subshell_occupations(
        energies    : np.ndarray,
        degeneracy  : np.ndarray,
        n_electrons : float,
        smearing    : float,
    ) -> np.ndarray:
        """
        Subshell occupations N_i = g_i / (1 + exp((eps_i - mu)/kT)) with sum_i N_i = n_electrons.

        smearing = kT (Ha). kT = 0 fills the subshells strictly in ascending eigenvalue
        order, the last one fractionally.
        """
        energies   = np.asarray(energies, dtype=float)
        degeneracy = np.asarray(degeneracy, dtype=float)
        capacity   = float(np.sum(degeneracy))
        if capacity <= n_electrons:
            raise ValueError(AUFBAU_CAPACITY_TOO_SMALL_ERROR.format(capacity, n_electrons))

        if smearing == 0.0:
            occupations = np.zeros_like(energies)
            remaining   = float(n_electrons)
            for i in np.argsort(energies, kind="stable"):
                occupations[i] = min(degeneracy[i], remaining)
                remaining     -= occupations[i]
                if remaining <= 0.0:
                    break
            return occupations

        mu_low  = float(np.min(energies)) - 40.0 * smearing - 1.0
        mu_high = float(np.max(energies)) + 40.0 * smearing + 1.0
        mu = brentq(OccupationInfo._fermi_dirac_electron_count_residual, mu_low, mu_high,
                    args=(energies, degeneracy, float(n_electrons), float(smearing)), xtol=1.0e-14)
        return degeneracy * expit((mu - energies) / smearing)


    @staticmethod
    def _fermi_dirac_electron_count_residual(
        mu          : float,
        energies    : np.ndarray,
        degeneracy  : np.ndarray,
        n_electrons : float,
        smearing    : float,
    ) -> float:
        """sum_i g_i f((eps_i - mu)/kT) - n_electrons, the root function for the chemical potential."""
        return float(np.sum(degeneracy * expit((mu - energies) / smearing))) - n_electrons


    @property
    def closed_shell_flag(self) -> bool:
        """
        Check if the atom is closed-shell.
        """
        for n, l, spin_up, spin_down in zip(self.occ_n, self.occ_l, self.occ_spin_up, self.occ_spin_down):
            if spin_up != l * 2 + 1 or spin_down != l * 2 + 1:
                return False
        return True



    def print_info(self):
        def _format_array_for_print(values: np.ndarray) -> np.ndarray:
            """
            Format array for display only.
            If all entries are integer-valued, print as ints.
            """
            array_values = np.asarray(values)
            if np.issubdtype(array_values.dtype, np.integer):
                return array_values
            if np.allclose(array_values, np.rint(array_values), rtol=0.0, atol=1e-12):
                return np.rint(array_values).astype(int)
            return array_values

        print("===========================================================================")
        print("                        OCCUPATION INFORMATION                             ")
        print("===========================================================================")
        print(f"\t z_valence (valence charge) : {self.z_valence}")
        print(f"\t z_nuclear (nuclear charge) : {self.z_nuclear}")
        print(f"\t n_electrons                : {self.n_electrons}")
        print(f"\t all_electron_flag          : {self.all_electron_flag}")
        print(f"\t occ_n                      : {_format_array_for_print(self.occ_n)}")
        print(f"\t occ_l                      : {_format_array_for_print(self.occ_l)}")
        print(f"\t occ_spin_up                : {_format_array_for_print(self.occ_spin_up)}")
        print(f"\t occ_spin_down              : {_format_array_for_print(self.occ_spin_down)}")
        print(f"\t occ_spin_up_plus_spin_down : {_format_array_for_print(self.occ_spin_up_plus_spin_down)}")
        if self.occupation_rule != "table":
            print(f"\t occupation_rule            : {self.occupation_rule} (smearing kT = {self.occupation_smearing} Ha)")
        print()




if __name__ == "__main__":
    for atomic_number in range(1, 93):
        occupation_info = OccupationInfo(z_nuclear=atomic_number, z_valence=atomic_number, all_electron_flag=True)
        if occupation_info.closed_shell_flag:
            print(f"atomic_number = {atomic_number} is closed shell")
        else:
            pass



