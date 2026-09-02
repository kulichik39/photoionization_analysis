import numpy as np
from fortran_output_analysis.types import ArrFloat64, ArrComplex128
from fortran_output_analysis.constants_and_parameters import (
    g_inverse_atomic_frequency_to_attoseconds,
)


def l_to_str(l: int) -> str:
    """
    Converts orbital momentum into the corresponding string literal s,p,d ...

    Args:
        l - orbital angular momentum.

    Returns:
        the string literal for the given momentum.
    """
    if l == 0:
        return "s"
    elif l == 1:
        return "p"
    elif l == 2:
        return "d"
    elif l == 3:
        return "f"
    elif l == 4:
        return "g"
    elif l == 5:
        return "h"
    else:
        raise ValueError(
            "l_to_str(): invalid or unimplemented orbital momentum value."
            "Function was given: l =",
            l,
        )


def l_to_int(l: str) -> int:
    """
    Opposite to l_to_str().
    Converts given string literal into the orbital momentum value.

    Args:
        l - string literal for the orbital momentum.

    Returns:
        value of the orbital momentum.
    """

    if l == "s":
        return 0
    elif l == "p":
        return 1
    elif l == "d":
        return 2
    elif l == "f":
        return 3
    elif l == "g":
        return 4
    elif l == "h":
        return 5
    else:
        raise ValueError(
            "l_to_int(): invalid or unimplemented orbital momentum string literal."
            "Function was given: l =",
            l,
        )


def final_sideband_energies_2sim(
    energies_emi: ArrFloat64, energies_abs: ArrFloat64, energies_mode: str
) -> ArrFloat64:
    """
    Prepares an array of final sideband energies in the case of 2 simulations.

    Args:
        energies_emi - energies of the emission path.
        energies_abs - energies of the absorption path.
        energies_mode - tells which energies should be taken.
                        Possible options:
                        "emi" - emission energies.
                        "abs" - absorption energies.
                        "both" - combined emission and absorption energies.

    Returns:
        an array of final sideband energies.
    """

    assert energies_mode in (
        "emi",
        "abs",
        "both",
    ), f"energies_mode for the final sideband energy from two simulations must be 'emi', 'abs' or 'both', not '{energies_mode}'!"

    if energies_mode == "emi":
        return energies_emi
    elif energies_mode == "abs":
        return energies_abs
    else:
        energies_concat = np.concatenate((energies_abs, energies_emi))
        energies_final = np.sort(np.unique(energies_concat))
        return energies_final


def match_matrix_elements_2sim(
    energies_final: ArrFloat64,
    energies_emi: ArrFloat64,
    energies_abs: ArrFloat64,
    mat_emi: ArrComplex128,
    mat_abs: ArrComplex128,
    match_mode: str,
) -> tuple[ArrComplex128, ArrComplex128]:
    """
    Matches absoprtion and emission matrix elements to the final sideband energies
    in the case of two simulations.

    Args:
        energies_final - final sideband energies.
        energies_emi - energies of the emission path.
        energies_abs - energies of  the absorption path.
        mat_emi - unmatched matrix elements for the emission path.
        mat_abs - unmatched matrix elements for the absorption path.
        match_mode - the mode of the matrix element matching.
                     Possible options:
                     "interp_both" - interpolate both absorption and emission paths for the final
                                     energies.
                     "lin_extrap_emi" - linearly extrapolate the emission path and interpolate the
                                        absorption path for the final energies.
                     "lin_extrap_emi_left" - linearly extrapolate the emission path to the left using
                                             the first two points and interpolate the absosrption
                                             path for the final energies.


    Returns:
        mat_emi_matched - matched matrix elements for the emission path.
        mat_abs_matched - matched matrix elements for the absorption path.
    """

    match_mode_options = (
        "interp_both",
        "lin_extrap_emi",
        "lin_extrap_emi_left",
    )
    assert (
        match_mode in match_mode_options
    ), f"match_mode for the matching of matrices from two simulations must be in {match_mode_options}, not '{match_mode}'!"

    if match_mode == "interp_both":
        mat_emi_matched = np.interp(energies_final, energies_emi, mat_emi)
        mat_abs_matched = np.interp(energies_final, energies_abs, mat_abs)

    elif match_mode == "lin_extrap_emi":
        # linear extrapolation for the emission path
        fit = np.polyfit(energies_emi, mat_emi, 1)  # fitting coefficients
        mat_emi_matched = fit[0] * energies_final + fit[1]
        # interpolation for the absorption path
        mat_abs_matched = np.interp(energies_final, energies_abs, mat_abs)

    elif match_mode == "lin_extrap_emi_left":
        # linearly extrapolate emi to the left, using emi = k * energies + b
        # since extrapolating to the left, k and b are derived using the first two points
        k = (mat_emi[1] - mat_emi[0]) / (energies_emi[1] - energies_emi[0])
        b = mat_emi[0] - k * energies_emi[0]
        mat_emi_matched = k * energies_final + b
        # interpolation for the absorption path
        mat_abs_matched = np.interp(energies_final, energies_abs, mat_abs)

    return mat_emi_matched, mat_abs_matched


def unwrap_phase_with_nans(phase):
    """
    Unwraps phase that contains NaN values by masking out the NaNs.
    """

    # np.unwrap can not handle NaNs, mask them out.
    nanmask = np.logical_not(np.isnan(phase))
    phase[nanmask] = np.unwrap(phase[nanmask])

    return phase


def compute_omega_diff(g_omega_IR_1: float, g_omega_IR_2: float | None = None) -> float:
    """
    Computes energy sepearation between the absorption and emission paths.
    Can compute for 1 or 2 simulations.

    Args:
        g_omega_IR_1 - energy of the IR photon in the first simulation.
        g_omega_IR_2 - energy of the IR photon in the second simulation

    Returns:
        omega_diff - energy difference between the absorption and emission paths.
    """
    if g_omega_IR_2:  # if two simulations are provided
        omega_diff = g_omega_IR_1 + g_omega_IR_2
    else:  # if only one simulation is provided
        omega_diff = 2.0 * g_omega_IR_1

    return omega_diff


def phase_to_delay(phase: ArrFloat64, omega_diff_hart: float) -> ArrFloat64:
    """
    Converts phase into delay.

    Args:
        phase - an array with phase.
        omega_diff_hart - energy separation between the absorption and emission paths in Hartree.

    Returns:
        an array with delay.
    """

    return phase * g_inverse_atomic_frequency_to_attoseconds / omega_diff_hart
