import numpy as np
import math
from scipy.special import legendre
from fortran_output_analysis.common_utility import assert_abs_or_emi

"""
This namespace contains functions for analyzing complex q parameter in the one and 
two photon cases.
"""


def get_epsilon(ekin_eV, abs_or_emi, g_omega_IR_eV, E_res, width_res):
    """
    Calculates an array of resclased energies eps used in the formula for complex parameter.

    Args:
        ekin_eV - array of electron kinetic energies
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string
        g_omega_IR_eV - energy of the IR photon in eV
        E_res - energy of the resonance in eV
        width_res - width of the resonance in eV

    Returns:
        eps - array with the rescaled energy values
    """

    assert_abs_or_emi(abs_or_emi)

    # create an array of the scaled energies that are used in the formula
    if abs_or_emi == "abs":
        eps = (ekin_eV - (E_res + g_omega_IR_eV)) / (width_res / 2)
    else:
        eps = (ekin_eV - (E_res - g_omega_IR_eV)) / (width_res / 2)

    return eps


def get_coefficients_for_integrated(
    abs_or_emi,
    eps,
    integrated_intensity,
    q,
    eps_1=1.0,
    eps_2=0.0,
):
    """
    Calculates A_bg and A_0 coefficients for the formula describing intenisty (amplitude)
    near a resonance in the angular integrated case.

    Args:
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string
        eps - array with the rescaled energy values
        integrated_intensity - array with the integrated intensity values
        q - shape parameter (introduced by Fano)
        eps_1 - the first rescaled energy point to estimate coefficients
        eps_2 - the second rescaled energy point to estimate coefficients
        NOTE: the code is sensitive to the choice of eps_1 and eps_2

    Returns:
        A_bg and A_0 - formula coefficients
    """

    assert_abs_or_emi(abs_or_emi)

    # find intenisties at the sepcified points
    intens_1 = np.interp(eps_1, eps, integrated_intensity)
    intens_2 = np.interp(eps_2, eps, integrated_intensity)
    # If the resonance was in the absorption path than the formula describes conjugated intensity
    if abs_or_emi == "abs":
        intens_1 = np.conjugate(intens_1)
        intens_2 = np.conjugate(intens_2)

    # compute the coefficients using the found intensities
    A_0 = (intens_2 - intens_1) / (
        (eps_2 + q) / (eps_2 + 1 * 1j) - (eps_1 + q) / (eps_1 + 1 * 1j)
    )
    A_bg = intens_1 - A_0 * (eps_1 + q) / (eps_1 + 1 * 1j)

    return A_bg, A_0


def get_complex_q_integrated(A_bg, A_0, q):
    """
    Computes complex q parameter for the formula describing intensity (amplitude) near a
    resonance in the angular integrated case.

    Args:
        A_bg, A_0 - coefficients of the formula
        q - shape parameter (introduced by Fano)

    Returns:
        value of the complex q parameter
    """

    if A_0 == 0 and A_bg == 0:
        return 0

    return (A_0 * q + 1j * A_bg) / (A_0 + A_bg)


def get_background_contribution_integrated(abs_or_emi, A_bg, A_0):
    """
    Computes background contribution (the term before energy dependence) for the formula
    describing intensity (amplitude) near a resonance in the angular integrated case.

    Args:
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string
        A_bg, A_0 - coefficients of the formula

    Returns:
        background_cont - value of the background contribution
    """

    assert_abs_or_emi(abs_or_emi)

    background_cont = A_0 + A_bg
    if abs_or_emi == "abs":
        background_cont = np.conjugate(background_cont)

    return background_cont


def get_coefficients_for_beta_param(
    abs_or_emi,
    eps,
    integrated_intensity,
    beta_param,
    q,
    eps_1=1.0,
    eps_2=0.0,
):
    """
    Finds A_0 and A_bg coefficients for the given beta parameter to consturct the formula
    describing intensity (amplitude) near a resonance when the angular dependence is included.

    Args:
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string
        eps - array with the rescaled energy values
        integrated_intensity - array with the integrated intensity values
        beta_param - array with the beta parameter values
        q - shape parameter (introduced by Fano)
        eps_1 - the first rescaled energy point to estimate coefficients
        eps_2 - the second rescaled energy point to estimate coefficients
        NOTE: the code is sensitive to the choice of eps_1 and eps_2

    Returns:
        A_0, A_bg - coefficients for the given beta parameter
    """

    assert_abs_or_emi(abs_or_emi)

    # extract the numertor in the formula for beta parameter
    beta_numerator = beta_param * integrated_intensity

    # find beta numerator at the specified points
    beta_numerator_1 = np.interp(eps_1, eps, beta_numerator)
    beta_numerator_2 = np.interp(eps_2, eps, beta_numerator)

    # If the resonance was in the absorption path than the formula describes conjugated intensity
    if abs_or_emi == "abs":
        beta_numerator_1 = np.conjugate(beta_numerator_1)
        beta_numerator_2 = np.conjugate(beta_numerator_2)

    # compute the coefficients using the found intensities
    A_0 = (beta_numerator_2 - beta_numerator_1) / (
        (eps_2 + q) / (eps_2 + 1 * 1j) - (eps_1 + q) / (eps_1 + 1 * 1j)
    )
    A_bg = beta_numerator_1 - A_0 * (eps_1 + q) / (eps_1 + 1 * 1j)

    return A_bg, A_0


def get_complex_q_angular(
    angle, A_bg_int, A_0_int, A_bg_b2, A_0_b2, A_bg_b4, A_0_b4, q
):
    """
    Computes complex q parameter for the formula describing intensity (amplitude) near a
    resonance when the angular dependence is included.

    Args:
        angle - value of the angle for which we estimate complex q
        A_bg_int, A_0_int - coefficients for the integrated case
        A_bg_b2, A_0_b2 - coefficients for the beta parameter of order 2
        A_bg_b4, A_0_b4 - coefficients for the beta parameter of order 4 (NOTE: put them to zero if
        you work with the one photon case)
        q - shape parameter (introduced by Fano)

    Returns:
        value of the complex q parameter
    """

    # Introduce some auxiliary values for calculations
    q_complex_int = get_complex_q_integrated(A_bg_int, A_0_int, q)
    q_complex_b2 = get_complex_q_integrated(A_bg_b2, A_0_b2, q)
    q_complex_b4 = get_complex_q_integrated(A_bg_b4, A_0_b4, q)
    A_int = A_0_int + A_bg_int
    A_b2 = (A_0_b2 + A_bg_b2) * legendre(2)(np.array(np.cos(math.radians(angle))))
    A_b4 = (A_0_b4 + A_bg_b4) * legendre(4)(np.array(np.cos(math.radians(angle))))

    return (A_int * q_complex_int + A_b2 * q_complex_b2 + A_b4 * q_complex_b4) / (
        A_int + A_b2 + A_b4
    )


def get_background_contribution_angular(
    angle, abs_or_emi, A_bg_int, A_0_int, A_bg_b2, A_0_b2, A_bg_b4, A_0_b4
):
    """
    Computes background contribution (the term before energy dependence) for the formula
    describing intensity (amplitude) near a resonance when the angular dependence is included.

    Args:
        angle - value of the angle for which we estimate complex q
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string
        A_bg_int, A_0_int - coefficients for the integrated case
        A_bg_b2, A_0_b2 - coefficients for the beta parameter of order 2
        A_bg_b4, A_0_b4 - coefficients for the beta parameter of order 4 (NOTE: put them to zero if
        you work with the one photon case)

    Returns:
        background_cont - value of the background contribution
    """

    assert_abs_or_emi(abs_or_emi)

    # Introduce some auxiliary values for calculations
    A_int = A_0_int + A_bg_int
    A_b2 = (A_0_b2 + A_bg_b2) * legendre(2)(np.array(np.cos(math.radians(angle))))
    A_b4 = (A_0_b4 + A_bg_b4) * legendre(4)(np.array(np.cos(math.radians(angle))))

    background_cont = A_int + A_b2 + A_b4
    if abs_or_emi == "abs":
        background_cont = np.conjugate(background_cont)

    return background_cont


def get_zeros(q_complex):
    """
    Computes zeros of the real and imaginary part of the phase through the complex q
    parameter. Works for both cases: angularly integrated and when angular dependence is included.

    Args:
        q_complex - value of the complex q parameter

    Returns:
        im_zero - zero of the imaginary part
        re_zero_1 - the first zero of the real part
        re_zero_2 - the second zero of the real part
    """

    im_zero = np.real(q_complex) / (np.imag(q_complex) - 1)
    re_zero_1 = -np.real(q_complex) / 2 + np.sqrt(
        (np.real(q_complex) / 2) ** 2 - np.imag(q_complex)
    )
    re_zero_2 = -np.real(q_complex) / 2 - np.sqrt(
        (np.real(q_complex) / 2) ** 2 - np.imag(q_complex)
    )

    return im_zero, re_zero_1, re_zero_2
