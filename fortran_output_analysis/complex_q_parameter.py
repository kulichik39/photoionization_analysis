import numpy as np
import math
from scipy.special import legendre
from scipy.optimize import curve_fit
from fortran_output_analysis.common_utility import assert_abs_or_emi

"""
This namespace contains functions for analyzing complex q parameter in the one and 
two photon cases.
"""


def get_epsilon(en, abs_or_emi, g_omega_IR, E_res, width_res):
    """
    Calculates an array of resclased energies eps used in the formula for complex parameter.

    Args:
        en - array of energies
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string
        g_omega_IR - energy of the IR photon
        E_res - energy of the resonance
        width_res - width of the resonance
        NOTE: en, g_omega_IR, E_res, width_res must be in the same units

    Returns:
        eps - array with the rescaled energy values
    """

    assert_abs_or_emi(abs_or_emi)

    # create an array of the scaled energies
    if abs_or_emi == "abs":
        eps = (en - (E_res + g_omega_IR)) / (width_res / 2)
    else:
        eps = (en - (E_res - g_omega_IR)) / (width_res / 2)

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
        angle - emission angle
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
        angle - emission angle
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


def get_coefficients_for_integrated_by_fit(
    abs_or_emi, q, int_intens, eps, eps_min, eps_max
):
    """
    Fits the model to the provided range of the integrated intensity (signal)
    and retrieves the coefficients.

    Args:
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string.
        q - Fano shape parameter.
        int_intens - array with the integrated intensity (signal).
        eps - array with the reduced energy values.
        eps_min, eps_max - the lower and upper boundaries of the range to fit.

    Returns:
        A_bg, A_0 - the model coefficients.
        pcov - covariance matrix containing the errors of the fitting.
    """

    assert_abs_or_emi(abs_or_emi)

    # extcact the fitting range
    mask_fit = np.logical_and(eps >= eps_min, eps <= eps_max)
    eps_fit = eps[mask_fit]
    int_intens_fit = int_intens[mask_fit]

    def func_fit(x, A_0_Re, A_0_Im, A_bg_Re, A_bg_Im):
        """The fitting function."""

        if abs_or_emi == "abs":
            return np.conjugate(
                (A_0_Re + 1j * A_0_Im) * (x + q) / (x + 1j * 1) + A_bg_Re + 1j * A_bg_Im
            )

        else:
            return (
                (A_0_Re + 1j * A_0_Im) * (x + q) / (x + 1j * 1) + A_bg_Re + 1j * A_bg_Im
            )

    def func_fit_flatten(x, A_0_Re, A_0_Im, A_bg_Re, A_bg_Im):
        """
        The helper function that flattens the complex fitting function to an array of a doubled size.
        Needed since the scipy's curve fit doesn't work with complex values.
        """
        # x is the double size vector composed of two identical arrays with the original points,
        # i.e x = [x, x]
        N = len(x)
        x_real = x[: N // 2]
        x_imag = x[N // 2 :]
        y_real = np.real(func_fit(x_real, A_0_Re, A_0_Im, A_bg_Re, A_bg_Im))
        y_imag = np.imag(func_fit(x_imag, A_0_Re, A_0_Im, A_bg_Re, A_bg_Im))

        return np.hstack([y_real, y_imag])

    y_true_real = np.real(int_intens_fit)
    y_true_imag = np.imag(int_intens_fit)

    y_true_flatten = np.hstack([y_true_real, y_true_imag])

    popt, pcov = curve_fit(
        func_fit_flatten, np.hstack([eps_fit, eps_fit]), y_true_flatten
    )

    A_0_Re, A_0_Im, A_bg_Re, A_bg_Im = popt

    A_bg = A_bg_Re + 1j * A_bg_Im
    A_0 = A_0_Re + 1j * A_0_Im

    return A_bg, A_0, pcov


def get_coefficients_for_beta_param_by_fit(
    abs_or_emi, q, beta_param, eps, eps_min, eps_max, A_bg_int, A_0_int
):
    """
    Fits the model to the provided range of the beta parameter
    and retrieves the coefficients.

    Args:
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string.
        q - Fano shape parameter.
        beta_param - array with the beta parameter values.
        eps - array with the reduced energy values.
        eps_min, eps_max - the lower and upper boundaries of the range to fit.
        A_bg_int, A_0_int - coefficients for the integrated intensity (signal).

    Returns:
        A_bg, A_0 - the model coefficients.
        pcov - covariance matrix containing the errors of the fitting.
    """

    assert_abs_or_emi(abs_or_emi)

    # extcact the fitting range
    mask_fit = np.logical_and(eps >= eps_min, eps <= eps_max)
    eps_fit = eps[mask_fit]
    beta_param_fit = beta_param[mask_fit]

    def func_fit(x, A_0_Re, A_0_Im, A_bg_Re, A_bg_Im):
        """The fitting function."""

        if abs_or_emi == "abs":
            return np.conjugate(
                (
                    (A_0_Re + 1j * A_0_Im) * (x + q) / (x + 1j * 1)
                    + A_bg_Re
                    + 1j * A_bg_Im
                )
                / (A_0_int * (x + q) / (x + 1j * 1) + A_bg_int)
            )

        else:
            return (
                (A_0_Re + 1j * A_0_Im) * (x + q) / (x + 1j * 1) + A_bg_Re + 1j * A_bg_Im
            ) / (A_0_int * (x + q) / (x + 1j * 1) + A_bg_int)

    def func_fit_flatten(x, A_0_Re, A_0_Im, A_bg_Re, A_bg_Im):
        """
        The helper function that flattens the complex fitting function to an array of a doubled size.
        Needed since the scipy's curve fit doesn't work with complex values.
        """
        # x is the double size vector composed of two identical arrays with the original points,
        # i.e x = [x, x]
        N = len(x)
        x_real = x[: N // 2]
        x_imag = x[N // 2 :]
        y_real = np.real(func_fit(x_real, A_0_Re, A_0_Im, A_bg_Re, A_bg_Im))
        y_imag = np.imag(func_fit(x_imag, A_0_Re, A_0_Im, A_bg_Re, A_bg_Im))

        return np.hstack([y_real, y_imag])

    y_true_real = np.real(beta_param_fit)
    y_true_imag = np.imag(beta_param_fit)

    y_true_flatten = np.hstack([y_true_real, y_true_imag])

    popt, pcov = curve_fit(
        func_fit_flatten, np.hstack([eps_fit, eps_fit]), y_true_flatten
    )

    A_0_Re, A_0_Im, A_bg_Re, A_bg_Im = popt

    A_bg = A_bg_Re + 1j * A_bg_Im
    A_0 = A_0_Re + 1j * A_0_Im

    return A_bg, A_0, pcov


def get_model_pred_for_integrated(abs_or_emi, eps, A_bg, A_0, q_complex):
    """
    Calculates model prediction for the integrated intensity (signal).

    Args:
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string.
        eps - array with the reduced energy values.
        A_bg, A_0 - coefficients for the integrated intensity.
        q_complex - complex q parameter for the integrated intensity.

    Returns:
        model prediction for the integrated intensity.
    """

    assert_abs_or_emi(abs_or_emi)

    # background contribution
    background_contr = get_background_contribution_integrated(abs_or_emi, A_bg, A_0)

    # energy-dependent term
    energy_contr = (eps + q_complex) / (eps + 1 * 1j)
    if abs_or_emi == "abs":  # conjugate for the absroption path
        energy_contr = np.conjugate(energy_contr)

    return energy_contr * background_contr


def get_model_pred_for_beta_param(
    abs_or_emi,
    eps,
    A_bg_beta,
    A_0_beta,
    q_complex_beta,
    A_bg_int,
    A_0_int,
    q_complex_int,
):
    """
    Calculates model prediction for the beta parameter.

    Args:
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string.
        eps - array with the reduced energy values.
        A_bg_beta, A_0_beta - coefficients for the beta parameter.
        q_complex_beta - complex q for the beta parameter.
        A_bg_int, A_0_int - coefficients for the integrated intensity (signal).
        q_complex_int - complex q for the integrated intensity.

    Returns:
        model prediction for the beta parameter.
    """

    assert_abs_or_emi(abs_or_emi)

    model_pred = (
        (A_0_beta + A_bg_beta)
        * (eps + q_complex_beta)
        / ((A_0_int + A_bg_int) * (eps + q_complex_int))
    )

    if abs_or_emi == "abs":
        model_pred = np.conjugate(model_pred)

    return model_pred


def get_model_pred_for_angular(
    abs_or_emi,
    eps,
    angle,
    A_bg_int,
    A_0_int,
    A_bg_b2,
    A_0_b2,
    A_bg_b4,
    A_0_b4,
    q_complex_ang,
):
    """
    Calculates model prediction for the angularly resolved intensity (signal).

    Args:
        abs_or_emi - tells if the resonance was in absorption or emission path, must be
        "abs" or "emi" string.
        eps - array with the reduced energy values.
        angle - emission angle.
        A_bg_int, A_0_int - coefficients for the integrated intensity (signal).
        A_bg_b2, A_0_b2 - coefficients for the second order beta parameter.
        A_bg_b4, A_0_b4 - coefficients for the fourth order beta parameter.
        q_complex_ang - complex q parameter for the emission angle.

    Returns:
        model prediction for the angularly resolved intensity.
    """

    assert_abs_or_emi(abs_or_emi)

    background_contr = get_background_contribution_angular(
        angle,
        abs_or_emi,
        A_bg_int,
        A_0_int,
        A_bg_b2,
        A_0_b2,
        A_bg_b4,
        A_0_b4,
    )

    energy_contr = (eps + q_complex_ang) / (eps + 1j)

    if abs_or_emi == "abs":
        energy_contr = np.conjugate(energy_contr)

    return energy_contr * background_contr


def get_crit_angles(
    angles,
    q,
    A_bg_int,
    A_0_int,
    A_bg_b2,
    A_0_b2,
    A_bg_b4,
    A_0_b4,
):
    """
    Retrieves the critical angles by looking where the imaginary part of the complex q parameter
    crosses zero.

    Args:
        angles - array of angles to search in.
        q - Fano shape parameter.
        A_bg_int, A_0_int - coefficients for the integrated intensity (signal).
        A_bg_b2, A_0_b2 - coefficients for the second order beta parameter.
        A_bg_b4, A_0_b4 - coefficients for the fourth order beta parameter.
                          NOTE: A_bg_b4 and A_0_b4 are zero for the Wigner case.

    Returns:
        crit_angles - array of critical angles if any.
        complex_q_arr - array with the values of the complex q parameter.
    """

    complex_q_arr = []  # array to store the values of the complex q parameter

    # fill the array with complex q
    for angle in angles:
        complex_q = get_complex_q_angular(
            angle, A_bg_int, A_0_int, A_bg_b2, A_0_b2, A_bg_b4, A_0_b4, q
        )
        complex_q_arr.append(complex_q)

    complex_q_arr = np.array(complex_q_arr)

    # search for the critical angles
    complex_q_imag_arr = np.imag(complex_q_arr)  # extract the imaginary part
    crit_angles = []
    for i in range(1, len(complex_q_imag_arr)):
        if np.sign(complex_q_imag_arr[i]) != np.sign(complex_q_imag_arr[i - 1]):
            crit_angle = (angles[i] + angles[i - 1]) / 2
            crit_angles.append(crit_angle)

    crit_angles = np.array(crit_angles)

    return crit_angles, complex_q_arr
