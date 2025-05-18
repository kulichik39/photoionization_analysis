import numpy as np
import os
import math
from typing import Optional
from scipy.special import legendre
from fortran_output_analysis.constants_and_parameters import (
    g_inverse_atomic_frequency_to_attoseconds,
)
from fortran_output_analysis.common_utility import (
    delay_to_phase,
    unwrap_phase_with_nans,
    exported_mathematica_tensor_to_python_list,
)
from fortran_output_analysis.onephoton.onephoton import OnePhoton
from fortran_output_analysis.onephoton.onephoton_utilities import get_prepared_matrices
from fortran_output_analysis.onephoton.onephoton_asymmetry_parameters import (
    one_photon_asymmetry_parameter,
)

"""
This namespace contains functions for analyzing delays and phases based on the data from 
the OnePhoton object.
"""


def compute_omega_diff(g_omega_IR_1, g_omega_IR_2=None):
    """
    Computes energy difference between absorption and emission paths.
    Can compute for 1 or 2 simulations.

    Params:
    g_omega_IR_1 - energy of the IR photon in Hartree in the first simulation
    g_omega_IR_2 - energy of the IR photon in Hartree in the second simulation

    Returns:
    omega_diff - energy difference between absorption and emission paths
    """
    if g_omega_IR_2:  # if two simulations are provided
        omega_diff = g_omega_IR_1 + g_omega_IR_2
    else:  # if only one simulation is provided
        omega_diff = 2.0 * g_omega_IR_1

    return omega_diff


def get_wigner_intensity(
    hole_kappa,
    M_emi,
    M_abs,
    path=os.path.join(
        os.path.sep.join(
            os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1]
        ),
        "formula_coefficients",
        "one_photon",
        "integrated_intensity",
    ),
):
    """
    Computes Wigner intensity for a photoelectron that has absorbed one photon.

    Params:
    hole_kappa - kappa value of the hole
    M_emi - matrix for emission path matched to the same final photoelectron
    energies
    M_abs - matrix for absorption path matched to the same final photoelectron
    energies
    path - path to the file with coefficients for Wigner intensity calculation

    Returns:
    wigner_intensity - array with Wigner intensity values
    """

    assert (
        M_emi.shape == M_abs.shape
    ), "The shapes of the input matrices must be the same!"

    length = M_emi.shape[1]

    if path[-1] is not os.path.sep:
        path = path + os.path.sep

    try:
        with open(path + f"integrated_intensity_{hole_kappa}.txt", "r") as coeffs_file:
            coeffs_file_contents = coeffs_file.readlines()
    except OSError as e:
        print(e)
        raise NotImplementedError(
            f"The hole kappa {hole_kappa} is not yet implemented, or the file containing the coefficients could not be found!"
        )

    coeffs = exported_mathematica_tensor_to_python_list(coeffs_file_contents[2])

    wigner_intensity = np.zeros(length, dtype="complex128")
    for i in range(3):
        wigner_intensity += coeffs[i] * M_emi[i] * np.conj(M_abs[i])

    return wigner_intensity


def get_integrated_wigner_delay(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    g_omega_IR_1,
    one_photon_2: Optional[OnePhoton] = None,
    g_omega_IR_2=None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes integrated Wigner delay. Can compute for 1 or 2 simulations.
    If 2 simulations are provided, then the first one_photon object corresponds
    to emission path, while the second one_photon object corresponds to absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    g_omega_IR_1 - energy of the IR photon in Hartree in the first simulation
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    g_omega_IR_2 - energy of the IR photon in Hartree in the second simulation
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    tau_int_wigner - values of the integrated Wigner delay
    """

    ekin_eV, M_emi_matched, M_abs_matched = get_prepared_matrices(
        one_photon_1,
        n_qn,
        hole_kappa,
        Z,
        g_omega_IR_1,
        one_photon_2=one_photon_2,
        g_omega_IR_2=g_omega_IR_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(g_omega_IR_1, g_omega_IR_2=g_omega_IR_2)

    tau_int_wigner = integrated_wigner_delay_from_intensity(
        hole_kappa, omega_diff, M_emi_matched, M_abs_matched
    )

    return ekin_eV, tau_int_wigner


def get_integrated_wigner_phase(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    g_omega_IR_1,
    one_photon_2: Optional[OnePhoton] = None,
    g_omega_IR_2=None,
    steps_per_IR_photon=None,
    energies_mode="both",
    unwrap=True,
):
    """
    Computes integrated wigner phase from integrated wigner delay. Can compute for 1 or 2
    simulations. If 2 simulations are provided, then the first one_photon object corresponds
    to emission path, while the second one_photon object corresponds to absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    g_omega_IR_1 - energy of the IR photon in Hartree in the first simulation
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    g_omega_IR_2 - energy of the IR photon in Hartree in the second simulation
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    phase_int_wigner - values of the integrated Wigner phase
    """

    ekin_eV, tau_int_wigner = get_integrated_wigner_delay(
        one_photon_1,
        n_qn,
        hole_kappa,
        Z,
        g_omega_IR_1,
        one_photon_2=one_photon_2,
        g_omega_IR_2=g_omega_IR_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(g_omega_IR_1, g_omega_IR_2=g_omega_IR_2)

    phase_int_wigner = delay_to_phase(tau_int_wigner, omega_diff)

    if unwrap:
        phase_int_wigner = unwrap_phase_with_nans(phase_int_wigner)

    return ekin_eV, phase_int_wigner


def get_angular_wigner_delay(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    g_omega_IR_1,
    angles,
    one_photon_2: Optional[OnePhoton] = None,
    g_omega_IR_2=None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes angular part of Wigner delay. Can compute for 1 or 2 simulations.
    If 2 simulations are provided, then the first one_photon object corresponds
    to emission path, while the second one_photon object corresponds to absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    g_omega_IR_1 - energy of the IR photon in Hartree in the first simulation
    angles - array of angles to compute the delay for
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    g_omega_IR_2 - energy of the IR photon in Hartree in the second simulation
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    tau_ang_wigner - values of the angular part of Wigner delay for specified angles
    """

    ekin_eV, M_emi_matched, M_abs_matched = get_prepared_matrices(
        one_photon_1,
        n_qn,
        hole_kappa,
        Z,
        g_omega_IR_1,
        one_photon_2=one_photon_2,
        g_omega_IR_2=g_omega_IR_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(g_omega_IR_1, g_omega_IR_2=g_omega_IR_2)

    tau_ang_wigner = angular_wigner_delay_from_asymmetry_parameter(
        hole_kappa, ekin_eV, omega_diff, M_emi_matched, M_abs_matched, angles
    )

    return ekin_eV, tau_ang_wigner


def get_angular_wigner_phase(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    g_omega_IR_1,
    angles,
    one_photon_2: Optional[OnePhoton] = None,
    g_omega_IR_2=None,
    steps_per_IR_photon=None,
    energies_mode="both",
    unwrap=True,
):
    """
    Computes angluar part of Wigner phase from the angular part of Wigner delay.
    Can compute for 1 or 2 simulations. If 2 simulations are provided, then the first one_photon
    object corresponds to emission path, while the second one_photon object corresponds to
    absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    g_omega_IR_1 - energy of the IR photon in Hartree in the first simulation
    angles - array of angles to compute the phase for
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    g_omega_IR_2 - energy of the IR photon in Hartree in the second simulation
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    phase_ang_wigner - values of the angular part of Wigner phase for specified angles
    """

    ekin_eV, tau_ang_wigner = get_angular_wigner_delay(
        one_photon_1,
        n_qn,
        hole_kappa,
        Z,
        g_omega_IR_1,
        angles,
        one_photon_2=one_photon_2,
        g_omega_IR_2=g_omega_IR_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(g_omega_IR_1, g_omega_IR_2=g_omega_IR_2)

    phase_ang_wigner = delay_to_phase(tau_ang_wigner, omega_diff)

    if unwrap:
        for i in range(len(angles)):
            phase_ang_wigner[i, :] = unwrap_phase_with_nans(phase_ang_wigner[i, :])

    return ekin_eV, phase_ang_wigner


def get_wigner_delay(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    g_omega_IR_1,
    angles,
    one_photon_2: Optional[OnePhoton] = None,
    g_omega_IR_2=None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes total Wigner delay: integrated + angular part. Can compute for 1 or 2 simulations.
    If 2 simulations are provided, then the first one_photon object corresponds to emission path,
    while the second one_photon object corresponds to absorption path.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    g_omega_IR_1 - energy of the IR photon in Hartree in the first simulation
    angles - array of angles to compute the delay for
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    g_omega_IR_2 - energy of the IR photon in Hartree in the second simulation
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    tau_wigner - array with total Wigner delays for specified angles
    """

    ekin_eV, M_emi_matched, M_abs_matched = get_prepared_matrices(
        one_photon_1,
        n_qn,
        hole_kappa,
        Z,
        g_omega_IR_1,
        one_photon_2=one_photon_2,
        g_omega_IR_2=g_omega_IR_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(g_omega_IR_1, g_omega_IR_2=g_omega_IR_2)

    tau_int_wigner = integrated_wigner_delay_from_intensity(
        hole_kappa, omega_diff, M_emi_matched, M_abs_matched
    )

    tau_ang_wigner = angular_wigner_delay_from_asymmetry_parameter(
        hole_kappa, ekin_eV, omega_diff, M_emi_matched, M_abs_matched, angles
    )

    N_angles = len(angles)

    tau_wigner = np.zeros((N_angles, len(ekin_eV)))

    for i in range(N_angles):
        tau_wigner[i, :] = tau_int_wigner + tau_ang_wigner[i, :]

    return ekin_eV, tau_wigner


def get_wigner_phase(
    one_photon_1: OnePhoton,
    n_qn,
    hole_kappa,
    Z,
    g_omega_IR_1,
    angles,
    one_photon_2: Optional[OnePhoton] = None,
    g_omega_IR_2=None,
    steps_per_IR_photon=None,
    energies_mode="both",
    unwrap=True,
):
    """
    Computes total Wigner phase: integrated + angular part from total Wigner delay.

    Params:
    one_photon_1 - object of the OnePhoton class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    g_omega_IR_1 - energy of the IR photon in Hartree in the first simulation
    angles - array of angles to compute the phase for
    one_photon_2 - second object of the OnePhoton class if we want to consider 2 simulations
    (first for emission, second for absorption)
    g_omega_IR_2 - energy of the IR photon in Hartree in the second simulation
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    phase_wigner - array with total Wigner phases for specified angles
    """

    ekin_eV, tau_wigner = get_wigner_delay(
        one_photon_1,
        n_qn,
        hole_kappa,
        Z,
        g_omega_IR_1,
        angles,
        one_photon_2=one_photon_2,
        g_omega_IR_2=g_omega_IR_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(g_omega_IR_1, g_omega_IR_2=g_omega_IR_2)

    N_angles = len(angles)

    phase_wigner = delay_to_phase(tau_wigner, omega_diff)

    if unwrap:
        for i in range(len(angles)):
            phase_wigner[i, :] = unwrap_phase_with_nans(phase_wigner[i, :])

    return ekin_eV, phase_wigner


def integrated_wigner_delay_from_intensity(
    hole_kappa, omega_diff, M_emi_matched, M_abs_matched
):
    """
    Computes integrated Wigner delay from Wigner intenisty.

    Params:
    hole_kappa - kappa value of the hole
    omega_diff - energy difference between absorption and emission paths
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies

    Returns:
    tau_int_wigner - array with integrated Wigner delay
    """

    wigner_intensity = get_wigner_intensity(hole_kappa, M_emi_matched, M_abs_matched)

    tau_int_wigner = (
        g_inverse_atomic_frequency_to_attoseconds
        * np.angle(wigner_intensity)
        / omega_diff
    )

    return tau_int_wigner


def angular_wigner_delay_from_asymmetry_parameter(
    hole_kappa,
    ekin_eV,
    omega_diff,
    M_emi_matched,
    M_abs_matched,
    angles,
):
    """
    Computes angular part of Wigner delay from the complex assymetry parameter.

    Params:
    hole_kappa - kappa value of the hole
    ekin_eV - array of photoelectron kinetic energies in eV
    omega_diff - energy difference between absorption and emission paths
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    angles - array of angles to compute the delay for

    Returns:
    tau_ang_wigner - array with angular part of Wigner delay for specified angles
    """

    b2_complex, _ = one_photon_asymmetry_parameter(
        hole_kappa, M_emi_matched, M_abs_matched, "cross"
    )  # complex assymetry parameter for one photon case

    N_angles = len(angles)

    tau_ang_wigner = np.zeros((N_angles, len(ekin_eV)))

    for i in range(N_angles):
        angle = angles[i]
        tau_ang_wigner[i, :] = (
            g_inverse_atomic_frequency_to_attoseconds
            * np.angle(
                1.0 + b2_complex * legendre(2)(np.array(np.cos(math.radians(angle))))
            )
            / omega_diff
        )

    return tau_ang_wigner
