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
from fortran_output_analysis.twophotons.twophotons import TwoPhotons
from fortran_output_analysis.twophotons.twophotons_utilities import (
    get_prepared_matrices,
)
from fortran_output_analysis.twophotons.twophotons_asymmetry_parameters import (
    two_photons_asymmetry_parameter,
)

"""
This namespace contains functions for analyzing delays and phases based on the data from 
the TwoPhotons object.
"""


def compute_omega_diff(
    two_photons_1: TwoPhotons, two_photons_2: Optional[TwoPhotons] = None
):
    """
    Computes energy difference between absorption and emission paths.
    Can compute for 1 or 2 simulations.

    Params:
    two_photons_1 - TwoPhotons object corresponding to the first simulation
    two_photons_2 - TwoPhotons object corresponding to the second simulation

    Returns:
    omega_diff - energy difference between absorption and emission paths
    """
    if two_photons_2:  # if two simulations are provided
        omega_diff = two_photons_1.g_omega_IR + two_photons_2.g_omega_IR
    else:  # if only one simulation is provided
        omega_diff = 2.0 * two_photons_1.g_omega_IR

    return omega_diff


def get_integrated_two_photons_intensity(
    hole_kappa,
    M_emi,
    M_abs,
    path=os.path.join(
        os.path.sep.join(
            os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1]
        ),
        "formula_coefficients",
        "two_photons",
        "integrated_intensity",
    ),
):
    """
    Computes the integrated signal intensity for a photoelectron that has absorbed two photons.
    M_emi and M_abs contain the matrix elements and other phases of the wave function for
    emission and absorption paths respectively.
    They are organized according to their final kappa like so:
    m = |hole_kappa|
    s = sign(hole_kappa)
    M = [s(m-2), -s(m-1), sm, -s(m+1), s(m+2)] (the values in the list are kappas of final states)
    Each part of the matrix corresponding to a particular final state contains 3 arrays of matrix
    elements for K = 0, 1, 2 where K is the rank of photon-interaction. So, M_emi and M_abs have
    the following shape: (5, 3, size_of_elements_array), where 5 is the number of final states,
    3 is three different values of K, and size_of_elements_array stands for length of each vector
    with matrix elements.
    NOTE: before providing you need to match the original emission and absorption matrices so
    that they correspond to the same final photoelectron energies.

    Params:
    hole_kappa - kappa value of the hole
    M_emi - matrix for emission path matched to the same final photoelectron
    energies
    M_abs - matrix for absorption path matched to the same final photoelectron
    energies
    path - path to the file with coefficients for Wigner intensity calculation

    Returns:
    two_photons_integrated_intensity - array with integrated intensity for two photons case
    """

    assert (
        M_emi.shape == M_abs.shape
    ), "The shapes of the input matrices must be the same!"

    energy_size = len(M_emi[0][1])

    if path[-1] is not os.path.sep:
        path = path + os.path.sep

    try:
        with open(path + f"integrated_intensity_{hole_kappa}.txt", "r") as coeffs_file:
            coeffs_file_contents = coeffs_file.readlines()
    except OSError as e:
        print(e)
        raise NotImplementedError(
            "the given initial kappa is not yet implemented, or the file containing the coefficients could not be found"
        )

    coeffs = exported_mathematica_tensor_to_python_list(coeffs_file_contents[2])

    two_photons_integrated_intensity = np.zeros(energy_size, dtype="complex128")
    for kappa_i in range(5):
        for K in range(3):
            two_photons_integrated_intensity += (
                coeffs[kappa_i][K] * M_emi[kappa_i][K] * np.conj(M_abs[kappa_i][K])
            )

    return two_photons_integrated_intensity


def get_integrated_atomic_delay(
    two_photons_1: TwoPhotons,
    n_qn,
    hole_kappa,
    Z,
    two_photons_2: Optional[TwoPhotons] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes integrated atomic delay. Can compute for 1 or 2 simulations.
    If 2 simulations are provided, then the first two_photons object corresponds
    to emission path, while the second two_photons object corresponds to absorption path.

    Params:
    two_photons_1 - object of the TwoPhotons class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    two_photons_2 - second object of the TwoPhotons class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    tau_int_atomic - values of the integrated atomic delay
    """

    ekin_eV, M_emi_matched, M_abs_matched = get_prepared_matrices(
        two_photons_1,
        n_qn,
        hole_kappa,
        Z,
        two_photons_2=two_photons_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(two_photons_1, two_photons_2=two_photons_2)

    tau_int_atomic = integrated_atomic_delay_from_intensity(
        hole_kappa, omega_diff, M_emi_matched, M_abs_matched
    )

    return ekin_eV, tau_int_atomic


def get_integrated_atomic_phase(
    two_photons_1: TwoPhotons,
    n_qn,
    hole_kappa,
    Z,
    two_photons_2: Optional[TwoPhotons] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
    unwrap=True,
):
    """
    Computes integrated atomic phase from integrated atomic delay. Can compute for 1 or 2
    simulations. If 2 simulations are provided, then the first two_photons object corresponds
    to emission path, while the second two_photons object corresponds to absorption path.

    Params:
    two_photons_1 - object of the TwoPhotons class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    two_photons_2 - second object of the TwoPhotons class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    phase_int_atomic - values of the integrated atomic phase
    """

    ekin_eV, tau_int_atomic = get_integrated_atomic_delay(
        two_photons_1,
        n_qn,
        hole_kappa,
        Z,
        two_photons_2=two_photons_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(two_photons_1, two_photons_2=two_photons_2)

    phase_int_atomic = delay_to_phase(tau_int_atomic, omega_diff)

    if unwrap:
        phase_int_atomic = unwrap_phase_with_nans(phase_int_atomic)

    return ekin_eV, phase_int_atomic


def get_angular_atomic_delay(
    two_photons_1: TwoPhotons,
    n_qn,
    hole_kappa,
    Z,
    angle,
    two_photons_2: Optional[TwoPhotons] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes angular part of atomic delay. Can compute for 1 or 2 simulations.
    If 2 simulations are provided, then the first two_photons object corresponds
    to emission path, while the second two_photons object corresponds to absorption path.

    Params:
    two_photons_1 - object of the TwoPhotons class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    angle - angle to compute the delay
    two_photons_2 - second object of the TwoPhotons class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    tau_ang_atomic - values of the angular part of atomic delay
    """

    ekin_eV, M_emi_matched, M_abs_matched = get_prepared_matrices(
        two_photons_1,
        n_qn,
        hole_kappa,
        Z,
        two_photons_2=two_photons_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(two_photons_1, two_photons_2=two_photons_2)

    tau_ang_atomic = angular_atomic_delay_from_asymmetry_parameters(
        hole_kappa, omega_diff, M_emi_matched, M_abs_matched, angle
    )

    return ekin_eV, tau_ang_atomic


def get_angular_atomic_phase(
    two_photons_1: TwoPhotons,
    n_qn,
    hole_kappa,
    Z,
    angle,
    two_photons_2: Optional[TwoPhotons] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
    unwrap=True,
):
    """
    Computes angluar part of atomic phase from the angular part of atomic delay.
    Can compute for 1 or 2 simulations. If 2 simulations are provided, then the first
    two_photons object corresponds to emission path, while the second two_photons object
    corresponds to absorption path.

    Params:
    two_photons_1 - object of the TwoPhotons class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    angle - angle to compute phase
    two_photons_2 - second object of the TwoPhotons class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    phase_ang_atomic - values of the angular part of atomic phase
    """

    ekin_eV, tau_ang_atomic = get_angular_atomic_delay(
        two_photons_1,
        n_qn,
        hole_kappa,
        Z,
        angle,
        two_photons_2=two_photons_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(two_photons_1, two_photons_2=two_photons_2)

    phase_ang_atomic = delay_to_phase(tau_ang_atomic, omega_diff)

    if unwrap:
        phase_ang_atomic = unwrap_phase_with_nans(phase_ang_atomic)

    return ekin_eV, phase_ang_atomic


def get_atomic_delay(
    two_photons_1: TwoPhotons,
    n_qn,
    hole_kappa,
    Z,
    angle,
    two_photons_2: Optional[TwoPhotons] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes total atomic delay: integrated + angular part. Can compute for 1 or 2 simulations.
    If 2 simulations are provided, then the first two_photons object corresponds to emission
    path, while the second two_photons object corresponds to absorption path.

    Params:
    two_photons_1 - object of the TwoPhotons class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    angle - angle to compute the delay
    two_photons_2 - second object of the TwoPhotons class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    tau_atomic - array with total atomic delays
    """

    ekin_eV, M_emi_matched, M_abs_matched = get_prepared_matrices(
        two_photons_1,
        n_qn,
        hole_kappa,
        Z,
        two_photons_2=two_photons_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(two_photons_1, two_photons_2=two_photons_2)

    tau_int_atomic = integrated_atomic_delay_from_intensity(
        hole_kappa, omega_diff, M_emi_matched, M_abs_matched
    )

    tau_ang_atomic = angular_atomic_delay_from_asymmetry_parameters(
        hole_kappa, omega_diff, M_emi_matched, M_abs_matched, angle
    )

    tau_atomic = tau_int_atomic + tau_ang_atomic  # total atomic delay

    return ekin_eV, tau_atomic


def get_atomic_phase(
    two_photons_1: TwoPhotons,
    n_qn,
    hole_kappa,
    Z,
    angle,
    two_photons_2: Optional[TwoPhotons] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
    unwrap=True,
):
    """
    Computes total atomic phase: integrated + angular part from total atomic delay.

    Params:
    two_photons_1 - object of the TwoPhotons class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    angle - angle to compute the phase
    two_photons_2 - second object of the TwoPhotons class if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.
    unwrap - if to unwrap phase using np.unwrap

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    phase_atomic - array with total atomic phases
    """

    ekin_eV, tau_atomic = get_atomic_delay(
        two_photons_1,
        n_qn,
        hole_kappa,
        Z,
        angle,
        two_photons_2=two_photons_2,
        steps_per_IR_photon=steps_per_IR_photon,
        energies_mode=energies_mode,
    )

    omega_diff = compute_omega_diff(two_photons_1, two_photons_2=two_photons_2)

    phase_atomic = delay_to_phase(tau_atomic, omega_diff)

    if unwrap:
        phase_atomic = unwrap_phase_with_nans(phase_atomic)

    return ekin_eV, phase_atomic


def integrated_atomic_delay_from_intensity(
    hole_kappa, omega_diff, M_emi_matched, M_abs_matched
):
    """
    Computes integrated atomic delay from two photons integrated intenisty.

    Params:
    hole_kappa - kappa value of the hole
    omega_diff - energy difference between absorption and emission paths
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies

    Returns:
    tau_int_atomic - array with integrated atomic delay
    """

    two_photons_integrated_intensity = get_integrated_two_photons_intensity(
        hole_kappa, M_emi_matched, M_abs_matched
    )

    tau_int_atomic = (
        g_inverse_atomic_frequency_to_attoseconds
        * np.angle(two_photons_integrated_intensity)
        / omega_diff
    )

    return tau_int_atomic


# TODO: remove inefficiency. Instead of calculating delay for one angle, we can immediately
# calculate for many angles provided in the list/array.
def angular_atomic_delay_from_asymmetry_parameters(
    hole_kappa,
    omega_diff,
    M_emi_matched,
    M_abs_matched,
    angle,
):
    """
    Computes angular part of atomic delay from the complex assymetry parameters.

    Params:
    hole_kappa - kappa value of the hole
    omega_diff - energy difference between absorption and emission paths
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    angle - angle to compute delay

    Returns:
    tau_ang_atomic - array with angular part of atomic delay
    """

    b2_complex, _ = two_photons_asymmetry_parameter(
        2, hole_kappa, M_emi_matched, M_abs_matched, "cross"
    )  # 2nd order complex assymetry parameter

    b4_complex, _ = two_photons_asymmetry_parameter(
        4, hole_kappa, M_emi_matched, M_abs_matched, "cross"
    )  # 2nd order complex assymetry parameter

    tau_ang_atomic = (
        g_inverse_atomic_frequency_to_attoseconds
        * np.angle(
            1.0
            + b2_complex * legendre(2)(np.array(np.cos(math.radians(angle))))
            + b4_complex * legendre(4)(np.array(np.cos(math.radians(angle))))
        )
        / omega_diff
    )

    return tau_ang_atomic
