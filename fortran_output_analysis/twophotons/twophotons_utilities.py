import numpy as np
from typing import Optional
from sympy.physics.wigner import wigner_3j, wigner_6j
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree
from fortran_output_analysis.common_utility import (
    j_from_kappa,
    coulomb_phase,
    final_energies_for_matching_1sim,
    match_matrix_elements_1sim,
    final_energies_for_matching_2sim,
    match_matrix_elements_2sim,
    assert_abs_or_emi,
)
from fortran_output_analysis.twophotons.twophotons import (
    TwoPhotons,
    final_kappas,
    Channels,
)

"""
This name space contains functions that may be required across different sections of two photons 
analysis (e.g. analysis of atomuc delays/phases).
"""


def get_omega_eV(two_photons: TwoPhotons, abs_or_emi, n_qn, hole_kappa):
    """
    Returns array of XUV photon energies in eV for the given hole.

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    abs_or_emi - tells if we want to get for absorption or emission path,
    can take only 'abs' or 'emi' values.
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole

    Returns:
    array of XUV photon energies in eV for the given hole
    """

    assert_abs_or_emi(abs_or_emi)
    two_photons.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

    return (
        get_omega_Hartree(two_photons, abs_or_emi, n_qn, hole_kappa) * g_eV_per_Hartree
    )


def get_omega_Hartree(two_photons: TwoPhotons, abs_or_emi, n_qn, hole_kappa):
    """
    Returns array of XUV photon energies in Hartree for the given hole.

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    abs_or_emi - tells if we want to get for absorption or emission path,
    can take only 'abs' or 'emi' values.
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole

    Returns:
    omega_Hartree - array of XUV photon energies in Hartree for the given hole
    """

    assert_abs_or_emi(abs_or_emi)
    two_photons.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

    channels: Channels = two_photons.get_channels_for_hole(abs_or_emi, n_qn, hole_kappa)

    omega_Hartree = (
        channels.get_raw_omega_data()
    )  # omega energies in Hartree from the output file.

    return omega_Hartree


def get_electron_kinetic_energy_Hartree(
    two_photons: TwoPhotons, abs_or_emi, n_qn, hole_kappa
):
    """
    Returns array of electron kinetic energies in Hartree for the given hole.

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    abs_or_emi - tells if we want to get for absorption or emission path,
    can take only 'abs' or 'emi' values.
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole

    Returns:
    array of electron kinetic energies in Hartree for the given hole
    """

    assert_abs_or_emi(abs_or_emi)
    two_photons.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

    channels: Channels = two_photons.get_channels_for_hole(abs_or_emi, n_qn, hole_kappa)
    hole = channels.get_hole_object()

    if not hole.binding_energy:
        raise RuntimeError(f"The binding energy for {hole.name} is not initialized!")

    return (
        get_omega_Hartree(two_photons, abs_or_emi, n_qn, hole_kappa)
        - hole.binding_energy
    )


def get_electron_kinetic_energy_eV(
    two_photons: TwoPhotons, abs_or_emi, n_qn, hole_kappa
):
    """
    Returns array of electron kinetic energies in eV for the given hole.

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    abs_or_emi - tells if we want to get for absorption or emission path,
    can take only 'abs' or 'emi' values.
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole

    Returns:
    array of electron kinetic energies in eV for the given hole
    """

    assert_abs_or_emi(abs_or_emi)
    two_photons.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

    return (
        get_electron_kinetic_energy_Hartree(two_photons, abs_or_emi, n_qn, hole_kappa)
        * g_eV_per_Hartree
    )


def get_matrix_elements_for_ionisation_path(
    two_photons: TwoPhotons,
    abs_or_emi,
    n_qn,
    hole_kappa,
    intermdediate_kappa,
    final_kappa,
):
    """
    Computes matrix elements with phase after two photons for the given hole and ionisation path
    (determined by intermediate_kappa and final_kappa).

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    abs_or_emi - tells if we want to get for absorption or emission path,
    can take only 'abs' or 'emi' values.
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    intermediate_kappa - kappa value of the intermediate state
    final_kappa - kappa value of the final state

    Returns:
    matrix elements with phase after two photons for the given hole and ionization path
    """

    assert_abs_or_emi(abs_or_emi)

    two_photons.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

    channels: Channels = two_photons.get_channels_for_hole(abs_or_emi, n_qn, hole_kappa)

    elements = channels.get_raw_matrix_elements_for_ionization_path(
        intermdediate_kappa, final_kappa
    )
    phase = channels.get_raw_phase_for_ionization_path(intermdediate_kappa, final_kappa)

    return elements * np.exp(1j * phase)


def get_coupled_matrix_elements(
    two_photons: TwoPhotons,
    abs_or_emi,
    n_qn,
    hole_kappa,
    final_kappa,
):
    """
    Computes matrix elements for the given hole and final state (final_kappa) summed over
    all intermediate states.

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    abs_or_emi - tells if we want to get for absorption or emission path,
    can take only 'abs' or 'emi' values.
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    final_kappa - kappa value of the final state

    Returns:
    matrix_elements_coupled - matrix elements for the final state summed over all intermediate
    states
    """

    assert_abs_or_emi(abs_or_emi)

    two_photons.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

    channels: Channels = two_photons.get_channels_for_hole(abs_or_emi, n_qn, hole_kappa)
    energy_size = len(
        channels.get_raw_omega_data()
    )  # size of the array with XUV photon energies

    # Initialize array with final matrix elements for 3 different values of K: 0, 1 and 2
    # K is the rank of the photon-interaction.
    matrix_elements_coupled = np.zeros((3, energy_size), dtype="complex128")

    ionisation_paths = channels.get_all_ionisation_paths()

    for key_tuple in ionisation_paths:
        intermediate_kappa_path, final_kappa_path = key_tuple

        if final_kappa_path == final_kappa:  # if we match the desired final state
            # Get j values
            hole_j = j_from_kappa(hole_kappa)
            intermediate_j = j_from_kappa(intermediate_kappa_path)
            final_j = j_from_kappa(final_kappa)

            matrix_elements = get_matrix_elements_for_ionisation_path(
                two_photons,
                abs_or_emi,
                n_qn,
                hole_kappa,
                intermediate_kappa_path,
                final_kappa,
            )

            for K in [0, 2]:
                # NOTE: the thing commented below is probably incorrect, but I keep it just in case
                # matrix_elements_coupled[K, :] += (
                #     np.power(-1, hole_j + final_j + K)
                #     * (2 * K + 1)
                #     * float(wigner_3j(1, 1, K, 0, 0, 0))
                #     * matrix_elements
                #     * float(wigner_6j(1, 1, K, hole_j, float(final_j), intermediate_j))
                # )

                # Multiply by the prefactor and store it in the final matrix
                matrix_elements_coupled[K, :] += (
                    (2 * K + 1)
                    * float(wigner_3j(1, 1, K, 0, 0, 0))
                    * matrix_elements
                    * float(wigner_6j(1, 1, K, hole_j, float(final_j), intermediate_j))
                )

    return matrix_elements_coupled


def get_coupled_matrix_elements_for_all_final_states(
    two_photons: TwoPhotons,
    abs_or_emi,
    n_qn,
    hole_kappa,
):
    """
    Computes coupled matrix elements for all final states in the two photons case.
    If a particular final state is forbidden (final_kappa=0), the matrix elements for this
    state will be 0. This behavior is required to get vectors of a certain shape, which is
    necessary for functions like two_photons_asymmetry_parameter() from
    twophotons_asymmetry_parameters.py.

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    abs_or_emi - tells if we want to get for absorption or emission path,
    can take only 'abs' or 'emi' values.
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole

    Returns:
    M - coupled matrix elements for all final states
    """

    assert_abs_or_emi(abs_or_emi)

    two_photons.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

    final_kappas_list = final_kappas(
        hole_kappa, only_reachable=False
    )  # list of ALL final states
    N_kappas_all = len(final_kappas_list)

    final_kappas_reachable = final_kappas(
        hole_kappa, only_reachable=True
    )  # list of reachable final states

    channels: Channels = two_photons.get_channels_for_hole(abs_or_emi, n_qn, hole_kappa)
    energy_size = len(
        channels.get_raw_omega_data()
    )  # size of the array with XUV photon energies

    # initialize array to store matrix elements with the following dimensions:
    # N_kappas_all - number of all final states
    # 3 - for three values of K (rank of the photon-interaction)
    # energy_size - size of the array with XUV photon energies
    M = np.zeros((N_kappas_all, 3, energy_size), dtype="complex128")

    for i in range(N_kappas_all):
        final_kappa = final_kappas_list[i]
        if (
            final_kappa in final_kappas_reachable
        ):  # filter reachable states and keep unreachable ones zero
            M[i, :, :] = get_coupled_matrix_elements(
                two_photons, abs_or_emi, n_qn, hole_kappa, final_kappa
            )

    return M


def get_coulomb_phase(two_photons: TwoPhotons, abs_or_emi, n_qn, hole_kappa, Z):
    """
    Computes Coulomb phase for all final states of the given hole in the two photons case.
    If a particular final state is forbidden (final_kappa=0), the coulomb phase for this
    path will be 0. This behavior is required to get vectors of a certain shape, which is
    necessary for functions like two_photons_asymmetry_parameter() from
    twophotons_asymmetry_parameters.py.

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    abs_or_emi - tells if we want to get for absorption or emission path,
    can take only 'abs' or 'emi' values.
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion

    Returns:
    coulomb_phase_arr - array with Coulomb phases
    """

    assert_abs_or_emi(abs_or_emi)

    two_photons.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

    final_kappas_list = final_kappas(
        hole_kappa, only_reachable=False
    )  # list of ALL final states
    N_kappas_all = len(final_kappas_list)

    final_kappas_reachable = final_kappas(
        hole_kappa, only_reachable=True
    )  # list of reachable final states

    ekin = get_electron_kinetic_energy_Hartree(
        two_photons, abs_or_emi, n_qn, hole_kappa
    )
    coulomb_phase_arr = np.zeros(
        (N_kappas_all, len(ekin))
    )  # vector to store coulomb phase

    g_omega_IR = two_photons.g_omega_IR  # energy of IR photon in Hartree

    for i in range(N_kappas_all):
        final_kappa = final_kappas_list[i]
        if (
            final_kappa in final_kappas_reachable
        ):  # filter reachable states and keep unreachable ones zero
            if abs_or_emi == "emi":
                coulomb_phase_arr[i, :] = coulomb_phase(
                    final_kappa, ekin - g_omega_IR, Z
                )
            elif abs_or_emi == "abs":
                coulomb_phase_arr[i, :] = coulomb_phase(
                    final_kappa, ekin + g_omega_IR, Z
                )

    return coulomb_phase_arr


def get_matrix_elements_with_coulomb_phase(
    two_photons: TwoPhotons, abs_or_emi, n_qn, hole_kappa, Z
):
    """
    Computes coupled matrix elements for the given hole and adds Coulomb phase to them in the
    two photons case.

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    abs_or_emi - tells if we want to get for absorption or emission path,
    can take only 'abs' or 'emi' values.
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion

    Returns:
    Matrix elements with Coulomb phase
    """

    assert_abs_or_emi(abs_or_emi)

    two_photons.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

    M = get_coupled_matrix_elements_for_all_final_states(
        two_photons,
        abs_or_emi,
        n_qn,
        hole_kappa,
    )
    coul_phase = get_coulomb_phase(
        two_photons, abs_or_emi, n_qn, hole_kappa, Z
    )  # Coulomb phase

    assert (
        M.shape[0],
        M.shape[2],
    ) == coul_phase.shape, (
        "Shapes of matrix with elements and matrix with Coulomb phase don't match!"
    )

    for i in range(M.shape[0]):  # iterate over final states
        for j in range(3):  # iterate over K (rank of the photon-interaction) values
            M[i, j, :] *= np.exp(1j * coul_phase[i, :])

    return M


def prepare_absorption_and_emission_matrices_1sim(
    two_photons: TwoPhotons, n_qn, hole_kappa, Z, steps_per_IR_photon=None
):
    """
    Works with the case of 1 simulation (only 1 TwoPhotons object). Constructs absorption and
    emission matrices and matches them so that they corrsepond to the same final photoelectron
    energies.

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy.
    If not specified, the the program calculates it based on the XUV energy data in the
    omega.dat file and value of the IR photon energy

    Returns:
    ekin_final - array of final phototelctron kinetic energies in eV
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    """

    two_photons.assert_hole_load("abs", n_qn, hole_kappa)
    two_photons.assert_hole_load("emi", n_qn, hole_kappa)

    ekin_eV = get_electron_kinetic_energy_eV(two_photons, "abs", n_qn, hole_kappa)

    g_omega_IR = two_photons.g_omega_IR  # frequncy of the IR photon (in Hartree)

    if steps_per_IR_photon is None:
        steps_per_IR_photon = int(
            g_omega_IR / ((ekin_eV[1] - ekin_eV[0]) / g_eV_per_Hartree)
        )

    M_emi = get_matrix_elements_with_coulomb_phase(
        two_photons, "emi", n_qn, hole_kappa, Z
    )
    M_abs = get_matrix_elements_with_coulomb_phase(
        two_photons, "abs", n_qn, hole_kappa, Z
    )
    ekin_final, M_emi_matched, M_abs_matched = (
        match_absorption_and_emission_matrices_1sim(
            ekin_eV, M_emi, M_abs, steps_per_IR_photon
        )
    )

    return ekin_final, M_emi_matched, M_abs_matched


def match_absorption_and_emission_matrices_1sim(
    energies, M_emi, M_abs, steps_per_IR_photon
):
    """
    Matches absorpiton and emission matrices so that their indices correspond to the same final
    photoelectron energy in the case of 1 simulation.

    Params:
    energies - array of energies
    M_emi - unmatched matrix elements for emission path
    M_abs - unmatched matrix elements for absorption path
    steps_per_IR_photon - the number of XUV energy steps fitted in the IR photon energy

    Returns:
    energies_final - the array of final phototelctron energies
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    """

    energies_final = final_energies_for_matching_1sim(energies, steps_per_IR_photon)

    assert (
        M_abs.shape[0] == M_emi.shape[0]
    ), "The number of final states in absorption and emission matrices is different!"

    M_emi_matched = np.zeros(
        (M_emi.shape[0], M_emi.shape[1], M_emi.shape[2] - 2 * steps_per_IR_photon),
        dtype="complex128",
    )

    M_abs_matched = np.zeros(
        (M_abs.shape[0], M_abs.shape[1], M_abs.shape[2] - 2 * steps_per_IR_photon),
        dtype="complex128",
    )

    for i in range(M_abs.shape[0]):  # iterate over final states
        for j in range(3):  # iterate over K (rank of the photon-interaction) values
            M_emi_matched[i, j, :], M_abs_matched[i, j, :] = match_matrix_elements_1sim(
                M_emi[i, j, :], M_abs[i, j, :], steps_per_IR_photon
            )

    return energies_final, M_emi_matched, M_abs_matched


def prepare_absorption_and_emission_matrices_2sim(
    two_photons_emi: TwoPhotons,
    two_photons_abs: TwoPhotons,
    n_qn,
    hole_kappa,
    Z,
    energies_mode="both",
):
    """
    Works with the case of 2 simulations (requires 2 TwoPhotons objects).
    Constructs emission from two_photons_emi and absorption from two_photons_abs matrices and
    matches them so that they corrsepond to the same final photoelectron energies.

    Params:
    two_photons_emi - object of the TwoPhotons class for emission matrix
    two_photons_abs - object of the TwoPhotons class for absorption matrix
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    energies_mode - tells which energies we take for matrices interpolation. Possible options:
    "emi" - energies from emission object, "abs" - energies from absorption object, "both" -
    combined array from both emission and absorption objects.

    Returns:
    ekin_final - array of final phototelctron kinetic energies in eV
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    """

    two_photons_emi.assert_hole_load("emi", n_qn, hole_kappa)
    two_photons_abs.assert_hole_load("abs", n_qn, hole_kappa)

    ekin_eV_emi = get_electron_kinetic_energy_eV(
        two_photons_emi, "emi", n_qn, hole_kappa
    )
    ekin_eV_abs = get_electron_kinetic_energy_eV(
        two_photons_abs, "abs", n_qn, hole_kappa
    )

    # frequncies of the IR photons (in Hartree)
    g_omega_IR_emi = two_photons_emi.g_omega_IR
    g_omega_IR_abs = two_photons_abs.g_omega_IR

    # match to the final photoelectron energies
    ekin_eV_emi -= g_omega_IR_emi * g_eV_per_Hartree
    ekin_eV_abs += g_omega_IR_abs * g_eV_per_Hartree

    M_emi = get_matrix_elements_with_coulomb_phase(
        two_photons_emi, "emi", n_qn, hole_kappa, Z
    )
    M_abs = get_matrix_elements_with_coulomb_phase(
        two_photons_abs, "abs", n_qn, hole_kappa, Z
    )

    ekin_final, M_emi_matched, M_abs_matched = (
        match_absorption_and_emission_matrices_2sim(
            ekin_eV_emi, ekin_eV_abs, M_emi, M_abs, energies_mode=energies_mode
        )
    )

    return ekin_final, M_emi_matched, M_abs_matched


def match_absorption_and_emission_matrices_2sim(
    energies_emi,
    energies_abs,
    M_emi,
    M_abs,
    energies_mode="both",
):
    """
    Matches absorpiton and emission matrices so that their indices correspond to the same final
    photoelectron energy in the case of 2 simulations.

    Params:
    energies_emi - array of energies for emission path
    energies_abs - array of energies for absorption path
    M_emi - unmatched matrix elements for emission path
    M_abs - unmatched matrix elements for absorption path
    energies_mode - tells which energies we take for matrices interpolation. Possible options:
    "emi" - energies from emission object, "abs" - energies from absorption object, "both" -
    combined array from both emission and absorption objects.

    Returns:
    energies_final - the array of final phototelctron energies
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    """

    energies_final = final_energies_for_matching_2sim(
        energies_emi, energies_abs, energies_mode=energies_mode
    )

    assert (
        M_abs.shape[0] == M_emi.shape[0]
    ), "The number of final states in absorption and emission matrices is different!"

    M_emi_matched = np.zeros(
        (M_emi.shape[0], M_emi.shape[1], len(energies_final)),
        dtype="complex128",
    )

    M_abs_matched = np.zeros(
        (M_abs.shape[0], M_abs.shape[1], len(energies_final)),
        dtype="complex128",
    )
    for i in range(M_abs.shape[0]):  # iterate over final states
        for j in range(3):  # iterate over K (rank of the photon-interaction) values
            M_emi_matched[i, j, :], M_abs_matched[i, j, :] = match_matrix_elements_2sim(
                energies_final,
                energies_emi,
                energies_abs,
                M_emi[i, j, :],
                M_abs[i, j, :],
            )

    return energies_final, M_emi_matched, M_abs_matched


def get_prepared_matrices(
    two_photons_1: TwoPhotons,
    n_qn,
    hole_kappa,
    Z,
    two_photons_2: Optional[TwoPhotons] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Returns prepared emission and absorption matrices as well as energy vector
    depending on how many simulations (1 or 2) were provided.

    Params:
    two_photons_1 - object of the TwoPhotons class corresponding to one simulation
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    two_photons_2 - second object of the TwoPhotons class if we want to consider 2 simulations
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, then the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.
    NOTE: if two simulations were provided, then the first one corresponds to emission path and
    the second one to absorption path

    Returns:
    ekin_eV - array of phototelctron kinetic energies in eV
    M_emi_matched - matrix elements for emission path matched to the final energies
    M_abs_matched - matrix elements for absorption path matched to the final energies
    """

    if two_photons_2:  # if the second simulation is provided
        two_photons_1.assert_hole_load("emi", n_qn, hole_kappa)
        two_photons_2.assert_hole_load("abs", n_qn, hole_kappa)
        ekin_eV, M_emi_matched, M_abs_matched = (
            prepare_absorption_and_emission_matrices_2sim(
                two_photons_1,
                two_photons_2,
                n_qn,
                hole_kappa,
                Z,
                energies_mode=energies_mode,
            )
        )
    else:  # if the second simulation is not provided
        two_photons_1.assert_hole_load("emi", n_qn, hole_kappa)
        two_photons_1.assert_hole_load("abs", n_qn, hole_kappa)
        ekin_eV, M_emi_matched, M_abs_matched = (
            prepare_absorption_and_emission_matrices_1sim(
                two_photons_1,
                n_qn,
                hole_kappa,
                Z,
                steps_per_IR_photon=steps_per_IR_photon,
            )
        )

    return ekin_eV, M_emi_matched, M_abs_matched
