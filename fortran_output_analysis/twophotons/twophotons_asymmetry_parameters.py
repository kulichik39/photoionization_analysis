import numpy as np
import os
from typing import Optional
from fortran_output_analysis.common_utility import (
    exported_mathematica_tensor_to_python_list,
    assert_abs_or_emi,
)
from fortran_output_analysis.twophotons.twophotons import TwoPhotons
from fortran_output_analysis.twophotons.twophotons_utilities import (
    get_electron_kinetic_energy_eV,
    get_matrix_elements_with_coulomb_phase,
    get_prepared_matrices,
)


def two_photons_asymmetry_parameter(
    n,
    hole_kappa,
    M1,
    M2,
    abs_emi_or_cross,
    path=os.path.join(
        os.path.sep.join(
            os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1]
        ),
        "formula_coefficients",
        "two_photons",
        "asymmetry_coeffs",
    ),
    threshold=1e-10,
):
    """
    Computes the value of the n'th asymmetry parameter for a state defined by hole_kappa in the
    two photons case.
    M1 and M2 contain the matrix elements and other phases of the wave function organized
    according to their final kappa like so:
    m = |hole_kappa|
    s = sign(hole_kappa)
    M = [s(m-2), -s(m-1), sm, -s(m+1), s(m+2)] (the values in the list are kappas of final states)
    Each part of the matrix corresponding to a particular final state contains 3 arrays of matrix
    elements for K = 0, 1, 2 where K is the rank of photon-interaction. So, M1 and M2 have the
    following shape: (5, 3, size_of_elements_array), where 5 is the number of final states, 3
    is three different values of K, and size_of_elements_array stands for length of each vector
    with matrix elements.

    If you want to calculate real asymmetry parameter, then simply pass the same matrix to
    M1 and M2 and specify abs_emi_or_cross as "abs" or "emi".
    If you want to calculate complex asymmetry parameter, at first you need to match the
    original emission and absorption matrices so that they correspond to the same final
    photoelectron energies. Then pass matched emission matrix as M1 and matched absorption matrix
    as M2 and specify abs_emi_or_cross as "cross".

    Params:
    n - order of the asymmetry parameter
    hole_kappa - kappa value of the hole
    M1, M2 - either the same matrix or emission and absorption matrices matched to the same
    final photoelectron energies
    abs_emi_or_cross - specify "abs" or "emi" when provide the same matrix and "cross"
    otherwise
    path - path to the file with coefficients for asymmetry parameter calculation
    threshold - required to check if the imaginary part of the real asymmetry parameter
    is small (less than this threshold value), since in theory it should be 0
    """

    if (
        abs_emi_or_cross != "abs"
        and abs_emi_or_cross != "emi"
        and abs_emi_or_cross != "cross"
    ):
        raise ValueError(
            f"abs_emi_or_cross can only be 'abs', 'emi' or 'cross' not {abs_emi_or_cross}"
        )

    assert M1.shape == M2.shape, "The shapes of the input matrices must be the same!"

    energy_size = len(M1[0][1])

    # If the path to the coefficient files does not end in a path separator, add it.
    if path[-1] is not os.path.sep:
        path = path + os.path.sep

    # Try opening the needed file.
    try:
        with open(path + f"asymmetry_coeffs_{n}_{hole_kappa}.txt", "r") as coeffs_file:
            coeffs_file_contents = coeffs_file.readlines()
    except OSError as e:
        print(e)
        raise NotImplementedError(
            f"the given combination of hole kappa {hole_kappa} and n {n} is not yet implemented, or the file containing the coefficients could not be found"
        )

    # Read in the coefficients in front of all the different combinations of matrix elements in the numerator.
    numerator_coeffs = np.array(
        exported_mathematica_tensor_to_python_list(coeffs_file_contents[3])
    )

    # Read in the coefficients in front of the absolute values in the denominator.
    denominator_coeffs = np.array(
        exported_mathematica_tensor_to_python_list(coeffs_file_contents[4])
    )

    numerator = np.zeros(energy_size, dtype="complex128")
    denominator = np.zeros(energy_size, dtype="complex128")
    for kappa_1 in range(5):
        for K_1 in range(3):
            # compute the integrated intensity denominator.
            denominator += (
                denominator_coeffs[kappa_1, K_1]
                * M1[kappa_1][K_1]
                * np.conj(M2[kappa_1][K_1])
            )
            for kappa_2 in range(5):
                for K_2 in range(3):
                    # Multiply each combination of matrix elements with its coefficient.
                    numerator += (
                        numerator_coeffs[kappa_1, kappa_2, K_1, K_2]
                        * M1[kappa_1][K_1]
                        * np.conj(M2[kappa_2][K_2])
                    )

    parameter = numerator / denominator

    if abs_emi_or_cross != "cross":
        # When looking at the real asymmetry parameter the result is a real number
        values = parameter[
            ~np.isnan(parameter)
        ]  # Filter out the nans first, as they mess up boolean expressions (nan is not itself).
        assert all(
            np.abs(np.imag(values)) < threshold
        ), "The real asymmetry parameter had a non-zero imaginary part when it shouldn't. Check the input matrix elements or change the threshold for the allowed size of the imaginary part"
        parameter = np.real(parameter)

    if abs_emi_or_cross == "cross":
        abs_emi_or_cross = "complex"
    label = f"$\\beta_{n}^{{{abs_emi_or_cross}}}$"

    return parameter, label


def get_real_asymmetry_parameter(
    two_photons: TwoPhotons, n, abs_or_emi, n_qn, hole_kappa, Z
):
    """
    Computes real asymmetry parameter in the two photons case.

    Params:
    two_photons - object of the TwoPhotons class with some loaded holes
    n - order of the asymmetry parameter
    abs_or_emi - tells if we want to get for absorption or emission path,
    can take only 'abs' or 'emi' values
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion

    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    b_real - values of the real asymmetry parameter
    """

    assert_abs_or_emi(abs_or_emi)
    two_photons.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

    ekin_eV = get_electron_kinetic_energy_eV(two_photons, abs_or_emi, n_qn, hole_kappa)

    M = get_matrix_elements_with_coulomb_phase(
        two_photons, abs_or_emi, n_qn, hole_kappa, Z
    )
    b_real, _ = two_photons_asymmetry_parameter(
        n, hole_kappa, M, M, abs_or_emi
    )  # one-photon real assymetry parameter

    return ekin_eV, b_real


def get_complex_asymmetry_parameter(
    two_photons_1: TwoPhotons,
    n,
    n_qn,
    hole_kappa,
    Z,
    two_photons_2: Optional[TwoPhotons] = None,
    steps_per_IR_photon=None,
    energies_mode="both",
):
    """
    Computes complex asymmetry parameter in the two photons case. Can compute for 1 or 2
    simulations. If 2 simulations are provided, then the first two_photons object corresponds
    to emission path, while the second two_photons object corresponds to absorption path.

    Params:
    two_photons_1 - object of the TwoPhotons class corresponding to the first simulation
    n - order of the asymmetry parameter
    n_qn - principal quantum number of the hole
    hole_kappa - kappa value of the hole
    Z - charge of the ion
    two_photons_2 - object of the TwoPhotons class specified if we want to consider 2 simulations
    (first for emission, second for absorption)
    steps_per_IR_photon - Required for 1 simulation only. Represents the number of XUV energy
    steps fitted in the IR photon energy. If not specified, the the program calculates it based
    on the XUV energy data in the omega.dat file and value of the IR photon energy.
    energies_mode - Required for 2 simulations only. Tells which energies we take for matrices
    interpolation. Possible options: "emi" - energies from emission object, "abs" - energies
    from absorption object, "both" - combined array from both emission and absorption objects.


    Returns:
    ekin_eV - array of photoelectron kinetic energies in eV
    b_complex - values of the complex asymmetry parameter
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

    b_complex, _ = two_photons_asymmetry_parameter(
        n, hole_kappa, M_emi_matched, M_abs_matched, "cross"
    )

    return ekin_eV, b_complex
