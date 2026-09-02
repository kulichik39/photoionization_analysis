import numpy as np
from scipy.special import gamma
from fortran_output_analysis.types import ArrFloat64, ArrComplex128
from fortran_output_analysis.global_utility import l_to_str, l_to_int
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree


def extract_data_from_file(file: str, bp_index: int) -> tuple[ArrFloat64, ArrFloat64]:
    """
    Extracts data, corresponding to the specified break point, from the given file.

    Args:
        file - path to the file.
        bp_index - the breakpoint index, starting with 1.

    Returns:
        array of energies and array of values.
    """

    raw = np.loadtxt(file, dtype=np.float64)

    assert bp_index < raw.shape[1], "the break point index is out of range!"

    return raw[:, 0], raw[:, bp_index]


def wavenumber(ekin_hart: ArrFloat64) -> ArrFloat64:
    """
    Computes the wavenumber (k-value) in atomic units.

    Args:
        ekin - array of kinetic energies (in Hartree).

    Returns:
        k - vector of wavenumbers.
    """

    ekin_copy = ekin_hart.copy()
    ekin_copy[ekin_copy < 0.0] = (
        0.0  # change all negative elements to 0 to avoid warnings in np.sqrt
    )

    k = np.sqrt(2 * ekin_copy)

    return k


def coulomb_phase(final_l: int | str, ekin_hart: ArrFloat64, Z: int) -> ArrFloat64:
    """
    This is the definition of the phase of the Coulomb function, both the angular momentum part and
    the so-called Coulomb phase. Electron kinetic energy should be given in atomic units.

    Args:
        final_l - orbital angular momentum of the final state, in the int or str format.
        ekin_hart - array of electron kinetic energies (in Hartree).
        Z - charge of the ion.
    """

    if type(final_l) is str:
        final_l = l_to_int(final_l)

    k = wavenumber(ekin_hart)

    nu = Z / k

    b = np.angle(gamma(final_l + 1 + 1j * nu))

    return -b - final_l * np.pi / 2


class Hole:

    def __init__(
        self,
        atom_name: str,
        l: int | str,
        n_qn: int,
        binding_energy: float | None = None,
    ) -> None:
        """
        Args:
            atom_name - name of the parent atom.
            l - orbital angular momentum of the hole, given in the form of int or str.
            n_qn - pricnipal quantum number of the hole.
            binding_energy - binding energy for the hole in Hartree. Allows you to specify a
            predifined alue for the hole's binding energy instead of loading it from the simulation
            data.
        """
        self.atom_name = atom_name
        if type(l) is str:
            l = l_to_int(l)
        self.l: int = l
        self.n = n_qn
        self.name = construct_hole_name(atom_name, l, n_qn)
        self.binding_energy = binding_energy  # in Hartree

    def load_binding_energy(self, path_to_energies: str) -> None:
        """
        Attempts to load hole's binding energy from the energies.dat file in the output folder.

        Args:
            path_to_energies - path to the energies.dat file.
        """

        try:
            with open(path_to_energies, "r") as f:

                for line in f:
                    line = line.strip()
                    line_split = line.split()

                    # check if we're at the end of the file
                    if int(line_split[0]) == 0 and int(line_split[1]) == 0:
                        raise RuntimeError(
                            "Reached the end of the energies.dat file. The hole's "
                            "binding energy is missing!"
                        )

                    # check for the hole parameters
                    if int(line_split[0]) == self.n and int(line_split[1]) == self.l:
                        bind_en_data = line_split[2]
                        # replace Fortran double-prec notation with Python's e0
                        bind_en_data = bind_en_data.replace("d0", "e0").replace(
                            "D0", "e0"
                        )

                        # locate the binding energy in the output
                        start = bind_en_data.index("(") + 1
                        end = bind_en_data.index(",")
                        bind_en = bind_en_data[start:end]
                        # in Hartree, NOTE the minus sign in front of the float()
                        bind_en = -float(bind_en) / g_eV_per_Hartree
                        self.binding_energy = bind_en
                        return

        except Exception as e:
            print(
                f"{self.name}: Failed to load binding energy from the energies.dat file.! Error: {e}"
            )

        # if we reached this part -> the energy was not loaded -> send a warning
        print(f"Warning: binding energy for the {self.name} hole is not loaded!")


def construct_hole_name(atom_name: str, l: int | str, n_qn: int) -> str:
    """
    Constructs a readable name for the hole with given parameters.

    Args:
        atom_name - name of the parent atom.
        l - orbital angular momentum of the hole.
        n_qn - pricnipal quantum number of the hole.

    Returns:
        name - readable name of the hole.
    """
    name = atom_name + " " + str(n_qn)

    if type(l) is int:
        name += l_to_str(l)
    else:
        name += l

    return name


def assert_out_as(out_as: str) -> None:

    assert out_as in (
        "int",
        "str",
    ), f'The out_as argument must be "int" or "str", not "{out_as}"!'


def final_sideband_energies_1sim(
    energies_emi: ArrFloat64,
    energies_abs: ArrFloat64,
    g_omega_IR: float,
    energies_mode: str,
) -> ArrFloat64:
    """
    Prepares an array of final sideband energies in the case of one simulation.

    Args:
        energies_emi - energies of the emission path.
        energies_abs - energies of the absorption path.
        g_omega_IR - energy of the IR photon.
        energies_mode - tells which energies we choose for the final sideband.
                        Possible options:
                        "emi" - emission energies starting from the first available absorption point.
                        "abs" - absorption energies cut at the last available emission point.

    Returns:
        an array of final sideband energies.
    """

    assert energies_mode in (
        "emi",
        "abs",
    ), f"energies_mode for the final sideband energy from one simulation must be 'emi' or 'abs', not '{energies_mode}'!"

    if energies_mode == "emi":
        en_mask = energies_emi >= (
            np.min(energies_emi) + 2 * g_omega_IR
        )  # start from the first absorption point
        return energies_emi[en_mask]
    else:
        en_mask = energies_abs <= (
            np.max(energies_abs) - 2 * g_omega_IR
        )  # cut at the last emission point
        return energies_abs[en_mask]


def match_matrix_elements_1sim(
    energies_final: ArrFloat64,
    energies_emi: ArrFloat64,
    energies_abs: ArrFloat64,
    mat_emi: ArrComplex128,
    mat_abs: ArrComplex128,
) -> tuple[ArrComplex128, ArrComplex128]:
    """
    Matches absoprtion and emission matrix elements to the final sideband energies
    in the case of one simulation.

    Args:
        energies_final - final sideband energies.
        energies_emi - energies of the emission path.
        energies_abs - energies of the absorption path.
        mat_emi - unmatched matrix elements for the emission path.
        mat_abs - unmatched matrix elements for the absorption path.

    Returns:
        mat_emi_matched - matched matrix elements for the emission path.
        mat_abs_matched - matched matrix elements for the absorption path.
    """

    mat_emi_matched = np.interp(energies_final, energies_emi, mat_emi)
    mat_abs_matched = np.interp(energies_final, energies_abs, mat_abs)

    return mat_emi_matched, mat_abs_matched
