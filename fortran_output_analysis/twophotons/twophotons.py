import numpy as np
import re  # Regular expressions
import os
from itertools import islice  # Slicing when reading lines from Fortran files.
from fortran_output_analysis.constants_and_parameters import (
    g_eV_per_Hartree,
    g_omega_IR,
)
from fortran_output_analysis.common_utility import (
    l_from_kappa,
    l_to_str,
    wigner_eckart_phase,
    wigner3j_numerical2,
    j_from_kappa,
    j_from_kappa_int,
    IonHole,
    phase,
    exported_mathematica_tensor_to_python_list,
    mag,
    cross,
    coulomb_phase,
)
from sympy.physics.wigner import wigner_3j, wigner_6j
from input_to_fortran.parse_user_input_file import parse_user_input_file


class IonisationPath:
    def __init__(self, intermediate_kappa, final_kappa, file_col_idx, col_idx):
        self.kappa_intermediate = intermediate_kappa
        self.l_intermediate = l_from_kappa(intermediate_kappa)
        self.j_intermediate = j_from_kappa(intermediate_kappa)
        self.kappa_final = final_kappa
        self.l_final = l_from_kappa(final_kappa)
        self.j_final = j_from_kappa(final_kappa)
        self.name_intermediate = (
            ""
            + l_to_str(self.l_intermediate)
            + ("_{%i/2}" % (j_from_kappa_int(intermediate_kappa)))
        )
        self.name_final = (
            "" + l_to_str(self.l_final) + ("_{%i/2}" % (j_from_kappa_int(final_kappa)))
        )

        # We don't store zero columns from the input file,
        # so the index in the file and the index used to retrieve data will differ.
        self.file_column_index = file_col_idx
        self.column_index = col_idx

    def full_name(self):
        return self.name_intermediate + " to " + self.name_final


class MatrixElements:
    # This class handles the fortran output data
    # for each channel described by a two-photon (XUV+IR) interaction:
    # hole - intermediate - final
    # It also handles the summation over intermediate states required
    # for a "measurable" description of the channel.
    def __init__(self, path, hole_kappa, hole_n, abs_or_emi, verbose=False):
        self.is_initialised = False
        self.path = path
        self.hole = IonHole(hole_kappa, hole_n)
        self.ionisation_paths = {}

        with open(path, "r") as file:

            # Sets up what the available ionisation paths are, and what columns in the file it is corresponding to.
            parse_first_line_from_fortran_matrix_element_output_file(
                file, self.hole, self.ionisation_paths
            )
            self.number_of_ionisation_paths = len(self.ionisation_paths)

            # We use this list to only pick the non-zero data points after parsing raw data below.
            self.path_col_indices = []
            for ionisation_path in self.ionisation_paths.values():
                self.path_col_indices.append(ionisation_path.file_column_index)

            # These will be filled with parsing the rest of the file.
            self.raw_data_real = np.zeros(1)
            self.raw_data_imag = np.zeros(1)
            # For any energy the matrix element is printed at each breakpoint from the Fortran program.
            # So here we can control what breakpoints we choose.
            self.breakpoint_step = 5

            self.raw_data_real, self.raw_data_imag = (
                parse_matrix_element_raw_data_from_fortran_output_file(
                    file,
                    self.number_of_ionisation_paths,
                    self.path_col_indices,
                    self.breakpoint_step,
                )
            )

        # Errors for this are catched outside this class.
        abs_or_emi_string = "emission"
        if abs_or_emi == "abs":
            abs_or_emi_string = "absorption"

        self.name = self.hole.name + " " + abs_or_emi_string + " matrix elements."

        if verbose:
            print("Added:")
            print(self.name)
        self.is_initialised = True

    def get_ionisation_path(self, intermediate_kappa, final_kappa):
        if self.is_initialised:
            key_tuple = (self.hole.kappa, intermediate_kappa, final_kappa)
            ionisation_path = self.ionisation_paths[key_tuple]
            column_index = ionisation_path.column_index
            z = (
                self.raw_data_real[:, column_index]
                + 1j * self.raw_data_imag[:, column_index]
            )
            name = self.hole.name + "$ to $" + ionisation_path.full_name()
            return z, name
        else:
            raise ValueError("MatrixElements not initialised!")

    def get_ionisation_path_summed_over_intermediate(self, final_kappa, mj):

        paths_summed_over_intermediate_states = (
            sum_over_intermediate_states_including_3j_symbols(
                self.hole.kappa,
                final_kappa,
                self.raw_data_real,
                self.raw_data_imag,
                self.ionisation_paths,
                mj,
            )
        )

        for state_tuple in paths_summed_over_intermediate_states.keys():

            if final_kappa == state_tuple[1]:
                final_label = l_to_str(l_from_kappa(final_kappa)) + (
                    "_{%i/2}" % (j_from_kappa_int(final_kappa))
                )
                name = self.hole.name + "$  to  $" + final_label
                return paths_summed_over_intermediate_states[state_tuple], name

        # If channel doesn't exist we return None and handle error outside this function.
        return None

    def get_summed_channels(self, mj):
        # This function just gets the hole- and final-kappas present, ie implies the sum over all intermediate.
        # It returns a list of the unique tuples (hole_kappa, final_kappa).
        # It can be used as input to getting all the data for these channels summed over intermediate states.
        hole_kappa = self.hole.kappa
        channels = []
        for path_tuple in self.ionisation_paths.keys():
            final_kappa = path_tuple[2]

            j_hole = j_from_kappa_int(hole_kappa)
            j_final = j_from_kappa_int(final_kappa)
            mjj = int(2 * mj)
            if np.abs(mjj) > j_hole or np.abs(mjj) > j_final:
                continue

            if (hole_kappa, final_kappa) not in channels:
                channels.append((hole_kappa, final_kappa))

        return channels


class TwoPhotons:
    def __init__(self, atom_name, g_omega_IR):
        self.atom_name = atom_name
        self.matrix_elements_abs = {}
        self.matrix_elements_emi = {}
        self.eV_per_Hartree = g_eV_per_Hartree
        self.omega_path = ""
        self.omega_Hartree = 0
        self.omega_eV = 0

        # energy of the IR photon used in Fortran simulations (in Hartree)
        self.g_omega_IR = g_omega_IR

    def add_omega(self, path):
        self.omega_path = path
        self.omega_Hartree = np.loadtxt(path)
        self.omega_eV = self.omega_Hartree * self.eV_per_Hartree

    # Add matrix elements after absorption or emission of an IR photon
    # for a particular hole (from XUV absorption),
    # using the output data from the Fortran program.
    def add_matrix_elements(self, path, abs_or_emi, hole_kappa, hole_n):

        if abs_or_emi == "abs":
            self.matrix_elements_abs[hole_kappa] = MatrixElements(
                path, hole_kappa, hole_n, abs_or_emi
            )

        elif abs_or_emi == "emi":
            self.matrix_elements_emi[hole_kappa] = MatrixElements(
                path, hole_kappa, hole_n, abs_or_emi
            )

        else:
            raise ValueError(
                "Need to specify emission ('emi') or absorption ('abs') when adding matrix elements data!"
            )

        return

    # This is a method to get the data that is summed over intermediate states and over m values.
    def get_matrix_elements_for_hole_to_final(
        self, hole_kappa, final_kappa, abs_or_emi, mj
    ):

        retval = np.zeros(1)
        if abs_or_emi == "abs":
            retval, name = self.matrix_elements_abs[
                hole_kappa
            ].get_ionisation_path_summed_over_intermediate(final_kappa, mj)

        elif abs_or_emi == "emi":
            retval, name = self.matrix_elements_emi[
                hole_kappa
            ].get_ionisation_path_summed_over_intermediate(final_kappa, mj)
        else:
            raise ValueError(
                "Need to specify emission ('emi') or absorption ('abs') when getting matrix element data!"
            )

        if len(retval) < 2:
            raise ValueError(
                "Couldn't get matrix elements for hole_kappa %i and final_kappa %i, \n"
                "get_ionisation_path_summed_over_intermediate_and_mj returned None."
                % (hole_kappa, final_kappa)
            )
        else:
            return retval, name

    def get_hole_to_final_channels(self, hole_kappa, abs_or_emi, mj):
        channels = []
        if abs_or_emi == "abs":
            channels = self.matrix_elements_abs[hole_kappa].get_summed_channels(mj)

        elif abs_or_emi == "emi":
            channels = self.matrix_elements_emi[hole_kappa].get_summed_channels(mj)
        else:
            raise ValueError(
                "Need to specify emission ('emi') or absorption ('abs') when getting matrix element data!"
            )

        return channels

    def get_matrix_element_for_intermediate_resolved_channel(
        self, hole_kappa, intermediate_kappa, final_kappa, abs_or_emi
    ):
        retval = np.zeros(1)
        if abs_or_emi == "abs":
            retval, name = self.matrix_elements_abs[hole_kappa].get_ionisation_path(
                intermediate_kappa, final_kappa
            )

        elif abs_or_emi == "emi":
            retval, name = self.matrix_elements_emi[hole_kappa].get_ionisation_path(
                intermediate_kappa, final_kappa
            )
        else:
            raise ValueError(
                "Need to specify emission ('emi') or absorption ('abs') when getting matrix element data!"
            )

        if len(retval) < 2:
            raise ValueError(
                "Couldn't get matrix elements for hole_kappa %i and final_kappa %i, \n"
                "get_ionisation_path_summed_over_intermediate_and_mj returned None."
                % (hole_kappa, final_kappa)
            )
        else:
            return retval, name

    def get_coupled_matrix_element(
        self, hole_kappa, abs_or_emi, final_kappa, verbose=False
    ):
        """Computes the value of the specified coupled matrix element.
        This function also adds in the non-matrix element phases from the fortran program
        """
        if abs_or_emi == "abs":
            fortranM = self.matrix_elements_abs[hole_kappa]
        elif abs_or_emi == "emi":
            fortranM = self.matrix_elements_emi[hole_kappa]
        else:
            raise ValueError(
                "Need to specify emission ('emi') or absorption ('abs') when getting matrix element data!"
            )

        # Get the length of the array of data points by looking up the index of the first open channel and checking
        # the row length at that column.
        random_col = fortranM.ionisation_paths[
            next(iter(fortranM.ionisation_paths.keys()))
        ].column_index
        energy_size = len(fortranM.raw_data_real[:, random_col])

        # Initialize the output array       3 different values for K: 0, 1 and 2
        coupled_matrix_element = np.zeros((3, energy_size), dtype="complex128")

        # Get the data from the correct phase file
        phase_data = self._raw_short_range_phase(abs_or_emi, hole_kappa)

        # Loop through all the ionisation paths
        for kappa_tuple in fortranM.ionisation_paths.keys():
            loop_hole_kappa, intermediate_kappa, loop_final_kappa = kappa_tuple

            # Only use those that match the requested initial and final state
            if loop_hole_kappa == hole_kappa and loop_final_kappa == final_kappa:
                # Get j values
                hole_j = j_from_kappa(hole_kappa)
                intermediate_j = j_from_kappa(intermediate_kappa)
                final_j = j_from_kappa(final_kappa)

                # Get the column index of the raw fortran output file that contains the requested ionisation path
                col_index = fortranM.ionisation_paths[kappa_tuple].column_index

                # Compute the value of the matrix element
                matrix_element = (
                    fortranM.raw_data_real[:, col_index]
                    + 1j * fortranM.raw_data_imag[:, col_index]
                )

                # Add in the short range phase from the fortran program
                matrix_element *= np.exp(1j * phase_data[:, col_index])

                for K in [0, 2]:
                    if verbose:
                        print(
                            f"{K=}, {hole_j=}, {intermediate_j=}, {final_j=} gives ",
                            (2 * K + 1)
                            * float(wigner_3j(1, 1, K, 0, 0, 0))
                            * float(
                                wigner_6j(1, 1, K, hole_j, final_j, intermediate_j)
                            ),
                        )
                        # print(f"{K=}, {hole_j=}, {intermediate_j=}, {final_j=} gives ", phase(hole_j + final_j + K)*(2*K+1)*float(wigner_3j(1,1,K,0,0,0))*float(wigner_6j(1,1,K,hole_j,final_j,intermediate_j)))
                        print(
                            f"The last point of this matrix element is {matrix_element[-1]}\n"
                        )

                    # Multiply by the prefactor and store it in the output matrix
                    coupled_matrix_element[K] += (
                        (2 * K + 1)
                        * float(wigner_3j(1, 1, K, 0, 0, 0))
                        * matrix_element
                        * float(wigner_6j(1, 1, K, hole_j, final_j, intermediate_j))
                    )
                    # coupled_matrix_element[K] += phase(hole_j + final_j + K)*(2*K+1)*float(wigner_3j(1,1,K,0,0,0))*matrix_element*float(wigner_6j(1,1,K,hole_j,final_j,intermediate_j))

        if verbose:
            print(
                f"in the end the 100th point of the coupled matrix element for {final_kappa=} ends up as",
                coupled_matrix_element[:, 100],
            )

        return [
            coupled_matrix_element[0],
            coupled_matrix_element[1],
            coupled_matrix_element[2],
        ]

    def _raw_short_range_phase(self, abs_or_emi, hole_kappa):
        """Returns the short range phase for the given channel.
        The channels are organized in the same way for the return value from this function
        As the raw_data_real and raw_data_imag items of a MatrixElement object"""

        if abs_or_emi == "abs":
            M = self.matrix_elements_abs[hole_kappa]
        elif abs_or_emi == "emi":
            M = self.matrix_elements_emi[hole_kappa]
        else:
            raise ValueError(f"abs_or_emi can only be 'abs' or 'emi' not {abs_or_emi}")

        # Determine the name of the data file
        phase_file_name = (
            "phase_"
            + abs_or_emi
            + f"_{hole_kappa}_{M.hole.n - l_from_kappa(hole_kappa)}.dat"
        )

        # Remove the string after the last "/" in the phase that points to the matrix element file
        # and replace it with the name of the phase data file
        phase_path = (
            os.path.sep.join(M.path.split(os.path.sep)[:-1])
            + os.path.sep
            + phase_file_name
        )

        raw_phase_data = np.loadtxt(phase_path)

        # Filter out all columns that are all zeros to make column index match with organization of matrix element data
        zero_col_indexes = np.argwhere(np.all(raw_phase_data[..., :] == 0, axis=0))
        raw_phase_data = np.delete(raw_phase_data, zero_col_indexes, axis=1)

        return raw_phase_data
