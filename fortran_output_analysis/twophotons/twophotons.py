import numpy as np
import os
import re  # Regular expressions
from itertools import islice  # Slicing when reading lines from Fortran files.

from fortran_output_analysis.common_utility import (
    l_from_kappa,
    l_to_str,
    j_from_kappa,
    j_from_kappa_int,
    Hole,
    load_raw_data,
    construct_hole_name,
)
from sympy.physics.wigner import wigner_3j, wigner_6j


class IonisationPath:
    """
    Contains attributes of an ionisation path in the two photons case.
    """

    def __init__(self, intermediate_kappa, final_kappa, file_col_idx, col_idx):
        """
        Params:
        intermediate_kappa - kappa value of the intermediate state
        final_kappa - kappa value of the final state
        file_col_idx - column index corresponding to the given ionization path in
        the Fortran output file
        col_idx - column index corresponding to the given ionization path in the data
        loaded to the Channels class
        NOTE: We don't store zero columns from the input file, so the index in the output file
        and the index used to retrieve data loaded to the Channels class will differ.
        """
        self.intermediate_kappa = intermediate_kappa
        self.intermediate_l = l_from_kappa(intermediate_kappa)
        self.intermediate_j = j_from_kappa(intermediate_kappa)
        self.final_kappa = final_kappa
        self.final_l = l_from_kappa(final_kappa)
        self.final_j = j_from_kappa(final_kappa)

        intermediate_name = (
            ""
            + l_to_str(self.intermediate_l)
            + ("_{%i/2}" % (j_from_kappa_int(intermediate_kappa)))
        )
        final_name = (
            "" + l_to_str(self.final_l) + ("_{%i/2}" % (j_from_kappa_int(final_kappa)))
        )
        self.name = intermediate_name + " to " + final_name

        self.file_column_index = file_col_idx
        self.column_index = col_idx


def assert_abs_or_emi(abs_or_emi):
    """
    Asserts that abs_or_emi parameter takes only "abs" or "emi" values.
    """

    assert abs_or_emi in (
        "abs",
        "emi",
    ), "abs_or_emi parameter can only be 'abs' or 'emi'!"


def convert_abs_or_emi_to_string(abs_or_emi):
    """
    Converts abs_or_emi parameter into readable string.
    """

    assert_abs_or_emi(abs_or_emi)

    if abs_or_emi == "abs":
        return "absorption"

    elif abs_or_emi == "emi":
        return "emission"


class Channels:
    """
    For the given hole, idenitifies possible ionization channels and loads raw Fortran data for
    them in the two photons case.
    """

    def __init__(
        self,
        path_to_omega,
        path_to_matrix_elements,
        hole: Hole,
        abs_or_emi,
        breakpoint_step,
        breakpoint_use,
    ):
        """
        path_to_omega - path to the omega.dat file for the given hole (usually in
        pert folders)
        path_to_matrix_elements - path to the file containing matrix elements for the hole
        hole - object of the Hole class containing hole's parameters
        abs_or_emi - tells if we load matrix elements for absorption or emission path,
        can take only 'abs' or 'emi' values.
        breakpoint_step - number of breakpoints used in Fortran simulations
        breakpoint_use - the breakpoint we want to use to fetch the data. Counted starting from
        1: e.g. 1, 2, 3, 4 ...
        NOTE: For any energy the matrix element is printed at each breakpoint from the Fortran
        program. So using the two parameters above we can control what breakpoints to
        choose.
        """

        self.__hole = hole

        assert_abs_or_emi(abs_or_emi)
        abs_or_emi_string = convert_abs_or_emi_to_string(abs_or_emi)
        self.name = hole.name + " " + abs_or_emi_string + " channels."

        # raw data on XUV photon energies
        self.__raw_omega_data = load_raw_data(path_to_omega)

        # initialize attributes for ionization paths and matrix elements
        self.__ionisation_paths = {}
        self.__raw_matrix_elements = None

        # load ionization paths and corresponding matrix elements
        self.__load_data(path_to_matrix_elements, breakpoint_step, breakpoint_use)

    def __load_data(self, path_to_matrix_elements, breakpoint_step, breakpoint_use):
        """
        Loads possible ionization paths and raw matrix elements based on the data
        from the Fortran output file.

        Params:
        path_to_matrix_elements - path to the file containing matrix elements for the hole
        breakpoint_step - number of breakpoints used in Fortran simulations
        breakpoint_use - the breakpoint we want to use to fetch the data. Counted starting from
        1: e.g. 1, 2, 3, 4 ...
        NOTE: For any energy the matrix element is printed at each breakpoint from the Fortran
        program. So using the two parameters above we can control what breakpoints to
        choose.
        """

        self.__load_ionisation_paths(path_to_matrix_elements)
        self.__load_matrix_elements(
            path_to_matrix_elements, breakpoint_step, breakpoint_use
        )

    def __load_ionisation_paths(self, path_to_matrix_elements):
        """
        Identifies and loads all possible ionization paths based
        on the data from the matrix elements file.

        Params:
        path_to_matrix_elements - path to the file containing matrix elements for the hole
        """

        with open(path_to_matrix_elements, "r") as file:
            # The first line contains information about what is in each column of the output file.
            # They are the kappas for the hole - intermediate - final channels, according to:
            # <hole> <offset_don't_care> <intermediate1> <intermediate2> <intermediate3> <final1> <final2> ... <final9>
            first_line = file.readline().rstrip()

        # split using regex - The kappas are separated by spaces.
        split_first_line = re.split("\s+", first_line)
        # For some reason we get an extra first element that is an empty string.
        # We discard this
        split_first_line = split_first_line[1:]

        # Catch input error here:
        hole_kappa = int(split_first_line[0])
        assert (
            hole_kappa == self.__hole.kappa
        ), "Mismatch between hole kappa read from file, and the input to Channels constructor"

        intermediate_kappas_str = split_first_line[2:5]
        final_kappas_str = split_first_line[5:]

        final_col_index = 0
        raw_data_col_index = 0
        intermediate_stride = 3  # There are three possible final states (including zero for non-channel) for each intermediate.
        intermediate_index = 0

        for intermediate_kappa_str in intermediate_kappas_str:

            intermediate_kappa = int(intermediate_kappa_str)

            i = intermediate_index
            for j in range(3):  # 3 finals per kappa.
                final_index = j + i * intermediate_stride
                final_kappa = int(final_kappas_str[final_index])
                if intermediate_kappa != 0 and final_kappa != 0:
                    self.__ionisation_paths[(intermediate_kappa, final_kappa)] = (
                        IonisationPath(
                            intermediate_kappa,
                            final_kappa,
                            final_col_index,
                            raw_data_col_index,
                        )
                    )
                    raw_data_col_index += 1

                final_col_index += 1  # note how this is inceremented even though we have a zero column.

            intermediate_index += 1

    def __load_matrix_elements(
        self, path_to_matrix_elements, breakpoint_step, breakpoint_use
    ):
        """
        Loads raw matrix elements from the Fortran output file.

        Params:
        path_to_matrix_elements - path to the file containing matrix elements for the hole
        breakpoint_step - number of breakpoints used in Fortran simulations
        breakpoint_use - the breakpoint we want to use to fetch the data. Counted starting from
        1: e.g. 1, 2, 3, 4 ...
        NOTE: For any energy the matrix element is printed at each breakpoint from the Fortran
        program. So using the two parameters above we can control what breakpoints to
        choose.
        """

        nonzero_col_indices = []
        for ionisation_path in self.__ionisation_paths.values():
            nonzero_col_indices.append(ionisation_path.file_column_index)

        N = 9  # we know we have 9 columns with real/imag pairs

        # arrays to store one row of the matrix
        real_line_np = np.zeros(
            N, dtype=np.double
        )  # for real part of the matrix element
        imag_line_np = np.zeros(
            N, dtype=np.double
        )  # for imaginary part of the matrix element

        # lists to store all rows of the matrix
        real_dynamic = []
        imag_dynamic = []

        breakpoint_index = breakpoint_use - 1

        with open(path_to_matrix_elements, "r") as file:
            file.readline()  # skip the first line

            # Read the rest of lines with actual matrix elements
            for line in islice(file, breakpoint_index, None, breakpoint_step):
                line = line.replace(" ", "")  # remove whitespace
                line = line.split(")(")  # split by parentheses
                line[0] = line[0].replace(
                    "(", ""
                )  # remove stray parenthesis from first element.
                line[-1] = line[-1].replace(")\n", "")  # remove crap from last element

                for i in range(len(line)):
                    real_imag_pair = line[i]
                    real, imag = real_imag_pair.split(",")
                    real_line_np[i] = np.double(real)
                    imag_line_np[i] = np.double(imag)

                real_dynamic.append(np.copy(real_line_np))
                imag_dynamic.append(np.copy(imag_line_np))

        # transform final list to arrays for convenience
        raw_matrix_real = np.array(real_dynamic)
        raw_matrix_imag = np.array(imag_dynamic)

        # keep only non-zero columns
        raw_matrix_real = raw_matrix_real[:, nonzero_col_indices]
        raw_matrix_imag = raw_matrix_imag[:, nonzero_col_indices]

        self.__raw_matrix_elements = raw_matrix_real + 1j * raw_matrix_imag

    def get_raw_omega_data(self):
        """
        Returns:
        raw photon energies in Hartree
        """
        return self.__raw_omega_data

    def get_all_ionisation_paths(self):
        """
        Returns:
        all loaded ionisation paths
        """

        return self.__ionisation_paths

    def assert_ionisation_path(self, intermediate_kappa, final_kappa):
        """
        Assertion of the ionisation path. Checks if the given ionisation path
        is within possible ionization paths.

        Params:
        intermediate_kappa - kappa value of the intermediate state
        final_kappa - kappa value of the final state
        """

        assert self.check_ionisation_path(
            intermediate_kappa, final_kappa
        ), f"The state with intermediate kappa {intermediate_kappa} and final kappa {final_kappa} is not within ionisation paths for {self.__hole.name} hole!"

    def check_ionisation_path(self, intermediate_kappa, final_kappa):
        """
        Checks if the given ionisation path (determined by intermediate_kappa and final_kappa)
        is within self.__ionisation_paths.

        Params:
        intermediate_kappa - kappa value of the intermediate state
        final_kappa - kappa value of the final state

        Returns:
        True if the given path is within self.__ionisation_paths, False otherwise
        """

        return (intermediate_kappa, final_kappa) in self.__ionisation_paths

    def get_ionisation_path(self, intermediate_kappa, final_kappa):
        """
        Returns ionization path corresponding to the given intermediate and final states.

        Params:
        intermediate_kappa - kappa value of the intermediate state
        final_kappa - kappa value of the final state

        Returns:
        object of IonisationPath class
        """

        self.assert_ionisation_path(intermediate_kappa, final_kappa)
        key_tuple = (intermediate_kappa, final_kappa)

        return self.__ionisation_paths[key_tuple]

    def get_raw_matrix_elements_for_ionization_path(
        self, ionisation_path: IonisationPath
    ):
        """
        Returns raw matrix elements corresponding to the given ionization path

        Params:
        ionisation_path - object of the IonisationPath class

        Returns:
        raw_matrix_elements - raw matrix elements for the ionization path
        """

        column_index = ionisation_path.column_index
        raw_matrix_elements = self.__raw_matrix_elements[:, column_index]

        return raw_matrix_elements

    def get_hole_object(self):
        """
        Returns:
        the Hole object corresponding to these channels.
        """

        return self.__hole


class TwoPhotons:
    """
    Grabs and stores data from Fortran simulations in the two photons case.
    """

    def __init__(self, atom_name, g_omega_IR):
        self.atom_name = atom_name

        # dicts to store ionisation channels for emission and absorption paths
        self.__channels_abs = {}
        self.__channels_emi = {}

        # IR photon energy used in Fortran simulations (in Hartree)
        self.g_omega_IR = g_omega_IR

    def load_hole(
        self,
        abs_emi_or_both,
        n_qn,
        hole_kappa,
        path_to_data,
        path_to_matrix_elements_emi=None,
        path_to_matrix_elements_abs=None,
        path_to_omega=None,
        binding_energy=None,
        path_to_hf_energies=None,
        path_to_sp_ekin=None,
        should_reload=False,
        breakpoint_step=5,
        breakpoint_use=3,
    ):
        """
        Initializes hole for absorption or emission or both paths, initializes
        corresponding ionization paths and loads data for them.

        Params:
        abs_emi_or_both - tells if we load matrix elements for absorption, emission or both paths,
        can take only 'abs', 'emi' or 'both' values.
        n_qn - principal quantum number of the hole
        hole_kappa - kappa value of the hole
        path_to_data - path to the output folder with Fortran simulation results
        path_to_matrix_elements_emi - path to the file containing matrix elements for emission
        path. NOTE: must be specified if abs_emi_or_both is 'emi' or 'both'!!
        path_to_matrix_elements_abs - path to the file containing matrix elements for absorption
        path. NOTE: must be specified if abs_emi_or_both is 'abs' or 'both'!!
        path_to_omega - path to the omega.dat file with XUV photon energies for the given hole
        (usually in pert folders). If not specified, constructed for path_to_data
        binding_energy - binding energy for the hole. Allows you to specify the predifined
        value for the hole's binding energy instead of loading it from the simulation data.
        path_to_hf_energies - path to the file with Hartree Fock energies for the given hole.
        If not specified, constructed for path_to_data
        path_to_sp_ekin - path to the file with kinetic energies for the given hole from
        secondphoton folder. If not specified, constructed for path_to_data
        should_reload - tells whether we should reload if the hole was previously
        loaded
        """

        self.assert_paths_for_loading(
            abs_emi_or_both, path_to_matrix_elements_emi, path_to_matrix_elements_abs
        )

        if abs_emi_or_both == "both":
            is_loaded = self.is_hole_loaded(
                "abs", n_qn, hole_kappa
            ) and self.is_hole_loaded("emi", n_qn, hole_kappa)

        else:
            is_loaded = self.is_hole_loaded(abs_emi_or_both, n_qn, hole_kappa)

        if not is_loaded or should_reload:
            hole = Hole(
                self.atom_name, hole_kappa, n_qn, binding_energy=binding_energy
            )  # initialize hole object

            if is_loaded and should_reload:
                message_string = (
                    "both"
                    if abs_emi_or_both == "both"
                    else convert_abs_or_emi_to_string(abs_emi_or_both)
                )
                print(f"Reload {hole.name} hole for {message_string } path!")

            # If the path to the omega file was not specified we assume that it is in the
            # pert folder.
            if path_to_omega is None:
                path_to_omega = (
                    path_to_data
                    + f"pert_{hole.kappa}_{hole.n - hole.l}"
                    + os.path.sep
                    + "omega.dat"
                )

            if (
                not binding_energy
            ):  # if the value for binding energy hasn't been provided - load it from data

                # if paths to the HF and kinetic energies are not specified, we assume
                # that they are in the seconphoton folder
                if path_to_hf_energies is None:
                    path_to_hf_energies = (
                        path_to_data
                        + "hf_wavefunctions"
                        + os.path.sep
                        + f"hf_energies_kappa_{hole.kappa}.dat"
                    )

                if path_to_sp_ekin is None:
                    path_to_sp_ekin = (
                        path_to_data
                        + "second_photon"
                        + os.path.sep
                        + f"energy_rpa_{hole.kappa}_{hole.n - hole.l}.dat"
                    )

                hole._load_binding_energy(
                    path_to_hf_energies,
                    path_to_omega=path_to_omega,
                    path_to_sp_ekin=path_to_sp_ekin,
                )

            # load data for ionization channels
            if abs_emi_or_both == "abs" or abs_emi_or_both == "both":
                self.__channels_abs[(n_qn, hole_kappa)] = Channels(
                    path_to_omega,
                    path_to_matrix_elements_abs,
                    hole,
                    "abs",
                    breakpoint_step,
                    breakpoint_use,
                )
            if abs_emi_or_both == "emi" or abs_emi_or_both == "both":
                self.__channels_emi[(n_qn, hole_kappa)] = Channels(
                    path_to_omega,
                    path_to_matrix_elements_emi,
                    hole,
                    "emi",
                    breakpoint_step,
                    breakpoint_use,
                )

    @staticmethod
    def assert_paths_for_loading(
        abs_emi_or_both, path_to_matrix_elements_emi, path_to_matrix_elements_abs
    ):
        """
        Asserts if the paths to matrix elements were correctly specified.
        If abs_emi_or_both is "abs" checks if path_to_matrix_elements_abs is not None.
        If abs_emi_or_both is "emi" checks if path_to_matrix_elements_emi is not None.
        If abs_emi_or_both is "both" checks if boths paths are not None.

        Params:
        abs_emi_or_both - tells if we load matrix elements for absorption, emission or both paths,
        can take only 'abs', 'emi' or 'both' values.
        path_to_matrix_elements_emi - path to the file containing matrix elements for emission
        path
        path_to_matrix_elements_abs - path to the file containing matrix elements for absorption
        path
        """
        assert abs_emi_or_both in (
            "abs",
            "emi",
            "both",
        ), "abs_emi_or_both parameter can only be 'abs', 'emi' or 'both'!"

        if abs_emi_or_both == "emi" or abs_emi_or_both == "both":
            assert (
                path_to_matrix_elements_emi
            ), "Please, specify the path to emission matrix elements!"

        if abs_emi_or_both == "abs" or abs_emi_or_both == "both":
            assert (
                path_to_matrix_elements_abs
            ), "Please, specify the path to absorption matrix elements!"

    def is_hole_loaded(self, abs_or_emi, n_qn, hole_kappa):
        """
        Checks if the hole is loaded for the given abs_or_emi path.

        Params:
        abs_or_emi - tells if we want to check for absorption or emission path,
        can take only 'abs' or 'emi' values.
        n_qn - principal quantum number of the hole
        hole_kappa - kappa value of the hole

        Returns:
        True if loaded, False otherwise.
        """

        assert_abs_or_emi(abs_or_emi)

        if abs_or_emi == "abs":
            return (n_qn, hole_kappa) in self.__channels_abs

        elif abs_or_emi == "emi":
            return (n_qn, hole_kappa) in self.__channels_emi

    def assert_hole_load(self, abs_or_emi, n_qn, hole_kappa):
        """
        Assertion that the hole was loaded for the given path (abs_or_emi).

        Params:
        abs_or_emi - tells if we want to check for absorption or emission path,
        can take only 'abs' or 'emi' values.
        hole - object of the Hole class containing hole's parameters
        n_qn - principal quantum number of the hole
        hole_kappa - kappa value of the hole
        """
        assert_abs_or_emi(abs_or_emi)

        assert self.is_hole_loaded(
            abs_or_emi, n_qn, hole_kappa
        ), f"The {construct_hole_name(self.atom_name, n_qn, hole_kappa)} hole is not loaded for {convert_abs_or_emi_to_string(abs_or_emi)} path!"

    def get_channels_for_hole(self, abs_or_emi, n_qn, hole_kappa):
        """
        Returns abosrption/emission ionization channels for the given hole.

        Params:
        abs_or_emi - tells if we want to get for absorption or emission path,
        can take only 'abs' or 'emi' values.
        n_qn - principal quantum number of the hole
        hole_kappa - kappa value of the hole

        Returns:
        channels - channels for the given hole and path
        """
        assert_abs_or_emi(abs_or_emi)

        self.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

        if abs_or_emi == "abs":
            channels = self.__channels_abs[(n_qn, hole_kappa)]

        elif abs_or_emi == "emi":
            channels = self.__channels_emi[(n_qn, hole_kappa)]

        return channels

    def get_all_channels(self, abs_or_emi):
        """
        Returns abosrption/emission ionization channels for all loaded holes.

        Params:
        abs_or_emi - tells if we want to get for absorption or emission path,
        can take only 'abs' or 'emi' values.

        Returns:
        loaded absosrption/emission channels for all holes
        """
        assert_abs_or_emi(abs_or_emi)

        if abs_or_emi == "abs":
            return self.__channels_abs

        elif abs_or_emi == "emi":
            return self.__channels_emi

    def get_channel_labels_for_hole(self, abs_or_emi, n_qn, hole_kappa):
        """
        Constructs labels for all ionization channels of the given hole and the
        given path (abs_or_emi).

        Params:
        abs_or_emi - tells if we want to get for absorption or emission path,
        can take only 'abs' or 'emi' values.
        n_qn - principal quantum number of the hole
        hole_kappa - kappa value of the hole

        Returns:
        channel_labels - list with labels of all ionization channels
        """

        assert_abs_or_emi(abs_or_emi)
        self.assert_hole_load(abs_or_emi, n_qn, hole_kappa)

        channel_labels = []
        channels = self.get_channels_for_hole(abs_or_emi, n_qn, hole_kappa)
        hole = channels.get_hole_object()
        hole_name = hole.name
        ionisation_paths = channels.get_all_ionisation_paths()
        for key_tuple in ionisation_paths.keys():
            ionisation_path = ionisation_paths[key_tuple]
            channel_labels.append(hole_name + " to " + ionisation_path.name)

        return channel_labels
