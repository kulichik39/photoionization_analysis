import numpy as np
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree
from fortran_output_analysis.common_utility import kappa_from_l_and_j, l_from_str

# NOTE(anton):
# 1 Mbarn = 1e-18 cm^2
# 1 a_0 = 5.29177210903 * 1e-9 cm.
# 1 a_0^2 = (5.29177210903)^2 * (1e-18) cm^2
# =>
# 1 Mbarn = (5.29177210903)^2 a_0^2
g_au_to_Mbarn = (0.529177210903) * (0.529177210903) * 100

# Resonance class
# This keeps q, value, epsilon etc for a given energy by user choice
# How this is calculated is shown in Lindroth PRA 1995 "Photodetachment of H- and Li-"
class Resonance:
    def __init__(self, eigval_idx, eigenvalue, matrix_element):
        self.C_const = (4.0*np.pi/3.0)*(1/137.035999074)*g_au_to_Mbarn
        self.eigenvalue_index = eigval_idx  # This is the index for the eigenvalue & matrix element for the resonance.
        self.eigenvalue = eigenvalue  # E + iGamma/2 complex number from diag_eigenvalues.dat.
        self.matrix_element = matrix_element  # Complex number from diag_matrix_elements.dat.
        self.R_k = np.real(self.matrix_element*self.matrix_element)
        self.I_k = np.imag(self.matrix_element*self.matrix_element)
        self.energy = np.real(self.eigenvalue)
        self.half_gamma = -np.imag(self.eigenvalue) # eigenvalue = E - iGamma/2 -> Gamma/2 = -imag(eigenvalue).
        self.b = compute_b_parameter(self.energy, self.R_k, self.I_k, self.half_gamma)
        self.q = compute_q_parameter(self.b)

    def compute_sigma0(self, q_idx=2):
        # We can either choose q or let the routines choose appropriate q according to
        # if I_k < 0, or I_k > 0.
        if q_idx >= 2:
            q_index = 0
            if self.I_k > 0:
                q_index = 1
        else:
            q_index = q_idx

        prefac = -(self.I_k*self.C_const)/(2.0*self.q[q_index])
        sigma0 = prefac*(self.energy/self.half_gamma-(self.R_k/self.I_k))

        return sigma0

    def get_fano_profile(self, omega, q_idx=2):
        # We can either choose q or let the routines choose appropriate q according to
        # if I_k < 0, or I_k > 0.
        if q_idx >= 2:
            q_index = 0
            if self.I_k > 0:
                q_index = 1
        else:
            q_index = q_idx

        epsilon = self.get_epsilon(omega)
        q = self.q[q_index]
        profile = ((q+epsilon)*(q+epsilon))/(1+epsilon*epsilon)

        return profile

    def get_epsilon(self, omega):
        epsilon = (omega-self.energy)/self.half_gamma
        return epsilon


# Start Channel
class Channel:
    def __init__(self):
        self.kappa_hole = 256
        self.kappa_final = 256
        self.n_hole = 256
        self.l_hole = 256
        self.j_hole = 256
        self.l_final = 256
        self.j_final = 256
        self.name = "NO_NAME"
        self.start_index = -1
        self.end_index = -1
        self.handle = -1

    def print_vars(self):
        print(vars(self))


#--- Start Diagonalisation class
class Diagonalisation:
    def __init__(self, name):
        self.name = name
        self.channels = []
        self.channels_dict = {}
        self.channels_initialised = False
        self.eigenvectors = np.array([0+0*1j])
        self.eigenvalues = np.array([0+0*1j])
        self.dipole_elements = np.array([0+0*1j])
        self.matrix_elements = np.array([0+0*1j])
        self.read_matrix_elements = False
        self.system_size = 0

    def load_diag_data_from_directory(self, data_dir):
        # Note that channel indices has to be read first.
        self.load_channel_data(data_dir+"diag_channel_indices.dat")
        self.load_eigenvalues(data_dir + "diag_eigenvalues.dat")
        self.load_matrix_elements(data_dir+"diag_matrix_elements.dat")

    def load_channel_data(self, path_to_channel_indices_file):
        self.channels, self.channels_dict = parse_channel_indices(path_to_channel_indices_file)
        self.channels_initialised = True

    def num_channels(self):
        return len(self.channels)

    def load_eigenvalues(self, path_to_eigenvalues_file):
        if(self.channels_initialised == False):
            raise ValueError("Diagonalisation(): Channels needs to be loaded before eigenvalues.")
        else:
            eigvals_fname = path_to_eigenvalues_file
            eigvals_raw = np.loadtxt(eigvals_fname)
            eigvals_re = eigvals_raw[:, 0]
            eigvals_im = eigvals_raw[:, 1]
            self.eigenvalues = np.zeros(len(eigvals_im), dtype=complex)
            self.eigenvalues = eigvals_re+1j*eigvals_im
            #print(np.real(self.eigenvalues).dtype) # check that this is actually complex double.

            print("Loaded %i complex eigenvalues from %s" % (len(eigvals_re), eigvals_fname))


    def load_eigenvectors(self, path_to_eigenvectors_file):
        if (self.channels_initialised == False):
            raise ValueError("Diagonalisation(): Channels needs to be loaded before eigenvectors.")
        else:
            eigvecs_fname = path_to_eigenvectors_file
            # NOTE(anton): Eigenvectors data has 2N columns,
            # corresponding to N complex numbers.
            # So the data is rows of (real,imag) pairs for each of the N eigenvalues.
            # The rows in a particular complex-number column (ie 2 cols in the raw data)
            # are all the coefficients for all channels (in the order explained by channel indices).
            eigvecs_raw = np.loadtxt(eigvecs_fname)
            eigvecs_re = eigvecs_raw[:, 0::2]  # Every other column starting from 0 is the real part
            eigvecs_im = eigvecs_raw[:, 1::2]  # Every other column starting from 1 is the imag part
            num_rows = eigvecs_re.shape[0]
            num_cols = eigvecs_re.shape[1]
            self.eigenvectors = np.zeros((num_rows, num_cols), dtype=complex)
            self.eigenvectors = eigvecs_re + eigvecs_im*1j

            print("Loaded %i x %i = %i complex elements for eigenvectors from %s" %
                  (num_rows, num_cols, num_cols*num_rows, eigvecs_fname))

            self.system_size = num_rows

    def load_dipole_elements(self, path_to_dipole_elems_file):
        if (self.channels_initialised == False):
            raise ValueError("Diagonalisation(): Channels needs to be loaded before dipole elements.")
        else:
            dip_elems_raw = np.loadtxt(path_to_dipole_elems_file)
            dip_elems_re = dip_elems_raw[:, 0]
            dip_elems_im = dip_elems_raw[:, 1]
            num_rows = dip_elems_raw.shape[0]
            self.dipole_elements = np.zeros(num_rows, dtype=complex)
            self.dipole_elements = dip_elems_re + 1j*dip_elems_im

            print("Loaded %i complex dipole elements from %s" %
                  (num_rows, path_to_dipole_elems_file))

    def load_matrix_elements(self, path_to_matrix_elems_file):
        if (self.channels_initialised == False):
            raise ValueError("Diagonalisation(): Channels needs to be loaded before dipole elements.")
        else:
            mat_elems_raw = np.loadtxt(path_to_matrix_elems_file)
            mat_elems_re = mat_elems_raw[:, 0]
            mat_elems_im = mat_elems_raw[:, 1]
            num_rows = mat_elems_raw.shape[0]
            self.matrix_elements = np.zeros(num_rows, dtype=complex)
            self.matrix_elements = mat_elems_re + 1j*mat_elems_im
            # Since the dipole transitions elements are defined with an overall extra i-factor in Fortran,
            # we just multiply by -i here again to remove that.
            # What is called "matrix element" M here is M_k = sum_i x_k^i * dipole_element_i,
            # where x_k^i are the coefficients in the array representing eigenvector x_k after diagonalisation.
            self.matrix_elements *= -1j

            print("Loaded %i complex matrix elements M_k = sum(psi_k * all_dipole) from %s" %
                  (num_rows, path_to_matrix_elems_file))

            self.read_matrix_elements = True

    def compute_matrix_elements(self):
        if (self.channels_initialised == False or len(self.dipole_elements) <= 1):
            raise ValueError("Diagonalisation(): Cannot compute matrix elements before data dependencies are loaded!")
        else:
            if(self.read_matrix_elements == True):
                raise ValueError("Diagonalisation(): Read matrix elements from file, do not recompute!")
            else:
                rhs = self.dipole_elements
                self.matrix_elements = np.zeros(self.system_size, dtype=complex)

                for k in range(self.system_size):
                     psi = self.eigenvectors[:, k]
                     self.matrix_elements[k] = np.sum(psi * rhs)

    def compute_absorption_cross_section(self, omega_list):
        N = len(omega_list)
        cs = np.zeros(N)
        for i in range(N):
            omega = omega_list[i]
            cs[i] = compute_diag_cross_section(self, omega)

        return cs





#-------------- End Diagonalisation

# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
#


def compute_b_parameter(E, R, I_k, half_gamma):
    E_by_halfGamma = E/half_gamma
    R_by_I = R/I_k
    enum = E_by_halfGamma*R_by_I+1.0
    denom = R_by_I - E_by_halfGamma
    b = enum/denom
    return b


def compute_q_parameter(b):
    q1 = b + np.sqrt(b*b + 1)
    q2 = b - np.sqrt(b*b + 1)

    return (q1, q2)


def compute_diag_cross_section(atom_system: Diagonalisation, photon_energy_ev):
    photon_energy_au = photon_energy_ev/g_eV_per_Hartree

    coeff_start = 0

    M = atom_system.matrix_elements[coeff_start:]
    E = atom_system.eigenvalues[coeff_start:]
    omega = (photon_energy_au+0*1j)
    # NOTE(anton): Since the matrix is constructed with an energy diagonal of
    # e_s - e_a (excited - ground), we need to subtract the photon energy here.
    # Compare to ie Jimmy's lic with a denominator (e_a - e_s + Omega)
    imag_term = np.sum(M*M/(E - omega))


    c_au = 137.035999074

    # NOTE(anton):
    # 1 Mbarn = 1e-18 cm^2
    # 1 a_0 = 5.29177210903 * 1e-9 cm.
    # 1 a_0^2 = (5.29177210903)^2 * (1e-18) cm^2
    # =>
    # 1 Mbarn = (5.29177210903)^2 a_0^2
    #au_to_Mbarn = (5.29177210903)*(5.29177210903)
    au_to_Mbarn = (0.529177210903) * (0.529177210903) * 100

    pi = np.pi

    # NOTE(anton): There used to be an unexplained overall factor -1 here,
    # but it comes from a definition of the dipole element with an extra i-factor in Fortran.
    # This resulted in a -1 factor when taking the square of M above.
    # Now the M are loaded with that i-factor removed, however!
    C = (4.0*pi/3.0)*(1.0/c_au)*au_to_Mbarn

    out_Mbarn = C*np.imag(imag_term)*np.real(omega)
    #out_Mbarn = C*np.imag(imag_term)*np.sqrt(np.real(omega))

    return out_Mbarn

def compute_diag_cross_section_single_element(atom_system: Diagonalisation, photon_energy_ev, element_index):
    photon_energy_au = photon_energy_ev/g_eV_per_Hartree

    i = element_index
    coeff_start = 0

    M = atom_system.matrix_elements[i]
    E = atom_system.eigenvalues[i]
    omega = (photon_energy_au+0*1j)
    # NOTE(anton): Since the matrix is constructed with an energy diagonal of
    # e_s - e_a (excited - ground), we need to subtract the photon energy here.
    # Compare to ie Jimmy's lic with a denominator (e_a - e_s + Omega)
    imag_term = M*M/(E - omega)


    c_au = 137.035999074

    # NOTE(anton):
    # 1 Mbarn = 1e-18 cm^2
    # 1 a_0 = 5.29177210903 * 1e-9 cm.
    # 1 a_0^2 = (5.29177210903)^2 * (1e-18) cm^2
    # =>
    # 1 Mbarn = (5.29177210903)^2 a_0^2
    # au_to_Mbarn = (5.29177210903)*(5.29177210903)
    au_to_Mbarn = (0.529177210903) * (0.529177210903) * 100

    pi = np.pi

    # TODO(anton): Is the minus sign in C (below) actually correct?
    # Ie is not the denominator -E+omega above, instead?
    # Fortran code seems to be consistent with (E-omega)...
    # But we are also having an overall minus in the old jimmy cross section calculations...
    C = -(4.0*pi/3.0)*(1.0/c_au)*au_to_Mbarn

    out_Mbarn = C*np.imag(imag_term)*np.real(omega)
    #out_Mbarn = C*np.imag(imag_term)*np.sqrt(np.real(omega))

    return out_Mbarn



# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
# Utility functions

def parse_channel_indices(path_to_file):
    channels_list = []
    channels_dict = {}

    num_channels = 0
    file = open(path_to_file, "r")

    for line in file:
        line = line.replace("\n","")
        # parse the # Channel row
        if(line.find("Channel") != -1):
            num_channels += 1
            line = line.split(":")
            channel = channel_from_string(num_channels-1, line[1])
            channels_list.append(channel)
            channels_dict[channel.name] = channel.handle
        else:
            # We check that we're not in a comment line (# start).
            # And we check that we're not on an empty line. (line.strip() != "")
            if(line.find("#") == -1):
                if(line.strip() != ""):
                    # Now we're on an index line.
                    line = line.split(",")
                    start_index = int(line[0])
                    end_index = int(line[1])
                    channels_list[num_channels-1].start_index = start_index
                    channels_list[num_channels-1].end_index = end_index

    file.close()

    return channels_list, channels_dict



def channel_from_string(handle_index, string):
    channel = Channel()
    channel.name = string
    channel.handle = handle_index

    string = string.split("->")
    hole_string = "".join(string[0].split())
    final_string = "".join(string[1].split())

    n = int(hole_string[0])
    channel.n_hole = n
    l_str = hole_string[1]
    l_hole = l_from_str(l_str)
    channel.l_hole = l_hole
    j_hole = int(hole_string[2])
    channel.j_hole = j_hole
    channel.kappa_hole = kappa_from_l_and_j(l_hole, j_hole)

    l_final = l_from_str(final_string[0])
    channel.l_final = l_final
    j_final = int(final_string[1])
    channel.j_final = j_final
    channel.kappa_final = kappa_from_l_and_j(l_final, j_final)

    return channel

# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
