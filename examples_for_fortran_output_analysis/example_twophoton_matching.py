import sys
import os
import numpy as np
from matplotlib import pyplot as plt


"""
Sometimes we may want to compute asymmetry parameters as well as atomic delay and phase 
based on the data from two different simultations: one for absorption path and one for 
emission path. In this scenario, we have to properly match matrices from both simulations 
and compute the asymmetry parameters/phase/delay on the interpolated values (matrix elements). 

The logic for handling this is included in the functions from the 
twophotons_asymmetry_parameters and twophotons_delays_and_phases namespaces. 
You just need to provide the second TwoPhoton object (corresponding to the second simulation) 
and a couple of additional parameters which are described below.

We'll show how it's supposed to work in the case of ionization of 5p_3/2 electron in 
Xenon. NOTE: all the detailed explanation for the case of one simulation only is shown 
in the example_usage_twophoton.py file.
"""

# append this repository to our system path to easily import all the necessary modules
sys.path.append(os.getcwd())


from fortran_output_analysis.twophotons.twophotons import TwoPhotons
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree


atom_name = "Xenon"
g_omega_IR = (
    1.55 / g_eV_per_Hartree
)  # energy of IR photon used in simulations (in Hartree)

two_photons_emi = TwoPhotons(
    atom_name, g_omega_IR
)  # object for the first simulation (emission path)
two_photons_abs = TwoPhotons(
    atom_name, g_omega_IR
)  # object for the second simulation (absorption path)


# initialize 5p_3/2 hole
hole_kappa_5p3half = -2
hole_n_5p3half = 5


# specify path to Fortran output data (I use backslash for the path since I work on
# Windows)
data_dir_emi = (
    "fortran_data\\xenon_emission\\"  # data for the first simulation (emission path)
)
data_dir_abs = "fortran_data\\xenon_absorption\\"  # data for the second simulation (absorption path)

path_to_matrix_emi = (
    data_dir_emi + "second_photon" + os.path.sep + "m_elements_eF1_-2_4.dat"
)
path_to_matrix_abs = (
    data_dir_abs + "second_photon" + os.path.sep + "m_elements_eF2_-2_4.dat"
)

path_to_phases_emi = data_dir_emi + "second_photon" + os.path.sep + "phase_eF1_-2_4.dat"
path_to_phases_abs = data_dir_abs + "second_photon" + os.path.sep + "phase_eF2_-2_4.dat"


# load hole for emission simulation
two_photons_emi.load_hole(
    "emi",
    hole_n_5p3half,
    hole_kappa_5p3half,
    data_dir_emi,
    path_to_matrix_elements_emi=path_to_matrix_emi,
    path_to_phases_emi=path_to_phases_emi,
)


# load hole for absorption simulation
two_photons_abs.load_hole(
    "abs",
    hole_n_5p3half,
    hole_kappa_5p3half,
    data_dir_abs,
    path_to_matrix_elements_abs=path_to_matrix_abs,
    path_to_phases_abs=path_to_phases_abs,
)

# get asymmetry parameters
import fortran_output_analysis.twophotons.twophotons_asymmetry_parameters as asym_p

Z = 1  # charge of the ion

# 2nd order complex asymmetry parameter
en, b2_complex = asym_p.get_complex_asymmetry_parameter(
    two_photons_emi,
    2,
    hole_n_5p3half,
    hole_kappa_5p3half,
    Z,
    two_photons_2=two_photons_abs,
    energies_mode="both",
)
plt.figure("Complex asymmetry parameter, 2nd order")
plt.plot(en, np.real(b2_complex), label="Real part")
plt.plot(en, np.imag(b2_complex), "--", label="Imaginary part")
plt.legend()
plt.title("Complex asymmetry parameter, 2nd order")

# 4th order complex asymmetry parameter
en, b4_complex = asym_p.get_complex_asymmetry_parameter(
    two_photons_emi,
    4,
    hole_n_5p3half,
    hole_kappa_5p3half,
    Z,
    two_photons_2=two_photons_abs,
    energies_mode="both",
)
plt.figure("Complex asymmetry parameter, 4th order")
plt.plot(en, np.real(b4_complex), label="Real part")
plt.plot(en, np.imag(b4_complex), "--", label="Imaginary part")
plt.legend()
plt.title("Complex asymmetry parameter, 4th order")


# get atomic phase and delay
import fortran_output_analysis.twophotons.twophotons_delays_and_phases as atomic

angles = np.array([0, 30, 60])  # angles to compute the total delay/phase for
Z = 1  # charge of the ion

en, delay = atomic.get_atomic_delay(
    two_photons_emi,
    hole_n_5p3half,
    hole_kappa_5p3half,
    Z,
    angles,
    two_photons_2=two_photons_abs,
    energies_mode="both",
)

plt.figure("Total atomic delay")
for i in range(len(angles)):
    plt.plot(en, delay[i, :], label=f"{angles[i]}")

plt.legend()
plt.title("Total atomic delay for 5p_3/2")


en, phase = atomic.get_atomic_phase(
    two_photons_emi,
    hole_n_5p3half,
    hole_kappa_5p3half,
    Z,
    angles,
    two_photons_2=two_photons_abs,
    energies_mode="both",
)

plt.figure("Total atomic phase")
for i in range(len(angles)):
    plt.plot(en, phase[i, :], label=f"{angles[i]}")

plt.legend()
plt.title("Total atomic phase for 5p_3/2")


plt.show()
input()


"""
The energies_mode parameter in the functions above specifies the array of energies 
that we use to match our simulations. Poissble options:
"both" (default) - concatenate energy arrays from both simulations
"abs" - take the energy array from the simulation for absorption path
"emi" - take the energy array from the simulation for emission path
"""
