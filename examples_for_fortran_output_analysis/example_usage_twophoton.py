import sys
import os
import numpy as np
from matplotlib import pyplot as plt

"""
This code will help you understand how to use TwoPhotons class and different twophotons_...  
namespaces to proccess Fortran output data for the two photons case.

At first, we need to import modules from this repository. We have two possible ways
to do this:
1. Add __init__.py file to this folder as well as all the folders from which we import code 
and then run this script using "python" command with "-m" flag (run module as a script). 
The full command is: "python -m examples_for_fortran_output_analysis.example_usage_twophoton".
2. Add this repository to our system path through sys.path.append("path/to/this/repo")
and run this script as usual.

In this example, we'll use the second approach.
"""

# append this repository to our system path
repo_path = "D:\\photoionization_analysis"
sys.path.append(repo_path)

# now we can easily import our TwoPhotons class
from fortran_output_analysis.twophotons.twophotons import TwoPhotons

# we also import some physical constants required for the analysis
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree


# ============== Initialization with TwoPhotons ==============
"""
The main purpose of the TwoPhotons class is to initialize the atom, the holes we want to 
consider ionization from and load the raw Fortran output data for them. 

In this guide we'll consider ionization from the Radon's 6p_3/2. 
"""
# create an instance
g_omega_IR = (
    1.55 / g_eV_per_Hartree
)  # energy of IR photon used in simulations (in Hartree)
atom_name = "Radon"
two_photons = TwoPhotons(atom_name, g_omega_IR)

# specify path to Fortran output data (I use backslash for the path since I work on Windows)
path_to_data = "fortran_data\\2_-4_64_radon\\"


"""
To load a hole we call a method named "load_hole" inside the two_photons object. 
The Fortran program outputs complex matrix elements for ionisation from a hole through 
the set of possible ionisation paths (channels). So, when we load a hole we also add all these 
"channels". Furthermore, "load_hole" method tries to find binding energy for the hole (if not 
specified) based on Hartree Fock energies or electron kinetic energies from the secondphoton 
folder. "load_hole" method also contains "should_reload" parameter that tells wheter we 
should reinitalize a hole if that hole was previously loaded (False by default).

The data for loaded holes are stored in the self.__channels dictionary attribute of the 
two_photons object and can be accessed via the "get_channel_for_hole" method.

NOTE: When we load data in the two photons case, we should provide three things: 
1. "path_to_data" - path to fortran output folder. This is needed to grab XUV photon energies data
and Hartree Fock energies or electron kinetic energies to load hole's binding energy 
(if not specified).
2. "path_to_matrix_elements_emi" or "path_to_matrix_elements_abs" or both - paths to the complex 
matrix elements. "abs_emi_or_both" parameter in "load_hole" method determines if we want to load
for absorption only (abs_emi_or_both="abs"). emission only (abs_emi_or_both="emi") or both 
(abs_emi_or_both="both"). If we want to load for absorption/emission only then we need to specify
only one corresponding path, if we want to load for both we must specify both paths, otherwise,
we'll get an assertion error.
3. "path_to_phase_emi" or "path_to_phase_abs" or both - similar to the point 2 above but these
are files containing phases.

Finally, I should say that many paths parameters in the "load_hole" method are kept as optional 
parameters (e.g. path_to_omega, path_to_hf_energies, path_to_sp_ekin). If some of them are not 
specified (None), those will be consturcted from "path_to_data". However, if you specify THEM ALL, 
then you may not need "path_to_data", and you can provide any garbage value to it. Since 
situations when a user specifies all the required paths are very rare, I decided to keep 
"path_to_data" as mandatory parameter with an option to put any garbage value if you don't really
need it.
"""

# load 6p_3/2 to two_photons object
hole_kappa_6p3half = -2
hole_n_6p3half = 6

path_to_matrix_emi = (
    path_to_data + "second_photon" + os.path.sep + "m_elements_eF1_-2_5.dat"
)
path_to_matrix_abs = (
    path_to_data + "second_photon" + os.path.sep + "m_elements_eF2_-2_5.dat"
)

path_to_phases_emi = path_to_data + "second_photon" + os.path.sep + "phase_eF1_-2_5.dat"
path_to_phases_abs = path_to_data + "second_photon" + os.path.sep + "phase_eF2_-2_5.dat"

two_photons.load_hole(
    "both",
    hole_n_6p3half,
    hole_kappa_6p3half,
    path_to_data,
    path_to_matrix_elements_emi=path_to_matrix_emi,  # must specify both paths to matrix elements
    path_to_matrix_elements_abs=path_to_matrix_abs,
    path_to_phases_emi=path_to_phases_emi,
    path_to_phases_abs=path_to_phases_abs,
)  # load for both path simultaneously

# try to realod hole with the same data, just for demonstration purposes
# we'll get information messages abot the reloading
two_photons.load_hole(
    "emi",
    hole_n_6p3half,
    hole_kappa_6p3half,
    path_to_data,
    path_to_matrix_elements_emi=path_to_matrix_emi,
    path_to_phases_emi=path_to_phases_emi,  # need to sepcify only one path to matrix elements
    should_reload=True,
)  # load for emission path
two_photons.load_hole(
    "abs",
    hole_n_6p3half,
    hole_kappa_6p3half,
    path_to_data,
    path_to_matrix_elements_abs=path_to_matrix_abs,
    path_to_phases_abs=path_to_phases_abs,  # need to sepcify only one path to matrix elements
    should_reload=True,
)  # load for absorption path

# We can get the labels for all possible inonization channels from 6p_3/2 hole:
labels_from_6p3half = two_photons.get_channel_labels_for_hole(
    "abs", hole_n_6p3half, hole_kappa_6p3half
)  # for absorption path
print(
    f"\n Possible ionisation channels for absorption path in Radon 6p_3/2: {labels_from_6p3half}\n"
)

# We can print binding energy for 6p_3/2
channels_6p3half = two_photons.get_channels_for_hole(
    "abs", hole_n_6p3half, hole_kappa_6p3half
)
hole_6p3half = channels_6p3half.get_hole_object()
print(f"\n Binding energy for Radon 6p_3/2 is {hole_6p3half.binding_energy}\n")


# NOTE: by analogy, you can add more holes (e.g. 6p_{1/2}) to the two_photons object

# channels = two_photons.get_channels_for_hole("abs", hole_n_6p3half, hole_kappa_6p3half)
# mat = channels.get_raw_matrix_elements_for_ionization_path(-1, -2)
# print(mat.shape)
# print(mat)
# phase = channels.get_raw_phase_for_ionization_path(-1, -2)
# print(phase.shape)
# print(phase)

# from fortran_output_analysis.twophotons.twophotons_utilities import (
#     get_matrix_elements_with_coulomb_phase,
#     get_coupled_matrix_elements_for_all_final_states,
# )

# Z = 1
# M = get_matrix_elements_with_coulomb_phase(
#     two_photons, "emi", hole_n_6p3half, hole_kappa_6p3half, Z
# )
# print(M.shape)
# print(M[-1])
