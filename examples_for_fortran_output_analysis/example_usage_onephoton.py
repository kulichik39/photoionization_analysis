import sys
import numpy as np
from matplotlib import pyplot as plt

"""
This code will help you understand how to use OnePhoton class and different onephoton_...  
namespaces to proccess Fortran output data for the one photon case.

At first, we need to import modules from this repository. We have two possible ways
to do this:
1. Add __init__.py file to this folder as well as all the folders from which we import code 
and then run this script using "python" command with "-m" flag (run module as a script). 
The full command is: "python -m examples_for_fortran_output_analysis.example_usage_onephoton".
2. Add this repository to our system path through sys.path.append("path/to/this/repo")
and run this script as usual.

In this example, we'll use the second approach.
"""

# append this repository to our system path
repo_path = "D:\\photoionization_analysis"
sys.path.append(repo_path)

# now we can easily import our OnePhoton class
from fortran_output_analysis.onephoton.onephoton import OnePhoton
from fortran_output_analysis.onephoton.onephoton import Channels

# we also import some physical constants required for the analysis
from fortran_output_analysis.constants_and_parameters import g_eV_per_Hartree


# ============== Initialization with OnePhoton ==============
"""
The main purpose of the OnePhoton class is to initialize the atom, the holes we want to 
consider ionization from and load the raw Fortran output data. 

In this guide we'll consider ionization from the Radon's 6p_3/2 and 6p_1/2 holes. 
"""
# create an instance
g_omega_IR = (
    1.55 / g_eV_per_Hartree
)  # energy of IR photon used in simulations (in Hartree)
atom_name = "Radon"
one_photon = OnePhoton(atom_name)

# specify path to Fortran output data (I use backslash for the path since I work on Windows)
path_to_data = "fortran_data\\2_-4_64_radon\\"

"""
To load diagonal matrix elements and diagonal eigenvalues (required for e.g. photoabsorption 
cross section) we use "load_diag_data" method inside the one_photon object. The method contains 
"should_reload" parameter that tells whether we should reinitalize diagonal data if they were
previously loaded (False by default).
"""

# load diagonal data
one_photon.load_diag_data(path_to_data)

# try to reload diagonal data (outputs information message)
one_photon.load_diag_data(path_to_data, should_reload=True)


"""
To load a hole we call a method named "load_hole" inside the one_photon object. 
The Fortran program outputs probability current for ionisation from a hole through 
the set of possible ionisation paths (channels). So, when we load a hole we also add all these 
"channels". Furthermore, "load_hole" method tries to find binding energy for the hole (if not 
specified) based on Hartree Fock energies or electron kinetic energies from the secondphoton 
folder. "load_hole" method also contains "should_reload" parameter that tells wheter we 
should reinitalize a hole if that hole was previously loaded (False by default).

The data for loaded holes are stored in the self.__channels dictionary attribute of the 
one_photon object and can be accessed via the "get_channel_for_hole" method.

Finally, I should say that many paths parameters in the "load_hole" method are kept as 
optional parameters (e.g. path_to_pcur_all, path_to_omega, path_to_hf_energies, 
path_to_sp_ekin). If some of them are not specified (None), those will be consturcted 
from "path_to_data" mandatory parameter. However, if you specify THEM ALL, then you may
not need "path_to_data", and you can provide any garbage value to it. Since situations 
when a user specifies all the required paths are very rare,I decided to keep 
"path_to_data" as mandatory parameter with an option to put any garbage value if you 
don't really need it.
"""

# load 6p_3/2 to one_photon object
hole_kappa_6p3half = -2
hole_n_6p3half = 6
one_photon.load_hole(hole_n_6p3half, hole_kappa_6p3half, path_to_data)

# We can get the labels for all possible inonization channels from 6p_3/2 hole:
labels_from_6p3half = one_photon.get_channel_labels_for_hole(
    hole_n_6p3half, hole_kappa_6p3half
)
print(f"\nPossible channels for Radon 6p_3/2: {labels_from_6p3half}\n")

# We can print binding energy for 6p_3/2
channels_6p3half: Channels = one_photon.get_channels_for_hole(
    hole_n_6p3half, hole_kappa_6p3half
)
hole_6p3half = channels_6p3half.get_hole_object()
print(f"\nBinding energy for Radon 6p_3/2 is {hole_6p3half.binding_energy}\n")


# load 6p_1/2 hole to one_photon object
hole_kappa_6p1half = 1
hole_n_6p1half = 6
one_photon.load_hole(hole_n_6p1half, hole_kappa_6p1half, path_to_data)

# We can get the labels for all possible inonization channels from 6p_1/2 hole:
labels_from_6p1half = one_photon.get_channel_labels_for_hole(
    hole_n_6p1half, hole_kappa_6p1half
)
print(f"\nPossible channels for Radon 6p_1/2: {labels_from_6p1half}\n")

# We can print binding energy for 6p_1/2
channels_6p1half: Channels = one_photon.get_channels_for_hole(
    hole_n_6p1half, hole_kappa_6p1half
)
hole_6p1half = channels_6p1half.get_hole_object()
print(f"\nBinding energy for Radon 6p_1/2 is {hole_6p1half.binding_energy}\n")

# try to reload 6p_1/2 hole with the same data (outputs information message)
one_photon.load_hole(
    hole_n_6p1half,
    hole_kappa_6p1half,
    path_to_data,
    should_reload=True,
)

# ============== Analysis with onephoton_... namespaces ==============
"""
onephoton folder contains different namespaces with functions that analyse raw output 
Fortran data and obtain meaningful physical properties. There is a namespace for 
cross sections, a namespace for asymmetry parameters, etc.

Almost all functions in these namespaces require an object of OnePhoton class with some
loaded holes as input.
"""

# 1. Photon and photoelctron kinetic energies with onephoton_utilities.
"""
It's usually important to check for which XUV photon/photoelectron energies our data 
were computed in the Fortran simulations. We can easily do this by calling the methods 
from onephoton_utilities namespace shown below:
"""
import fortran_output_analysis.onephoton.onephoton_utilities as util

# energies for 6p_3/2 (similarly for 6p_1/2)

# XUV photon energies in eV
en = util.get_omega_eV(one_photon, hole_n_6p3half, hole_kappa_6p3half)

# XUV photon energies in Hartree
en = util.get_omega_Hartree(one_photon, hole_n_6p3half, hole_kappa_6p3half)

# photoelectron kinetic energies in eV
en = util.get_electron_kinetic_energy_eV(one_photon, hole_n_6p3half, hole_kappa_6p3half)

# photoelectron kinetic energies in Hartree
en = util.get_electron_kinetic_energy_Hartree(
    one_photon, hole_n_6p3half, hole_kappa_6p3half
)

"""
We can also get matrix elements with added Coulomb phase for all the final states 
of the loaded hole.
"""
Z = 1  # charge of the ion

# matrix elements for 6p_{3/2}
M = util.get_matrix_elements_with_coulomb_phase(
    one_photon, hole_n_6p3half, hole_kappa_6p3half, Z
)

# 2. Integrated cross sections with onephoton_cross_sections
"""
Usually we want to look at the integrated (over all angles) photoionisation 
cross sections after absorption of the XUV photon. All the required methods are 
contained in onephoton_cross_sections namespace.
We can calculate cross sections from probability current using mode="pcur" parameter in
the functions and from matrix amplitudes using mode="amp".

# NOTE: All these methods for cross sections below also return the corresponding
# photoelectron kinetic energies in eV.
# """
import fortran_output_analysis.onephoton.onephoton_cross_sections as cs

# partial integrated cross section for the ionziation 6p_3/2 -> d_3/2
# calculated in two ways: with probability current or with matrix amplitudes
kappa_d3half = 2
en, cs_pcur = cs.get_partial_integrated_cross_section_1_channel(
    one_photon, hole_n_6p3half, hole_kappa_6p3half, kappa_d3half, mode="pcur"
)
en, cs_amp = cs.get_partial_integrated_cross_section_1_channel(
    one_photon, hole_n_6p3half, hole_kappa_6p3half, kappa_d3half, mode="amp"
)

plt.figure("Partial integrated crossection for 6p_3/2 -> d_3/2 ")
plt.plot(en, cs_pcur, label="pcur")
plt.plot(en, cs_amp, "--", label="amp")
plt.legend()
plt.title("Partial integrated crossection for 6p_3/2 -> d_3/2 ")

# partial integrated cross section for two channels: 6p_3/2 -> d_3/2 and 6p_3/2 -> d_5/2
kappa_d5half = -3
final_kappas = [kappa_d3half, kappa_d5half]
en, cs_pcur = cs.get_partial_integrated_cross_section_multiple_channels(
    one_photon, hole_n_6p3half, hole_kappa_6p3half, final_kappas, mode="pcur"
)
en, cs_amp = cs.get_partial_integrated_cross_section_multiple_channels(
    one_photon, hole_n_6p3half, hole_kappa_6p3half, final_kappas, mode="amp"
)
plt.figure(
    "Partial integrated crossection for two channels: 6p_3/2 -> d_3/2 and 6p_3/2 -> d_5/2"
)
plt.plot(en, cs_pcur, label="pcur")
plt.plot(en, cs_amp, "--", label="amp")
plt.legend()
plt.title(
    "Partial integrated crossection for two channels: 6p_3/2 -> d_3/2 and 6p_3/2 -> d_5/2"
)

# total integrated cross section for 6p_3/2
en, cs_pcur = cs.get_total_integrated_cross_section_for_hole(
    one_photon, hole_n_6p3half, hole_kappa_6p3half, mode="pcur"
)
en, cs_amp = cs.get_total_integrated_cross_section_for_hole(
    one_photon, hole_n_6p3half, hole_kappa_6p3half, mode="amp"
)
plt.figure("Total integrated crossection for hole 6p_3/2")
plt.plot(en, cs_pcur, label="pcur")
plt.plot(en, cs_amp, "--", label="amp")
plt.title(f"Total integrated crossection for hole 6p_3/2")

# Integrated photoelectron emission cross section. Can be computed in two energy modes:
# 1. "omega" mode when we just compute the sum of cross sections for all loaded holes
# and return the result for photon energies.
# 2. "ekin" mode when we compute cross sections for electron kinetic energies (which
# are different for different holes) and then interpolate them so that they match the
# same final kinetic energies.

# "omega" mode
omega, cs_pcur = cs.get_integrated_photoelectron_emission_cross_section(
    one_photon, mode_energies="omega", mode_cs="pcur"
)
omega, cs_amp = cs.get_integrated_photoelectron_emission_cross_section(
    one_photon, mode_energies="omega", mode_cs="amp"
)
plt.figure("Integrated photonelectron emission cross section for photon energies")
plt.plot(omega, cs_pcur, label="pcur")
plt.plot(omega, cs_amp, "--", label="amp")
plt.legend()
plt.title(f"Integrated photonelectron emission cross section for photon energies")

# "ekin" mode
ekin, cs_pcur = cs.get_integrated_photoelectron_emission_cross_section(
    one_photon, mode_energies="ekin", mode_cs="pcur"
)

ekin, cs_amp = cs.get_integrated_photoelectron_emission_cross_section(
    one_photon, mode_energies="ekin", mode_cs="amp"
)

plt.figure(
    "Integrated photonelectron emission cross section for photoelectron energies"
)
plt.plot(ekin, cs_pcur, label="pcur")
plt.plot(ekin, cs_amp, "--", label="amp")
plt.legend()
plt.title("Integrated photonelectron emission cross section for photoelectron energies")

# 3. Photoabsorption cross section with onephoton_cross_sections
"""
Photoabsorption cross section is slightly different from the cross sections shown above.
We need diagonal data (eigenvalues and matrix elements) for computations, so be sure 
that they are loaded (check the initialization section). We aslo must provide array
of photon energies (in eV) for which we want to compute the cross section.
"""

# Photoabsorption cross section. Requires diagonal eigenvalues and matrix elements to
# be loaded.
omega = util.get_omega_eV(one_photon, hole_n_6p3half, hole_kappa_6p3half)
cs_photoabs = cs.get_photoabsorption_cross_section(one_photon, omega)

plt.figure("Photoabsorption cross section")
plt.plot(omega, cs_photoabs)
plt.title("Photoabsorption cross section")

# 4. Angular part of a hole's cross section
"""
For in depth analysis of ionization from a hole, we should also consider angular part 
of its total cross section. The angular part is computed thorugh the real asymmetry 
parameter. The methods for asymmetry parameters are contained in 
onephoton_asymmetry_parameter namespace.

# NOTE: All methods also return corresponding photoelectron energy in eV.
# """
import fortran_output_analysis.onephoton.onephoton_asymmetry_parameters as asym_p

# specify angles to compute angular part of the hole's cross section
angles = np.array([0, 30, 45, 60])

# We also need ion charge which is usually just 1
Z = 1

# compute real asymmetry parameter
en, b2_real = asym_p.get_real_asymmetry_parameter(
    one_photon, hole_n_6p3half, hole_kappa_6p3half, Z
)
plt.figure("Real asymmetry parameter")
plt.plot(en, b2_real)
plt.title("Real asymmetry parameter")

# angular part of the cross section for 6p_3/2
ekin, ang_cs = cs.get_angular_part_of_cross_section(
    one_photon, hole_n_6p3half, hole_kappa_6p3half, Z, angles
)
plt.figure("Angular part of cross section for 6p_3/2")
for i in range(len(angles)):
    plt.plot(ekin, ang_cs[i, :], label=f"{angles[i]}")
plt.title("Angular part of cross section for 6p_3/2")
plt.legend()

# total cross section for 6p_3/2 (angular + integrated parts)
ekin, total_cs = cs.get_total_cross_section_for_hole(
    one_photon, hole_n_6p3half, hole_kappa_6p3half, Z, angles
)
plt.figure("Total cross section for 6p_3/2")
for i in range(len(angles)):
    plt.plot(ekin, total_cs[i, :], label=f"{angles[i]}")
plt.title("Total cross section for 6p_3/2")
plt.legend()

# 5. Wigner delay and phases
"""
The last but not least property we usually want to investigate in the one photon case 
is the Wigner delay (phase). And we actually want to consider both: integrated and 
angular parts.
The integrated part is computed from the so-called "Wigner intensity". The angular part 
is computed through the complex asymmetry parameter. onephoton_delays_and_phases 
namespace includes all the necessary methods for such computations for both delay and 
phase.
The usage examples are shown below.

NOTE: all the method below also return corresponding photoelectron kinetic energies in 
eV.
"""
import fortran_output_analysis.onephoton.onephoton_delays_and_phases as wigner

# specify angles to compute angular part of the Wigner delay
angles = np.array([0, 30, 45, 60])
# angles = np.array([0])

# We also need ion charge which is usually just 1
Z = 1

# complex asymmetry parameter
en, b2_complex = asym_p.get_complex_asymmetry_parameter(
    one_photon,
    hole_n_6p3half,
    hole_kappa_6p3half,
    Z,
    g_omega_IR,
)
# integrated Wigner delay for 6p_3/2
en, delay = wigner.get_integrated_wigner_delay(
    one_photon,
    hole_n_6p3half,
    hole_kappa_6p3half,
    Z,
    g_omega_IR,
)
plt.figure("Integrated Wigner delay for 6p_3/2")
plt.plot(en, delay)
plt.title("Integrated Wigner delay for 6p_3/2")

# integrated Wigner phase for 6p_3/2
en, phase = wigner.get_integrated_wigner_phase(
    one_photon,
    hole_n_6p3half,
    hole_kappa_6p3half,
    Z,
    g_omega_IR,
)
plt.figure("Integrated Wigner phase for 6p_3/2")
plt.plot(en, phase)
plt.title("Integrated Wigner phase for 6p_3/2")

# angular part of Wigner delay for 6p_3/2
en, delay = wigner.get_angular_wigner_delay(
    one_photon,
    hole_n_6p3half,
    hole_kappa_6p3half,
    Z,
    g_omega_IR,
    angles,
)
plt.figure("Angular part of Wigner delay for 6p_3/2")
for i in range(len(angles)):
    plt.plot(en, delay[i, :], label=f"{angles[i]}")
plt.legend()
plt.title("Angular part of Wigner delay for 6p_3/2")

# angular part of Wigner phase for 6p_3/2
plt.figure("Angular part of Wigner phase for 6p_3/2")
en, phase = wigner.get_angular_wigner_phase(
    one_photon,
    hole_n_6p3half,
    hole_kappa_6p3half,
    Z,
    g_omega_IR,
    angles,
)
for i in range(len(angles)):
    plt.plot(en, phase[i, :], label=f"{angles[i]}")
plt.legend()
plt.title("Angular part of Wigner phase for 6p_3/2")

# Total (integrated + angular) Wigner delay for 6p_3/2
en, delay = wigner.get_wigner_delay(
    one_photon,
    hole_n_6p3half,
    hole_kappa_6p3half,
    Z,
    g_omega_IR,
    angles,
)
plt.figure("Total wigner delay")
for i in range(len(angles)):
    plt.plot(
        en,
        delay[i, :],
        label=f"{angles[i]}",
    )
plt.legend()
plt.title("Total wigner delay for 6p_3/2")

# Total (integrated + angular) Wigner phase for 6p_3/2
en, phase = wigner.get_wigner_phase(
    one_photon,
    hole_n_6p3half,
    hole_kappa_6p3half,
    Z,
    g_omega_IR,
    angles,
)
plt.figure("Total wigner phase")
for i in range(len(angles)):
    plt.plot(en, phase[i, :], label=f"{angles[i]}")
plt.legend()
plt.title("Total wigner phase for 6p_3/2")

"""
Optional: "steps_per_IR_photon" gives the XUV step size (as g_omega_IR/steps_per_IR_photon).
NOTE(Leon): What does this mean in the context of non-linear energy grids?

If no value is given, it is calculated from "omega.dat".
"""
plt.show()
input()
