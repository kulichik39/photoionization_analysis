import numpy as np
import matplotlib.pyplot as plt

# required for imports
import sys
import os

repo_path = "D:\\photoionization_analysis"
sys.path.append(repo_path)


# Constants and Parameters
from fortran_output_analysis.constants_and_parameters import (
    g_eV_per_Hartree,
)

g_omega_IR_eV = 1.55  # energy of the IR photon (in eV)
g_omega_IR_hart = (
    g_omega_IR_eV / g_eV_per_Hartree
)  # energy of the IR photon (in Hartree)
Z = 1  # charge of the ion


"""    
       #####################################
                  Load Raw Data
       #####################################
                
First of all, you should load raw simulation data. The OnePhoton and TwoPhoton classes provide the
interface for it.

As an example, I'll use two xenon 5p_{3/2} simulations: one for the emission path and one for 
the absorption path. The simulations are focused on the 5p_{1/2}8d and 5p_{1/2}10s resonances.
"""

from fortran_output_analysis.onephoton.onephoton import OnePhoton
from fortran_output_analysis.twophotons.twophotons import TwoPhotons

# parameters of the 5p_{3/2} hole
xenon_5p3h_kappa = -2
xenon_5p3h_n = 5

# data paths
data_path = repo_path + os.path.sep + "radon_xenon_paper" + os.path.sep
xenon_data_emi = data_path + "xenon_emi" + os.path.sep  # emission simulation
xenon_data_abs = data_path + "xenon_abs_8d_10s" + os.path.sep  # absorption simulation

# load the one photon (1ph) data
xenon_1ph_emi = OnePhoton("Xenon")  # object for the emission simulation
xenon_1ph_abs = OnePhoton("Xenon")  # object for the absorption simulation

xenon_1ph_emi.load_hole(xenon_5p3h_n, xenon_5p3h_kappa, xenon_data_emi)
xenon_1ph_abs.load_hole(xenon_5p3h_n, xenon_5p3h_kappa, xenon_data_abs)

# load the two photon (2ph) data
xenon_2ph_emi = TwoPhotons(
    "Xenon", g_omega_IR_hart
)  # object for the emission simulation
xenon_2ph_abs = TwoPhotons(
    "Xenon", g_omega_IR_hart
)  # object for the absorption simulation

path_to_xenon_2ph_emi = (
    xenon_data_emi + "second_photon" + os.path.sep
)  # 2ph emission data
path_to_xenon_2ph_abs = (
    xenon_data_abs + "second_photon" + os.path.sep
)  # 2ph absorption data

# files with the 2ph matrix elements
path_to_xenon_2ph_emi_mat = path_to_xenon_2ph_emi + "m_elements_eF1_-2_4.dat"
path_to_xenon_2ph_abs_mat = path_to_xenon_2ph_abs + "m_elements_eF1_-2_4.dat"
# files with the 2ph phases
path_to_xenon_2ph_emi_phase = path_to_xenon_2ph_emi + "phase_eF1_-2_4.dat"
path_to_xenon_2ph_abs_phase = path_to_xenon_2ph_abs + "phase_eF1_-2_4.dat"

xenon_2ph_emi.load_hole(
    "emi",
    xenon_5p3h_n,
    xenon_5p3h_kappa,
    xenon_data_emi,
    path_to_matrix_elements_emi=path_to_xenon_2ph_emi_mat,
    path_to_phases_emi=path_to_xenon_2ph_emi_phase,
)

xenon_2ph_abs.load_hole(
    "abs",
    xenon_5p3h_n,
    xenon_5p3h_kappa,
    xenon_data_abs,
    path_to_matrix_elements_abs=path_to_xenon_2ph_abs_mat,
    path_to_phases_abs=path_to_xenon_2ph_abs_phase,
)


"""
       #####################################
                 Matrix elements
       #####################################
              
Now, you need to get the matrix elements that include all the necessary stuff (like Non-Coulomb and 
Coulomb phase shifts) and that are matched to the final sideband energies. All this logic is 
incorporated in the get_prepared_matrices() functions (for the 1ph and 2ph cases).

From these matrix elements the qunatities of interest, like integrated phases and beta parameters, 
can be computed. 
"""

import fortran_output_analysis.onephoton.onephoton_utilities as utils_1ph
import fortran_output_analysis.twophotons.twophotons_utilities as utils_2ph

# 1ph matrix elements
ekin_eV, M_emi_1ph_matched, M_abs_1ph_matched = utils_1ph.get_prepared_matrices(
    xenon_1ph_emi,
    xenon_5p3h_n,
    xenon_5p3h_kappa,
    Z,
    g_omega_IR_hart,
    one_photon_2=xenon_1ph_abs,
    g_omega_IR_2=g_omega_IR_hart,
    energies_mode="abs",
    match_mode="interp_both",
)

# 2ph matrix elements
_, M_emi_2ph_matched, M_abs_2ph_matched = utils_2ph.get_prepared_matrices(
    xenon_2ph_emi,
    xenon_5p3h_n,
    xenon_5p3h_kappa,
    Z,
    two_photons_2=xenon_2ph_abs,
    energies_mode="abs",
    match_mode="interp_both",
)  # skip ekin_eV in the output here since it's the same as above


"""
       #####################################
               Necessary quantities 
       #####################################
            
To apply the complex q analysis in the angularly integrated case, you need to get the integrated
sideband interference term (two photons) or its Wigner contribution (one photon). 
In my code, the integrated one photon contribution can be obtained from the get_wigner_intensity()
function, while the integrated sideband interference term from the get_integrated_two_photons_intensity()
function.

For the angularly resolved case, you also need the complex beta parameters. They are calculated below
as well.
"""

import fortran_output_analysis.onephoton.onephoton_delays_and_phases as wigner
import fortran_output_analysis.twophotons.twophotons_delays_and_phases as atomic
import fortran_output_analysis.onephoton.onephoton_asymmetry_parameters as asym_p_1ph
import fortran_output_analysis.twophotons.twophotons_asymmetry_parameters as asym_p_2ph

# integrated Wigner contirbution to the sidebnad term (1ph)
int_sideband_1ph = wigner.get_wigner_intensity(
    xenon_5p3h_kappa, M_emi_1ph_matched, M_abs_1ph_matched
)
# integrated sideband term (2ph)
int_sideband_2ph = atomic.get_integrated_two_photons_intensity(
    xenon_5p3h_kappa, M_emi_2ph_matched, M_abs_2ph_matched
)
# NOTE: the Wigner phase is the phase of the 1ph sideband, the atomic phase is the phase of the 2ph
# sideband,

# complex asymmetry parameter for 1ph. In this case, only the second order parameter is non-zero.
b2_1ph, _ = asym_p_1ph.one_photon_asymmetry_parameter(
    xenon_5p3h_kappa, M_emi_1ph_matched, M_abs_1ph_matched, "cross"
)

# complex asymmetry parameters for 2ph. In this case, the second order and the fourth order
# parameters are non-zero.
b2_2ph, _ = asym_p_2ph.two_photons_asymmetry_parameter(
    2, xenon_5p3h_kappa, M_emi_2ph_matched, M_abs_2ph_matched, "cross"
)
b4_2ph, _ = asym_p_2ph.two_photons_asymmetry_parameter(
    4, xenon_5p3h_kappa, M_emi_2ph_matched, M_abs_2ph_matched, "cross"
)

"""
       #####################################
            Angularly Integrated Case 
       #####################################

To remind, we're considering the 8d and 10s resonances in xenon. To get the model coefficients, we 
should fit the integrated terms (1ph and 2ph) to the formula for each resonance separately. 
The code provides two options for the fitting:
1) Extract a range of the data specified by the min and max reduced energies (eps_min, eps_max).
   Use this range in the scipy fitting routine to get the coefficient. This way is shown below.

2) Use only two points (eps_1, eps_2) to estimate the coefficient. Can be called through the
   complex_q.get_coefficients_for_integrated() function. Not shown below.

NOTE: Both methods are highly sensitive to the choice of the two eps values.
"""

import fortran_output_analysis.complex_q_parameter as complex_q

# -------- 8d resonance --------

E_res_8d = 0.891  # resonance position, eV
width_res_8d = 0.0284  # resonance width, eV
q_8d = 1.52  # Fano shape parameter

# get the reduced energies
eps_8d = complex_q.get_epsilon(ekin_eV, "abs", g_omega_IR_eV, E_res_8d, width_res_8d)

# boundaries for the fitting
E_8d_min = 2.42
E_8d_max = 2.44
eps_8d_min = complex_q.get_epsilon(
    E_8d_min, "abs", g_omega_IR_eV, E_res_8d, width_res_8d
)
eps_8d_max = complex_q.get_epsilon(
    E_8d_max, "abs", g_omega_IR_eV, E_res_8d, width_res_8d
)

# fit to the 1ph data
A_bg_8d_int_1ph, A_res_8d_int_1ph, _ = complex_q.get_coefficients_for_integrated_by_fit(
    "abs", q_8d, int_sideband_1ph, eps_8d, eps_8d_min, eps_8d_max
)
# fit to the 2ph data
A_bg_8d_int_2ph, A_res_8d_int_2ph, _ = complex_q.get_coefficients_for_integrated_by_fit(
    "abs", q_8d, int_sideband_2ph, eps_8d, eps_8d_min, eps_8d_max
)

# get the complex q parameters
q_complex_8d_int_1ph = complex_q.get_complex_q_integrated(
    A_bg_8d_int_1ph, A_res_8d_int_1ph, q_8d
)
q_complex_8d_int_2ph = complex_q.get_complex_q_integrated(
    A_bg_8d_int_2ph, A_res_8d_int_2ph, q_8d
)

# report the computed coefficients/parameters
print("")
print("##### Angularly Integrated Case #####")
print("")
print("-- 8d resonance --")
print("")
print(f"1ph A_bg: {A_bg_8d_int_1ph}")
print(f"1ph A_res: {A_res_8d_int_1ph}")
print(f"1ph complex q: {q_complex_8d_int_1ph}")
print("")
print(f"2ph A_bg: {A_bg_8d_int_2ph}")
print(f"2ph A_res: {A_res_8d_int_2ph}")
print(f"2ph complex q: {q_complex_8d_int_2ph}")
print("")


# -------- 10s resonance --------

# 10s resonance parameters
E_res_10s = 0.930  # resonance position, eV
width_res_10s = 0.0008  # resonance width, eV
q_10s = -136.62  # Fano shape parameter

# get the reduced energy
eps_10s = complex_q.get_epsilon(ekin_eV, "abs", g_omega_IR_eV, E_res_10s, width_res_10s)

# boundaries for the fitting
E_10s_min = 2.474
E_10s_max = 2.485
eps_10s_min = complex_q.get_epsilon(
    E_10s_min, "abs", g_omega_IR_eV, E_res_10s, width_res_10s
)
eps_10s_max = complex_q.get_epsilon(
    E_10s_max, "abs", g_omega_IR_eV, E_res_10s, width_res_10s
)

# fit to the 1ph data
A_bg_10s_int_1ph, A_res_10s_int_1ph, _ = (
    complex_q.get_coefficients_for_integrated_by_fit(
        "abs", q_10s, int_sideband_1ph, eps_10s, eps_10s_min, eps_10s_max
    )
)
# fit to the 2ph data
A_bg_10s_int_2ph, A_res_10s_int_2ph, _ = (
    complex_q.get_coefficients_for_integrated_by_fit(
        "abs", q_10s, int_sideband_2ph, eps_10s, eps_10s_min, eps_10s_max
    )
)

# get the complex q parameters
q_complex_10s_int_1ph = complex_q.get_complex_q_integrated(
    A_bg_10s_int_1ph, A_res_10s_int_1ph, q_10s
)
q_complex_10s_int_2ph = complex_q.get_complex_q_integrated(
    A_bg_10s_int_2ph, A_res_10s_int_2ph, q_10s
)

# report the computed coefficients/parameters
print("-- 10s resonance --")
print("")
print(f"1ph A_bg: {A_bg_10s_int_1ph}")
print(f"1ph A_res: {A_res_10s_int_1ph}")
print(f"1ph complex q: {q_complex_10s_int_1ph}")
print("")
print(f"2ph A_bg: {A_bg_10s_int_2ph}")
print(f"2ph A_res: {A_res_10s_int_2ph}")
print(f"2ph complex q: {q_complex_10s_int_2ph}")
print("")


# construct the model prediction
int_pred_8d_1ph = complex_q.get_model_pred_for_integrated(
    "abs", eps_8d, A_bg_8d_int_1ph, A_res_8d_int_1ph, q_complex_8d_int_1ph
)
int_pred_8d_2ph = complex_q.get_model_pred_for_integrated(
    "abs", eps_8d, A_bg_8d_int_2ph, A_res_8d_int_2ph, q_complex_8d_int_2ph
)
int_pred_10s_1ph = complex_q.get_model_pred_for_integrated(
    "abs", eps_10s, A_bg_10s_int_1ph, A_res_10s_int_1ph, q_complex_10s_int_1ph
)
int_pred_10s_2ph = complex_q.get_model_pred_for_integrated(
    "abs", eps_10s, A_bg_10s_int_2ph, A_res_10s_int_2ph, q_complex_10s_int_2ph
)

# we can check how the phase was approximated by the model near the two resonances
# calculate the actual phases and unwrap them to the 2pi period
wigner_int_phase_real = np.unwrap(np.angle(int_sideband_1ph))
atomic_int_phase_real = np.unwrap(np.angle(int_sideband_2ph))
# calculate the predicted phases for the 8d and 10s resonances and unwrap them to the 2pi period
wigner_int_phase_8d_pred = np.unwrap(np.angle(int_pred_8d_1ph))
atomic_int_phase_8d_pred = np.unwrap(np.angle(int_pred_8d_2ph))
wigner_int_phase_10s_pred = np.unwrap(np.angle(int_pred_10s_1ph))
atomic_int_phase_10s_pred = np.unwrap(np.angle(int_pred_10s_2ph))

# plot the actual phases vs model prediction
mask_to_plot_8d = np.logical_and(ekin_eV >= 2.4, ekin_eV <= 2.46)
mask_to_plot_10s = np.logical_and(ekin_eV >= 2.47, ekin_eV <= 2.49)
fontsize = 11
plt.figure("Int. Phases vs Model")
plt.plot(
    ekin_eV,
    wigner_int_phase_real,
    color="lightskyblue",
    linewidth=3.5,
    label="$\\phi_{W,0}$",
)
plt.plot(
    ekin_eV,
    atomic_int_phase_real,
    color="violet",
    linewidth=2.2,
    label="$\\phi_{A,0}$",
)
plt.plot(
    ekin_eV[mask_to_plot_8d],
    wigner_int_phase_8d_pred[mask_to_plot_8d],
    linestyle=":",
    color="black",
    linewidth=2.5,
    label="model",
)
plt.plot(
    ekin_eV[mask_to_plot_8d],
    atomic_int_phase_8d_pred[mask_to_plot_8d],
    linestyle=":",
    color="black",
    linewidth=2.5,
)
plt.plot(
    ekin_eV[mask_to_plot_10s],
    wigner_int_phase_10s_pred[mask_to_plot_10s],
    linestyle=":",
    color="black",
    linewidth=2.5,
)
plt.plot(
    ekin_eV[mask_to_plot_10s],
    atomic_int_phase_10s_pred[mask_to_plot_10s],
    linestyle=":",
    color="black",
    linewidth=2.5,
)
plt.legend(fontsize=fontsize, loc=(0.65, 0.01))
plt.xlabel("Electron energy (eV)", fontsize=fontsize)
plt.ylabel("$\\phi_{0}$ (rad)", fontsize=fontsize)
plt.xticks([2.4 + i * 0.02 for i in range(6)], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim([2.4, 2.5])

# plot the resonance regions
ax = plt.gca()
ymin, ymax = ax.get_ylim()
ymin = -1.3 * np.pi
ymax = 1.7 * np.pi
ax.set_ylim([ymin, ymax])
ax.set_ymargin(0)
ax.set_xmargin(0)
xenon_res_params = [
    (0.891, 0.0284, "$8d$", 1.52),
    (0.930, 0.0008, "$10s$", -136.62),
]
for i, (r, width, name, _) in enumerate(xenon_res_params):
    r += g_omega_IR_eV
    plt.fill_betweenx(
        [ymin, ymax], r - width / 2, r + width / 2, color="gray", alpha=0.2
    )
    ax.plot([r, r], [ymin, ymax], color="gray", alpha=0.2)
    ax.text(r, ymax + (ymax - ymin) / 40, name, rotation=40, fontsize=fontsize)


"""
       #####################################
            Angularly Resolved Case 
       #####################################

Here, we'll consider only the 10s resonance since the 8d doesn't have any critical angles.
In the angularly resolved case, we need to combine the model predictions for the integrated scenario
shown above and predictions for the complex beta parameters that comes below. 

As before, we'll use the first fitting method. The second one (with just two energy values) can 
be called through the complex_q.get_coefficients_for_beta_param() function and is not shown below.
"""

# Fit the model to the beta parameters

# In the 1ph case, there's only one beta parameter of the second order.
A_bg_10s_b2_1ph, A_res_10s_b2_1ph, _ = complex_q.get_coefficients_for_beta_param_by_fit(
    "abs",
    q_10s,
    b2_1ph,
    eps_10s,
    eps_10s_min,
    eps_10s_max,
    A_bg_10s_int_1ph,
    A_res_10s_int_1ph,
)
q_complex_10s_b2_1ph = complex_q.get_complex_q_integrated(
    A_bg_10s_b2_1ph, A_res_10s_b2_1ph, q_10s
)

# In the 2ph case, there're two beta parameters of the second and fourth order.
A_bg_10s_b2_2ph, A_res_10s_b2_2ph, _ = complex_q.get_coefficients_for_beta_param_by_fit(
    "abs",
    q_10s,
    b2_2ph,
    eps_10s,
    eps_10s_min,
    eps_10s_max,
    A_bg_10s_int_2ph,
    A_res_10s_int_2ph,
)
q_complex_10s_b2_2ph = complex_q.get_complex_q_integrated(
    A_bg_10s_b2_2ph, A_res_10s_b2_2ph, q_10s
)

A_bg_10s_b4_2ph, A_res_10s_b4_2ph, _ = complex_q.get_coefficients_for_beta_param_by_fit(
    "abs",
    q_10s,
    b4_2ph,
    eps_10s,
    eps_10s_min,
    eps_10s_max,
    A_bg_10s_int_2ph,
    A_res_10s_int_2ph,
)
q_complex_10s_b4_2ph = complex_q.get_complex_q_integrated(
    A_bg_10s_b4_2ph, A_res_10s_b4_2ph, q_10s
)

# construct the model predictions
b2_1ph_10s_pred = complex_q.get_model_pred_for_beta_param(
    "abs",
    eps_10s,
    A_bg_10s_b2_1ph,
    A_res_10s_b2_1ph,
    q_complex_10s_b2_1ph,
    A_bg_10s_int_1ph,
    A_res_10s_int_1ph,
    q_complex_10s_int_1ph,
)
b2_2ph_10s_pred = complex_q.get_model_pred_for_beta_param(
    "abs",
    eps_10s,
    A_bg_10s_b2_2ph,
    A_res_10s_b2_2ph,
    q_complex_10s_b2_2ph,
    A_bg_10s_int_2ph,
    A_res_10s_int_2ph,
    q_complex_10s_int_2ph,
)
b4_2ph_10s_pred = complex_q.get_model_pred_for_beta_param(
    "abs",
    eps_10s,
    A_bg_10s_b4_2ph,
    A_res_10s_b4_2ph,
    q_complex_10s_b4_2ph,
    A_bg_10s_int_2ph,
    A_res_10s_int_2ph,
    q_complex_10s_int_2ph,
)

# compare the model predictions with the actual second order parameters
plt.figure("Beta Params vs Model")
plt.subplot(2, 1, 1)
plt.plot(
    ekin_eV,
    np.real(b2_1ph),
    color="limegreen",
    linewidth=3.5,
    label="$\\Re(\\tilde{\\beta})$",
)
plt.plot(
    ekin_eV,
    np.imag(b2_1ph),
    color="red",
    linewidth=1.8,
    label="$\\Im(\\tilde{\\beta})$",
)
plt.plot(
    ekin_eV[mask_to_plot_10s],
    np.real(b2_1ph_10s_pred)[mask_to_plot_10s],
    color="black",
    linewidth=2.5,
    linestyle=":",
)
plt.plot(
    ekin_eV[mask_to_plot_10s],
    np.imag(b2_1ph_10s_pred)[mask_to_plot_10s],
    color="black",
    linewidth=2.5,
    linestyle=":",
)
plt.ylabel("$\\tilde{\\beta}_2^{W}$", fontsize=fontsize)
plt.legend(fontsize=fontsize, loc=(0.01, 0.01))
plt.xticks([])
plt.yticks(fontsize=fontsize)
plt.xlim([2.45, 2.5])
plt.text(2.451, 2.1, "(a) $\\tilde{\\beta}_2^{W}$", fontsize=fontsize + 3)

ax = plt.gca()
ymin, ymax = ax.get_ylim()
ymin = -3
ymax = 3
ax.set_ylim([ymin, ymax])
ax.set_ymargin(0)
ax.set_xmargin(0)
for i, (r, width, name, _) in enumerate(xenon_res_params[1:]):
    r += g_omega_IR_eV
    plt.fill_betweenx(
        [ymin, ymax], r - width / 2, r + width / 2, color="gray", alpha=0.2
    )
    ax.plot([r, r], [ymin, ymax], color="gray", alpha=0.2)
    ax.text(r, ymax + (ymax - ymin) / 40, name, rotation=40, fontsize=fontsize)


plt.subplot(2, 1, 2)
plt.plot(
    ekin_eV,
    np.real(b2_2ph),
    color="limegreen",
    linewidth=3.5,
    label="$\\Re(\\tilde{\\beta})$",
)
plt.plot(
    ekin_eV,
    np.imag(b2_2ph),
    color="red",
    linewidth=1.8,
    label="$\\Im(\\tilde{\\beta})$",
)
plt.plot(
    ekin_eV[mask_to_plot_10s],
    np.real(b2_2ph_10s_pred)[mask_to_plot_10s],
    color="black",
    linewidth=2.5,
    linestyle=":",
)
plt.plot(
    ekin_eV[mask_to_plot_10s],
    np.imag(b2_2ph_10s_pred)[mask_to_plot_10s],
    color="black",
    linewidth=2.5,
    linestyle=":",
)

plt.ylabel("$\\tilde{\\beta}_2^{A}$", fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks([2.4 + i * 0.02 for i in range(6)], fontsize=fontsize)
plt.xlim([2.45, 2.5])
plt.xlabel("Electron energy (eV)", fontsize=fontsize)
plt.text(2.451, 12.55, "(b) $\\tilde{\\beta}_2^{A}$", fontsize=fontsize + 3)

ax = plt.gca()
ymin, ymax = ax.get_ylim()
ymin = -14
ymax = 17
ax.set_ylim([ymin, ymax])
ax.set_ymargin(0)
ax.set_xmargin(0)
for i, (r, width, name, _) in enumerate(xenon_res_params[1:]):
    r += g_omega_IR_eV
    plt.fill_betweenx(
        [ymin, ymax], r - width / 2, r + width / 2, color="gray", alpha=0.2
    )
    ax.plot([r, r], [ymin, ymax], color="gray", alpha=0.2)
    ax.text(r, ymax + (ymax - ymin) / 40, name, rotation=40, fontsize=fontsize)

# So, both the beta parameters and the integrated part above are described well by the model, which
# means that the angularly resolved sideband, which is the combination of the two, should also align
# with the analytical formula.

# Now, having all the model coefficients, we can try to predict the critical angles for the 10s
# resonance.

# We can predict them numerically by checking where the imaginary part of q(theta) crosses zero:
# Wigner phase
angles = np.arange(0, 90, 0.01)
crit_angles_1ph_numeric, complex_q_arr_1ph = complex_q.get_crit_angles(
    angles,
    q_10s,
    A_bg_10s_int_1ph,
    A_res_10s_int_1ph,
    A_bg_10s_b2_1ph,
    A_res_10s_b2_1ph,
    0,
    0,
)  # NOTE: here, the last two arguments are 0 since in the 1ph case the b4 parameter is zero.
# Atomic phase
crit_angles_2ph_numeric, complex_q_arr_2ph = complex_q.get_crit_angles(
    angles,
    q_10s,
    A_bg_10s_int_2ph,
    A_res_10s_int_2ph,
    A_bg_10s_b2_2ph,
    A_res_10s_b2_2ph,
    A_bg_10s_b4_2ph,
    A_res_10s_b4_2ph,
)

# Or we can predict them by using an analytical formula:
# Wigner phase
crit_angles_1ph_analyt = complex_q.get_crit_angles_analyt(
    q_10s,
    [A_res_10s_int_1ph, A_res_10s_b2_1ph, 0],
    [A_bg_10s_int_1ph, A_bg_10s_b2_1ph, 0],
    wigner=True,
)
# Atmoic Phase
crit_angles_2ph_analyt = complex_q.get_crit_angles_analyt(
    q_10s,
    [A_res_10s_int_2ph, A_res_10s_b2_2ph, A_res_10s_b4_2ph],
    [A_bg_10s_int_2ph, A_bg_10s_b2_2ph, A_bg_10s_b4_2ph],
    wigner=False,
)

# print out the predicted and true angles
print("##### Angularly Resolved Case #####")
print("")
print("-- 10s resonance --")
print("")
print(f"1ph crit angles, numeric: {crit_angles_1ph_numeric}")
print(f"1ph crit angles, analyt: {crit_angles_1ph_analyt}")
print(f"1ph crit angles, true: {[79.4]}")
print("")
print(f"2ph crit angles, numeric: {crit_angles_2ph_numeric}")
print(f"2ph crit angles, analyt: {crit_angles_2ph_analyt}")
print(f"1ph crit angles, true: {[]}")


# As we see there's one critical angle in the Wigner phase and zero in the atomic phase.
# The model correctly identifies the number of critical angles in both cases, and in the Wigner phase
# the agreement with the true value is good.
# Also, the numeric and analytical ways of predicting the critical angles give very close results.

# plot the imaginary part of q(theta)
plt.figure("Im(q(theta))")
plt.plot(
    angles,
    np.imag(complex_q_arr_2ph),
    linestyle="-",
    linewidth=1.75,
    color="firebrick",
    label="$10s$, atomic",
)
plt.plot(
    angles,
    np.imag(complex_q_arr_1ph),
    linestyle="--",
    linewidth=1.75,
    color="red",
    label="$10s$, Wigner",
)
plt.scatter(
    crit_angles_1ph_analyt,
    [0 for _ in range(len(crit_angles_1ph_analyt))],
    marker="o",
    s=40,
    color="black",
    label="Crit. angles",
)
plt.axhline(0, angles[0], angles[-1], linestyle="--", linewidth=1, color="black")

plt.ylabel("$\\Im(\\tilde{q}(\\theta))$", fontsize=fontsize)
plt.xlabel("$\\theta$ [deg]", fontsize=fontsize)
plt.yticks([0, 1, 2], fontsize=fontsize)
plt.legend(fontsize=fontsize, loc=(0.01, 0.3))

plt.show()
