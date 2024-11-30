from qutip import *
import qutip as qt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
T = tensor
import pandas as pd
import math
from scipy.stats import ks_2samp
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from scipy.interpolate import interp1d

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 22})

N = 30
fact = 1e6
g_o = 2 * np.pi * 50e6/fact
omega_p = 2 * np.pi * 5.5e9/fact
delta = 2 * np.pi * 400e6/fact

#opt_range = np.linspace(2 * np.pi * 300e6/fact, 2 * np.pi * 900e6/fact, 50)
opt_range = np.array([2 * np.pi * 200e6/fact])
gamma_ph_arr = np.array([ 2 * np.pi * 5e3 / fact ])

F_dro_arr = []

for gamma_ph in gamma_ph_arr:

    F_dro = []

    for Omega_opt in opt_range:

        #Omega_opt = 2 * np.pi * 600e6/fact

        g_eff = g_o * Omega_opt / (2 * omega_p)
        chi = g_eff * g_eff / delta
        n_crit = delta * delta / (g_eff * g_eff)
        wr = 5.5 * 2 * np.pi * 1e9/fact   # resonator frequency
        wq = wr - delta
        # wq = 3.0 * 2 * np.pi    # qubit frequency
        #chi =   # parameter in the dispersive hamiltonian
        #gamma_ph = 2 * np.pi * 5e3 / fact
        gamma_e = 2 * np.pi * 0.1e9/fact
        gamma_E = gamma_e / n_crit

        print(g_eff/(2*np.pi), chi/(2*np.pi), gamma_ph/(2*np.pi), n_crit, gamma_E/(2*np.pi))



        # cavity operators
        a = tensor(destroy(N), qeye(2))
        nc = a.dag() * a
        xc = a + a.dag()

        # atomic operators
        sm = tensor(qeye(N), destroy(2))
        sz = tensor(qeye(N), sigmaz())
        sx = tensor(qeye(N), sigmax())
        sy = tensor(qeye(N), sigmay())
        nq = sm.dag() * sm
        xq = sm + sm.dag()

        I = tensor(qeye(N), qeye(2))
        #psi_g = tensor(coherent(N, np.sqrt(4)), basis(2,0))
        #psi_e = tensor(coherent(N, np.sqrt(4)), basis(2,1))
        psi_g = tensor(coherent(N, np.sqrt(4)), (basis(2,0)).unit())

        # #psi_g = tensor(basis(N, 0), basis(2,1))
        # #psi_e = tensor(basis(N, 1), basis(2,0))

        # dispersive hamiltonian
        # H = wr * (a.dag() * a + I/2.0) + (wq / 2.0) * sz + chi * (a.dag() * a + I/2) * sz

        H =  chi * nc * sz - chi * (I - sz)/2
        H2 = 0 * nc * sz - 0 * (I - sz)/2
        tlist = np.linspace(0, 1000, 800)

        c_ops1 = [np.sqrt(gamma_E)* (sx - 1j * sy)/2, np.sqrt(gamma_ph)*a]
        c_ops2 = [np.sqrt(gamma_ph)*a]
        #res_g = mesolve(H, psi_g, tlist, [], [], options=Odeoptions(nsteps=5000))
        #res_e = mesolve(H2, psi_g, tlist, [], [], options=Odeoptions(nsteps=5000))
        #res_e = mesolve(H, psi_e, tlist, c_ops, [], options=Odeoptions(nsteps=5000))

        corr_vec_g = correlation(H, psi_g, None, tlist, c_ops2, a.dag(), a)
        corr_vec_e = correlation(H2, psi_g, None, tlist, c_ops2, a.dag(), a)

        w_g, S_g = spectrum_correlation_fft(tlist, corr_vec_g)
        w_e, S_e = spectrum_correlation_fft(tlist, corr_vec_e)


        #print(ks_distance)
        #nc_list = expect(nc, res.states)
        #nq_list = expect(nq, res.states)
        wr = 0

        S_g = S_g - np.min(S_g)
        S_e = S_e - np.min(S_e)

        area_g = np.trapz(S_g, w_g)
        area_e = np.trapz(S_e, w_e)

        # S_g_n = S_g/area_g
        # S_e_n = S_e/area_e

        # print(area_g)
        # print(area_e)

        # min_spectrum = np.minimum(S_g_n, S_e_n)

        # print(min_spectrum)
        # print(S_g_n)
        # print(S_e_n)
        # overlap_area = np.trapz(min_spectrum, w_e)
        # fidelity = overlap_area

        # print(f"Numerical Integration-Based Readout Fidelity: {fidelity}")

        # # Plot the normalized spectra
        # fig, ax = plt.subplots(figsize=(8, 5))

        # #ax.plot(w_g / chi, S_g_n, color='red', label='N = 0')
        # #ax.plot(w_e / chi, S_e_n, color='blue', label='N = 1')
        # ax.plot(w_g, min_spectrum, color='green', label='min')
        # ax.set_xlabel(r'$\frac{\omega - \omega_{r}}{\chi}$', fontsize=18)
        # ax.set_ylabel('Normalized Transmittance', fontsize=18)
        # ax.legend()

        # plt.tight_layout()
        # plt.show()


        ###########################################################################

        # Define a finer frequency grid
        fine_w_g = np.linspace(min(w_g), max(w_g), 1000)  # Example: 1000 points for finer grid
        fine_w_e = np.linspace(min(w_e), max(w_e), 1000)


        # Interpolating S_g and S_e onto finer grids
        S_g_interp = interp1d(w_g, S_g, kind='cubic', fill_value="extrapolate")
        S_e_interp = interp1d(w_e, S_e, kind='cubic', fill_value="extrapolate")

        # Spectra on the finer grid
        fine_S_g = S_g_interp(fine_w_g)
        fine_S_e = S_e_interp(fine_w_e)

        # Normalize the interpolated spectra
        fine_area_g = np.trapz(fine_S_g, fine_w_g)
        fine_area_e = np.trapz(fine_S_e, fine_w_e)

        fine_S_g_n = fine_S_g / fine_area_g
        fine_S_e_n = fine_S_e / fine_area_e


        # Recalculate overlap and fidelity
        min_spectrum_fine = np.minimum(fine_S_g_n, fine_S_e_n)
        overlap_area_fine = np.trapz(min_spectrum_fine, fine_w_g)
        fidelity_fine = 1 - overlap_area_fine

        # print(f"Improved Readout Fidelity with Finer Grid: {fidelity_fine}")


        # # Plot the normalized interpolated spectra
        # fig, ax = plt.subplots(figsize=(8, 5))

        # ax.plot(fine_w_g / chi, fine_S_g_n, color='red', label='N = 0 (finer grid)')
        # ax.plot(fine_w_e / chi, fine_S_e_n, color='blue', label='N = 1 (finer grid)')
        # ax.fill_between(fine_w_g / chi, min_spectrum_fine, color='gray', alpha=0.5, label='Overlap Area (finer grid)')
        # ax.set_xlabel(r'$\frac{\omega - \omega_{r}}{\chi}$', fontsize=18)
        # ax.set_ylabel('Normalized Transmittance', fontsize=18)
        # ax.legend()

        # plt.tight_layout()
        # plt.show()
        print("done")
        F_dro.append(fidelity_fine)
    
    F_dro_arr.append(F_dro)

print(F_dro_arr[0])
# print(F_dro_arr[1])
# print(F_dro_arr[2])
#print(opt_range)

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(fine_w_g / chi, fine_S_g_n, color='red', label='N = 0')
ax.plot(fine_w_e / chi, fine_S_e_n, color='blue', label='N = 1')
#ax.fill_between(fine_w_g / chi, min_spectrum_fine, color='gray', alpha=0.5, label='Overlap Area')
ax.set_xlabel(r'$\frac{\omega - \omega_{r}}{\chi}$')
ax.set_xlim(-2, 3)
ax.set_ylabel('Normalized Transmittance')
ax.legend()


ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

#plt.plot(opt_range, F_dro_arr[0], color = 'red')
plt.show()

