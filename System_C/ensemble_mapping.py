from qutip import *
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
T = tensor
import pandas as pd
import math

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})

if __name__ == '__main__':


    mw_tot = []
    Na = 3
    N_coh = 14
    timescale = 100000
    t_up = 1e-4 #sec
    t0 = 5e-5 
    T_c = 5e-7 #sec
    c = 3.0e8
    delta_t = t_up/timescale
    time = np.linspace(0, t_up, timescale)

    
    
    t1 = t0-6*T_c
    t2 = t0+6*T_c

    ind1 = int(t1/delta_t)
    ind2 = int(t2/delta_t)

    time_new = time[ind1:ind2+1]
    timescale_new = len(time_new)

    kappa = 2*np.pi*2.42e6 #Hz
    lamb = np.sqrt(kappa)
    kappa_loss = 0#0.33e6 *2*np.pi
    gamma_ph = 2*np.pi*3.03e2 #Hz

    #C = g_mp*g_mp/((kappa + kappa_loss) * gamma_ph)

    # Ad = gamma_ph * C * T_c
    # eta_max = kappa/(kappa + kappa_loss) *  C/(1+C)
    Delta = 0
    T_pk = 4 * np.sqrt(3)/np.pi * T_c

    target = []
    fid = []
    fock = 1e-2

    ant_in = np.sqrt(fock)/np.sqrt(T_pk) * 1/np.cosh(2/T_pk * (time-t0))
    ant_in_1 = ant_in[ind1:ind2+1]
    C_in = [delta_t*ant_in[0]**2]

    for n in np.arange(timescale):

        if (n>0):
            d = delta_t*ant_in[n]**2 + C_in[n-1]
            C_in.append(d)
    
    C_in = np.array(C_in)
    C_in_1 = C_in[ind1:ind2+1]

    #g_ens_arr = np.array([2*np.pi*1e7]) #Hz
    #g_ens_arr = 2*np.pi*np.geomspace(1e6, 1e8, 3)
    #g_mp_arr = 2*np.pi*np.geomspace(1e4, 1e8, 4)

    #g_ens_arr = 2*np.pi*np.linspace(1e6, 5e7, 10)
    #g_mp_arr = 2*np.pi*np.linspace(1e6, 5e7, 10)

    #g_ens_arr = 2*np.pi*np.linspace(5e6, 5e7, 50)
    #g_mp_arr = 2*np.pi*np.linspace(5e6, 5e7, 50)
    #T2_arr = np.geomspace(1e-8, 1e-7, 50)
    T2 = 1.3e-2 #sec 
    g_ens_arr = 2*np.pi*np.array([2e7])
    g_mp_arr = 2*np.pi*np.array([1e7])
    #T2_st = 2e-8
    dark = []
    bright = []
    i = 0

    for g_ens in g_ens_arr:

        dark_st = []
        for g_mp in g_mp_arr:

            T2_st = 1.5e-8 #sec
            #g_ens = 2*np.pi*2e7
            #g_mp = 2*np.pi*4.9e6


            omega_c = 5e9
            a = destroy(N_coh)
            b = destroy(2)

            zero_G = basis(4,0)
            one_G = basis(4,2)
            zero_B = basis(4,1)
            zero_D = basis(4,3)

            sigma_z = (zero_D*zero_D.dag() - zero_G*zero_G.dag())


            H_0 =  g_mp * (T(a, one_G *zero_G.dag()) + T(a.dag(), zero_G * one_G.dag()))

            H_ant = T(a+a.dag(), qeye(4))
            H_pe = g_ens * T(qeye(N_coh), one_G*zero_B.dag() + zero_B*one_G.dag())

            H_I1 = [H_ant, lamb * ant_in_1]
            H = [H_0 + H_pe, H_I1]

            c_ops = [np.sqrt(kappa + kappa_loss)*T(a, qeye(4)), np.sqrt(gamma_ph)*T(qeye(N_coh), zero_G*one_G.dag()), np.sqrt(1/T2_st) * T(qeye(N_coh), zero_D*zero_B.dag()), np.sqrt(1/T2) * T(qeye(N_coh), sigma_z) ]

            psi_0 = T(basis(N_coh,0), zero_G)

            rho_aa = T(a.dag()*a, qeye(4))
            rho_bb = T(qeye(N_coh), zero_B*zero_B.dag())
            rho_dd = T(qeye(N_coh), zero_D*zero_D.dag())
            rho_ee = T(qeye(N_coh), one_G*one_G.dag())
            #rho_rg = T(basis(2,0)*basis(2,0).dag(), basis(2,0)*basis(2,0).dag(), zero_B*zero_G.dag())
            rho_rg = T(qeye(N_coh), zero_B*zero_G.dag())

            rho_re = T(qeye(N_coh), zero_B*one_G.dag())

            e_ops = [rho_bb, rho_dd, rho_rg, rho_re, rho_aa]

            output = mesolve(H, psi_0, time_new, c_ops, e_ops)

            #bright = np.array(output.expect[0])
            #dark = np.array(output.expect[1]/fock)
            mw = np.array(output.expect[4]/fock)

            bright = np.array(output.expect[0]/fock)
            #dark_st = np.array(output.expect[1]/fock)
            eff = output.expect[1][-1]/fock
            dark_st.append(eff)
            i +=1
            print(i)
            #mw.append(output.expect[4])
        dark.append(dark_st)

    print(dark)
    
    #print(fid)
    # x = np.log(Tc_arr)
    # y = np.log(kappa_arr)
    # z = mw_tot

    # plt.imshow(z, extent=[np.min(x), np.max(x), np.min(y), np.max(y)])

    # print(max(dark))
    # print(g_mp_arr[np.argmax(np.array(dark))]/(2*np.pi))


    #plt.figure(figsize=(10, 8))

    # Use pcolormesh for log scale plotting
    # print(dark)
    # plt.pcolormesh(g_mp_arr / (2 * np.pi), g_ens_arr / (2 * np.pi), dark, shading='auto', cmap='viridis')

    # # Set both axes to log scale
    # #plt.xscale('log')
    # #plt.yscale('log')

    # # Add color bar
    # plt.colorbar(label=r'$\nu$')

    # # Label the axes
    # plt.xlabel(r'$g_{mp}$ (Hz)')
    # plt.ylabel(r'$g_{ens}$ (Hz)')
    # #plt.title('Colormap of Dark Values')

    # # Show the plot
    # plt.show()





    # print(mw)
    # # plt.xlabel('T_c')
    # # plt.ylabel('kappa')
    # # plt.colorbar()
    # # plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # plt.plot(1e6*time_new, ant_in_1*ant_in_1/ max(ant_in_1*ant_in_1), '-', color = 'red', label = r'$|E_{in}|_{n}^{2}$')
    # plt.plot(1e6*time_new, 1*time_new/time_new, '-', color = 'blue', label = r'$g_{pe,n}$')
    # # # #plt.plot(fock_arr, target, 'v-', color = 'red', label = 'efficiency')
    # plt.plot(1e6*time_new, dark_st, '-', color = 'green', label = r'$\nu_{dark}$')
    # plt.plot(1e6*time_new, bright, '-', color = 'gold', label = r'$\nu_{bright}$')
    # # #plt.plot(time_new, C_in_1, 'o-', color = 'violet', label = 'totol photon')
    # plt.plot(1e6*time_new, mw, '-', color = 'brown', label = r'$\nu_{MW}$')
    # # # plt.plot(kappa_arr, mw_tot[1], '*-', color = 'blue', label = 'Tc = 6e-7')
    # # # plt.plot(kappa_arr, mw_tot[2], '*-', color = 'green', label = 'Tc = 7e-7')
    # # # plt.plot(kappa_arr, mw_tot[3], '*-', color = 'gold', label = 'Tc = 8e-7')
    # # # plt.plot(kappa_arr, mw_tot[4], '*-', color = 'violet', label = 'Tc = 9e-7')
    # # # #plt.plot(time_new, mw_in, '*-', color = 'gold', label = 'MW cavity tot')
    # # # #plt.plot(time_new, eta_2)
    # # # #print(E_in/(np.sqrt(C_in)))
    # # # # print(max(np.array(output.expect[3])))
    # # # # print(max(tar))
    # # plt.xscale('log')
    # ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
 
    # plt.xlabel(r"$t (\mu s)$")
    # #plt.ylabel(r'$\nu$')
    # plt.legend()
    # plt.show()

                    
    # for t in time:
    #     pulse_term = np.exp(-1*((t-center)**2)/(2* (t_width**2)))
    #     pulse_shape_las_norm.append(pulse_term)

    # pulse_shape_las = beta * 1/(t_width * np.sqrt(2*np.pi)) * np.array(pulse_shape_las_norm)


    # # plt.plot(time, pulse_shape_las, '-', color = 'blue')
    # # plt.show()





    # a = T(destroy(Na), qeye(4)) #Antenna_Cavity

    # E = basis(4,3) # |0,E> for spin-ensemble 
    # G = basis(4,1) # |1,G> for spin ensemble
    # D = basis(4,2) # |0,D> dark mode for spin ensemble
    # G0 = basis(4,0) # |0,G> for spin ensemble
    # sig_z = T(qeye(Na), (E * E.dag() - G * G.dag()))
    # tr_ge = T(qeye(Na),  E * G.dag())
    # tr_ed = T(qeye(Na),  D * E.dag())
    # tr_dg = T(qeye(Na),  G * D.dag())

    # tr_gg0 = T(qeye(Na),  G0 * G.dag())


    # omega_r = 2*np.pi*5e9
    # omega_ph = 2*np.pi*5e9
    # omega_spin = 2*np.pi*5e9
    # w0 = 2*np.pi*5e9

    # g_mp = 2*np.pi*0.3e6 #Hz
    # g_ens = 2*np.pi*5e9 #Hz

    # k_mw = 1e2 #Hz
    # k_ph = 1e5 #Hz
    # T2_star = 1e-3 #sec 
    # T2 = 1e-3 #sec

    # c_mw = np.sqrt(k_mw) * a
    # c_ph = np.sqrt(k_ph) * tr_gg0
    # c_bm = np.sqrt(1/T2_star) * tr_ed
    # c_dm = np.sqrt(1/T2) * tr_dg 

    # c_ops = [c_mw, c_ph, c_bm, c_dm]

    # psi_0 = T(basis(Na, 0), G0)

    # H0 = (omega_r-w0) * a.dag() * a + (omega_ph-w0) * ( tr_gg0.dag() * tr_gg0 ) + 0.5 * (omega_spin-w0) * sig_z
    # H_int = g_mp * (a * tr_gg0.dag() + a.dag() * tr_gg0) + g_ens * ( tr_ge + tr_ge.dag())
    # H_drive = [(a + a.dag()), np.array(pulse_shape_las)]

    # H1 = H0 + H_int
    # H = [H0 + H_int, H_drive]

    # e_ops = [a.dag()*a, tr_ge*tr_ge.dag(), tr_ed*tr_ed.dag(), tr_dg*tr_dg.dag(), tr_gg0*tr_gg0.dag()]

    # output = mesolve(H, psi_0, time, c_ops, e_ops)#, ntraj = 1000)

    # plt.plot(time, output.expect[0], '-', color = 'red', label = 'MW')
    # plt.plot(time, output.expect[1], '-', color = 'blue', label = 'bright mode')
    # plt.plot(time, output.expect[2], '-', color = 'green', label = 'dark mode')
    # # plt.plot(time, output.expect[3], 'o', color = 'orange', label = 'ground state')
    # #plt.plot(time, output.expect[4], '-', color = 'black', label = 'ground state 0')
    # #plt.plot(time, pulse_shape_las_norm, '-', color = 'blue', label = 'wavepacket')

    # plt.legend()
    # plt.show()







