from qutip import *
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
T = tensor
import pandas as pd
import math

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})



if __name__ == '__main__':

    

    Tc_arr = np.array([5e-7])#,6e-7,7e-7,8e-7,9e-7])#np.geomspace(1e-7, 1e-6, 10)
    #Tc_arr = np.linspace(1e-7, 1e-6, 50)
    kappa_arr = np.array([2*np.pi*2.42e6])#np.geomspace(1e4, 1e8, 70)  #np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])

    mw_tot = []

    for T_c in Tc_arr:

        mw = []

        for kappa in kappa_arr:

            Na = 3
            N_coh = 14
            timescale = 100000
            t_up = 1e-4 #sec
            t0 = 5e-5 
            #T_c = 5e-7 #sec
            c = 3.0e8
            delta_t = t_up/timescale
            time = np.linspace(0, t_up, timescale)
            
            t1 = t0-6*T_c
            t2 = t0+6*T_c

            ind1 = int(t1/delta_t)
            ind2 = int(t2/delta_t)

            time_new = time[ind1:ind2+1]
            timescale_new = len(time_new)

            #kappa = 2*np.pi*0.77e5 #Hz
            lamb = np.sqrt(kappa)
            kappa_loss = 0#0.33e6 *2*np.pi
            gamma_ph = 2*np.pi*3.03e2 #Hz

            g_mp = 2*np.pi*4.9e6 #Hz

            C = g_mp*g_mp/((kappa + kappa_loss) * gamma_ph)

            Ad = gamma_ph * C * T_c
            eta_max = kappa/(kappa + kappa_loss) *  C/(1+C)
            Delta = 0
            T_pk = 4 * np.sqrt(3)/np.pi * T_c

            target_s = []
            fid_s = []

            target_g = []
            fid_g = []

            wavepacket = []
            print(eta_max)

            fock_arr = np.linspace(1e-2, 15, 100)#np.array([1e0, 1e1, 1e2, 1e3])
            #gamma_e_arr = 2*np.pi*np.array([1e2, 1e3, 1e4, 1e5, 1e6])
            x_arr = np.array([-1.303, -0.636, -0.03])#np.linspace(-3, 3, 100)
            i = 0
            for x in x_arr:

                i +=1
                print(i)
                fock = 1e-2

                t_shift_2 = -x*T_c
                shift_index_2 = int(t_shift_2/delta_t)

                ant_in = np.sqrt(fock)/np.sqrt(T_pk) * 1/np.cosh(2/T_pk * (time-t0))
                ant_in_1 = ant_in[ind1+shift_index_2:ind2+1+shift_index_2]

                wavepacket.append(ant_in_1*ant_in_1/ max(ant_in_1*ant_in_1))


                C_in = [delta_t*ant_in[0]**2]

                ant_g = np.sqrt(fock)/np.sqrt(T_pk * np.sqrt(2*np.pi)) * np.exp(-(time-t0)**2/(4*T_pk*T_pk))
                ant_g_1 = ant_g[ind1:ind2+1]

                for n in np.arange(timescale):

                    if (n>0):
                        d = delta_t*ant_in[n]**2 + C_in[n-1]
                        C_in.append(d)
                
                C_in = np.array(C_in)
                C_in_1 = C_in[ind1:ind2+1]
                
                t_shift = 0*T_c
                shift_index = int(t_shift/delta_t)

                g_pe = (gamma_ph*(1+C) + 1j* Delta)/(np.sqrt(2*gamma_ph*(1+C))) * ant_in/(np.sqrt(C_in)) * np.exp(-1j* (Delta/(2*gamma_ph*(1+C)))*np.log(C_in))   
                g_pe_1 = g_pe[ind1+shift_index:ind2+1+shift_index]

                omega_c = 5e9
                a = destroy(N_coh)
                b = destroy(2)

                zero_down = basis(3,0)
                one_down = basis(3,2)
                zero_up = basis(3,1)

                sigma_z = (zero_up*zero_up.dag() - zero_down*zero_down.dag())

                gamma_e = 2*np.pi*1e4 #Hz

                H_0 =  g_mp * (T(a, one_down *zero_down.dag()) + T(a.dag(), zero_down * one_down.dag()))


                H_ant = T(a+a.dag(), qeye(3))
                H_pe = T(qeye(N_coh), one_down*zero_up.dag() + zero_up*one_down.dag())

                H_I1 = [H_ant, lamb * ant_in_1]
                H_I1_g = [H_ant, lamb * ant_g_1]
                H_I2 = [H_pe,g_pe_1]

                H_sec = [H_0, H_I1, H_I2]
                H_gauss = [H_0, H_I1_g, H_I2]

                c_ops = [np.sqrt(kappa + kappa_loss)*T(a, qeye(3)), np.sqrt(gamma_ph)*T(qeye(N_coh), zero_down*one_down.dag()), np.sqrt(gamma_e)*T(qeye(N_coh), sigma_z)]

                psi_0 = T(basis(N_coh,0), zero_down)

                rho_aa = T(a.dag()*a, qeye(3))
                rho_rr = T(qeye(N_coh), zero_up*zero_up.dag())
                rho_ee = T(qeye(N_coh), one_down*one_down.dag())
                #rho_rg = T(basis(2,0)*basis(2,0).dag(), basis(2,0)*basis(2,0).dag(), zero_up*zero_down.dag())
                rho_rg = T(qeye(N_coh), zero_up*zero_down.dag())

                rho_re = T(qeye(N_coh), zero_up*one_down.dag())

                e_ops = [rho_rr, rho_rg, rho_re, rho_aa]

                output_sec = mesolve(H_sec, psi_0, time_new, c_ops, e_ops)
                output_gauss = mesolve(H_gauss, psi_0, time_new, c_ops, e_ops)

                tar =  np.array(output_sec.expect[0]/fock)
                wavepacket.append(tar)
                target_s.append(output_sec.expect[0][-1])
                fid_s.append(output_sec.expect[0][-1]*1/fock)

                target_g.append(output_gauss.expect[0][-1])
                fid_g.append(output_gauss.expect[0][-1]*1/fock)
                #mw = np.array(output.expect[3])

                #print(fid)

            # mw_in = [mw[0]]
            # for n in np.arange(len(time_new)):
            #     if (n>0):
            #         e = delta_t*mw[n] + mw_in[n-1]
            #         mw_in.append(e)
            
            # mw_in = np.array(mw_in)
        mw_tot.append(mw)

    
    #print(fid)
    # x = np.log(Tc_arr)
    # y = np.log(kappa_arr)
    # z = mw_tot

    # plt.imshow(z, extent=[np.min(x), np.max(x), np.min(y), np.max(y)])


    # plt.xlabel('T_c')
    # plt.ylabel('kappa')
    # plt.colorbar()
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()

    #print(x_arr)
    #print(fid_s)
    
    plt.plot(1e6*time_new, wavepacket[0], '-', color = 'red', label = r'$|E_{in}|_{n}^{2}, \Delta t_{shift} = -1.3T_{c}$')
    plt.plot(1e6*time_new, wavepacket[1], '--', color = 'green', label =  r'$|E_{in}|_{n}^{2}, \Delta t_{shift} = -0.64T_{c}$')
    plt.plot(1e6*time_new, wavepacket[2], '-', color = 'gold', label =  r'$|E_{in}|_{n}^{2}, \Delta t_{shift} = -0.03T_{c}$')
    # plt.plot(1e6*time_new, wavepacket[3], '-', color = 'gold', label = r'$\eta/n, \gamma_{e}/2\pi = 10^{5} Hz$')
    # plt.plot(1e6*time_new, wavepacket[4], '-', color = 'orange', label = r'$\eta/n, \gamma_{e}/2\pi = 10^{6} Hz$')
    plt.plot(1e6*time_new, np.absolute(g_pe_1)/max(np.absolute(g_pe_1)), '-', color = 'blue', label = r'$g_{pe,n}$')
    # plt.plot(1e6*time_new, ant_in_1*ant_in_1/ max(ant_in_1*ant_in_1), '-', color = 'red', label = r'$|E_{in}|_{n}^{2}$')

    # plt.plot(fock_arr, target_s, '-', color = 'red', label = r'$\eta_{sec}$')
    # plt.plot(fock_arr, fid_s, '-', color = 'green', label = r'$\nu_{sec}$')
    # # plt.plot(x_arr, 0.8*x_arr/x_arr, '--', color = 'red', label = 'Threshold')
    # plt.plot(fock_arr, target_g, '--', color = 'blue', label = r'$\eta_{gaus}$')
    # plt.plot(fock_arr, fid_g, '--', color = 'gold', label = r'$\nu_{gaus}$')
    #plt.plot(1e6*time_new, tar, '-', color = 'green', label = r'$\nu$')
    #plt.plot(time_new, C_in_1, 'o-', color = 'violet', label = 'totol photon')
    #plt.plot(1e6*time_new, mw, '-', color = 'gold', label = 'MW cavity')
    # plt.plot(kappa_arr, mw_tot[1], '*-', color = 'blue', label = 'Tc = 6e-7')
    # plt.plot(kappa_arr, mw_tot[2], '*-', color = 'green', label = 'Tc = 7e-7')
    # plt.plot(kappa_arr, mw_tot[3], '*-', color = 'gold', label = 'Tc = 8e-7')
    # plt.plot(kappa_arr, mw_tot[4], '*-', color = 'violet', label = 'Tc = 9e-7')
    # #plt.plot(time_new, mw_in, '*-', color = 'gold', label = 'MW cavity tot')
    # #plt.plot(time_new, eta_2)
    # #print(E_in/(np.sqrt(C_in)))
    # # print(max(np.array(output.expect[3])))
    # # print(max(tar))
    #plt.xscale('log')

    plt.xlabel(r"$t (\mu s)$")
    #plt.xlabel(r"$n$")
    #plt.xlabel(r"$\Delta t_{shift}/T_{c}$")
    #plt.ylabel(r"$\nu$")
    #ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.legend()
    plt.show()

                    




    