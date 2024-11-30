'''
Author: Pratyush Anand (anand43@mit.edu)
Date: Jan 3, 2024
'''


from qutip import *
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpmath import mp
T = tensor
import pandas as pd

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 22})

# options = Options()
# # options.num_cpus = 3
# # #options.atol = 1e-10
# options.nsteps = 100000000

Na = 2 #number of atomic levels in NV
g_state = basis(Na, 0)
e_state = basis(Na, 1)

N_ph = 2
N_mw = 2

if __name__ == '__main__':

    psi_mw1 = T(basis(N_mw, 1), basis(N_ph, 0), g_state)

    time_scale = 10000
    t_up = 9.0e-7
    delta_t = t_up/time_scale
    tlist = np.linspace(0,t_up,time_scale)
    numb = 100
    ntraj = 500

    #Defining |MW mode, NV spin> == |b, nv>
    b = destroy(N_mw) 
    p = destroy(N_ph)

    

    w_spin = 2*np.pi*5.5e9 #Hz
    w_mw = 2*np.pi*5.5e9 #Hz
    w_ph = 2*np.pi*5.5e9 #Hz

    g_mwp = 2*np.pi*10e6 #Hz
    g_ps = 2*np.pi*1e6 #Hz

    gamma_mw = 2*np.pi*10e3 #Hz
    gamma_ph = 2*np.pi*1e2 #Hz
    gamma_e = 2*np.pi*1e4 #Hz

    tau = 510e-9

    #sech_vec = np.vectorize(mp.sech)
    #Defining time-dependent mw-phonon coupling
    t_mwp = 50.e-9 #sec                   
    #pulse_shape_mw = g_mwp * np.exp(-(2 * g_mwp** 2)*(tlist - t_mwp)** 2)

    #Defining time-dependent phonon-spin coupling
    t_ps = t_mwp + tau #sec                   
    #pulse_shape_ph = g_ps * np.exp(-(2 * g_ps** 2)*(tlist - t_ps)** 2)

    t_down1 = 0                  # Gaussian pulse parameter
    t_up1 = 1.0e-6

    t_down2 = 0                  # Gaussian pulse parameter
    t_up2 = 1.0e-6

    pulse_shape_mw_arr = []
    pulse_shape_ph_arr = []

    for time in tlist:
        if ((time<t_down1) or (time > t_up1)):
            pulse_shape_mw_arr.append(0.0) #np.exp(-(tlist - t_offset_G) ** 2 /(2 * tp_G ** 2))
        else:
            pulse_shape_mw_arr.append(1/np.cosh(2*g_mwp*(time - t_mwp)))
                #g_mwp * np.exp(-(2 * g_mwp** 2)*(time - t_mwp)** 2))#g_mwp * 1/np.cosh(2*g_mwp*(time - t_mwp)))
                #np.exp(-(0.5 * (10*g_mwp)** 2)*(time - t_mwp)** 2)
    for time in tlist:
        if ((time<t_down2) or (time > t_up2)):
            pulse_shape_ph_arr.append(0.0) #np.exp(-(tlist - t_offset_G) ** 2 /(2 * tp_G ** 2))
        else:
            pulse_shape_ph_arr.append(1/np.cosh(2*g_ps*(time - t_ps)))
                #g_ps * np.exp(-(2 * g_ps** 2)*(time - t_ps)** 2))#g_ps * 1/np.cosh(2*g_ps*(time - t_ps)))
                #np.exp(-(0.5 * (10*g_ps)** 2)*(time - t_ps)** 2)

    pulse_shape_mw = np.array(pulse_shape_mw_arr)
    pulse_shape_ph = np.array(pulse_shape_ph_arr)



    H_0 = T(qeye(N_mw), qeye(N_ph), w_spin * e_state * e_state.dag()) + T(w_mw * b.dag() * b, qeye(N_ph), qeye(Na)) + T(qeye(N_mw), w_ph * p.dag() * p, qeye(Na))

    H_int1_mw = g_mwp * T(b, p.dag(),qeye(Na))  
    H_int2_mw = H_int1_mw.dag()
    H_int_mw = H_int1_mw + H_int2_mw

    H_mw = [H_int_mw, pulse_shape_mw]

    H_int1_ph = g_ps * T(qeye(N_mw), p, e_state *g_state.dag())  
    H_int2_ph = H_int1_ph.dag()
    H_int_ph = H_int1_ph + H_int2_ph

    H_ph = [H_int_ph, pulse_shape_ph]

    H = [0*H_0, H_mw, H_ph]



    ### Defining collapse operators

    c1 = T(np.sqrt(gamma_mw)*b, qeye(N_ph), qeye(Na))
    c2 = T( qeye(N_mw), np.sqrt(gamma_ph)*p, qeye(Na)) 
    c3 = T(qeye(N_mw), qeye(N_ph), np.sqrt(gamma_e) * (e_state * g_state.dag() + g_state * e_state.dag()))

    c_ops = [c1, c2, c3]
    e_ops = [T(qeye(N_mw), p.dag()*p, qeye(Na)), T(b.dag()*b, qeye(N_ph), qeye(Na)), T(qeye(N_mw), qeye(N_ph), qt.ket2dm(e_state))]


    #R_fl = T(qeye(N_mw), qt.ket2dm(d_state))

    mc1 = mesolve(H, psi_mw1, tlist, c_ops, e_ops, options=Options(store_states=True))#, [ntraj])

    rho_in = mc1.states[0].ptrace(0)
    rho_f = mc1.states[-1].ptrace(2)

    print(rho_in)
    print(rho_f)

    # state = 0*qt.ket2dm(mc1.states[0][time_scale-1])
    
    # for iter in np.arange(ntraj):
    #     state += qt.ket2dm(mc1.states[iter][time_scale-1])/ntraj

    # rho_f = state.ptrace(2)

    F = ((rho_in.sqrtm() * rho_f * rho_in.sqrtm()).sqrtm()).tr()

    print(F)

    fig = plt.figure()
    ax = fig.add_subplot()

    plt.plot(1e9*tlist, mc1.expect[0], '-', color='red', label = 'Phonon')
    plt.plot(1e9*tlist, mc1.expect[1], '-', color='blue', label ='MW')
    plt.plot(1e9*tlist, mc1.expect[2], '-', color='green', label ='El spin')
    # plt.plot(1e9*tlist, pulse_shape_ph,  '-', color='black', label =r'$g_{pe}(t)$')
    # plt.plot(1e9*tlist, pulse_shape_mw, '-', color='orange', label =r'$g_{mp}(t)$')
    # #plt.plot(tlist, pulse_shape_laser, 'o', color='purple', label ='Laser drive')
    #plt.legend(loc='upper right', fontsize='small')

    plt.xlabel(r'$t (ns)$')
    plt.ylabel("Population")

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    #plt.legend(bbox_to_anchor=(1.05, 1000), loc='upper right')

    plt.legend(loc =7)
    plt.show()

