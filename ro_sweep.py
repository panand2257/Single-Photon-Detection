'''
Author: Pratyush Anand (anand43@mit.edu)
Date: Jan 3, 2024
'''


from qutip import *
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
T = tensor
import pandas as pd
import math


Na = 4 #number of atomic levels in NV
down = basis(Na, 0)
up = basis(Na, 1)
down_pr = basis(Na, 2)
up_pr = basis(Na, 3)

N_c = 5 #Set the limit on fock state for the cavity field


if __name__ == '__main__':

    psi1 = T(basis(N_c, 0), up)
    psi0 = T(basis(N_c, 0), down)

    time_scale = 10000
    t_up = 1.0e-6
    delta_t = t_up/time_scale
    tlist = np.linspace(0,t_up,time_scale)
    numb = 100
    ntraj = 1000

    #Defining |MW mode, NV spin> == |b, nv>
    c = destroy(N_c) 

    w_las = 2*np.pi*406.7e12
    w_up_up = 2*np.pi*406.7e12 #Hz
    Delta = 2*np.pi*2e9 #Hz
    w_down_down = (w_up_up - Delta) #Hz 

    g = 2*np.pi *8e9 #Hz
    kappa = 2*np.pi*21e9 #Hz
    epsilon = np.sqrt(0.01 * 2 * g*g/kappa)
    t_ro_arr = np.arange(130.1e-9, 1.0e-6, 10e-9) #sec
    t_ro = 240e-9 #sec

    eta_arr = np.array([10**-3, 10**(-2.5), 0.01, 10**(-1.5), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    P_succ_arr = []

    with open("eta_sweep_new.txt", "a") as mf:
        print(('{0:9s}   {1:9s}'.format('eta', 'P_sum')), file=mf)

  
    for eta in eta_arr:

        pulse_shape_las = []                 

        for time in tlist:
            if (time > t_ro):
                pulse_shape_las.append(0.0) 
            else:
                pulse_shape_las.append(1.0)

        pulse_shape_las = np.array(pulse_shape_las)

        H_0 = -Delta * T(qeye(N_c), down_pr * down_pr.dag()) + 1j *g * T(c, up_pr * up.dag()) - 1j *g * T(c.dag(), up * up_pr.dag()) + 1j *g * T(c, down_pr * down.dag()) - 1j *g * T(c.dag(), down * down_pr.dag())

        # Laser driving Hamiltonian
        H_las1 = np.sqrt(kappa)* epsilon * T(c + c.dag(), qeye(Na))
        H_las = [H_las1, pulse_shape_las]

        H = [H_0, H_las]

        cyc = 5
        

        Gamma_up = 2*np.pi * 0.123e9
        Gamma_down = 2*np.pi * 0.123e9

        C = 2*g*g/(kappa*Gamma_up)

        gamma_0 = Gamma_up * cyc/(1+cyc)
        gamma_1 = Gamma_up * 1/(1+cyc)
        gamma_2 = Gamma_down * 1/(1+cyc)
        gamma_3 = Gamma_down * cyc/(1+cyc)
        gamma_d = 2*np.pi * 0.25e6

        ### Defining collapse operators

        c0 = T(np.sqrt(kappa)*c, qeye(Na))
        c1 = T(qeye(N_c), np.sqrt(gamma_0) * up * up_pr.dag())
        c2 = T(qeye(N_c), np.sqrt(gamma_1) * down * up_pr.dag())
        c3 = T(qeye(N_c), np.sqrt(gamma_2) * up * down_pr.dag())
        c4 = T(qeye(N_c), np.sqrt(gamma_3) * down * down_pr.dag())
        c5 = T(qeye(N_c), np.sqrt(2*gamma_d) * (up_pr * up_pr.dag() + down_pr * down_pr.dag()))

        c_ops = [c0, c1, c2, c3, c4, c5]
        e_ops = [T(qeye(N_c), qt.ket2dm(up)), T(qeye(N_c), qt.ket2dm(down)), T(c.dag()*c, qeye(Na)), c0]


        me1 = mesolve(H, psi1, tlist, c_ops, e_ops)
        me0 = mesolve(H, psi0, tlist, c_ops, e_ops)

        N_0 = eta * np.sum(np.absolute(np.array(me1.expect[3]))**2)*delta_t
        N_1 = eta * np.sum(np.absolute(np.array(me0.expect[3]))**2)*delta_t
        # mc_mw0 = mcsolve(H, psi_mw0, tlist, c_ops, [], [numb])

        M = int(np.floor((N_1-N_0)/(np.log(N_1)-np.log(N_0))))

        P_sum = 0.5

        for j in np.arange(0, M+1):

            term = 0.5*(np.power(N_0,j)*np.exp(-1*N_0) - np.power(N_1,j)*np.exp(-1*N_1))/math.factorial(j) 
            P_sum += term


        with open("eta_sweep_new.txt", "a") as mf:
            print(('{0:.3e}   {1:.6e}'.format(eta, P_sum)), file=mf)

        print("done")
        # print(N_0)
        # print(N_1)
        # print(M)
        # print(P_sum)


    # plt.ylabel('Population')
    # plt.xlabel('time')
    # # plt.plot(np.arange(numb), sum, 'o', color='red', label = 'Quantum Jumps')
    # plt.plot(tlist, me1.expect[0], 'o', color='red', label = 'up occupation')
    # plt.plot(tlist, me1.expect[1], 'o', color='blue', label = 'down occupation')
    #plt.plot(t_ro_arr, P_succ_arr, 'o', color='green', label = 'P_succ vs T_ro')
    # # # # plt.plot(tlist, mc.expect[1], 'o', color='blue', label ='MW occupation')
    # # # # plt.plot(tlist, mc.expect[2], 'o', color='green', label ='spin occupation')
    # # # # plt.plot(tlist, pulse_shape_ph,  'o', color='black', label ='phonon-spin pulse')
    # # # # plt.plot(tlist, pulse_shape_mw, 'o', color='orange', label ='MW-phonon pulse')
    # # # plt.plot(tlist, pulse_shape_laser, 'o', color='purple', label ='Laser drive')
    # plt.legend()
    # plt.show()
