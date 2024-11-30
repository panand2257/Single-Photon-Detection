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
    t_ro = 240e-9 #sec

    # with open("histo.txt", "a") as mf:
    #     print(('{0:9s}   {1:9s}   {2:9s}'.format('iter', 'N0', 'N1')), file=mf)


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
    eta = 0.85
    

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
    c4 = T(qeye(N_c), np.sqrt(gamma_0) * down * down_pr.dag())
    c5 = T(qeye(N_c), np.sqrt(2*gamma_d) * (up_pr * up_pr.dag() + down_pr * down_pr.dag()))

    c_ops = [c0, c1, c2, c3, c4, c5]
    #e_ops = [T(qeye(N_c), qt.ket2dm(up)), T(qeye(N_c), qt.ket2dm(down)), T(c.dag()*c, qeye(Na)), c0]


    me1 = mcsolve(H, psi1, tlist, c_ops, [], [ntraj])#, options=Options(store_states=True))
    me0 = mcsolve(H, psi0, tlist, c_ops, [], [ntraj])#, options=Options(store_states=True))

    # N_0 = eta * np.sum(np.absolute(np.array(me1.expect[3]))**2)*delta_t
    # N_1 = eta * np.sum(np.absolute(np.array(me0.expect[3]))**2)*delta_t
    # mc_mw0 = mcsolve(H, psi_mw0, tlist, c_ops, [], [numb])
    
    N_0 = []
    N_1 = []

    for iter in np.arange(ntraj):
        n_0 = 0
        n_1 = 0
        a1 = me1.col_which[iter]
        a0 = me0.col_which[iter]

        t1 = me1.col_times[iter]
        t0 = me0.col_times[iter]

        for iter1 in a1:
            if(iter1 == 0):
                n_0 += 1

        for iter2 in a0:
            if(iter2 == 0):
                n_1 += 1

        # with open("histo.txt", "a") as mf:
        #     print(('{0:.3e}   {1:.6e}   {2:.6e}'.format(iter, n_0, n_1)), file=mf)

        N_0.append(n_0)
        N_1.append(n_1)

    N_0 = np.array(N_0)
    N_1 = np.array(N_1)

    TN = 0
    FP = 0
    FN = 0
    TP = 0
    p = 0.5
    q = 0.5
    MI_arr = []
    S_arr = []

    k_arr = np.arange(61)

    for k in k_arr:

        for x in np.arange(ntraj):

            if(N_0[x] < k):
                TP += 1/ntraj
            if(N_0[x] > k):
                FN += 1/ntraj
            if(N_1[x] > k):
                TN += 1/ntraj
            if(N_1[x] < k):
                FP += 1/ntraj

        MI = p*TN * np.log(TN*p/(p*(TN*p + FN*q))) + p*FP * np.log(FP*p/(p*(FP*p + TP*q))) + q*FN * np.log(FN*q/(q*(TN*p + FN*q))) + q*TP * np.log(TP*q/(q*(FP*p + TP*q)))
        S = TP + TN

        MI_arr.append(MI)
        S_arr.append(S)

    


    




    # N_0_av = np.sum(np.array(N_0))/ntraj
    # N_1_av = np.sum(np.array(N_1))/ntraj

    # M = int(np.floor((N_1_av-N_0_av)/(np.log(N_1_av)-np.log(N_0_av))))

    # P_sum = 0.5

    # for j in np.arange(0, M+1):

    #     term = 0.5*(np.power(N_0_av,j)*np.exp(-1*N_0_av) - np.power(N_1_av,j)*np.exp(-1*N_1_av))/math.factorial(j) 
    #     P_sum += term

    # print(N_0)
    # print(N_1)
    # print(N_0_av)
    # print(N_1_av)
    # print(M)
    # print(P_sum)


    # fig = plt.figure()
    # ax = fig.add_subplot()
    # # plt.ylabel('Population')
    # # plt.xlabel('time')
   
    # plt.hist(N_0, density=True, alpha=0.5, color='red', label = '')
    # plt.hist(N_1, density=True, alpha=0.5, color='blue', label = '')
    # plt.xlabel('Counts/readout')
    # plt.ylabel('Number')
    # # plt.plot(tlist, me1.expect[0], 'o', color='red', label = 'up occupation')
    # # plt.plot(tlist, me1.expect[1], 'o', color='blue', label = 'down occupation')
    # #plt.plot(t_ro_arr, P_succ_arr, 'o', color='green', label = 'P_succ vs T_ro')
    # # # # plt.plot(tlist, mc.expect[1], 'o', color='blue', label ='MW occupation')
    # # # # plt.plot(tlist, mc.expect[2], 'o', color='green', label ='spin occupation')
    # # # # plt.plot(tlist, pulse_shape_ph,  'o', color='black', label ='phonon-spin pulse')
    # # # # plt.plot(tlist, pulse_shape_mw, 'o', color='orange', label ='MW-phonon pulse')
    # # # plt.plot(tlist, pulse_shape_laser, 'o', color='purple', label ='Laser drive')
    # ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    # #plt.legend()
    # plt.show()
