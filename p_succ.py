'''
Author: Pratyush Anand (anand43@mit.edu)
Date: Jan 3, 2024
'''

import numpy as np
import matplotlib.pyplot as plt
import math
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 22})

eta_t_n = np.arange(0.1, 100.1, 0.1)

C_list = [0.5, 5, 10, 50, 100]
P_c = []

for C in C_list:
    P_s = []
    for x in eta_t_n:
        N_0 = x/((1+C)**2)
        N_1 = x
        M = int(np.floor((N_1-N_0)/(np.log(N_1)-np.log(N_0))))
        P_sum = 0.5

        for j in np.arange(0, M+1):

            term = 0.5*(np.power(N_0,j)*np.exp(-1*N_0) - np.power(N_1,j)*np.exp(-1*N_1))/math.factorial(j) 
            P_sum += term

        P_s.append(P_sum)
    P_c.append(P_s)

fig = plt.figure()
ax = fig.add_subplot()

plt.semilogx(eta_t_n, P_c[0], '-', color = 'red', label='C = 0.5')
plt.semilogx(eta_t_n, P_c[1], '-', color = 'blue', label='C = 5')
plt.semilogx(eta_t_n, P_c[2], '-', color = 'green', label='C = 10')
plt.semilogx(eta_t_n, P_c[3], '-', color = 'black', label='C = 50')
plt.semilogx(eta_t_n, P_c[4], '-', color = 'purple', label='C = 100')
plt.xlabel(r'$\eta n_{pump} T$')
plt.ylabel("Success probability")

ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

plt.legend()
plt.show()






