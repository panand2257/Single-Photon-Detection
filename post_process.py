'''
Author: Pratyush Anand (anand43@mit.edu)
Date: Jan 3, 2024
'''


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 22})
import math


# Using readlines()
file1 = open('histo.txt', 'r')
Lines = file1.readlines()


#eta = []
N0 = []
N1 = []
#P_s = []

for L in Lines[1:]:

    data = L.split()
    N0.append(int(float(data[1])))
    N1.append(int(float(data[2])))

fig = plt.figure()
ax = fig.add_subplot()

# t_ro_fil = np.concatenate((t_ro[:19], t_ro[20:]))
# P_succ_fil = np.concatenate((P_succ[:19],P_succ[20:]))


#plt.semilogx(eta, P_s, 'o', color = 'red')
# Plot the optimal point
# plt.plot(1e9*np.array(t_ro_fil), P_succ_fil, 'o', color='red', label='')

# t_ro_opt = t_ro[19]
# P_succ_opt = P_succ[19]
N_0_av = np.sum(np.array(N0))/1000
N_1_av = np.sum(np.array(N1))/1000

M = int(np.floor((N_1_av-N_0_av)/(np.log(N_1_av)-np.log(N_0_av))))

P_sum = 0.5

for j in np.arange(0, M+1):

    term = 0.5*(np.power(N_0_av,j)*np.exp(-1*N_0_av) - np.power(N_1_av,j)*np.exp(-1*N_1_av))/math.factorial(j) 
    P_sum += term

print(N_0_av)
print(N_1_av)
print(M)
print(P_sum)

# Plot the optimal point
#print("Histogram N0:", np.histogram(N0, density=True))

# plt.hist(N0, density=True, alpha=0.4, color='red', label = '')
# plt.hist(N1, density=True, alpha=0.4, color='blue', label = '')
#plt.xlabel('Counts/readout')
# plt.plot(t_ro_op, P_succ_opt, '*', color='lime', markersize = 10,  label='')
# plt.xlabel(r'Photon collection efficiency $\eta$')
# plt.ylabel(r'Success probability $P_s$')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# #plt.legend()
# plt.show()














