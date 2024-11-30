# Single-Photon-Detection

Description of the files:

System A contains following codes:

1. p_succ.py -- This is to estimate Detection success probability as a function of laser readout time, Cooperativity, n_pump, without including the spin-flip Hamiltoninan

2. qst.py -- This implements the QuTiP simulation for performing quantum state transduction from MW --> Phonon --> Spin
   
3. ro.py -- This implements the QuTiP simulation for laser-based cavity-enhanced single-shot readout of electron spin, and takes into account 6 different collapse operators (2 spin-conserving transitions, 2 spin-flipping transitions, 1 dephasing channel, 1 cavity decay channel)
   
4. qst_sweep.py -- This sweeps the Transduction fidelity for different values of inter-pulse delays, in order to optimize for the fidelity.

5. ro_sweep.py -- This sweeps the detection success probability for different values of laser readout time, in order to optimize for the success probability.

6. ro_histo.py -- This performs the QuTiP based Quantum Monte-Carlo simulation for the readout step, in order to obtain the histogram plots for bright and dark states.

7. post_process.py -- This provides an easy-to-use template to post-process the data in '.txt' file and process as numpy array, for processing the '.txt' files generated by 'qst_sweep.py', 'ro_sweep.py', 'ro_histo.py'.

System B contains following codes:

1. 

