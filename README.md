# Single-Photon-Detection
Link for the manuscript: https://arxiv.org/abs/2401.10455
Description of the files:

**System A contains following codes:**

1. p_succ.py -- This is to estimate Detection success probability as a function of laser readout time, Cooperativity, n_pump, without including the spin-flip Hamiltoninan. Used to generate Fig. 4d from the manuscript.

2. m_info.py -- This is to estimate mutual information paramter for the single shot readout part. Used to generate Fig. 4e, 4f from the manuscript.

3. qst.py -- This implements the QuTiP simulation for performing quantum state transduction from MW --> Phonon --> Spin. Used to generate Fig. 6a, 6d, 6e, 6f from the manuscript.

4. qst_sweep.py -- This sweeps the Transduction fidelity for different values of inter-pulse delays, in order to optimize for the fidelity. Used to generate Fig. 6b, 6c from the manuscript.
   
5. ro.py -- This implements the QuTiP simulation for laser-based cavity-enhanced single-shot readout of electron spin, and takes into account 6 different collapse operators (2 spin-conserving transitions, 2 spin-flipping transitions, 1 dephasing channel, 1 cavity decay channel). Used to generate Fig. 9a, 9b, 9c from the manuscript.

6. ro_sweep.py -- This sweeps the detection success probability for different values of laser readout time, in order to optimize for the success probability. Used to generate Fig. 9a from the manuscript.

7. ro_histo.py -- This performs the QuTiP based Quantum Monte-Carlo simulation for the readout step, in order to obtain the histogram plots for bright and dark states. Used to generate Fig. 9d from the manuscript.

8. post_process.py -- This provides an easy-to-use template to post-process the data in '.txt' file and process as numpy array, for processing the '.txt' files generated by 'qst_sweep.py', 'ro_sweep.py', 'ro_histo.py'.

**System B contains following codes:**

1. adiabatic_mapping.py -- This code implements the QuTiP code for implementing the optimal adiabatic mapping for system B. Similar or some minor variations is used to generate Fig. 7a, 7b, 7c, 7e from the manuscript.

2. t_wind.py -- Based on the previous code, this implements the detection_window vs cut-off fidelity. Used to generate Fig. 7d from the manuscript.

**System C contains following codes:**

1. ensemble_mapping.py -- This implements the QuTiP code for ensemble mapping. Used to generate Fig. 8a from the manuscript.

2. ensemble_mesh_plot.py -- This used to generate the tradespace plots for ensemble mapping. Used to generate Fig. 8b, 8c, 8d from the manuscript.

3. fidelity_vs_drive.py -- This evaluates the dispersive readout fidelity as a function of the optical drive. Used to generate Fig. 9e from the manuscript.

4. dispersive_readout.py -- This evaluates the QuTiP code for performing dispersive readout on the ensemble system. Used to generate Fig. 9f from the manuscript.

