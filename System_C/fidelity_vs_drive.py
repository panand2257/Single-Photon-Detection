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

F1 = [0.8665501044007489, 0.8782465355019407, 0.8892544545044951, 0.8983242389963553, 0.9041678491688804, 0.909809206327276, 0.9148039207049697, 0.9200550077507584, 0.9249455334280469, 0.929519961663758, 0.9343727525247569, 0.9392200013721463, 0.9436921611441241, 0.94713687496761, 0.9487132733036789, 0.9495917337681454, 0.9529468381566057, 0.9576931350540443, 0.958289705751656, 0.9589413601620681, 0.9638271227028623, 0.9631043963289961, 0.9660428277289639, 0.9675032958302376, 0.968591419452368, 0.9702763411452344, 0.9719689707502354, 0.9716479296765074, 0.9753080666505306, 0.9740461871297025, 0.97553537339027, 0.9783649524823999, 0.978009619656693, 0.9774428319194963, 0.9791114729019798, 0.9812494393764011, 0.9824542782119995, 0.9829664065636681, 0.9832263252276532, 0.9835554317919969, 0.9840626436696815, 0.9847716900162534, 0.9856407596591253, 0.9865457030216039, 0.9871974578416072, 0.9872399521055054, 0.9865588837880958, 0.9864041996151294, 0.9880991631339744, 0.9893773288967468]
F2 = [0.7465085316086832, 0.7656611175048396, 0.7813551997643913, 0.7969978535460125, 0.8098819556872509, 0.8229148455840315, 0.8340021883016261, 0.8445030657289524, 0.8540497750028919, 0.8624981040528344, 0.8707696722436146, 0.8781699287806954, 0.8848526078097344, 0.8912585583575539, 0.8971460636917379, 0.9025308547741531, 0.9075001952997304, 0.9122499423223953, 0.9166452876705827, 0.9207163054976681, 0.924534211086386, 0.9280869930118321, 0.9314148863983782, 0.9345524487065039, 0.9375093938553549, 0.9402890014144779, 0.9429122904547249, 0.945367657322919, 0.9477021329721753, 0.9498918007320205, 0.9519659601446241, 0.9539372355479822, 0.9557960228739859, 0.9575702822340351, 0.9592462311106127, 0.9608467734067614, 0.9623580050429225, 0.9637975182180959, 0.9651619623495079, 0.966473948870787, 0.9677207371317719, 0.9689092422140774, 0.9700396800519809, 0.9711301636731764, 0.9721655421863413, 0.9731551723756086, 0.9741015979848611, 0.9750122180804898, 0.9758798720964843, 0.9767209845472136]
F3 = [0.6419541860726552, 0.6661378808777347, 0.6866334723991758, 0.7071875718655516, 0.7245933395839617, 0.7422360807192241, 0.7575042882508012, 0.7720637029283983, 0.7854458303126396, 0.7974083943872745, 0.8091332153987878, 0.8197032831401981, 0.8293255769604, 0.838571955708949, 0.8470968884063734, 0.8549414362005792, 0.8622056441431303, 0.8691388905072677, 0.875590699331944, 0.8815925607514374, 0.8872027035477622, 0.8924546398818143, 0.8973816141746562, 0.902018963554059, 0.9064090540175394, 0.9105375461192884, 0.9144283463589264, 0.9180933975837808, 0.9215571764804221, 0.9248288524320559, 0.9279233838148626, 0.9308550312956605, 0.9336414539497968, 0.9362893485302045, 0.9387986727443739, 0.9411839485760293, 0.943447621675096, 0.9456007162815139, 0.947647099537636, 0.9496123137359238, 0.9514818795979949, 0.953262167453061, 0.9549573747195856, 0.956590093657194, 0.9581448894539574, 0.9596284308877995, 0.9610553693909699, 0.9624200133913365, 0.9637233328289083, 0.9649801502759024]
opt = [1884.95559215, 1961.8925551,  2038.82951804, 2115.76648099, 2192.70344393, 2269.64040688, 2346.57736982, 2423.51433277, 2500.45129571, 2577.38825866,
 2654.3252216,  2731.26218455, 2808.19914749, 2885.13611044, 2962.07307338,
 3039.01003633, 3115.94699927, 3192.88396222, 3269.82092516, 3346.75788811,
 3423.69485105, 3500.631814, 3577.56877695, 3654.50573989, 3731.44270284,
 3808.37966578, 3885.31662873, 3962.25359167, 4039.19055462, 4116.12751756,
 4193.06448051, 4270.00144345, 4346.9384064,  4423.87536934, 4500.81233229,
 4577.74929523, 4654.68625818, 4731.62322112, 4808.56018407, 4885.49714701,
 4962.43410996, 5039.3710729,  5116.30803585, 5193.24499879, 5270.18196174,
 5347.11892468, 5424.05588763, 5500.99285057, 5577.92981352, 5654.86677646]


fig = plt.figure()
ax = fig.add_subplot()

ax.plot(opt, F1, color='red', label= r'$\gamma_{ph}/2\pi = 2 kHz$')
ax.plot(opt, F2, color='blue', label= r'$\gamma_{ph}/2\pi = 4 kHz$')
ax.plot(opt, F3, color='green', label= r'$\gamma_{ph}/2\pi = 6 kHz$')
#ax.fill_between(fine_w_g / chi, min_spectrum_fine, color='gray', alpha=0.5, label='Overlap Area')
ax.set_xlabel(r'$\Omega_{opt}/2\pi (MHz)$')
ax.set_ylabel(r'Readout Fidelity $\mathcal{F}$')
ax.legend()
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

plt.show()