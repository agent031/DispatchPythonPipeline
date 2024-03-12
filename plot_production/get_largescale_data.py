import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.constants import M_sun
import numpy as np
import os
import sys

sys.path.append('../my_funcs/')

from pipeline_main_nosink import pipeline_nosink
from pipeline_2D_nosink import to_osyris_ivs

data = pipeline_nosink(snap = 236, run = 'sink80core01', sink_pos = np.array([0,0,0]))

width_au = 4 * data.sn.cgs.pc / data.sn.cgs.au 

variable = ['d']
data.to_osyris_ivs(variables=variable, resolution=1500, view = width_au, dz = width_au, viewpoint=np.array([1,0,0]), plot = False)

np.savetxt('data_for_plotting/fullbox_1500_s80.txt', data.osyris_ivs[0])