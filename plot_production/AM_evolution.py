import sys
import numpy as np
from scipy.integrate import simps

sys.path.append('../my_funcs/')
from pipeline_main import pipeline, serialize_directory
from pipeline_1D import *


#### DATA TO EXTRACT ####
# sink122core02 343	395
# goodold13     223	281
# sink24core02  213	256 snapshot 233 is corrupted
# sink13core02  177 217 snapshot 198 is corrupted
# sink178core03 404	442
# sink225core03 446	524

data_name = ['sink122core02', 'goodold13','sink178core03','sink225core03']
initial_snap = [343, 223, 404, 446]
final_snap = [395, 282, 442, 528]
sink_id = [122, 13, 13, 178, 225]
save_as = ['122', 'old13', '178', '225']


for data_i in range(len(data_name)):
    datai = pipeline(snap = initial_snap[data_i], run = data_name[data_i], sink_id = sink_id[data_i], loading_bar = False, verbose = 0)
    data = pipeline(snap = final_snap[data_i], run = data_name[data_i], sink_id = sink_id[data_i], loading_bar = False, verbose = 0)    

    snapshots = np.arange(datai.sn.iout, data.sn.iout)
    dict_to_save = {key: np.zeros_like(snapshots, dtype=float) for key in ['disk_size', 'L_tot', 'L_kep', 'time']}


    for snaps_j in range(len(snapshots)):
        data_loop = pipeline(snap = datai.sn.iout + snaps_j , run = data_name[data_i], sink_id = sink_id[data_i], loading_bar = False, verbose = 0)
        data_loop.recalc_L(verbose = 0)
        data_loop.calc_disksize(verbose=0, plot = False)

        dict_to_save['disk_size'][snaps_j] = np.copy(data_loop.disk_size)

        data_loop.to_1D(plot = False, r_in= 5, r_out = 100, Nr = 100, verbose = 0)
        data_loop.get_1D_param(Ω = True, verbose = 0, get_units=False)
        H = data_loop.H_1D[:,0]
        r = data_loop.r_1D * data.sn.scaling.l
        Σ = data_loop.Σ_1D[:,0]
        v_φ = data_loop.vφ_1D[:,0]
        v_kep = data_loop.kepVφ_1D

        L_tot = simps(2 * np.pi * Σ * v_φ * r**2, r)
        L_kep = simps(2 * np.pi * Σ * v_kep * r**2, r)  
        dict_to_save['L_tot'][snaps_j] = L_tot * 1e-50
        dict_to_save['L_kep'][snaps_j] = L_kep * 1e-50
        dict_to_save['time'][snaps_j] = data_loop.time - datai.time + 500

    serialize_directory(filename = 'AM_evolution_s' + save_as[data_i] + '.pkl', directory=dict_to_save, store = True)
    print('Done with ' + data_name[data_i])