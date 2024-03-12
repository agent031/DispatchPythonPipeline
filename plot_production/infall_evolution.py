import sys
import numpy as np
from astropy.constants import M_sun, G
import astropy.units as u
from scipy.integrate import simps
from scipy.interpolate import interp1d

sys.path.append('../my_funcs/')
from pipeline_main import pipeline, serialize_directory
from pipeline_streamers import infall_sphere 


#### DATA TO EXTRACT ####
# sink122core02 343	395
# goodold13     223	281
# sink24core02  213	256 snapshot 233 is corrupted
# sink13core02  177 217 snapshot 198 is corrupted
# sink178core03 404	442
# sink225core03 446	524

data_name = ['sink122core02', 'goodold13','sink13core02','sink178core03','sink225core03']
initial_snap = [343, 223, 177, 404, 446]
final_snap = [395, 282, 217, 442, 528]
sink_id = [122, 13, 13, 178, 225]
save_as = ['122', 'old13', '13', '178', '225_2']

for data_i in range(3,4):
    datai = pipeline(snap = initial_snap[data_i], run = data_name[data_i], sink_id = sink_id[data_i], loading_bar = False, verbose = 0)
    data = pipeline(snap = final_snap[data_i], run = data_name[data_i], sink_id = sink_id[data_i], loading_bar = False, verbose = 0)
    data.recalc_L(verbose = 0)

    data.sink_evolution(start = initial_snap[data_i], verbose = 0)

    snapshots = np.arange(datai.sn.iout, data.sn.iout)
    infall_spheres = [50, 200, 400, 700, 1000]
    hammer_data = {key: np.zeros(len(data.snaps)) for key in infall_spheres}
    hammer_data['M_dot'] = data.sink_accretion.copy()
    hammer_data['M_star'] = data.sink_mass.copy()
    hammer_data['sink_time'] = data.t_eval.copy()
    hammer_data['time'] = np.zeros(len(data.snaps))
    hammer_data['t_ff'] = np.zeros((len(infall_spheres), len(data.snaps)))

    # Take about 36min to run
    for snaps_j in range(len(data.snaps.keys())):
        
        data_loop = pipeline(snap =  initial_snap[data_i] + snaps_j , run = data_name[data_i], sink_id = sink_id[data_i], loading_bar = False, verbose = 0)
        
        data_loop.calc_L(verbose = 0); data_loop.calc_cyl(); data_loop.calc_trans_xyz(verbose = 0)

        for shells_k, shells_radius in enumerate(infall_spheres):
            _, _, _, hammer_data[shells_radius][snaps_j] = data_loop.infall_sphere(shell_r=shells_radius, get_data=True, plot = False, verbose = 0)
            hammer_data['time'][snaps_j] = data_loop.time - datai.time + 500

            #### The code below is the calculate the free-fall time###
            #### t_ff is calculated by rough estimate of the gas density within the sphere and the mass of the sink at that time####
            #### the sphere density is almost negligible #### 
            mass_tot = 0; vol_tot = 0
            for p in data_loop.sn.patches:
                if np.linalg.norm(p.rel_ppos) * data_loop.au_length <= shells_radius:
                    mass_tot += np.sum(p.m); vol_tot += np.prod(p.ds) * np.prod(p.var('d').shape)
            mean_dens = mass_tot / vol_tot * data_loop.cgs_density
            sphere_dens = mean_dens  +  (data_loop.M_star / (4/3 * (shells_radius * u.au)**3 * np.pi)).to('g/cm^3').value   
            hammer_data['t_ff'][shells_k, snaps_j] = np.sqrt((3 * np.pi / (32 * G * sphere_dens * (u.g/u.cm**3)))).to('yr').value

    serialize_directory(filename = 'accretion_spheres_s' + save_as[data_i], directory=hammer_data, store = True)
    print('Done with ' + data_name[data_i])
    

