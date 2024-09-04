import numpy as np
from astropy.constants import M_sun, G
import astropy.units as u
import tqdm
import os
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import multiprocessing as mp
import pickle

top = os.getenv('HOME')+'/codes/dispatch2/'
os.chdir(top+'experiments/ISM/python')
import sys
sys.path.insert(0,top+'utilities/python')
import dispatch as dis


dist = lambda dist1, dist2: np.sqrt(np.sum((dist1 - dist2)**2))
calc_ang = lambda vector1, vector2: np.rad2deg(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))
patch_diag = lambda patch: 0.5 * (np.sum(patch.size**2))**0.5


sinks = [6, 13, 14, 25, 82, 122, 162, 180, 225]
true_sinks = [6, 13, 13, 24, 80, 122, 161, 178, 225]
first_sink_snap = [158, 223, 177, 213, 236, 342, 402, 404, 446]

sink_positions_array = np.array([[0.175920050, -0.450297541, 0.281144660],
                           [0.191895130, -0.435872910, 0.288312320], 
                           [-0.441553111, -0.324398830, 0.262413070],
                          [-0.184876630, 0.485943820, -0.436633163],
                          [0.335228030, 0.439704800, -0.193611020],
                          [0.218421970, 0.094505260, -0.157203590],
                          [-0.278816220, -0.433494570, -0.477272036],
                          [0.463706970, -0.032524150, 0.469566290],
                          [-0.181617690, 0.249763490, 0.457740760]])
                          

sink_positions = {sink: sink_positions_array[i] for i, sink in enumerate(sinks)}

#__________________________________USED FOR SAVING DIRECTORIES OF DATA_____________________________#
def serialize_dictionary(filename, store, dictionary = None, folder = '/groups/astro/kxm508/codes/python_dispatch/data_for_plotting/'):
    if store and dictionary == None:
        print('Overwriting existing filename - please choose directory when saving data')
    elif store:
        with open(folder + filename, 'wb') as file:
            pickle.dump(dictionary, file)
    
    elif not store:
        with open(folder + filename, 'rb') as file:
            return pickle.load(file)


class pipeline():
    def __init__(self, snap, run, sink_id, initialize = True, verbose = 1, loading_bar = True, data = '../data/'):
        self.sink_id = sink_id
        self.loading_bar = loading_bar
        self.init_class(snap, run,  initialize = initialize, verbose = verbose, loading_bar=loading_bar, data = data)

    def init_class(self, snap, run, initialize, verbose, loading_bar, μ = 2.34, data = '../data/'):
        self.sn = dis.snapshot(snap, run, data = data)
        self.lmax = np.array([p.level for p in self.sn.patches]).max()
        self.star_pos = self.sn.sinks[self.sink_id][0].position
        self.star_vel = self.sn.sinks[self.sink_id][0].velocity
        self.msun_mass = self.sn.scaling.m / self.sn.cgs.m_sun
        self.yr_time = self.sn.scaling.t / self.sn.cgs.yr    # [yr]
        self.au_length = self.sn.scaling.l / self.sn.cgs.au  # [au]
        self.cgs_density = self.sn.scaling.d                  # [g/cm^3]
        self.cms_velocity = (self.sn.scaling.l / self.sn.scaling.t) 
        self.cgs_pressure = self.sn.scaling.e / self.sn.scaling.l**3
        self.cgs_AM = self.sn.scaling.m * self.sn.scaling.l**2 / self.sn.scaling.t
        self.M_star = self.sn.sinks[self.sink_id][0].mass * self.msun_mass * M_sun
        self.time = self.sn.time * self.yr_time # [yr]
        self.cyl_calculated = False
        self.μ = μ
        

        #_____________________________THE FOLLOWING CONSTANTS AND FUNCTIONS ARE DEFINED IN REGARDS TO THE GAS BEING POLYTROPIC___________________#

        # Calculate normalization constant
        # Polytropic exponents
        g1 = 1.0
        g2 = 1.1
        g3 = 1.4
        g4 = 1.1
        g5 = 5.0 / 3.0
        # upper density limits in numerical units
        r1 = 2.5e-16 / self.cgs_density  # 7.88e4
        r2 = 3.84e-13 / self.cgs_density # 1.21e8
        r3 = 3.84e-8 / self.cgs_density  # 1.21e13
        r4 = 3.84e-3 / self.cgs_density  # 1.21e18
        # normalization constants for each segment using P = k rho^gamma
        # needed to make the pressure continous
        sound_speed = 1
        k1 = sound_speed**2
        k2 = k1 * r1**(g1 - g2)
        k3 = k2 * r2**(g2 - g3)
        k4 = k3 * r3**(g3 - g4)
        k5 = k4 * r4**(g4 - g5)

         # The function below is from: utilities/python/dispatch/EOS/polytrope.py
        # 2. Equation in https://arxiv.org/pdf/2004.07523.pdf
        def calc_gamma(rho):
            result = np.empty_like(rho)
            w1 = rho < r1
            w2 = np.logical_and(rho >= r1, rho < r2)
            w3 = np.logical_and(rho >= r2, rho < r3)
            w4 = np.logical_and(rho >= r3, rho < r4)
            w5 = rho >= r4
            result[w1] = g1
            result[w2] = g2
            result[w3] = g3
            result[w4] = g4
            result[w5] = g5
            return result
        
        def calc_pressure(rho):
            P = np.empty_like(rho)
            w1 = rho < r1
            w2 = np.logical_and(rho >= r1, rho < r2)
            w3 = np.logical_and(rho >= r2, rho < r3)
            w4 = np.logical_and(rho >= r3, rho < r4)
            w5 = rho >= r4
            P[w1] = k1 * rho[w1]**g1
            P[w2] = k2 * rho[w2]**g2
            P[w3] = k3 * rho[w3]**g3
            P[w4] = k4 * rho[w4]**g4
            P[w5] = k5 * rho[w5]**g5
            return P
        
        # From Åke:
        # The Python interface does not properly detect and handle the polytropic EOS case, which would need a hack (to allow a p.var('T')shortcut)   
        # As per what I wrote above, you can just do it yourself instead, as
        # T=p.var(4)/p.var(0)*sn.scaling.temp*mu
   
        if initialize:
            if verbose > 0:
                print('Initialising patch data')
                print('Assigning relative cartesian velocities and coordinates to all cells')
                print('Assigning masses to all cells')
                print('Calculating adiabatic index γ and pressure (polytropic) for all cells')
            #print(f'Assigning temperature to all cells, μ = {μ}')
            for p in tqdm.tqdm(self.sn.patches, disable = not self.loading_bar): # Should take 3s to loop over the patches like this
                XX, YY, ZZ = np.meshgrid(p.xi, p.yi, p.zi, indexing='ij')
                p.xyz = np.array([XX, YY, ZZ]); 
                p.rel_ppos = p.position - self.star_pos
                p.rel_ppos[p.rel_ppos < -0.5] += 1
                p.rel_ppos[p.rel_ppos > 0.5] -= 1
                p.rel_xyz = p.xyz - self.star_pos[:, None, None, None]
                p.rel_xyz[p.rel_xyz < -0.5] += 1
                p.rel_xyz[p.rel_xyz > 0.5] -= 1
                p.dist_xyz = np.linalg.norm(p.rel_xyz, axis = 0) 
                p.vel_xyz = np.asarray([p.var('ux'), p.var('uy'), p.var('uz')]) 
                p.vrel = p.vel_xyz - self.star_vel[:, None, None, None]
                p.m = p.var('d') * np.prod(p.ds)
                p.γ = calc_gamma(p.var('d')) 
                #p.T = p.var('eth')/p.var('d') * self.sn.scaling.temp*μ
                p.P = calc_pressure(p.var('d'))
                

    def sink_evolution(self, start = None, end = None, verbose = 1):
        if verbose > 0: print('Loading all snapshots - this might take awhile')
        if end == None or end > self.sn.iout: end = self.sn.iout
        self.snaps = {}
        t_eval = []; sink_mass = []
        for io in tqdm.tqdm(range(start, end + 1), disable = not self.loading_bar):
            try:
                sn = dis.snapshot(io, self.sn.run)
                unique_sink_datapoints = [sink_eval for i, sink_eval in enumerate(sn.sinks[self.sink_id]) if sn.sinks[self.sink_id][i].time !=sn.sinks[self.sink_id][i-1].time]
                self.snaps[io] = sn
            except:
                continue
            
            sink_times = [_.time for _ in unique_sink_datapoints] 
            sink_masses = [_.mass for _ in unique_sink_datapoints]
            
            t_eval.extend(sink_times)
            sink_mass.extend(sink_masses)        

        t = np.asarray(t_eval).flatten()
        m = np.asarray(sink_mass).flatten()
        ##### The following while loop is to sort out repeated or unphysical data: either backwards timesteps or decrease in sink mass ####
        while True:
            index_to_remove = []
            for i in range(1, len(t)):
                if m[i] < m[i - 1]:
                    index_to_remove.append(i - np.argmax([m[i], m[i - 1]]))
                elif t[i] < t[i - 1]:
                    index_to_remove.append(i)

            if len(index_to_remove) == 0:
                break
            else:
                t = np.delete(t, np.array(index_to_remove))
                m = np.delete(m, np.array(index_to_remove))  
        t_eval = t       
        sink_mass = m
        self.t_eval = np.asarray(t_eval).flatten() * self.yr_time
        self.sink_mass = np.asarray(sink_mass).flatten() * self.msun_mass
        self.sink_accretion = np.gradient(self.sink_mass, self.t_eval, edge_order = 2)


    def change_snapshot(self, change_to, initialize = False):
        self.init_class(change_to, run = self.sn.run, initialize = initialize)

    #Calculate mean angular momentum vector
    def calc_L(self, radius = 100,  angle_to_calc = None, verbose = 0):
        L = np.zeros(3)
        d = radius / self.au_length
        if verbose != 0: 
            patches_skipped = 0 
            contained = 0

        pp = [p for p in self.sn.patches if (p.dist_xyz < d).any()]
        if verbose != 0: print(f'Looping through {len(pp)} patches')

        for p in pp:
            idx = np.nonzero(p.dist_xyz < d)

            if (p.dist_xyz < d).all() and verbose != 0:
             contained += 1
        
            if (p.dist_xyz > d).all():
                if verbose != 0: patches_skipped += 1
                continue

            L_patch = np.cross(p.rel_xyz, p.vel_xyz * p.m , axisa=0, axisb=0, axisc=0)
            L += np.array([np.sum(L_patch[axis][idx]) for axis in range(3)])
        if verbose != 0:
            print("Completely contained patchess:", contained)
            print('Patches skipped:', patches_skipped)
        if isinstance(angle_to_calc, np.ndarray):
            print(f'Angle between the given vector and the mean angular momentum vector: {calc_ang(angle_to_calc, L):2.1f} deg')
  
        self.L =  L / np.linalg.norm(L)
        self.total_L = np.linalg.norm(L)


    # Coordinate transformation into cylindrical coordinates
    def calc_cyl(self):
        try: self.L
        except: self.calc_L()
       
        for p in self.sn.patches:
            p.cyl_z = np.sum(self.L[:, None, None, None] * p.rel_xyz, axis = 0)     # z-coordinate in new axis (Cylindrical)
            p.cyl_r = p.rel_xyz -  p.cyl_z * self.L[:, None, None, None]            # r-coordinate in plane r' = r - ez * r #### NOTE: Is p.cyl_r[0] the transformed x-coordinate? must be...
            p.cyl_R = np.linalg.norm(p.cyl_r, axis = 0) 
            p.e_r = p.cyl_r / p.cyl_R
            p.e_φ = np.cross(self.L, p.e_r, axisa=0, axisb=0, axisc=0)
            p.vφ = np.sum(p.vrel * p.e_φ, axis = 0)
            p.vr = np.sum(p.vrel * p.e_r, axis = 0)
            p.vz = np.sum(p.vrel * self.L[:,None,None,None], axis = 0)
            p.position_cylZ = np.dot(self.L, p.position - self.star_pos)
            p.position_cylr = (p.position - self.star_pos) - p.position_cylZ * self.L
        self.cyl_calculated = True
    

    #Recalculate mean angular momentum vector after transforming into cylindrical coordiantes
    def recalc_L(self, height = 15, radius = 150, err_deg = 5, verbose = 1):
        if not self.cyl_calculated: self.calc_cyl()
        height /= self.au_length; radius /= self.au_length
        pp = [p for p in self.sn.patches if (p.dist_xyz < radius).any()]
        w= np.array([p.level for p in pp]).argsort()[::-1]
        sorted_patches = [pp[w[i]] for i in range(len(pp))]

        def recalc():
            L_new = np.zeros(3)
            for p in sorted_patches:
                nbors = [self.sn.patchid[i] for i in p.nbor_ids if i in self.sn.patchid]
                children = [ n for n in nbors if n.level == p.level + 1]
                leafs = [n for n in children if ((n.position - p.position)**2).sum() < ((p.size)**2).sum()/12]   
                if len(leafs) == 8: continue

                to_extract = (p.cyl_R < radius) & ((abs(p.cyl_z) < height) | (abs(p.cyl_z / p.cyl_R) < 0.3))
                for lp in leafs: 
                    leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
                    covered_bool = ~np.all((p.xyz > leaf_extent[:, 0, None, None, None]) & (p.xyz < leaf_extent[:, 1, None, None, None]), axis=0)
                    to_extract *= covered_bool 
                
                L_patch = np.cross(p.rel_xyz, p.vel_xyz * p.m , axisa=0, axisb=0, axisc=0)
                L_new += np.array([np.sum(L_patch[axis, to_extract]) for axis in range(3)])
            return L_new
        L_new =  recalc()
        L_i = 0
        while calc_ang(self.L, L_new) > err_deg:
            self.L = L_new / np.linalg.norm(L_new); self.total_L = np.linalg.norm(L_new)
            self.calc_cyl()
            L_new = recalc()
            L_i += 1
            if L_i > 20: break
        if verbose != 0: print(f'Converged mean angular momentum vector after {L_i} iteration(s)')


    # Caculate the disk size and thereby also the azimuthal velocity
    def calc_disksize(self, height = 20, radius = 1000, r_in = 10, radial_bins = 200, a = 0.8, plot = True, avg_cells = 10, verbose = 1):
        if not self.cyl_calculated: self.calc_cyl()

        rad_bins = np.logspace(np.log10(r_in), np.log10(radius), radial_bins) / self.au_length    
        height /= self.au_length; radius /= self.au_length

        h_mass_tot = np.zeros(len(rad_bins) - 1)
        h_vφ_tot = np.zeros(len(rad_bins) - 1)
        h_vφ_tot2 = np.zeros(len(rad_bins) - 1)
        if verbose > 0:
            print('Looping through patches, assigning azimuthal velocities to all cells and extracing them within given cylindrical coordiantes')
        for p in tqdm.tqdm(self.sn.patches, disable = not self.loading_bar):
            #Cutting which pathces to look through cells (in height) encompass very large and thereby low level patches not representing the orbital velocity in the disk.
            #The strict cut in pacthes has to made in height - several combination have tested
            if (abs(p.cyl_z) <= height).any and (p.cyl_R < radius).any():
                
                h_mass, _ = np.histogram(p.cyl_R, bins = rad_bins, weights =  p.m)
                h_vφ, _ = np.histogram(p.cyl_R, bins = rad_bins, weights =  p.vφ * p.m)
                h_vφ2, _ = np.histogram(p.cyl_R, bins = rad_bins, weights =  p.vφ**2 * p.m)
                
                h_vφ_tot += h_vφ
                h_mass_tot += h_mass
                h_vφ_tot2 += h_vφ2
        self.vφ = (h_vφ_tot/h_mass_tot) * self.cms_velocity;
        self.vφ2 = (h_vφ_tot2/h_mass_tot) * self.cms_velocity**2;

        r_plot = rad_bins[:-1] + 0.5 * np.diff(rad_bins)
        
        kep_vel = (((G * self.M_star) / (r_plot * self.au_length * u.au))**0.5).to('cm/s').value

        orbitvel_ratio_mean = uniform_filter1d(self.vφ / kep_vel, size = avg_cells)

        for i in range(len(self.vφ)):
            if orbitvel_ratio_mean[i] < a:
                self.disk_size = r_plot[i] * self.au_length
                if verbose > 0: print(f'Disk size: {self.disk_size:2.1f} au')
                break
            else:
                self.disk_size = np.nan
                if verbose > 0: print('No disk size found')


        if plot:
            fig, axs = plt.subplots(1, 2, figsize = (20,6),gridspec_kw={'width_ratios': [2, 1.5]})

            σ_φ = np.sqrt(self.vφ2 - self.vφ**2)

            axs[0].loglog(r_plot * self.au_length, kep_vel, label = 'Keplerian Orbital Velocity', color = 'black')
            axs[0].loglog(r_plot * self.au_length, self.vφ , label = 'Azimuthal velocity v$_φ$', c = 'blue')
            axs[0].fill_between(r_plot * self.au_length, self.vφ - σ_φ, self.vφ + σ_φ, alpha = 0.5, label = '$\pm1\sigma_{φ}$')

            axs[0].set(xlabel = 'Distance from sink [au]', ylabel = 'Orbital speed [cm/s]')

            axs[0].legend(frameon = False)
            axs[1].semilogx(r_plot * self.au_length, orbitvel_ratio_mean, label = 'v$_φ$/v$_K$ ratio', color = 'black', lw = 0.8)
            axs[1].axhline(a, color = 'red', ls = '--', label = f'a = {a}')
            axs[1].axhline(1, color = 'black', ls = '-', alpha = 0.7)
            axs[1].set(xlabel = 'Distance from sink [au]', ylim = (0.5, 1.1))
            axs[1].legend(frameon = False)
    
    def calc_trans_xyz(self, verbose = 1, top = 'L'):
        if not self.cyl_calculated: self.calc_cyl()

        #https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
        def rotation_matrix_func(axis, theta):
            """
            Return the rotation matrix associated with counterclockwise rotation about
            the given axis by theta radians.
            """
            axis = np.asarray(axis)
            axis = axis / np.sqrt(np.dot(axis, axis))
            a = np.cos(theta / 2.0)
            b, c, d = -axis * np.sin(theta / 2.0)
            aa, bb, cc, dd = a * a, b * b, c * c, d * d
            bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
            return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        rotation_axis = np.cross(np.array([0, 0, 1]), self.L)
        theta = np.arccos(np.dot(np.array([0, 0, 1]), self.L))
        rotation_matrix = rotation_matrix_func(rotation_axis, theta)
        self.rotation_matrix = rotation_matrix

        if verbose > 0:
            print('Transforming old z-coordinate into mean angular momentum vector')
        self.new_x = np.dot(self.rotation_matrix, np.array([1,0,0])); self.new_y = np.dot(self.rotation_matrix, np.array([0,1,0]))
        if top != 'L':
            if top == 'x':
                new_x = self.new_y.copy(); new_y = self.L.copy(); new_L = self.new_x.copy() 
            if top == 'y':
                new_x = self.L.copy(); new_y = self.new_x.copy(); new_L = self.new_y.copy()
            self.new_x = new_x; self.new_y = new_y; self.L = new_L 
        for p in tqdm.tqdm(self.sn.patches, disable = not self.loading_bar):
            p.trans_xyz = np.array([np.sum(coor[:, None, None, None] * p.rel_xyz, axis = 0) for coor in [self.new_x, self.new_y, self.L]])
            p.trans_vrel = np.array([np.sum(coor[:, None, None, None] * p.vrel, axis = 0) for coor in [self.new_x, self.new_y, self.L]])
            #p.trans_xyz = np.sum(rotation_matrix[:, :, None, None, None] * p.rel_xyz, axis = 1)
            #p.trans_ppos = np.dot(rotation_matrix, (p.position - self.star_pos))
            p.trans_ppos = np.array([np.dot(coor, p.rel_ppos) for coor in [self.new_x, self.new_y, self.L]])
            proj_r = np.sum(p.cyl_r * self.new_x[:, None, None, None], axis = 0)
            proj_φ = np.sum(p.cyl_r * self.new_y[:, None, None, None], axis = 0)
            p.φ = np.arctan2(proj_φ, proj_r) + np.pi
            

    







        

        




        


