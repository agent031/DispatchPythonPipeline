import numpy as np
from astropy.constants import M_sun, G
import astropy.units as u
import tqdm
import os
import matplotlib.pyplot as plt

top = os.getenv('HOME')+'/codes/dispatch2/'
os.chdir(top+'experiments/ISM/python')
import sys
sys.path.insert(0,top+'utilities/python')
import dispatch as dis


dist = lambda dist1, dist2: np.sqrt(np.sum((dist1 - dist2)**2))
calc_ang = lambda vector1, vector2: np.rad2deg(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))
patch_diag = lambda patch: 0.5 * (np.sum(patch.size**2))**0.5

class pipeline_nosink():
    def __init__(self, snap, run, sink_pos, initialize = True, data = '../data/'):
        self.init_class(snap, run,  sink_pos=sink_pos, initialize = initialize, data = data)

    def init_class(self, snap, run, sink_pos, initialize, data = '../data/'):
        self.sn = dis.snapshot(snap, run, data = data)
        self.star_pos = sink_pos #self.sn.sinks[self.sink_id][0].position
        #self.star_vel = self.sn.sinks[self.sink_id][0].velocity
        self.msun_mass = self.sn.scaling.m / self.sn.cgs.m_sun
        self.yr_time = self.sn.scaling.t / self.sn.cgs.yr    # [yr]
        self.au_length = self.sn.scaling.l / self.sn.cgs.au  # [au]
        self.cgs_density = self.sn.scaling.d                  # [g/cm^3]
        self.cms_velocity = (self.sn.scaling.l / self.sn.scaling.t) 
        #self.M_star = self.sn.sinks[self.sink_id][0].mass * self.msun_mass * M_sun
        self.time = self.sn.time * self.yr_time # [yr]
        self.cyl_calculated = False
   
        if initialize:
            print('Initialising patch data')
            for p in tqdm.tqdm(self.sn.patches): # Should take 3s to loop over the patches like this
                XX, YY, ZZ = np.meshgrid(p.xi, p.yi, p.zi, indexing='ij')
                p.xyz = np.array([XX, YY, ZZ]); 
                p.rel_ppos = p.position - self.star_pos
                p.rel_xyz = p.xyz - self.star_pos[:, None, None, None]
                p.dist_xyz = np.linalg.norm(p.rel_xyz, axis = 0) 
                p.vel_xyz = np.asarray([p.var('ux'), p.var('uy'), p.var('uz')]) 
                p.m = p.var('d') * np.prod(p.ds) 

    def sink_evolution(self, start = 223, end = None):
        print('Loading all snapshots - this might take awhile')
        if end == None or end > self.sn.iout: end = self.sn.iout
        self.snaps = {}
        self.t_eval = []; self.sink_mass = []
        for io in tqdm.tqdm(range(start, end + 1)):
            sn = dis.snapshot(io, self.sn.run)
            self.snaps[io] = sn
            self.t_eval.append(sn.sinks[self.sink_id][0].time * self.yr_time)
            self.sink_mass.append(sn.sinks[self.sink_id][0].mass * self.msun_mass)
        self.t_eval = np.asarray(self.t_eval); self.mass_evo = np.asarray(self.sink_mass)
        self.sink_accretion = np.gradient(self.sink_mass, self.t_eval, edge_order = 2)

    def change_snapshot(self, change_to, initialize = False):
        self.init_class(change_to, run = self.sn.run, initialize = initialize)

    #Calculate mean angular momentum vector
    def calc_L(self, distance = 100,  angle_to_calc = None, verbose = 0):
        L = np.zeros(3)
        d = distance / self.au_length
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
            p.position_cylZ = np.dot(self.L, p.position - self.star_pos)
            p.position_cylr = (p.position - self.star_pos) - p.position_cylZ * self.L
        self.cyl_calculated = True
    

    #Recalculate mean angular momentum vector after transforming into cylindrical coordiantes
    def recalc_L(self, height = 10, radius = 100, err_deg = 1, verbose = 1):
        if not self.cyl_calculated: self.calc_cyl()
        height /= self.au_length; radius /= self.au_length

        def recalc():
            L_new = np.zeros(3)
            for p in self.sn.patches:
                idx = np.nonzero((p.cyl_R < radius) & ((abs(p.cyl_z) < height) | (abs(p.cyl_z / p.cyl_R) < 0.3)))
                L_patch = np.cross(p.rel_xyz, p.vel_xyz * p.m , axisa=0, axisb=0, axisc=0)
                L_new += np.array([np.sum(L_patch[axis][idx]) for axis in range(3)])
            return L_new / np.linalg.norm(L_new)
        L_new =  recalc()
        L_i = 0
        while calc_ang(self.L, L_new) > err_deg:
            self.L = L_new
            self.calc_cyl()
            L_new = recalc()
            L_i += 1
        if verbose != 0: print(f'Converged mean angular momentum vector after {L_i} iteration(s)')


    # Caculate the disk size and thereby also the azimuthal velocity
    def calc_disksize(self, height = 10, radius = 1000, r_in = 10, radial_bins = 100, a = 0.8, plot = True):
        if not self.cyl_calculated: self.calc_cyl()

        rad_bins = np.logspace(np.log10(r_in), np.log10(radius), radial_bins) / self.au_length    
        height /= self.au_length; radius /= self.au_length

        h_mass_tot = np.zeros(len(rad_bins) - 1)
        h_vφ_tot = np.zeros(len(rad_bins) - 1)
        h_vφ_tot2 = np.zeros(len(rad_bins) - 1)
        print('Looping through patches, assigning azimuthal velocities to all cells and extracing them within given cylindrical coordiantes')
        for p in tqdm.tqdm(self.sn.patches):
            p.vrel = p.vel_xyz - self.star_vel[:, None, None, None]
            p.vφ = np.sum(p.vrel * p.e_φ, axis = 0)
            #Cutting which pathces to look through cells (in height) encompass very large and thereby low level patches not representing the orbital velocity in the disk.
            #The strict cut in pacthes has to made in height - several combination have tested
            if abs(p.position_cylZ) <= height and (p.cyl_R < radius).any():
                
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

        for i in range(len(self.vφ)):
            if self.vφ[i] / kep_vel[i] < a:
                self.disk_size = r_plot[i] * self.au_length
                print(f'Disk size: {self.disk_size:2.1f} au')
                break

        if plot:
            fig, axs = plt.subplots(1, 2, figsize = (20,6),gridspec_kw={'width_ratios': [2, 1.5]})

            σ_φ = np.sqrt(self.vφ2 - self.vφ**2)

            axs[0].loglog(r_plot * self.au_length, kep_vel, label = 'Keplerian Orbital Velocity', color = 'black')
            axs[0].loglog(r_plot * self.au_length, self.vφ , label = 'Azimuthal velocity v$_φ$', c = 'blue')
            axs[0].fill_between(r_plot * self.au_length, self.vφ - σ_φ, self.vφ + σ_φ, alpha = 0.5, label = '$\pm1\sigma_{φ}$')

            axs[0].set(xlabel = 'Distance from sink [AU]', ylabel = 'Orbital speed [cm/s]')

            axs[0].legend(frameon = False)
            axs[1].semilogx(r_plot * self.au_length, self.vφ / kep_vel, label = 'v$_φ$/v$_K$ ratio', color = 'black', lw = 0.8)
            axs[1].axhline(a, color = 'red', ls = '--', label = f'a = {a}')
            axs[1].axhline(1, color = 'black', ls = '-', alpha = 0.7)
            axs[1].set(xlabel = 'Distance from sink [AU]', ylim = (0.5, 1.1))
            axs[1].legend(frameon = False)
    
    def calc_trans_xyz(self):
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

        print('Transforming old z-coordinate into mean angular momentum vector')
        new_x = np.dot(self.rotation_matrix, np.array([1,0,0])); new_y = np.dot(self.rotation_matrix, np.array([0,1,0]))
        for p in tqdm.tqdm(self.sn.patches):
            p.trans_xyz = np.sum(rotation_matrix[:, :, None, None, None] * p.rel_xyz, axis = 1)
            p.trans_ppos = np.dot(rotation_matrix, (p.position - self.star_pos))
            proj_r = np.sum(p.cyl_r * new_x[:, None, None, None], axis = 0)
            proj_φ = np.sum(p.cyl_r * new_y[:, None, None, None], axis = 0)
            p.φ = np.arctan2(proj_φ, proj_r) + np.pi
            

    







        

        




        


