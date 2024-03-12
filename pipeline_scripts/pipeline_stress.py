import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import G
from scipy import interpolate
from scipy.integrate import simps
from matplotlib import colors 
import warnings

from pipeline_main import pipeline

def _fill_2Dhist(self, hist, orig_coor, new_coor, method = 'nearest', periodic_x = True):
    x, y = orig_coor; x_new, y_new = new_coor
    if periodic_x:
        hist = np.vstack((hist[-10:], hist, hist[:10]))
        x = np.concatenate((x[0] - x[:10][::-1], x, x[-1] + x[:10]))

    xx, yy = np.meshgrid(x, y, indexing = 'ij')
    xx_new, yy_new = np.meshgrid(x_new, y_new, indexing='ij')
    ma = np.ma.masked_array(hist.flatten(), mask = np.isnan(hist.flatten()))
    interpolation = interpolate.griddata(np.hstack((xx.flatten()[:,None][~ma.mask], yy.flatten()[:,None][~ma.mask])), ma[~ma.mask], xi = (xx_new, yy_new), method = method, fill_value=-np.nanmin(abs(hist)))
    return interpolation

pipeline._fill_2Dhist = _fill_2Dhist


def L_transport(self, radius = 90, Nh = 100, N_phi = 200, refine_grid = 2, shell_Δpct = 0.05, plot = True, verbose = 1):
    Nr = int(2 * Nh); N_phi_v = N_phi // 2; height = radius
    G_cgs = G.to('cm**3 / (g * s**2)').value
    radius /= self.au_length; height /= self.au_length; 
    
    # The shell is given in terms of the radius; Default is +- 1% -> 0.01
    shell_Δ = np.maximum(shell_Δpct * radius, 0.5**(self.lmax))
    selection_radius = np.sqrt(radius**2 + height**2) * 2

    pp = [p for p in self.sn.patches if np.linalg.norm(p.rel_ppos, axis = 0) < selection_radius]
    w= np.array([p.level for p in pp]).argsort()[::-1]
    sorted_patches = [pp[w[i]] for i in range(len(pp))]

    if verbose == 1: print(f'Using {len(sorted_patches)} patches to retrive angular momentum change')


    #__________________________________________________EXTRACTING KNOWLEDGE OF THE RADIAL PART OF THE STRESS___________________________________________________


    extracted_values =  {key: [] for key in range(7)}
    if verbose == 1: print('Calculating radial part')
    for p in sorted_patches:
        nbors = [self.sn.patchid[i] for i in p.nbor_ids if i in self.sn.patchid]
        children = [ n for n in nbors if n.level == p.level + 1]
        leafs = [n for n in children if ((n.position - p.position)**2).sum() < ((p.size)**2).sum()/12]   
        if len(leafs) == 8: continue

        to_extract = (p.cyl_R > radius - shell_Δ) & (p.cyl_R <  radius + shell_Δ)
        p.B = np.concatenate([p.var(f'b'+axis)[None,...] for axis in ['x','y','z']], axis = 0)
        p.gradφ = np.array(np.gradient(p.var('phi'), p.ds[0], edge_order = 2))
        p.Bφ = np.sum(p.B * p.e_φ, axis = 0)
        p.Br = np.sum(p.B * p.e_r, axis = 0)
        p.vφ = np.sum(p.vrel * p.e_φ, axis = 0)
        p.vr = np.sum(p.vrel * p.e_r, axis = 0)
        p.gradφ_φ = np.sum(p.gradφ * p.e_φ, axis = 0)
        p.gradφ_r = np.sum(p.gradφ * p.e_r, axis = 0)
        
        
        for lp in leafs: 
            leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
            covered_bool = ~np.all((p.xyz > leaf_extent[:, 0, None, None, None]) & (p.xyz < leaf_extent[:, 1, None, None, None]), axis=0)
            to_extract *= covered_bool 
        
        z_coor = p.cyl_z[to_extract].T 
        φ_coor = p.φ[to_extract].T
        vel_φr = p.vr[to_extract].T * p.vφ[to_extract].T
        B_φr =  p.Br[to_extract].T * p.Bφ[to_extract].T
        mass_val = p.m[to_extract].T 
        gradφ_φr = p.gradφ_r[to_extract].T * p.gradφ_φ[to_extract].T
        
        extracted_values[0].extend(z_coor.tolist())
        extracted_values[1].extend(φ_coor.tolist())
        extracted_values[2].extend(vel_φr.tolist())
        extracted_values[3].extend(mass_val.tolist())
        extracted_values[4].extend(p.ds[0]**3 * np.ones(len(mass_val)))
        extracted_values[5].extend(B_φr.tolist())
        extracted_values[6].extend(gradφ_φr.tolist())

    for key in extracted_values:
        extracted_values[key] = np.array(extracted_values[key])

    #Making grid in height and phi direction:
    z_grid = np.linspace(-height, height, Nh); phi_grid = np.linspace(0, 2 * np.pi, N_phi)

    #Binning values
    hist_mass, binedges_phi, binedges_z = np.histogram2d(extracted_values[1], extracted_values[0], bins = (phi_grid, z_grid), weights = extracted_values[3])
    hist_vol, _, _ =  np.histogram2d(extracted_values[1], extracted_values[0], bins = (phi_grid, z_grid), weights = extracted_values[4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        hist_vφvr = np.histogram2d(extracted_values[1], extracted_values[0],  bins = (phi_grid, z_grid), weights = extracted_values[2] * extracted_values[3])[0] / hist_mass 
        hist_BφBr = np.histogram2d(extracted_values[1], extracted_values[0],  bins = (phi_grid, z_grid), weights = extracted_values[5] * extracted_values[3])[0] / hist_mass 
        hist_gradφ_φr =np.histogram2d(extracted_values[1], extracted_values[0],  bins = (phi_grid, z_grid), weights = extracted_values[6] * extracted_values[3])[0] / hist_mass 

        hist_ρ = hist_mass/hist_vol

    if (hist_mass == 0).any and verbose == 1: print('Radial 2D histogram not completely covered')
    reynolds_radial = - hist_vφvr * hist_ρ * self.cms_velocity**2 * self.cgs_density
    maxwell_radial = hist_BφBr / (4 * np.pi) * self.sn.scaling.b**2
    grav_radial = hist_gradφ_φr / (4 * np.pi * G_cgs) * (self.cms_velocity / self.sn.scaling.t)**2

    z_bins = z_grid[:-1] + 0.5 * np.diff(binedges_z)
    phi_bins = phi_grid[:-1] + 0.5 * np.diff(binedges_phi)

    znew_grid = np.linspace(-height, height, Nh * refine_grid); phinew_grid = np.linspace(0, 2 * np.pi, N_phi * refine_grid)

    ### PRINTING TO SEE WHAT THE RATIO BETWEEN EMPTY AND FULL CELLS IS
    if verbose == 2 or verbose == 1: print(f'Ratio of nan-valued cells to be filled by interpolation: {np.sum(np.isnan(reynolds_radial))/np.prod(reynolds_radial.shape) * 100:1.2f} %')
    reynolds_Ir = self._fill_2Dhist(reynolds_radial, orig_coor=[phi_bins, z_bins], new_coor=[phinew_grid, znew_grid], periodic_x=True)
    maxwell_Ir = self._fill_2Dhist(maxwell_radial, orig_coor=[phi_bins, z_bins], new_coor=[phinew_grid, znew_grid], periodic_x=True)
    grav_Ir = self._fill_2Dhist(grav_radial, orig_coor=[phi_bins, z_bins], new_coor=[phinew_grid, znew_grid], periodic_x=True)


    #__________________________________________________PLOTTING THE RADIAL PART OF THE STRESS_______________________________________________________

    tick_labels = ['$\pi$/3','2$\pi$/3','$\pi$', '4$\pi$/3', '5$\pi$/3', '2$\pi$']
    tick_values = [np.pi/3, 2*np.pi/3, np.pi, np.pi/3 + np.pi, 2*np.pi/3 + np.pi, 2*np.pi]
    stress_names = ['Reynolds', 'Maxwells', 'Grav. instability', 'Total']

    if plot:
        fig, axs = plt.subplots(4,1, figsize = (20,16))
        for ax in axs.flatten():
            ax.set(ylabel = '[AU]', xlim = (0, 2*np.pi))

        ytick = radius * self.au_length // 2

        ax = axs[0]
        ax.set_xticks(tick_values); ax.set_xticklabels(tick_labels); ax.xaxis.tick_top()

        for ax in axs.flatten()[1:3]:
            ax.set_xticklabels([])
            
        total_stress_R = reynolds_Ir + maxwell_Ir + grav_Ir
        ax = axs[-1]
        ax.set_xticks(tick_values); ax.set_xticklabels(tick_labels);
        ax.pcolormesh(phinew_grid, znew_grid * self.au_length, total_stress_R.T, norm = colors.SymLogNorm(linthresh=1e-10, linscale=0.5, vmin = -1e-5, vmax = 1e-5), snap = True, shading = 'gouraud', cmap = 'coolwarm')
        ax.set_yticks([-ytick,0, ytick])
        ax.text(0.99, 0.95, 'Total', transform=ax.transAxes, ha='right', va='top', fontsize = 24)


        for ax, stress, name in zip(axs.flatten(), [reynolds_Ir, maxwell_Ir, grav_Ir], stress_names):
            cs = ax.pcolormesh(phinew_grid, znew_grid * self.au_length, stress.T, norm = colors.SymLogNorm(linthresh=1e-10, linscale=0.5, vmin = -1e-5, vmax = 1e-5), snap = True, shading = 'gouraud', cmap = 'coolwarm')
            ax.set_yticks([-ytick,0, ytick])
            ax.text(0.99, 0.95, name, transform=ax.transAxes, ha='right', va='top', fontsize = 24)


        fig.subplots_adjust(wspace=0, hspace=0)
        cbar = fig.colorbar(cs, ax=axs.ravel().tolist(), fraction = 0.04, pad = 0.01)
        cbar.set_label('Energy density [Ba]',fontsize = 18)
    
    
    #___________________________________CALCULATING THE INTEGRAL (RADIAL) TO GET TOTAL CHANGE I ANGULAR MOMENTUM__________________________________________
    R = radius * self.sn.scaling.l
    ΔL_Rr = simps(simps(reynolds_Ir * R**2, phinew_grid, axis = 0), znew_grid * self.sn.scaling.l)
    ΔL_Mr = simps(simps(maxwell_Ir * R**2, phinew_grid, axis = 0), znew_grid * self.sn.scaling.l)
    ΔL_Gr = simps(simps(grav_Ir * R**2, phinew_grid, axis = 0), znew_grid * self.sn.scaling.l)


    #__________________________________________EXTRACTING KNOWLEDGE OF THE VERTICAL PART OF THE STRESS___________________________________________________

    #Looping over top and then bottom of the cylinder:
    for top in [1, 0]:

        extracted_values =  {key: [] for key in range(7)} # So far this is only for densities and velocities
        if verbose == 1: print('Calculating vertical part')
        for p in sorted_patches:
            nbors = [self.sn.patchid[i] for i in p.nbor_ids if i in self.sn.patchid]
            children = [ n for n in nbors if n.level == p.level + 1]
            leafs = [n for n in children if ((n.position - p.position)**2).sum() < ((p.size)**2).sum()/12]   
            if len(leafs) == 8: continue

            if top == True:
                to_extract = (p.cyl_z > height - shell_Δ) & (p.cyl_z <  height + shell_Δ)
            elif top == False:
                to_extract = (p.cyl_z > - height - shell_Δ) & (p.cyl_z < - height + shell_Δ)

            p.Bz = np.sum(p.B * self.L[:,None, None, None], axis = 0)
            p.vz = np.sum(p.vrel * self.L[:,None, None, None], axis = 0)
            p.gradφ_z = np.sum(p.gradφ * self.L[:,None, None, None], axis = 0)
            
            for lp in leafs: 
                leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
                covered_bool = ~np.all((p.xyz > leaf_extent[:, 0, None, None, None]) & (p.xyz < leaf_extent[:, 1, None, None, None]), axis=0)
                to_extract *= covered_bool 
            
            r_coor = p.cyl_R[to_extract].T 
            φ_coor = p.φ[to_extract].T
            vel_φz = p.vz[to_extract].T * p.vφ[to_extract].T
            B_φz =  p.Bz[to_extract].T * p.Bφ[to_extract].T
            mass_val = p.m[to_extract].T 
            gradφ_φz = p.gradφ_r[to_extract].T * p.gradφ_z[to_extract].T
            
            extracted_values[0].extend(r_coor.tolist())
            extracted_values[1].extend(φ_coor.tolist())
            extracted_values[2].extend(vel_φz.tolist())
            extracted_values[3].extend(mass_val.tolist())
            extracted_values[4].extend(p.ds[0]**3 * np.ones(len(mass_val)))
            extracted_values[5].extend(B_φz.tolist())
            extracted_values[6].extend(gradφ_φz.tolist())

        for key in extracted_values:
            extracted_values[key] = np.array(extracted_values[key])

        #Making grid in height and phi direction:
        r_grid = np.logspace(np.log10(1e-3 / self.au_length), np.log10(radius), Nr); phi_grid = np.linspace(0, 2 * np.pi, N_phi_v)

        #Binning values
        hist_mass, binedges_phiv, binedges_r = np.histogram2d(extracted_values[1], extracted_values[0], bins = (phi_grid, r_grid), weights = extracted_values[3])
        hist_vol, _, _ =  np.histogram2d(extracted_values[1], extracted_values[0], bins = (phi_grid, r_grid), weights = extracted_values[4])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            hist_vφvz = np.histogram2d(extracted_values[1], extracted_values[0],  bins = (phi_grid, r_grid), weights = extracted_values[2] * extracted_values[3])[0] / hist_mass 
            hist_BφBz = np.histogram2d(extracted_values[1], extracted_values[0],  bins = (phi_grid, r_grid), weights = extracted_values[5] * extracted_values[3])[0] / hist_mass 
            hist_gradφ_φz = np.histogram2d(extracted_values[1], extracted_values[0],  bins = (phi_grid, r_grid), weights = extracted_values[6] * extracted_values[3])[0] / hist_mass 

            hist_ρ = hist_mass/hist_vol
        if (hist_mass == 0).any and top and verbose == 1: print('Vertical (top) 2D histogram not completely covered')
        if (hist_mass == 0).any and top and verbose == 1: print('Vertical (bottom) 2D histogram not completely covered')
        if top == True:
            reynolds_vertical = - hist_vφvz * hist_ρ * self.cms_velocity**2 * self.cgs_density
            maxwell_vertical = hist_BφBz / (4 * np.pi) * self.sn.scaling.b**2
            grav_vertical = - hist_gradφ_φz / (4 * np.pi * G_cgs) * (self.cms_velocity / self.sn.scaling.t)**2
            
        elif top == False:
            reynolds_vertical = hist_vφvz * hist_ρ * self.cms_velocity**2 * self.cgs_density
            maxwell_vertical = - hist_BφBz / (4 * np.pi) * self.sn.scaling.b**2
            grav_vertical = hist_gradφ_φz / (4 * np.pi * G_cgs) * (self.cms_velocity / self.sn.scaling.t)**2

        r_bins = r_grid[:-1] + 0.5 * np.diff(binedges_r)
        phi_binsv = phi_grid[:-1] + 0.5 * np.diff(binedges_phiv)

        rnew_grid =np.logspace(np.log10(1e-3 / self.au_length), np.log10(radius), Nr * refine_grid); 
        phinew_gridv = np.linspace(0, 2 * np.pi, N_phi_v * refine_grid)


        if top == True:
            reynolds_Iv = self._fill_2Dhist(reynolds_vertical, orig_coor=[phi_binsv, r_bins], new_coor=[phinew_gridv, rnew_grid], periodic_x=True)
            maxwell_Iv = self._fill_2Dhist(maxwell_vertical, orig_coor=[phi_binsv, r_bins], new_coor=[phinew_gridv, rnew_grid], periodic_x=True)
            grav_Iv = self._fill_2Dhist(grav_vertical, orig_coor=[phi_binsv, r_bins], new_coor=[phinew_gridv, rnew_grid], periodic_x=True)

        elif top == False:
            reynolds_Iv += self._fill_2Dhist(reynolds_vertical, orig_coor=[phi_binsv, r_bins], new_coor=[phinew_gridv, rnew_grid], periodic_x=True)
            maxwell_Iv += self._fill_2Dhist(maxwell_vertical, orig_coor=[phi_binsv, r_bins], new_coor=[phinew_gridv, rnew_grid], periodic_x=True)
            grav_Iv += self._fill_2Dhist(grav_vertical, orig_coor=[phi_binsv, r_bins], new_coor=[phinew_gridv, rnew_grid], periodic_x=True)

        

    #__________________________________________________PLOTTING THE RADIAL PART OF THE STRESS_______________________________________________________


    if plot:
        fig, axs = plt.subplots(2,2, figsize = (20,20), subplot_kw={'projection' :'polar'})

        for ax in axs.flatten():
            ax.set_xticks([]); ax.set_xticklabels([])
            ax.set_yticks([])
            
        total_stress_V = reynolds_Iv + maxwell_Iv + grav_Iv
        ax = axs.flatten()[-1]
        ax.set_xticks([]); ax.set_xticklabels([]);
        ax.pcolormesh(phinew_gridv, rnew_grid * self.au_length, total_stress_V.T, norm = colors.SymLogNorm(linthresh=1e-10, linscale=0.5, vmin = -1e-5, vmax = 1e-5), snap = True, shading = 'gouraud', cmap = 'coolwarm')
        ax.set(title = 'Total')

        for ax, stress, name in zip(axs.flatten(), [reynolds_Iv, maxwell_Iv, grav_Iv], stress_names):
            cs = ax.pcolormesh(phinew_gridv, rnew_grid * self.au_length, stress.T, norm = colors.SymLogNorm(linthresh=1e-10, linscale=0.5, vmin = -1e-5, vmax = 1e-5), snap = True, shading = 'gouraud', cmap = 'coolwarm')
            ax.set(title = name)

        fig.subplots_adjust(wspace=0, hspace=0.01)
        cbar = fig.colorbar(cs, ax=axs.ravel().tolist(), fraction = 0.04, pad = 0.02)
        cbar.set_label('Energy density [Ba]',fontsize = 18)


    #___________________________________CALCULATING THE INTEGRAL (VERTICAL) TO GET TOTAL CHANGE IN ANGULAR MOMENTUM__________________________________________


    r = rnew_grid * self.sn.scaling.l
    ΔL_Rv = simps(simps(reynolds_Iv * r**2, phinew_gridv, axis = 0), r)
    ΔL_Mv = simps(simps(maxwell_Iv * r**2, phinew_gridv, axis = 0), r)
    ΔL_Gv = simps(simps(grav_Iv * r**2, phinew_gridv, axis = 0), r)



    #________________________________________________CALCULATING TOTAL ANGULAR MOMENTUM WITHIN CYLINDER_______________________________________________


    def calc_L(height = height, radius = radius):
        if verbose == 1: print('Calculating total angular momentum within the cylinder')
        if not self.cyl_calculated: self.calc_cyl()

        pp = [p for p in self.sn.patches if np.linalg.norm(p.rel_ppos, axis = 0) < selection_radius]
        w= np.array([p.level for p in pp]).argsort()[::-1]
        sorted_patches = [pp[w[i]] for i in range(len(pp))]

        L_new = np.zeros(3)
        for p in sorted_patches:
            nbors = [self.sn.patchid[i] for i in p.nbor_ids if i in self.sn.patchid]
            children = [ n for n in nbors if n.level == p.level + 1]
            leafs = [n for n in children if ((n.position - p.position)**2).sum() < ((p.size)**2).sum()/12]   
            if len(leafs) == 8: continue

            to_extract = (p.cyl_R < radius) & ((abs(p.cyl_z) < height))
            for lp in leafs: 
                leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
                covered_bool = ~np.all((p.xyz > leaf_extent[:, 0, None, None, None]) & (p.xyz < leaf_extent[:, 1, None, None, None]), axis=0)
                to_extract *= covered_bool 

            L_patch = np.cross(p.rel_xyz, p.vel_xyz * p.m , axisa=0, axisb=0, axisc=0)
            L_new += np.array([np.sum(L_patch[axis][to_extract]) for axis in range(3)])
        return L_new * self.sn.scaling.m * self.cms_velocity * self.sn.scaling.l
    
    L_total = np.linalg.norm(calc_L())

    # The order of the computed loss of angular momentum is:
    # 'Reynolds', 'Maxwells', 'Grav. instability', 'Total'
    # 1. The cylinder walls, 2. The cylinder top and bottom compined
    # Everything is in cgs units so g*cm^2 / s^2 for the integral parts

    def stresses(self, verbose = 1):
        if verbose == 1: 
            print(f'Stresses for radius = {radius * self.au_length:2.0f} au, height = {2 * radius * self.au_length:2.0f} au')
            print('Order of stresses:',stress_names,'\n0: Radial\n1: Vertical\n2: Total angular momentum')
            print('All values are givin in cgs-units')
        radial = np.array([ΔL_Rr, ΔL_Mr, ΔL_Gr, ΔL_Rr + ΔL_Mr + ΔL_Gr])
        vertical = np.array([ΔL_Rv, ΔL_Mv, ΔL_Gv, ΔL_Rv + ΔL_Mv + ΔL_Gv])
        return radial, vertical, L_total
    
    pipeline.stresses = stresses
    
pipeline.L_transport = L_transport




###################################################################################################################
############################################ HERE THE MASSFLUX FUNCION STARTS #####################################

def massflux(self, radius = 90, Nh = 100, N_phi = 200, refine_grid = 2, shell_Δpct = 0.01, plot = True, verbose = 1):
    Nr = int(2 * Nh); N_phi_v = N_phi // 2; height = radius
    
    radius /= self.au_length; height /= self.au_length; 

    # The shell is given in terms of the radius; Default is +- 1% -> 0.01
    shell_Δ = np.maximum(shell_Δpct * radius, 0.5**(self.lmax))
    selection_radius = np.sqrt(radius**2 + height**2) * 2

    pp = [p for p in self.sn.patches if np.linalg.norm(p.rel_ppos, axis = 0) < selection_radius]
    w= np.array([p.level for p in pp]).argsort()[::-1]
    sorted_patches = [pp[w[i]] for i in range(len(pp))]

    if verbose == 1: print(f'Using {len(sorted_patches)} patches to retrive mass flux')


    #__________________________________________________EXTRACTING KNOWLEDGE OF THE RADIAL PART OF THE MASS FLUX___________________________________________________


    extracted_values =  {key: [] for key in range(5)}
    if verbose == 1: print('Calculating radial part')
    for p in sorted_patches:
        nbors = [self.sn.patchid[i] for i in p.nbor_ids if i in self.sn.patchid]
        children = [ n for n in nbors if n.level == p.level + 1]
        leafs = [n for n in children if ((n.position - p.position)**2).sum() < ((p.size)**2).sum()/12]   
        if len(leafs) == 8: continue

        to_extract = (p.cyl_R > radius - shell_Δ) & (p.cyl_R <  radius + shell_Δ)
        p.vφ = np.sum(p.vrel * p.e_φ, axis = 0)
        p.vr = np.sum(p.vrel * p.e_r, axis = 0)

        for lp in leafs: 
            leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
            covered_bool = ~np.all((p.xyz > leaf_extent[:, 0, None, None, None]) & (p.xyz < leaf_extent[:, 1, None, None, None]), axis=0)
            to_extract *= covered_bool
         
        z_coor = p.cyl_z[to_extract].T 
        φ_coor = p.φ[to_extract].T
        vel_r = p.vr[to_extract].T 
        mass_val = p.m[to_extract].T

        extracted_values[0].extend(z_coor.tolist())
        extracted_values[1].extend(φ_coor.tolist())
        extracted_values[2].extend(vel_r.tolist())
        extracted_values[3].extend(mass_val.tolist())
        extracted_values[4].extend(p.ds[0]**3 * np.ones(len(mass_val)))
    
    for key in extracted_values:
        extracted_values[key] = np.array(extracted_values[key])
    
    #Making grid in height and phi direction:
    z_grid = np.linspace(-height, height, Nh); phi_grid = np.linspace(0, 2 * np.pi, N_phi)

    #Binning values
    hist_mass, binedges_phi, binedges_z = np.histogram2d(extracted_values[1], extracted_values[0], bins = (phi_grid, z_grid), weights = extracted_values[3])
    hist_vol, _, _ =  np.histogram2d(extracted_values[1], extracted_values[0], bins = (phi_grid, z_grid), weights = extracted_values[4])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        hist_vr = np.histogram2d(extracted_values[1], extracted_values[0],  bins = (phi_grid, z_grid), weights = extracted_values[2] * extracted_values[3])[0] / hist_mass 
        hist_ρ = hist_mass/hist_vol
    
    if (hist_mass == 0).any and verbose == 1: print('Radial 2D histogram not completely covered')

    massflux_radial = - hist_vr * hist_ρ * self.cms_velocity * self.cgs_density 

    z_bins = z_grid[:-1] + 0.5 * np.diff(binedges_z)
    phi_bins = phi_grid[:-1] + 0.5 * np.diff(binedges_phi)

    znew_grid = np.linspace(-height, height, Nh * refine_grid); phinew_grid = np.linspace(0, 2 * np.pi, N_phi * refine_grid)

    massflux_Ir = self._fill_2Dhist(massflux_radial, orig_coor=[phi_bins, z_bins], new_coor=[phinew_grid, znew_grid], periodic_x=True)

    #__________________________________________________PLOTTING THE RADIAL PART OF THE STRESS_______________________________________________________

    tick_labels = ['$\pi$/3','2$\pi$/3','$\pi$', '4$\pi$/3', '5$\pi$/3', '2$\pi$']
    tick_values = [np.pi/3, 2*np.pi/3, np.pi, np.pi/3 + np.pi, 2*np.pi/3 + np.pi, 2*np.pi]

    if plot:
        fig, ax = plt.subplots(figsize = (14, 6))
        ax.set(ylabel = '[AU]', xlim = (0, 2*np.pi))
        ax.set_xticks(tick_values); ax.set_xticklabels(tick_labels)

        cs = ax.pcolormesh(phinew_grid, znew_grid * self.au_length, massflux_Ir.T, norm = colors.SymLogNorm(linthresh=1e-16, linscale=1, vmin = -1e-10, vmax = 1e-10), snap = True, shading = 'gouraud', cmap = 'coolwarm')
        cbar = fig.colorbar(cs, ax=ax, fraction = 0.2, pad = 0.06, location = 'top')
        cbar.set_label('Mass flux [g/cm$^2$/s]',fontsize = 18)

    #_______________________________________CALCULATING THE INTEGRAL (RADIAL) TO GET TOTAL ACCRETION RATE__________________________________________
    R = radius * self.sn.scaling.l

    Mdot_r = simps(simps(massflux_Ir * R, phinew_grid, axis = 0), znew_grid * self.sn.scaling.l)


    #__________________________________________EXTRACTING KNOWLEDGE OF THE VERTICAL PART OF THE MASS FLUX___________________________________________________


    #Looping over top and then bottom of the cylinder:
    for top in [1, 0]:
        extracted_values =  {key: [] for key in range(5)} # So far this is only for densities and velocities
        if verbose == 1: print('Calculating vertical part')
        for p in sorted_patches:
            nbors = [self.sn.patchid[i] for i in p.nbor_ids if i in self.sn.patchid]
            children = [ n for n in nbors if n.level == p.level + 1]
            leafs = [n for n in children if ((n.position - p.position)**2).sum() < ((p.size)**2).sum()/12]   
            if len(leafs) == 8: continue

            if top == True:
                to_extract = (p.cyl_z > height - shell_Δ) & (p.cyl_z <  height + shell_Δ)
            elif top == False:
                to_extract = (p.cyl_z > - height - shell_Δ) & (p.cyl_z < - height + shell_Δ)

            p.vz = np.sum(p.vrel * self.L[:,None, None, None], axis = 0)

            for lp in leafs: 
                leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
                covered_bool = ~np.all((p.xyz > leaf_extent[:, 0, None, None, None]) & (p.xyz < leaf_extent[:, 1, None, None, None]), axis=0)
                to_extract *= covered_bool

            r_coor = p.cyl_R[to_extract].T
            φ_coor = p.φ[to_extract].T
            vel_z = p.vz[to_extract].T
            mass_val = p.m[to_extract].T

            extracted_values[0].extend(r_coor.tolist())
            extracted_values[1].extend(φ_coor.tolist())
            extracted_values[2].extend(vel_z.tolist())
            extracted_values[3].extend(mass_val.tolist())
            extracted_values[4].extend(p.ds[0]**3 * np.ones(len(mass_val)))
        
        for key in extracted_values:
            extracted_values[key] = np.array(extracted_values[key])

        #Making grid in height and phi direction:
        r_grid = np.logspace(np.log10(1e-3 / self.au_length), np.log10(radius), Nr); phi_grid = np.linspace(0, 2 * np.pi, N_phi_v)

        #Binning values
        hist_mass, binedges_phiv, binedges_r = np.histogram2d(extracted_values[1], extracted_values[0], bins = (phi_grid, r_grid), weights = extracted_values[3])
        hist_vol, _, _ =  np.histogram2d(extracted_values[1], extracted_values[0], bins = (phi_grid, r_grid), weights = extracted_values[4])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            hist_vz = np.histogram2d(extracted_values[1], extracted_values[0],  bins = (phi_grid, r_grid), weights = extracted_values[2] * extracted_values[3])[0] / hist_mass 
            hist_ρ = hist_mass/hist_vol

        if (hist_mass == 0).any and top and verbose == 1: print('Vertical (top) 2D histogram not completely covered') 
        if (hist_mass == 0).any and top and verbose == 1: print('Vertical (bottom) 2D histogram not completely covered')

        if top == True:
            massflux_vertical = - hist_vz * hist_ρ * self.cms_velocity * self.cgs_density
        elif top == False:
            massflux_vertical = hist_vz * hist_ρ * self.cms_velocity * self.cgs_density


        r_bins = r_grid[:-1] + 0.5 * np.diff(binedges_r)
        phi_binsv = phi_grid[:-1] + 0.5 * np.diff(binedges_phiv)

        rnew_grid =np.logspace(np.log10(1e-3 / self.au_length), np.log10(radius), Nr * refine_grid); 
        phinew_gridv = np.linspace(0, 2 * np.pi, N_phi_v * refine_grid)

        if top == True:
            massflux_Iv = self._fill_2Dhist(massflux_vertical, orig_coor=[phi_binsv, r_bins], new_coor=[phinew_gridv, rnew_grid], periodic_x=True)
        elif top == False:
            massflux_Iv += self._fill_2Dhist(massflux_vertical, orig_coor=[phi_binsv, r_bins], new_coor=[phinew_gridv, rnew_grid], periodic_x=True)

    #__________________________________________________PLOTTING THE VERTICAL PART OF THE MASS FLUX_______________________________________________________
            
    if plot:
        fig, ax = plt.subplots(figsize = (10,10), subplot_kw={'projection' :'polar'})
        ax.set_xticks([]); ax.set_xticklabels([])
        ax.set_yticks([])

        cs = ax.pcolormesh(phinew_gridv, rnew_grid * self.au_length, massflux_Iv.T, norm = colors.SymLogNorm(linthresh=1e-16, linscale=1, vmin = -1e-10, vmax = 1e-10), snap = True, shading = 'gouraud', cmap = 'coolwarm')
        cbar = fig.colorbar(cs, ax=ax, fraction = 0.1, pad = 0.06)
        cbar.set_label('Mass flux [g/cm$^2$/s]',fontsize = 18)

    
    #___________________________________CALCULATING THE INTEGRAL (VERTICAL) TO GET TOTAL MASS ACCRETION__________________________________________
        
    r = rnew_grid * self.sn.scaling.l
    Mdot_v = simps(simps(massflux_Iv * r, phinew_gridv, axis = 0), r)

    if verbose == 1: print('Returns total accretion rate in radial and vertical direction resp.\nUnit: [g/s]')
    return np.array([Mdot_r, Mdot_v])

pipeline.massflux = massflux