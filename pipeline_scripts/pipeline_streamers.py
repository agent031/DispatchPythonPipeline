# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.9.13 (main, Aug 25 2022, 23:51:50) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: /lustre/hpc/astro/kxm508/codes/dispatch2/experiments/ISM/python/plot_production/../my_funcs/pipeline_streamers.py
# Compiled at: 2024-02-07 15:53:34
# Size of source mod 2**32: 12285 bytes
import numpy as np, tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from scipy.integrate import simps, dblquad
import matplotlib as mpl
from scipy import interpolate
import warnings
from pipeline_main import pipeline
from pipeline_stress import _fill_2Dhist

def infall_sphere(self, shell_r=50, shell_Δpct=0.05, lon_N=360, lat_N=180, range_plot=1e-09, linear_threshold=1e-13, dpi=100, get_data=True, plot=True, verbose=1):
    shell_r /= self.au_length
    Δ_r = np.maximum(shell_Δpct * shell_r, 2.5 * 0.5 ** self.lmax) 
    patch_values = []
    patch_cartcoor = []
    longtitude = []
    latitude = []
    patch_mass = []
    if verbose > 0: print('Loop through patch present in defined shell')

    pp = [p for p in self.sn.patches if (p.dist_xyz < shell_r * 2).any() and p.level > 5]
    w = np.array([p.level for p in pp]).argsort()[::-1]
    sorted_patches = [pp[w[i]] for i in range(len(pp))]

    for p in tqdm.tqdm(sorted_patches, disable=(not self.loading_bar)):
        nbors = [self.sn.patchid[i] for i in p.nbor_ids if i in self.sn.patchid]
        children = [n for n in nbors if n.level == p.level + 1]
        leafs = [n for n in children if ((n.position - p.position) ** 2).sum() < (p.size ** 2).sum() / 12]
        if len(leafs) == 8: continue
        to_extract = (p.dist_xyz < shell_r + Δ_r) & (p.dist_xyz > shell_r - Δ_r)
        for lp in leafs:
            leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
            covered_bool = ~np.all(((p.xyz > leaf_extent[:, 0, None, None, None]) & (p.xyz < leaf_extent[:, 1, None, None, None])), axis=0)
            to_extract *= covered_bool

        new_xyz = p.trans_xyz[:, to_extract].T
        new_R = np.linalg.norm(new_xyz, axis=1)
        new_value = (p.var('d') * np.sum((p.trans_vrel *  p.trans_xyz / np.linalg.norm(p.trans_xyz, axis=0)), axis=0))[to_extract].T
        mass = p.m[to_extract].T
        longtitude.extend(np.arctan2(new_xyz[:, 1], new_xyz[:, 0]).tolist())

        ### Np.arcccos gives values wihtin [0, π] but pcolormesh only plots values [-π/2, π/2] and there I add π/2 ###
        latitude.extend((np.arccos(new_xyz[:, 2] / new_R)  - np.pi / 2).tolist()) 
        patch_values.extend(new_value.tolist())
        patch_cartcoor.extend(new_xyz.tolist())
        patch_mass.extend(mass.tolist())


    longtitude = np.asarray(longtitude)
    latitude = np.asarray(latitude)
    patch_values = np.asarray(patch_values)
    patch_mass = np.asarray(patch_mass)
    lon = np.linspace(-np.pi, np.pi, lon_N)
    lat = np.linspace(-np.pi/2, np.pi/2, lat_N)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        counts, binedges_lon, binedges_lat = np.histogram2d(longtitude, latitude, bins=(lon, lat))
        hist_values = np.histogram2d(longtitude, latitude, bins=(lon, lat), weights=(patch_values))[0] / counts

    lon_bins = lon[:-1] + 0.5 * np.diff(binedges_lon)
    lat_bins = lat[:-1] + 0.5 * np.diff(binedges_lat)
    Lon, Lat = np.meshgrid(lon_bins, lat_bins, indexing='ij')
    
    if lon_N == 360 and lat_N == 180:
        # Area of spherical cells in code-units
        cell_areas = np.genfromtxt('/groups/astro/kxm508/codes/python_dispatch/pipeline_scripts/default_spherical_cellsize.txt') * shell_r**2 
    else:    
        @np.vectorize
        def calc_area_sphere(R, φi, φf, θi, θf):
            return dblquad((lambda θ, φ: R ** 2 * np.sin(φ)), θi, θf, φi, φf)[0]
        lon_new = lon + np.pi
        lat_new = lat + 0.5 * np.pi
        cell_areas = np.array([calc_area_sphere(shell_r, lon_new[0], lon_new[1], lat_new[:-1], lat_new[1:]) for _ in range(lon_N - 1)])

    proj_data = self._fill_2Dhist(hist_values, orig_coor=[lon_bins, lat_bins], new_coor=[lon_bins, lat_bins], method='linear', periodic_x=False)
    proj_data *= cell_areas
    proj_data *= -self.msun_mass / self.sn.cgs.yr   
    total_infall = np.sum(proj_data)   
        
    if plot:
        fig = plt.figure(figsize=(10, 7), dpi=dpi)
        ax = fig.add_subplot(111, projection='hammer')
        ax.set_yticks([])
        ax.set_xticks([])
        im = ax.pcolormesh(Lon, Lat, proj_data, cmap='coolwarm', snap=True, norm=colors.SymLogNorm(linthresh=linear_threshold, linscale=0.5, vmin=(-range_plot), vmax=range_plot), shading='gouraud')
        cbar = fig.colorbar(im, orientation='horizontal')
        cbar.set_label('Mass accretion [M$_\\odot$yr$^{-1}$]', labelpad=(-80), y=2, rotation=0, fontsize=16)
        ax.set(title=f"Radius = {shell_r * self.au_length:2.0f}$\\pm${Δ_r * self.au_length:1.0f} au, Total infall {total_infall * 1000000.0:2.1f} 10$^{{-6}}$ M$_\\odot$yr$^{-1}$")
        plt.tight_layout()
    if get_data:
        return (Lon, Lat, proj_data, total_infall, cell_areas)


pipeline.infall_sphere = infall_sphere


def flow_fraction(self, hammer_data, verbose = 1):
    if verbose > 0:
        print('Output along axis = 1:\nMass flux [M_sun/yr]\nCell size [au^2]\nLatitude [rad]\nMass flux [M_sun/yr au^2]')
    Lon, Lat, proj_data, total_infall, cell_areas = hammer_data
    
    bool_pos = proj_data > 0
    bool_neg = proj_data < 0

    outflow_cells = proj_data[bool_neg], cell_areas[bool_neg], Lat[bool_neg], proj_data[bool_neg] / cell_areas[bool_neg]
    infall_cells = proj_data[bool_pos], cell_areas[bool_pos], Lat[bool_pos],  proj_data[bool_pos] / cell_areas[bool_pos]

    outflow_cells = np.asarray(outflow_cells).T
    infall_cells = np.asarray(infall_cells).T

    infall_index = np.argsort(infall_cells[:,-1])[::-1]     # Sorts after the largest inflowrates first
    infall_new = np.copy(infall_cells[infall_index])

    outflow_index = np.argsort(outflow_cells[:,-1])         # Sorts after the largest negative outflow first
    outflow_new = np.copy(outflow_cells[outflow_index])
    return infall_new, outflow_new, total_infall, cell_areas

pipeline.flow_fraction = flow_fraction


################ NOW WORKING WITH IVS LIST; IS FIXED #################
###############################################################################


def phi_average(self, variables = None, variables_weight = None, radius=50, height=20, NR=80, Nh_half=30, origo_close=1, phi_extent=None, quiver_dens=0.6, log_vmin=-20, log_vmax=-12, plot=True):
    radius /= self.au_length
    height /= self.au_length
    selection_radius = np.sqrt(radius ** 2 + height ** 2) * 1.2
    pp = [p for p in self.sn.patches if np.linalg.norm((p.rel_ppos), axis=0) < selection_radius]
    w = np.array([p.level for p in pp]).argsort()[::-1]
    sorted_patches = [pp[w[i]] for i in range(len(pp))]
    extracted_values = {key: [] for key in range(6)}
    if variables != None: extracted_values.update({key: [] for key in variables})

    try:
        self.rotation_matrix
    except:
        self.calc_trans_xyz()

    for p in sorted_patches:
        nbors = [self.sn.patchid[i] for i in p.nbor_ids if i in self.sn.patchid]
        children = [n for n in nbors if n.level == p.level + 1]
        leafs = [n for n in children if ((n.position - p.position) ** 2).sum() < (p.size ** 2).sum() / 12]
        if len(leafs) == 8:
            pass
        to_extract = (p.cyl_R < radius) & (abs(p.cyl_z) < height)
        p.vz = np.sum((p.vrel * self.L[:, None, None, None]), axis=0)
        p.vr = np.sum((p.vrel * p.e_r), axis=0)
        if phi_extent != None:
            to_extract *= (p.φ > phi_extent[0]) & (p.φ < phi_extent[1])

        for lp in leafs:
            leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
            covered_bool = ~np.all(((p.xyz > leaf_extent[:, 0, None, None, None]) & (p.xyz < leaf_extent[:, 1, None, None, None])), axis=0)
            to_extract *= covered_bool
 
        vel_r = p.vr[to_extract].T
        vel_z = p.vz[to_extract].T
        z_coor = p.cyl_z[to_extract].T
        R_coor = p.cyl_R[to_extract].T
        mass_val = p.m[to_extract].T
        if variables != None:
            for i, iv in enumerate(variables):
                if hasattr(p, iv):
                    value = getattr(p, iv)[to_extract].T
                else:
                    value = p.var(iv)[to_extract].T
                extracted_values[iv].extend(value)          

        extracted_values[0].extend(R_coor.tolist())
        extracted_values[1].extend(z_coor.tolist())
        extracted_values[2].extend(vel_r.tolist())
        extracted_values[3].extend(vel_z.tolist())
        extracted_values[4].extend(mass_val.tolist())
        extracted_values[5].extend(p.ds[0] ** 3 * np.ones(len(mass_val)))

    for key in extracted_values:
        extracted_values[key] = np.array(extracted_values[key])

    R_grid = np.logspace(np.log10(origo_close), np.log10(radius * self.au_length), NR) / self.au_length
    R_grid = np.insert(R_grid, 0, 0)
    z_grid = np.logspace(np.log10(origo_close), np.log10(height * self.au_length), Nh_half) / self.au_length
    z_grid = np.insert(z_grid, 0, 0)
    z_grid = np.concatenate((-np.logspace(np.log10(origo_close), np.log10(height * self.au_length), Nh_half)[::-1] / self.au_length, z_grid))
    hist_mass, binedges_R, binedges_z = np.histogram2d((extracted_values[0]), (extracted_values[1]), bins=(R_grid, z_grid), weights=(extracted_values[4]))
    hist_vol, _, _ = np.histogram2d((extracted_values[0]), (extracted_values[1]), bins=(R_grid, z_grid), weights=(extracted_values[5]))
    hist_ρ = hist_mass * self.sn.scaling.m / (hist_vol * self.sn.scaling.l ** 3)

    R_bins = R_grid[:-1] + 0.5 * np.diff(binedges_R)
    z_bins = z_grid[:-1] + 0.5 * np.diff(binedges_z)

    quiver_shift = 1
    quivergrid_vr = np.linspace(origo_close / 2, R_grid.max() * self.au_length - quiver_shift, int(NR * quiver_dens)) / self.au_length
    quivergrid_vz = np.linspace(origo_close / 2, z_grid.max() * self.au_length - quiver_shift, int(Nh_half * quiver_dens)) / self.au_length
    quivergrid_vz = np.concatenate((-np.linspace(origo_close / 2, z_grid.max() * self.au_length - quiver_shift, int(Nh_half * quiver_dens))[::-1] / self.au_length, quivergrid_vz))

    mass, qbinedgesR, qbinedgesz = np.histogram2d((extracted_values[0]), (extracted_values[1]), bins=(quivergrid_vr, quivergrid_vz), weights = extracted_values[4])
    hist_vr, _, _ = np.histogram2d((extracted_values[0]), (extracted_values[1]), bins=(quivergrid_vr, quivergrid_vz), weights=(extracted_values[4] * extracted_values[2] * self.cms_velocity))
    hist_vz, _, _ = np.histogram2d((extracted_values[0]), (extracted_values[1]), bins=(quivergrid_vr, quivergrid_vz), weights=(extracted_values[4] * extracted_values[3] * self.cms_velocity))    
    
    arrow_length = np.sqrt((hist_vr / mass) ** 2 + (hist_vz / mass) ** 2)
    arrow_length[np.isnan(arrow_length)] = 1e-20

    vR_bins = quivergrid_vr[:-1] + 0.5 * np.diff(qbinedgesR)
    vz_bins = quivergrid_vz[:-1] + 0.5 * np.diff(qbinedgesz)

    rr_ρ, zz_ρ = np.meshgrid(R_bins, z_bins, indexing='ij')
    rr_v, zz_v = np.meshgrid(vR_bins, vz_bins, indexing='ij')

    mask = np.isnan(hist_ρ.flatten())
    masked_hist_ρ = np.ma.masked_array((hist_ρ.flatten()), mask=mask)
    interpolation = interpolate.griddata((np.hstack((rr_ρ.flatten()[:, None][~mask], zz_ρ.flatten()[:, None][~mask]))), (masked_hist_ρ[~mask]), xi=(rr_ρ, zz_ρ), method='nearest')

    data = {}
    data['r_bins'] = R_bins; 
    data['z_bins'] = z_bins; 
    data['d'] = interpolation; 
    data['quiver_r_bins'] = rr_v; 
    data['quiver_z_bins'] = zz_v; 
    data['hist_vr'] = hist_vr / mass / arrow_length; 
    data['hist_vz'] =  hist_vz / mass / arrow_length; 
    data['arrow_length'] = arrow_length
    
    if variables != None:
        for i, iv in enumerate(variables):
            if variables_weight[i] == 'mass' : weight = extracted_values[4]; weight_hist = hist_mass
            elif variables_weight[i] == 'volume' : weight = extracted_values[5]; weight_hist = hist_vol

            hist_val, _, _ = np.histogram2d((extracted_values[0]), (extracted_values[1]), bins=(R_grid, z_grid), weights=(weight * extracted_values[iv]))
            hist_val /= weight_hist

            masked_hist_val = np.ma.masked_array((hist_val.flatten()), mask=mask)

            data[iv] = interpolate.griddata((np.hstack((rr_ρ.flatten()[:, None][~mask], zz_ρ.flatten()[:, None][~mask]))), (masked_hist_val[~mask]), xi=(rr_ρ, zz_ρ), method='nearest')     
        
    if plot:
        fig, axs = plt.subplots(figsize=(20, 8))
        cs = axs.contourf((R_bins * self.au_length), (z_bins * self.au_length), (np.log10(interpolation.T)), vmin=log_vmin, vmax=log_vmax, origin='lower', levels=200, cmap='gist_heat')
        cbar = fig.colorbar(ScalarMappable(norm=(cs.norm), cmap=(cs.cmap)), ticks=(range(log_vmin, log_vmax + 1, 1)), ax=axs, fraction=0.1, pad=0.06, location='top')
        quiver = axs.quiver((rr_v * self.au_length), (zz_v * self.au_length), (hist_vr / mass / arrow_length), (hist_vz / mass / arrow_length), (np.log10(arrow_length)), cmap=(mpl.cm.Greys_r),
            headwidth=2.5,
            headaxislength=2.3,
            headlength=2.3,
            pivot='mid',
            scale=(100 / (0.6 / quiver_dens)))
        cbar_vel = fig.colorbar(quiver, pad=0.005)
        cbar_vel.set_label('$\\log_{10}(V)$ [cm/s]')
        cbar.set_label('$\\log_{10}(ρ)$\n[ρ]: g/cm$^3$', labelpad=(-60), x=(-0.08), rotation=0, fontsize=18)
        axs.set(ylabel='Height over midplane [AU]', xlabel='Distance from star  [AU]')
        if phi_extent != None:
            axs.set_title(f"Averaged over φ: [{phi_extent[0]:1.2f},{phi_extent[1]:1.2f}] rad")
        axs.axhline(0, c='black', alpha=0.4)
        fig.tight_layout()

    return data


pipeline.phi_average = phi_average