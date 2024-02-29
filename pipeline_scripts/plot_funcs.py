import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import pickle

planes = ['ZY', 'ZX', 'YX']

def plot_density(dens_data, width, snapshot, lmax = 20,  axis = 0, hide_ticks = True):

    ds = 0.5**lmax
    size = width * ds
    extent = size * snapshot.scaling.l / snapshot.cgs.au
    cgs_density = snapshot.scaling.d
    cgs_to_yr = snapshot.scaling.t / snapshot.cgs.yr
    

    fig, axs = plt.subplots(figsize = (10, 10))
    plot_data_d = np.log10(np.take(dens_data * cgs_density, np.shape(dens_data)[axis] // 2, axis=axis)).transpose()

    cs = axs.imshow(plot_data_d, cmap = 'gist_heat', extent = (-extent/2, extent/2, -extent/2, extent/2), origin = 'lower')
    cbar = fig.colorbar(cs, ax = axs, fraction = 0.04);
    cbar.set_label('$\log_{10}(ρ)$\n[ρ]: g/cm$^3$', labelpad = -40, y = 1.1, rotation = 0, fontsize = 18)

    axs.set_title(f'Log10 Density, {planes[axis]}-plane, Time = {snapshot.time * cgs_to_yr:4.0f} yr', fontsize = 20)
    if hide_ticks:
        axs.set_xticks([]); axs.set_yticks([])
        scalebar = AnchoredSizeBar(axs.transData, 10, '10 AU', 'lower center', 
                           pad=0.5,
                           color='white',
                           frameon=False,
                           fontproperties = fm.FontProperties(size=25),
                           size_vertical=1,
                           label_top=True)

        axs.add_artist(scalebar)
    else:
        axs.set_xlabel(f'{planes[axis][-1]}, Distance [AU]', fontsize = 16); axs.set_ylabel(f'{planes[axis][0]}, Distance [AU]', fontsize = 16)

    plt.tight_layout()


def plot_velocity(vel_data, width, snapshot, axis = 0, colorbar_lim = 30, add_title='', lmax = 20):

    ds = 0.5**lmax
    size = width * ds
    extent = size * snapshot.scaling.l / snapshot.cgs.au
    kms_velocity = (snapshot.scaling.l / snapshot.scaling.t) / 1e5
    cgs_to_yr = snapshot.scaling.t / snapshot.cgs.yr

    fig, axs = plt.subplots(figsize = (10, 10))
    plot_data = np.take(vel_data, np.shape(vel_data)[axis] // 2, axis=axis).transpose() * kms_velocity

    cs = axs.imshow(plot_data, cmap = 'coolwarm', extent = (-extent/2, extent/2, -extent/2, extent/2), origin = 'lower', vmin = -colorbar_lim, vmax = colorbar_lim)
    cbar = fig.colorbar(cs, ax = axs, fraction = 0.04);
    cbar.set_label('Velocity\n[km/s]', labelpad = -40, y = 1.1, rotation = 0, fontsize = 16)

    axs.set_title(add_title + f'Velocity, {planes[axis]}-plane, Time = {snapshot.time * cgs_to_yr:4.0f} yr', fontsize = 20)
    axs.set_xlabel(f'{planes[axis][-1]}, Distance [AU]', fontsize = 16); axs.set_ylabel(f'{planes[axis][0]}, Distance [AU]', fontsize = 16)

    plt.tight_layout()

