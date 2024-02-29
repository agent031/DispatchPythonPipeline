import numpy as np
from astropy.constants import M_sun, G
import astropy.units as u
from lmfit import Model
import tqdm
import os
import matplotlib.pyplot as plt

from pipeline_main import pipeline

# Calculate the surface density of the disk
def to_1D(self, r_in = 5, r_out = 100, Nr = 100, plot = True, dpi = 100):
    if not self.cyl_calculated: self.recalc_L(verbose = 0)

    self.Nr = Nr
    self.r_bins = np.logspace(np.log10(r_in), np.log10(r_out), Nr) / self.au_length #[au]
    self.r_1D = self.r_bins[:-1] + 0.5 * np.diff(self.r_bins) 

    def Hp_func(x, Σ, H): return (Σ) / (np.sqrt(2 * np.pi) * H) * np.exp( - x**2 / (2 * H**2)) 

    def fit_scaleheight(ρ, h, x0):
        model = Model(Hp_func)
        params = model.make_params(Σ = x0[0], H = x0[1])
        result = model.fit(ρ, x = h, params = params)
        fit_params = np.array(list(result.best_values.values()))
        fit_err = np.array([par.stderr for _, par in result.params.items()])
        fit_params[0] *= self.sn.cgs.au ; fit_err[0] *= self.sn.cgs.au

        return np.array([fit_params[0], fit_err[0]]), np.array([fit_params[1], fit_err[1]]) 


    densities = {key: [] for key in range(Nr - 1)}
    heights = {key: [] for key in range(Nr - 1)}
    print('Looping through patches to extract densities and heights')
    for p in tqdm.tqdm(self.sn.patches):
        # Radial cut in the bins are saved for later since the cut will be used repeatedly
        p.bin_idx1D = np.digitize(p.cyl_R.flatten(), bins = self.r_bins)                                   #Assigning each point to their repective bin 
        if (p.bin_idx1D == Nr).all() or (p.bin_idx1D == 0).all(): continue                                    #If the points are outside of the given bin the index is 0 or 25 (dependent on the length of radial bins)
        for bin in np.unique(p.bin_idx1D):
            if bin == 0 or bin == Nr: continue                                            
            h_idx = np.nonzero(abs(p.cyl_z.flatten()[p.bin_idx1D == bin]) < 2 * self.r_1D[bin - 1])        # Now I make a cut only taking cells within 2 * the radial bins. I want the densities directory to be 0-indexet hence bin - 1
            if len(h_idx) == 0: continue                                             
            
            densities[bin - 1].extend(p.var('d').flatten()[p.bin_idx1D == bin][h_idx])
            heights[bin - 1].extend(p.cyl_z.flatten()[p.bin_idx1D == bin][h_idx])

    for key in densities:
        densities[key] = np.array(densities[key]) * self.cgs_density
        heights[key] = np.array(heights[key]) * self.au_length
    
    self.Σ_1D = np.zeros((Nr - 1, 2))
    self.H_1D = np.zeros((Nr - 1, 2))
    x0 = np.array([1e3, 7])

    print('Fitting surface density and scaleheight in each radial bin')
    for i in tqdm.tqdm(range(Nr - 1)):    
        self.Σ_1D[i], self.H_1D[i] = fit_scaleheight(ρ = densities[i], h = heights[i], x0 = x0)
        x0 = np.array([self.Σ_1D[i, 0], self.H_1D[i, 0]])
    

    def calc_sigma(fitted_Hp):
        annulus_m_sum = np.zeros(Nr - 1)
        annulus_V_sum = np.zeros(Nr - 1)     
        for p in self.sn.patches:                                
                if (p.bin_idx1D == Nr).all() or (p.bin_idx1D == 0).all(): continue                    
                
                for bin in np.unique(p.bin_idx1D):
                    if bin == 0 or bin == Nr: continue                                            
                    h_idx = np.nonzero(abs(p.cyl_z.flatten()[p.bin_idx1D == bin]) < (fitted_Hp[bin - 1]) / self.au_length)    
                    if len(h_idx) == 0: continue                                             

                    annulus_m_sum[bin - 1] += np.sum(p.m.flatten()[p.bin_idx1D == bin][h_idx])
                    annulus_V_sum[bin - 1] += np.prod(p.ds) * len(p.m.flatten()[p.bin_idx1D == bin][h_idx])                               # The cell volume multiplied with the number of cells within the given cut

        annulus_vol = np.pi * (np.roll(self.r_bins, -1)[:-1]**2 - self.r_bins[:-1]**2) * 2 * fitted_Hp / self.au_length               #np.pi*((radius + Δ_radius)**2 - (radius -  Δ_radius)**2) * H_p
        annulus_area = np.pi * (np.roll(self.r_bins, -1)[:-1]**2 - self.r_bins[:-1]**2) * self.sn.scaling.l**2                         #Area of each annulus in [cm**2]
        annulus_mtot = annulus_m_sum /annulus_V_sum  * annulus_vol * self.sn.scaling.m                                                 #Average of density from total cell mass over total cell volume

        Σ_calc = annulus_mtot / annulus_area
        return Σ_calc
    
    if plot:
        print('Validating fit...')
        sigmas = np.asarray([calc_sigma(σ * self.H_1D[:,0]) for σ in range(1, 3)])
        fig, axs = plt.subplots(1,3, figsize = (20, 6), dpi = dpi)
        ax = axs[0]
        ax.grid(which = 'minor', ls = '--', c = 'grey');

        ax.loglog(self.r_1D * self.au_length, self.Σ_1D[:,0], color = 'blue', label = 'Σ$_{Fit}$')
        for i in reversed(range(1, 3)):
            ax.loglog(self.r_1D * self.au_length, sigmas[i - 1], color = 'red', label = 'Σ$_{Calc}$'+f'$\propto\int\pm{i}H_p$', alpha = i/2, lw = 0.8)
        ax.fill_between(self.r_1D * self.au_length, self.Σ_1D[:,0] + self.Σ_1D[:,1], self.Σ_1D[:,0] - self.Σ_1D[:,1], alpha = 0.45, color = 'blue')
        ax.set(ylabel = 'Σ$_{gas}$ [g/cm$^2$]', xlabel = 'Distance from sink [au]', title = 'Surface density Σ$_{gas}$(r)')
        ax.legend(frameon = False)

        ax = axs[1]
        ax.grid(which = 'both', ls = '--')
        ax.loglog(self.r_1D * self.au_length, self.H_1D[:,0], label = 'Scaleheight H$_p$', color = 'green')
        ax.fill_between(self.r_1D * self.au_length, self.H_1D[:,0] + self.H_1D[:,1], self.H_1D[:,0] - self.H_1D[:,1], alpha = 0.3, color = 'green', label = '$\pm σ_H$')
        ax.set(ylabel = 'Scaleheight [au]', xlabel = 'Distance from sink [au]', title = 'Scaleheight  H$_p$(r)')
        ax.legend(frameon = False)

        ax = axs[2]
        self.φ =  np.vstack((np.sin(self.H_1D[:,0] / (self.r_1D * self.au_length)), np.cos(self.H_1D[:,1] / (self.r_1D * self.au_length)) * (self.r_1D * self.au_length)**(-1) * self.H_1D[:,1])).T
        ax.grid()

        ax.semilogx(self.r_1D * self.au_length, self.φ[:,0], color = 'purple', label = 'Opening angle H$_p$')
        ax.fill_between(self.r_1D * self.au_length, self.φ[:,0] + self.φ[:,1], self.φ[:,0] - self.φ[:,1], color = 'purple', alpha = 0.3, label = '$\pm σ_φ$')

        #Values for ticks
        values = np.linspace(0, np.pi/2, 5)
        names = ['$0$', '$π/8$', '$π/4$', '$3π/8$', '$π/2$']
        ax.set_yticks(values); ax.set_yticklabels(names)
        ax2 = ax.twinx()
        ax2.set_yticks(np.rad2deg(values))
        ax2.set_yticklabels([f'{deg:2.0f}'+'$^{\circ}$' for deg in np.rad2deg(values)])
        ax.set(ylabel = 'Opening angle [rad/deg]', xlabel = 'Distance from sink [au]', title = 'Opening angle H$_p$/r(r)')
        ax.legend(frameon = False)
        fig.suptitle(f'Estimated disk size from azimuthal velocities: {self.disk_size:2.1f} au')
        plt.tight_layout()

pipeline.to_1D = to_1D

def get_1D_param(self, N_σ = 3, Ω = False, cs = False, Q = False, B = False, get_units = True):
    try: self.Σ_1D
    except: self.to_1D()
    self.units_1D = ['Σ = g/cm2', 'φ = rad', 'H = au', 'vφ = cm/s', 'Ω = 1/s', 'cs = cm/s', 'Q = dimensionless', 'B = Gauss']


    def calc_omega():
        
        annulus_m_sum = np.zeros_like(self.r_1D)
        annulus_vφ_tot = np.zeros_like(self.r_1D)
        annulus_vφ2_tot = np.zeros_like(self.r_1D)

        print('Calculating azimuthal and angular velocities')
        for p in tqdm.tqdm(self.sn.patches):                                
            if (p.bin_idx1D == self.Nr).all() or (p.bin_idx1D == 0).all(): continue

            for bin in np.unique(p.bin_idx1D):
                if bin == 0 or bin == self.Nr: continue                                            
                h_idx = np.nonzero(abs(p.cyl_z.flatten()[p.bin_idx1D == bin]) < ( N_σ * self.H_1D[bin - 1, 0]) / self.au_length)    
                if len(h_idx) == 0: continue                                             
                annulus_m_sum[bin - 1] += np.sum(p.m.flatten()[p.bin_idx1D == bin][h_idx])
                annulus_vφ_tot[bin - 1] += np.sum(p.m.flatten()[p.bin_idx1D == bin][h_idx] * (p.vφ.flatten()[p.bin_idx1D == bin][h_idx]))
                annulus_vφ2_tot[bin - 1] += np.sum(p.m.flatten()[p.bin_idx1D == bin][h_idx] * (p.vφ.flatten()[p.bin_idx1D == bin][h_idx])**2)
        
        self.mcode_1D = annulus_m_sum
        vφ_1D = (annulus_vφ_tot/self.mcode_1D) * self.cms_velocity
        vφ2_1D = (annulus_vφ2_tot/self.mcode_1D) * self.cms_velocity**2
        σvφ_1D = np.sqrt(vφ2_1D - vφ_1D**2)
        self.kepVφ_1D = (((G * self.M_star) / (self.r_1D * self.au_length * u.au))**0.5).to('cm/s').value
        self.vφ_1D = np.hstack((vφ_1D[:,None], σvφ_1D[:,None]))

        Ω_1D = vφ_1D / (self.r_1D * self.sn.scaling.l); σΩ_1D = σvφ_1D / (self.r_1D * self.sn.scaling.l)
        self.Ω_1D = np.hstack((Ω_1D[:,None], σΩ_1D[:,None]))

        self.kepΩr_1D = self.kepVφ_1D  / (self.r_1D * self.sn.scaling.l)
    if Ω: calc_omega()


    def calc_cs():
        print('Calculating sound speed from scaleheights and angular velocities')
        H_1D_to_cm = self.H_1D * self.sn.cgs.au
        c_s = self.Ω_1D[:,0] * H_1D_to_cm[:,0]
        σ_cs = np.sqrt(self.Ω_1D[:,0]**2 *  H_1D_to_cm[:,1]**2 + self.Ω_1D[:,1]**2 *  H_1D_to_cm[:,0]**2)
        self.cs_1D = np.hstack((c_s[:,None], σ_cs[:,None]))
    if cs: 
        try: 
            calc_cs()
        except:
             calc_omega()
             calc_cs()


    def calc_Q():
        try:
            self.cs_1D; self.Ω_1D
        except:
            calc_cs(); calc_omega()
        print('Caclulating Toomre Q parameter without magnetic fields')
        G_cgs = G.to('cm3/(g * s**2)').value
        Q_1D = (self.cs_1D[:,0] * self.Ω_1D[:,0]) / (np.pi * G_cgs * self.Σ_1D[:,0])
        σQ_1D = np.sqrt((np.pi * G_cgs)**(-2) 
                        * (self.cs_1D[:,1]**2 * (self.Ω_1D[:,0] / self.Σ_1D[:,0])**2 
                        + self.Ω_1D[:,1]**2 * (self.cs_1D[:,0] / self.Σ_1D[:,0])**2
                        + self.Σ_1D[:,1]**2 * (self.cs_1D[:,0] * self.Ω_1D[:,0] / self.Σ_1D[:,0]**2)**2))
        self.Q_1D = np.hstack((Q_1D[:,None], σQ_1D[:,None]))
    if Q: calc_Q()
    
    def calc_B():
        annulus_B_tot = np.zeros_like(self.r_1D)
        annulus_B2_tot = np.zeros_like(self.r_1D)

        print('Extracting magnetic field data from patches into 1D')
        for p in tqdm.tqdm(self.sn.patches):
            p.B = np.sqrt(p.var('(bx**2+by**2+bz**2)'))                               
            if (p.bin_idx1D == self.Nr).all() or (p.bin_idx1D == 0).all(): continue

            for bin in np.unique(p.bin_idx1D):
                if bin == 0 or bin == self.Nr: continue                                            
                h_idx = np.nonzero(abs(p.cyl_z.flatten()[p.bin_idx1D == bin]) < ( N_σ * self.H_1D[bin - 1, 0]) / self.au_length)    
                if len(h_idx) == 0: continue                                             
                annulus_B_tot[bin - 1] += np.sum(p.m.flatten()[p.bin_idx1D == bin][h_idx] * (p.B.flatten()[p.bin_idx1D == bin][h_idx]))
                annulus_B2_tot[bin - 1] += np.sum(p.m.flatten()[p.bin_idx1D == bin][h_idx] * (p.B.flatten()[p.bin_idx1D == bin][h_idx])**2)

        B_1D = (annulus_B_tot/self.mcode_1D) * self.sn.scaling.b
        B2_1D = (annulus_B2_tot/self.mcode_1D) * self.sn.scaling.b**2
        σB_1D = np.sqrt(B2_1D - B_1D**2)
        self.B_1D = np.hstack((B_1D[:,None], σB_1D[:,None]))
    if B: calc_B()

    if get_units: 
        for unit in self.units_1D: print(unit)

pipeline.get_1D_param = get_1D_param