import numpy as np
from astropy.constants import m_p, G, k_B 
import astropy.units as u
from lmfit import Model
import tqdm
import os
import matplotlib.pyplot as plt

from pipeline_main import pipeline

#________________________________THIS UPDATED VERSION OF THE 1D PIPELINE TAKES CARE OF OVERLAPPING PATCHES________________________#

# Calculate the surface density of the disk
def to_1D(self, r_in = 5, r_out = 100, Nr = 100, plot = True, MMSN = True, dpi = 100, verbose = 1):
    if not self.cyl_calculated: self.recalc_L(verbose = 0)

    self.Nr = Nr
    self.r_bins = np.logspace(np.log10(r_in), np.log10(r_out), Nr) / self.au_length #[au]
    self.r_1D = self.r_bins[:-1] + 0.5 * np.diff(self.r_bins) 

    def Hp_func(x, Σ, H): return (Σ) / (np.sqrt(2 * np.pi) * H) * np.exp( - x**2 / (2 * H**2)) 

    def fit_scaleheight(ρ, h, x0):
        model = Model(Hp_func)
        params = model.make_params(Σ = x0[0], H = x0[1])
        #params['Σ'].min = 0; params['H'].min = 0   # Ensure H is always positive
        result = model.fit(ρ, x = h, params = params)
        fit_params = np.array(list(result.best_values.values()))
        fit_err = np.array([par.stderr for _, par in result.params.items()])
        fit_params[0] *= self.sn.cgs.au ; fit_err[0] *= self.sn.cgs.au

        return np.array([fit_params[0], fit_err[0]]), np.array([fit_params[1], fit_err[1]]) 


    densities = {key: [] for key in range(Nr - 1)}
    heights = {key: [] for key in range(Nr - 1)}
    if verbose > 0: print('Looping through patches to extract densities and heights')


    selection_radius = (r_out * 2) / self.au_length
    pp = [p for p in self.sn.patches if np.linalg.norm(p.rel_ppos, axis = 0) < selection_radius]
    w= np.array([p.level for p in pp]).argsort()[::-1]
    sorted_patches = [pp[w[i]] for i in range(len(pp))]
    self.sorted_patches1D = sorted_patches.copy()

    for p in self.sorted_patches1D:
        nbors = [self.sn.patchid[i] for i in p.nbor_ids if i in self.sn.patchid]
        children = [ n for n in nbors if n.level == p.level + 1]
        leafs = [n for n in children if ((n.position - p.position)**2).sum() < ((p.size)**2).sum()/12]   
        if len(leafs) == 8: continue

        to_extract = np.ones((16,16,16), dtype='bool')
        for lp in leafs: 
            leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
            covered_bool = ~np.all((p.xyz > leaf_extent[:, 0, None, None, None]) & (p.xyz < leaf_extent[:, 1, None, None, None]), axis=0)
            to_extract *= covered_bool 
        p.to_extract1D = to_extract.copy()
        # Radial cut in the bins are saved for later since the cut will be used repeatedly
        p.bin_idx1D = np.digitize(p.cyl_R[to_extract].flatten(), bins = self.r_bins)                                   #Assigning each point to their repective bin 
        if (p.bin_idx1D == Nr).all() or (p.bin_idx1D == 0).all(): continue                                    #If the points are outside of the given bin the index is 0 or 25 (dependent on the length of radial bins)
        for bin in np.unique(p.bin_idx1D):
            if bin == 0 or bin == Nr: continue                                            
            h_idx = np.nonzero(abs(p.cyl_z[to_extract].flatten()[p.bin_idx1D == bin]) < 2 * self.r_1D[bin - 1])        # Now I make a cut only taking cells within 2 * the radial bins. I want the densities directory to be 0-indexet hence bin - 1
            if len(h_idx) == 0: continue                                             
            
            densities[bin - 1].extend(p.var('d')[to_extract].flatten()[p.bin_idx1D == bin][h_idx])
            heights[bin - 1].extend(p.cyl_z[to_extract].flatten()[p.bin_idx1D == bin][h_idx])

    for key in densities:
        densities[key] = np.array(densities[key]) * self.cgs_density
        heights[key] = np.array(heights[key]) * self.au_length
    
    self.Σ_1D = np.zeros((Nr - 1, 2))
    self.H_1D = np.zeros((Nr - 1, 2))
    x0 = np.array([1e3, 7])

    if verbose > 0: print('Fitting surface density and scaleheight in each radial bin')
    for i in tqdm.tqdm(range(Nr - 1), disable = not self.loading_bar):    
        self.Σ_1D[i], self.H_1D[i] = fit_scaleheight(ρ = densities[i], h = heights[i], x0 = x0)
        #x0 = np.array([self.Σ_1D[i, 0], self.H_1D[i, 0]])
    

    def calc_sigma(fitted_Hp):
        annulus_m_sum = np.zeros(Nr - 1)
        annulus_V_sum = np.zeros(Nr - 1)   
        for p in sorted_patches:
                #Some patches in sorted_patches does not contain information within the specied bins and p.bin_idx1D is therefore non-existent       
                try: p.bin_idx1D     
                except: continue           
                if (p.bin_idx1D == Nr).all() or (p.bin_idx1D == 0).all(): continue                    
                
                for bin in np.unique(p.bin_idx1D):
                    if bin == 0 or bin == Nr: continue                                            
                    h_idx = np.nonzero(abs(p.cyl_z[p.to_extract1D].flatten()[p.bin_idx1D == bin]) < (fitted_Hp[bin - 1]) / self.au_length)    
                    if len(h_idx) == 0: continue                                             

                    annulus_m_sum[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx])
                    annulus_V_sum[bin - 1] += np.prod(p.ds) * len(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx])                               # The cell volume multiplied with the number of cells within the given cut

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

        ax.loglog(self.r_1D * self.au_length, self.Σ_1D[:,0], color = 'blue', label = 'Σ$_{Fit}$')
        for i in reversed(range(1, 3)):
            ax.loglog(self.r_1D * self.au_length, sigmas[i - 1], color = 'red', label = 'Σ$_{Calc}$'+f'$\propto\int\pm{i}H$', alpha = i/2, lw = 0.8)
        ax.fill_between(self.r_1D * self.au_length, self.Σ_1D[:,0] + self.Σ_1D[:,1], self.Σ_1D[:,0] - self.Σ_1D[:,1], alpha = 0.45, color = 'blue')
        ax.set(ylabel = 'Σ$_{gas}$ [g/cm$^2$]', xlabel = 'Distance from sink [au]', title = 'Surface density Σ$_{gas}$(r)')

        if MMSN:
            Σ_MMSN = lambda r: 1700 * (r)**(-3/2)
            r = self.r_1D * self.au_length
            #ax.text(r[0], Σ_MMSN(r)[0] - 25, 'Σ$_{MMSN}\propto r^{-3/2}$', va = 'top', ha = 'left', rotation = -26, color = 'grey')
            ax.loglog(r, Σ_MMSN(r), color = 'grey', ls = '--', label = 'Σ$_{MMSN}\propto r^{-3/2}$')

        ax.legend(frameon = False)

        ax = axs[1]
        ax.loglog(self.r_1D * self.au_length, self.H_1D[:,0], label = 'Scale height H', color = 'green')
        ax.fill_between(self.r_1D * self.au_length, self.H_1D[:,0] + self.H_1D[:,1], self.H_1D[:,0] - self.H_1D[:,1], alpha = 0.3, color = 'green', label = '$\pm σ_H$')
        ax.set(ylabel = 'Scale height [au]', xlabel = 'Distance from sink [au]', title = 'Scale height  H(r)')
        ax.legend(frameon = False)

        ax = axs[2]
        self.φ =  np.vstack((self.H_1D[:,0] / (self.r_1D * self.au_length), self.H_1D[:,1] / (self.r_1D * self.au_length))).T

        ax.semilogx(self.r_1D * self.au_length, self.φ[:,0], color = 'purple', label = 'Opening angle H/r')
        ax.fill_between(self.r_1D * self.au_length, self.φ[:,0] + self.φ[:,1], self.φ[:,0] - self.φ[:,1], color = 'purple', alpha = 0.3, label = '$\pm σ_φ$')

        #Values for ticks
        values = np.linspace(0, np.pi/2, 5)
        names = ['$0$', '$π/8$', '$π/4$', '$3π/8$', '$π/2$']
        ax.set_yticks(values); ax.set_yticklabels(names)
        ax2 = ax.twinx()
        ax2.set_yticks(np.rad2deg(values))
        ax2.set_yticklabels([f'{deg:2.0f}'+'$^{\circ}$' for deg in np.rad2deg(values)])
        ax.set(ylabel = 'Opening angle [rad/deg]', xlabel = 'Distance from sink [au]', title = 'Opening angle H/r(r)')
        ax.legend(frameon = False)
        plt.tight_layout()

pipeline.to_1D = to_1D

def get_1D_param(self, N_σ = 3, Ω = False, cs = False, Q = False, B = False, T = False, get_units = True, verbose = 1):
    try: self.Σ_1D
    except: self.to_1D()
    self.units_1D = ['Σ = g/cm2', 'φ = rad', 'H = au', 'vφ = cm/s', 'Ω = 1/s', 'cs = cm/s', 'Q = dimensionless', 'B = Gauss', 'T = K']


    def calc_omega():  
        annulus_m_sum = np.zeros_like(self.r_1D)
        annulus_γ_tot = np.zeros_like(self.r_1D)
        annulus_γ2_tot = np.zeros_like(self.r_1D)
        annulus_vφ_tot = np.zeros_like(self.r_1D)
        annulus_vφ2_tot = np.zeros_like(self.r_1D)

        if verbose > 0 : print('Extracting azimuthal, angular velocities and adiabatic index data from patches into 1D')
        for p in tqdm.tqdm(self.sorted_patches1D, disable = not self.loading_bar):  
            #Some patches in sorted_patches does not contain information within the specied bins and p.bin_idx1D is therefore non-existent
            try: p.bin_idx1D    
            except: continue

            if (p.bin_idx1D == self.Nr).all() or (p.bin_idx1D == 0).all(): continue

            for bin in np.unique(p.bin_idx1D):
                if bin == 0 or bin == self.Nr: continue                                            
                h_idx = np.nonzero(abs(p.cyl_z[p.to_extract1D].flatten()[p.bin_idx1D == bin]) < ( N_σ * self.H_1D[bin - 1, 0]) / self.au_length)    
                if len(h_idx) == 0: continue                                             
                annulus_m_sum[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx])
                annulus_γ_tot[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx] * (p.γ[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx]))
                annulus_γ2_tot[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx] * (p.γ[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx])**2)
                annulus_vφ_tot[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx] * (p.vφ[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx]))
                annulus_vφ2_tot[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx] * (p.vφ[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx])**2)
        
        self.mcode_1D = annulus_m_sum
        γ = annulus_γ_tot/self.mcode_1D
        γ2 = annulus_γ2_tot/self.mcode_1D
        vφ_1D = (annulus_vφ_tot/self.mcode_1D) * self.cms_velocity
        vφ2_1D = (annulus_vφ2_tot/self.mcode_1D) * self.cms_velocity**2
        σ_γ = np.sqrt(γ2 - γ**2)
        σvφ_1D = np.sqrt(vφ2_1D - vφ_1D**2)
        self.kepVφ_1D = (((G * self.M_star) / (self.r_1D * self.au_length * u.au))**0.5).to('cm/s').value
        self.vφ_1D = np.hstack((vφ_1D[:,None], σvφ_1D[:,None]))

        Ω_1D = vφ_1D / (self.r_1D * self.sn.scaling.l); σΩ_1D = σvφ_1D / (self.r_1D * self.sn.scaling.l)

        self.γ_1D = np.hstack((γ[:,None], σ_γ[:,None]))
        self.Ω_1D = np.hstack((Ω_1D[:,None], σΩ_1D[:,None]))

        self.kepΩr_1D = self.kepVφ_1D  / (self.r_1D * self.sn.scaling.l)
    if Ω: calc_omega()

    def calc_T():
        try: self.cs_1D
        except: calc_cs1D()
        #annulus_T_tot = np.zeros_like(self.r_1D)
        #annulus_T2_tot = np.zeros_like(self.r_1D)
   
        #print(f'Extracting temperature and from patches into 1D')
        #for p in tqdm.tqdm(self.sorted_patches1D):
        #    try: p.bin_idx1D
        #    except: continue                            
        #    if (p.bin_idx1D == self.Nr).all() or (p.bin_idx1D == 0).all(): continue#

        #    for bin in np.unique(p.bin_idx1D):
        #        if bin == 0 or bin == self.Nr: continue                                            
        #        h_idx = np.nonzero(abs(p.cyl_z[p.to_extract1D].flatten()[p.bin_idx1D == bin]) < ( N_σ * self.H_1D[bin - 1, 0]) / self.au_length)    
        #        if len(h_idx) == 0: continue         
                                    
        #        annulus_T_tot[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx] * (p.T[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx]))
        #        annulus_T2_tot[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx] * (p.T[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx])**2)

        #T_1D = (annulus_T_tot/self.mcode_1D) 
        #T2_1D = (annulus_T2_tot/self.mcode_1D) 
        #σT_1D = np.sqrt(T2_1D - T_1D**2)

        if verbose > 0 : print(f'Calculating temperature from sound speed')
        T_1D = self.cs_1D[:,0]**2 * self.μ * m_p.to('g').value / (self.γ_1D[:,0] * k_B.to('erg/K').value)
        σT_1D = self.μ * m_p.to('g').value / k_B.to('erg/K').value * np.sqrt((2 * self.cs_1D[:,0] / self.γ_1D[:,0])**2 * self.cs_1D[:,1]**2 + (self.cs_1D[:,0]**2 / self.γ_1D[:,0]**2)**2 * self.γ_1D[:,1]**2)
        
        self.T_1D = np.hstack((T_1D[:,None], σT_1D[:,None]))

    if T: calc_T()  


    def calc_cs1D(self, method = 'data'):
        try: self.Ω_1D
        except: calc_omega()
        if method != 'temperature' and method != 'data' and method != 'settled_disk':
            if verbose > 0 :
                print('Not valid method for extracting sound speed')
                print('Valid methods are:\ntemperature\ndata\nsettled_disk')

        if method == 'settled_disk':
            if verbose > 0 : print('Calculating sound speed assuming a thin settled disk c_s = ΩΗ')
            try: self.Ω_1D    
            except: calc_omega() 
            H_1D_to_cm = self.H_1D * self.sn.cgs.au
            c_s = self.Ω_1D[:,0] * H_1D_to_cm[:,0]
            σ_cs = np.sqrt(self.Ω_1D[:,0]**2 *  H_1D_to_cm[:,1]**2 + self.Ω_1D[:,1]**2 *  H_1D_to_cm[:,0]**2)

        if method == 'data':
            if verbose > 0 : print('Calculating isothermal sound speed c_s = (γP/ρ)^0.5')
          
            annulus_cs_tot = np.zeros_like(self.r_1D)
            annulus_cs2_tot = np.zeros_like(self.r_1D)
            for p in tqdm.tqdm(self.sorted_patches1D, disable = not self.loading_bar):
                #Some patches in sorted_patches does not contain information within the specied bins and p.bin_idx1D is therefore non-existent
                p.cs = np.sqrt(p.γ * p.P / p.var('d'))         
                try: p.bin_idx1D    
                except: continue
                if (p.bin_idx1D == self.Nr).all() or (p.bin_idx1D == 0).all(): continue

                for bin in np.unique(p.bin_idx1D):
                    if bin == 0 or bin == self.Nr: continue                                            
                    h_idx = np.nonzero(abs(p.cyl_z[p.to_extract1D].flatten()[p.bin_idx1D == bin]) < ( N_σ * self.H_1D[bin - 1, 0]) / self.au_length)    
                    if len(h_idx) == 0: continue       

                    annulus_cs_tot[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx] * (p.cs[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx]))
                    annulus_cs2_tot[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx] * (p.cs[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx])**2)

            c_s = (annulus_cs_tot/self.mcode_1D) * self.cms_velocity
            c_s2 = (annulus_cs2_tot/self.mcode_1D) * self.cms_velocity**2
            σ_cs = np.sqrt(c_s2 - c_s**2)

        # !!! OBS The method below should not be used if i calculate the temperaure from the sounds speed !!!
        if method == 'temperature':
            try: self.T_1D
            except: calc_T()
            if verbose > 0 : print('Caclulating sound speed from extracted temperature')
            c_s = np.sqrt(self.γ_1D[:,0] * k_B.to('erg/K').value * self.T_1D[:,0] /(self.μ * m_p.to('g').value))
            σ_cs = np.sqrt(k_B.to('erg/K').value / (2 * self.μ * m_p.to('g').value)) * np.sqrt(self.γ_1D[:,0] / self.T_1D[:,0] * self.T_1D[:,1]**2 + self.T_1D[:,0] / self.γ_1D[:,0] * self.γ_1D[:,1]**2) 
        self.cs_1D = np.hstack((c_s[:,None], σ_cs[:,None]))

    pipeline.calc_cs1D = calc_cs1D
    if cs:  
        self.calc_cs1D()

    def calc_Q():
        try:
            self.cs_1D; self.Ω_1D
        except:
            self.calc_cs1D(); calc_omega()
        if verbose > 0 : print('Caclulating Toomre Q parameter without magnetic fields')
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

        if verbose > 0 : print('Extracting magnetic field data from patches into 1D')
        for p in tqdm.tqdm(self.sorted_patches1D, disable = not self.loading_bar):
            p.B = np.sqrt(p.var('(bx**2+by**2+bz**2)'))   
            try: p.bin_idx1D
            except: continue                            
            if (p.bin_idx1D == self.Nr).all() or (p.bin_idx1D == 0).all(): continue

            for bin in np.unique(p.bin_idx1D):
                if bin == 0 or bin == self.Nr: continue                                            
                h_idx = np.nonzero(abs(p.cyl_z[p.to_extract1D].flatten()[p.bin_idx1D == bin]) < ( N_σ * self.H_1D[bin - 1, 0]) / self.au_length)    
                if len(h_idx) == 0: continue                                             
                annulus_B_tot[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx] * (p.B[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx]))
                annulus_B2_tot[bin - 1] += np.sum(p.m[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx] * (p.B[p.to_extract1D].flatten()[p.bin_idx1D == bin][h_idx])**2)

        B_1D = (annulus_B_tot/self.mcode_1D) * self.sn.scaling.b
        B2_1D = (annulus_B2_tot/self.mcode_1D) * self.sn.scaling.b**2
        σB_1D = np.sqrt(B2_1D - B_1D**2)
        self.B_1D = np.hstack((B_1D[:,None], σB_1D[:,None]))
    if B: calc_B()
    

    if get_units: 
        for unit in self.units_1D: print(unit)

pipeline.get_1D_param = get_1D_param