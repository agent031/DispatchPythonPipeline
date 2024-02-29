import numpy as np
import tqdm 
import osyris

from pipeline_main_nosink import pipeline_nosink


def to_osyris_ivs(self, variables, view = 200, dz = None, resolution = 400, viewpoint = None, plot = False):
    #selection_radius =  (np.sqrt(2 * (0.5*view)**2) * 1.5)/ self.au_length # Not all data is needed to be read for a single slap of data

    print('Looping over DISPATCH data to extract data at highest level')
    pp = [p for p in self.sn.patches]#if np.linalg.norm(p.rel_ppos, axis = 0) < selection_radius]
    w= np.array([p.level for p in pp]).argsort()[::-1]
    sorted_patches = [pp[w[i]] for i in range(len(pp))]

    patch_cartcoor = []
    patch_ds = []
    patch_values = {key: [] for key in range(len(variables))}
    self.osyris_ivs = {key: [] for key in range(len(variables))}
    for p in tqdm.tqdm(sorted_patches[:-10]):
        nbors = [self.sn.patchid[i] for i in p.nbor_ids if i in self.sn.patchid]
        children = [ n for n in nbors if n.level == p.level + 1]
        leafs = [n for n in children if ((n.position - p.position)**2).sum() < ((p.size)**2).sum()/12]
        if len(leafs) == 8: continue
        to_extract = np.ones(p.n, dtype=bool)
        for lp in leafs:
            leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
            covered_bool = ~np.all((p.xyz > leaf_extent[:, 0, None, None, None]) & (p.xyz < leaf_extent[:, 1, None, None, None]), axis=0)
            to_extract *= covered_bool

        #Looping over the data to extract from DISPATCH
        for i, ivs in enumerate(variables):
            new_value = p.var(ivs)[to_extract].T
            patch_values[i].extend(new_value.tolist())
        
        #Extracting the position and cellsize data each is fixed across all data
        new_xyz = p.rel_xyz[:,to_extract].T
        patch_ds.extend(p.ds[0] * np.ones(len(new_value)))
        patch_cartcoor.extend(new_xyz.tolist())
        
    
    patch_ds = np.asarray(patch_ds)
    patch_cartcoor = np.asarray(patch_cartcoor)
    for key in patch_values:
        patch_values[key] = np.array(patch_values[key])

    print('Setting up Osyris data structure')
    ds = osyris.Dataset(nout = None)
    # overwrite units
    ds.meta['unit_l'] = self.sn.scaling.l
    ds.meta['unit_t'] = self.sn.scaling.t
    ds.meta['unit_d'] = self.sn.scaling.d
    ds.set_units()
    ds.meta["ndim"] = 3

    if type(viewpoint) != np.ndarray: viewpoint = self.L
    to_view = osyris.Vector(x=viewpoint[0],y=viewpoint[1],z=viewpoint[2])

    view *= osyris.units('au')
    if dz == None: dz = 0.1 * view
    else: dz *= osyris.units('au')
    center = osyris.Vector(x=0,y=0,z=0,unit='au')
    plot_height = dz / osyris.units('au') * self.sn.cgs.au

    ds['amr'] = osyris.Datagroup()
    ds['amr']['dx'] = osyris.Array(patch_ds*self.sn.scaling.l, unit='cm')
    ds['amr']['position'] = osyris.Vector(x=patch_cartcoor[:,0]*self.sn.scaling.l, y=patch_cartcoor[:,1]*self.sn.scaling.l, z=patch_cartcoor[:,2]*self.sn.scaling.l, unit='cm')
    ds['hydro'] = osyris.Datagroup()
    for i, ivs in enumerate(variables):
        ds['hydro'][f'{i}'] = osyris.Array(patch_values[i], unit = 'dimensionless')
        ret = osyris.map({"data": ds['hydro'][f'{i}'], "norm": "log"}, dx=view, dz = dz, origin=center, resolution=resolution, direction=to_view, plot=plot)
        if ivs == 'd': self.osyris_ivs[i] = ret.layers[0]['data']
        else:
            self.osyris_ivs[i] = ret.layers[0]['data'] / plot_height

pipeline_nosink.to_osyris_ivs = to_osyris_ivs