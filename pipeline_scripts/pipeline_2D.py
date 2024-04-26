import numpy as np
import tqdm 
import sys

sys.path.insert(0, '/groups/astro/kxm508/codes/osyris/src')
import osyris

from pipeline_main import pipeline

#### YOU CAN SEARCH IN VAR() VARIABLES OR THE ATTACHED ATRIBUTES E.G. p.P or p.vφ
#### ADDING AN ARGUMENT FOR EXTRACTING VECTOR. THE VECTORE MOST ALREADY BE MADE LIKE p.B:
#### p.B = np.concatenate([p.var(f'b'+axis)[None,...] for axis in ['x','y','z']], axis = 0)

def to_osyris_ivs(self, variables, data_name, view = 200, dz = None, resolution = 400, viewpoint = 'top', weights = 'mass', vectors = None, verbose = 1):
    selection_radius =  (np.sqrt(2 * (0.5*view)**2) * 2)/ self.au_length # Not all data is needed to be read for a single slap of data

    if verbose > 0: print('Looping over DISPATCH data to extract data at highest level')
    pp = [p for p in self.sn.patches if np.linalg.norm(p.rel_ppos, axis = 0) < selection_radius]
    w= np.array([p.level for p in pp]).argsort()[::-1]
    sorted_patches = [pp[w[i]] for i in range(len(pp))]

    patch_cartcoor = []
    patch_ds = []
    patch_values = {key: [] for key in range(len(variables))}
    patch_weight = []
    if vectors is not None: patch_vectors = {key: [] for key in range(len(vectors))}

    try: self.osyris_ivs[data_name]
    except: self.osyris_ivs = {data_name: {}}

    self.osyris_ivs[data_name] = {key: [] for key in variables}
    for p in tqdm.tqdm(sorted_patches, disable = not self.loading_bar):
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
            if hasattr(p, ivs):
                new_value = getattr(p, ivs)[to_extract].T               
            else:
                new_value = p.var(ivs)[to_extract].T
            patch_values[i].extend(new_value.tolist())

        if vectors is not None:
            for i, ivs in enumerate(vectors):
                new_vector = getattr(p, ivs)[:, to_extract].T
                patch_vectors[i].extend(new_vector.tolist())
     
        #Extracting the position, cellsize, and weight of the extracted data each is fixed across all data
        if weights != None:
            if weights == 'mass':
                weight = p.m[to_extract].T
            if weights == 'volume':
                weight = np.prod(p.ds) * np.ones(len(new_value))
        else:
            ### If no accpted weight is assigned and the final result is a combined sum over dz   #####
            weight = np.ones(len(new_value))
        patch_weight.extend(weight.tolist())
        new_xyz = p.rel_xyz[:,to_extract].T        
        patch_ds.extend(p.ds[0] * np.ones(len(new_value)))
        patch_cartcoor.extend(new_xyz.tolist())
    
    patch_weight = np.asarray(patch_weight)
    patch_ds = np.asarray(patch_ds)
    patch_cartcoor = np.asarray(patch_cartcoor)
    for key in patch_values:
        patch_values[key] = np.array(patch_values[key])
    if vectors is not None:
        for key in patch_vectors:
            patch_vectors[key] = np.array(patch_vectors[key])

    print('Setting up Osyris data structure')
    ds = osyris.Dataset(nout = None)
    # overwrite units
    ds.meta['unit_l'] = self.sn.scaling.l
    ds.meta['unit_t'] = self.sn.scaling.t
    ds.meta['unit_d'] = self.sn.scaling.d
    ds.set_units()
    ds.meta["ndim"] = 3

    #### Defining viewpoint coordinate system #####
    try: self.new_x
    except: self.calc_trans_xyz(verbose = verbose)

    try: self.L
    except: self.recalc_L() 

    if (viewpoint == np.array(['x', 'y', 'z'])).any():
        to_view = viewpoint
    else:
        dir_vecs = {}
        dir_vecs['pos_u'] = osyris.Vector(x=self.new_x[0],  y = self.new_x[1], z=self.new_x[2])
        dir_vecs['pos_v'] = osyris.Vector(x=self.new_y[0],  y = self.new_y[1], z=self.new_y[2])
        dir_vecs['normal'] = osyris.Vector(x=self.L[0],     y=self.L[1],       z=self.L[2])
    if viewpoint == 'top':
        to_view = dir_vecs
    elif viewpoint == 'edge':
        dir_vecs2 = dir_vecs.copy()
        dir_vecs2['normal'] = dir_vecs['pos_u']
        dir_vecs2['pos_u'] = dir_vecs['pos_v']
        dir_vecs2['pos_v'] = dir_vecs['normal']
        to_view = dir_vecs2
        
    view *= osyris.units('au')
    if dz == None: dz = 0.1 * view
    else: dz *= osyris.units('au')
    center = osyris.Vector(x=0,y=0,z=0,unit='au')
    plot_height = dz / osyris.units('au') * self.sn.cgs.au

    ds['amr'] = osyris.Datagroup()
    ds['amr']['dx'] = osyris.Array(patch_ds*self.sn.scaling.l, unit='cm')
    ds['amr']['position'] = osyris.Vector(x=patch_cartcoor[:,0]*self.sn.scaling.l, y=patch_cartcoor[:,1]*self.sn.scaling.l, z=patch_cartcoor[:,2]*self.sn.scaling.l, unit='cm')
    ds['hydro'] = osyris.Datagroup()

    #Looping over the scalar variables set for extraction
    for i, ivs in enumerate(variables):
        ds['hydro'][ivs] = osyris.Array(patch_weight * patch_values[i], unit = 'dimensionless')
        ret = osyris.map({"data": ds['hydro'][ivs], "norm": "log"}, dx=view, dz = dz, 
                         origin=center, resolution=resolution, direction=to_view, plot=False, operation = "sum")
        
        if weights != None:
            ds['hydro']['w'] = osyris.Array(patch_weight, unit = 'dimensionless')
            ret_weight = osyris.map({"data": ds['hydro']['w'], "norm": "log"}, dx=view, dz = dz, 
                             origin=center, resolution=resolution, direction=to_view, plot=False, operation = "sum")
            final_weights = ret_weight.layers[0]['data']

        else: final_weights = np.ones_like(ret.layers[0]['data'])
        
        self.osyris_ivs[data_name][ivs] = ret.layers[0]['data'] / final_weights
    
    #Looping over the vector variables set for extraction
    if vectors is not None:
        for i, ivs in enumerate(vectors):
            ds['hydro'][ivs] = osyris.Vector(x = patch_vectors[i][:,0], y = patch_vectors[i][:,1], z = patch_vectors[i][:,2], unit='dimensionless')
            ret = osyris.map({"data": ds['hydro'][ivs], "mode": "vec"}, dx=view, dz = dz, origin=center, resolution=resolution, direction=to_view, plot=False)
            self.osyris_ivs[data_name][i + len(variables)] = ret.layers[0]['data'] / plot_height

pipeline.to_osyris_ivs = to_osyris_ivs

