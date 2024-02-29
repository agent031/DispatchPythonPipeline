import numpy as np

dist = lambda dist1, dist2: np.sqrt(np.sum((dist1 - dist2)**2))
calc_ang = lambda vector1, vector2: np.rad2deg(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))
patch_diag = lambda patch: 0.5 * (np.sum(patch.size**2))**0.5

# Calculates the mean angular momentum vector within a given radius (Unit: au) 
# This function only estimates L from the mean of all the patches

def calc_mean_angvec(snapshot, distance = 110, angle_to_calc = None):
    star_pos = snapshot.sinks[13][0].position
    au_length = snapshot.scaling.l / snapshot.cgs.au
    d = distance / au_length
    
    pp = [p for p in snapshot.patches if dist(p.position, star_pos) < d]
    L = np.zeros(3)
    for i, p in enumerate(pp):
        r_vector = p.position - star_pos
        patch_mass = p.var('d').mean() * np.prod(p.size)
        V_mean = np.array([p.var('ux').mean(), p.var('uy').mean(), p.var('uz').mean()])
        L += np.cross(r_vector, patch_mass * V_mean)

    if isinstance(angle_to_calc, np.ndarray):
        print(f'Angle between the given vector and the mean angular momentum vector: {calc_ang([angle_to_calc], L):2.1f} deg') 
    return L / np.linalg.norm(L)


# Calculates the mean angular momentum vector within a given radius (Unit: au) 
# This function estimates L from all the cells within the given distance
def calc_meanL(snapshot, distance = 100,  angle_to_calc = None, verbose = 0):
    L = np.zeros(3)
    star_pos = snapshot.sinks[13][0].position
    au_length = snapshot.scaling.l / snapshot.cgs.au
    d = distance / au_length
    patches_skipped = 0
    contained = 0
    pp = [p for p in snapshot.patches if dist(p.position, star_pos) < d + patch_diag(p)]; 
    if verbose != 0: print(f'Looping through {len(pp)} patches')
    
    for p in pp:
        XX, YY, ZZ = np.meshgrid(p.xi - star_pos[0], p.yi - star_pos[1], p.zi - star_pos[2], indexing='ij')
        patch_rvector = np.array([XX, YY, ZZ]); 

        patch_velocities = np.asarray([p.var('ux'), p.var('uy'), p.var('uz')])  
        patch_masses = p.var('d') * np.prod(p.ds) 
        patch_distances = np.linalg.norm(patch_rvector, axis = 0) 

        idx = np.nonzero(patch_distances < d)

        if (patch_distances < d).all() and verbose != 0:
            contained += 1
        
        if (patch_distances > d).all():
            if verbose != 0: patches_skipped += 1
            continue
        L_patch = np.cross(patch_rvector, patch_velocities * patch_masses , axisa=0, axisb=0, axisc=0)
        L += np.array([np.sum(L_patch[axis][idx]) for axis in range(3)])

    if isinstance(angle_to_calc, np.ndarray):
        print(f'Angle between the given vector and the mean angular momentum vector: {calc_ang(angle_to_calc, L):2.1f} deg')
    if verbose != 0:
        print("Completely contained patchess :", contained)
        print('Patches skipped', patches_skipped)
    return L / np.linalg.norm(L)