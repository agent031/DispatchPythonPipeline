import numpy as np
import time
from pipeline_main import pipeline

def prolong(a,m):
    """ Prolong a vector along axis 2, repeating it m times
        in along the 2nmd axis, and replicating m times along
        the 0th and 1st axis
    """
    return np.tile(np.repeat(a,m),(m,m,1))

def inject(b,c,i,j,k,verbose=0):
    """ Injext the array b intro array c, starting at indices i,j,k,
        while respecting the index bounds of c
    """
    c_shape=np.array(c.shape)
    b_shape=np.array(b.shape)
    # d_shape is the size the output should have contain the full b
    d_shape=np.maximum(c_shape,b_shape+np.array([i,j,k]))
    # reducing b's shape with the excess leaves a piece that fits
    b_shape=b_shape-(d_shape-c_shape)
    if any(b_shape<0):
        return
    if verbose:
        print('reduced shape:',b_shape)
    i0=max(0,-i)
    j0=max(0,-j)
    k0=max(0,-k)
    if any(b_shape-np.array((i0,j0,k0))<0):
        return
    if verbose:
        print(i0,i,i+b_shape[0])
    c[i+i0:i+b_shape[0],j+j0:j+b_shape[1],k+k0:k+b_shape[2]]=\
    b[0+i0:  b_shape[0],0+j0:  b_shape[1],0+k0:  b_shape[2]]

def fill_zeros(f):
    """ Fill zero values in a 3D array with the smallest positive value
        Arrays with values of both signs are left untouched.
    """
    if f.min()==0.0:
        fn=np.copy(f.flatten())
        w=np.where(fn != 0.0)[0]
        if len(w) > 0:
            fmin=fn[w].min()
            if fmin>0.0:
                for i in range(f.shape[0]):
                    for j in range(f.shape[1]):
                        w=np.where(f[i,j,:]==0.0)[0]
                        f[i,j,w]=fmin
    return f

def amr2raw(self, ivs='logd',lmax=20,lmin=18,complete=True,
            fill=None,width=196,write=True,file='paraview',verbose=0):
    """ Collect data from levels lmin-lmax into a raw data cube
    """
    start=time.time()
    if fill is None:
        fill=False if complete else True
    pp=self.sn.patches
    nw=width
    if not type(ivs) in (tuple,list):
        ivs=[ivs]
    if verbose:
        print('variable list:',ivs)
    nv=len(ivs)

    # Select patches in level bracket
    if verbose>1:
        print('{:6d} patches'.format(len(pp)))
    pp=[p for p in pp if (p.level<=lmax and p.level>=lmin)]
    if verbose>1:
        print('{:6d} patches in selected levels {}-{}'.format(len(pp),lmin,lmax))

    # Lower and upper coordinate boundaries
    width=np.array(width)
    ds=0.5**lmax
    size=width*ds
    l=-size/2
    u=+size/2
    for p in self.sn.patches:
        p.x_trans, p.y_trans, p.z_trans = np.sum(self.rotation_matrix[:,:, None] * np.array((p.xi, p.yi, p.zi)), axis = 1)
    # Select patches for complete fill, or not
    if complete:     
        pp=[p for p in pp if any((p.trans_ppos+p.size/2 >= l) & (p.trans_ppos-p.size/2 <= u))]
        #pp=[p for p in pp if  any(abs(p.res_pos - p.size/2) <= u)] 
        #pp=[p for p in pp if  any(abs(p.res_pos - p.size/2) <= u)]
    else:
        pp=[p for p in pp if all(p.position-p.size/2 > l) and all(p.trans_ppos+p.size/2 < u)]
    if verbose>1:
        print('{:6d} patches contribute'.format(len(pp)))

    # Sort by level
    levels=[p.level for p in pp]
    w=np.array(levels).argsort()
    pp=[pp[w[i]] for i in range(len(pp))]
    
    # Loop over overlapping patches
    prev=start
    n=pp[0].n[0]
    ff=np.zeros((nv,nw,nw,nw))
    fv=np.zeros((nv,n,n,n))
    active=lmin
    for p in pp:
        if verbose>1 and p.level!=active:
            now=time.time()
            print('{:5.1f}s for level {}'.format(now-prev,active))
            prev=now
            active=p.level
        dw=p.size[0]/2
        llc = p.trans_ppos - dw
        #llc=p.res_pos-dw                               
        i1,j1,k1=((llc-l)/ds + 0.5).astype(int)
        m=2**(lmax-p.level)
        n=p.n[0]
        nm=n*m
        
        # Cache variable arrays
        for jv,iv in enumerate(ivs):
            fv[jv]=p.var(iv)

        # level==lmax
        if m==1:
            # Patches that are fully inside
            if i1>=0 and i1+n<=nw and j1>=0 and j1+n<=nw and k1>=0 and k1+n<=nw:
                for jv in range(nv):
                    ff[jv,i1:i1+n,j1:j1+n,k1:k1+n] = fv[jv]

            # Patches that overlap partially
            elif complete:
                for jv in range(nv):
                    inject(fv[jv],ff[jv],i1,j1,k1)

        # level < lmax
        else:            
            xyz = np.array((p.x_trans[0], p.y_trans[0], p.z_trans[0]))
            #xyz=np.array((p.xi[0],p.yi[0],p.zi[0]))
            #xyz= p.rel_xyz[:,0,0,0]                        
            ijk=((xyz-l)/ds+0.5).astype(int)
            i1=ijk[0]
            # Loop over 0th index in source
            for i0,x in enumerate(p.x_trans):
                j1=ijk[1]
                # Loop over 1st index in source
                for j0,y in enumerate(p.y_trans):
                    for jv in range(nv):
                        # Inject array along axis 2
                        zval=fv[jv,i0,j0,:]
                        zval=prolong(zval,m)
                        inject(zval,ff[jv],i1,j1,k1)
                        j1=j1+m
                i1=i1+m

    # Optionally, fill in values in incomplete commponents
    if fill and not complete:
        for iv in range(ff.shape[0]):
            fill_zeros(ff[iv])
    if verbose>1:
        now=time.time()
        print('{:5.1f}s for level {}'.format(now-prev,lmax))
    if verbose:
        print('{:5.1f}s for {} patches'.format(time.time()-start,len(pp)))
    if write:
        if verbose:
            file='{}{}.raw'.format(file,nw)
            print('writing raw file',file)
        d=np.memmap(file,shape=ff.shape,order='F',mode='w+',dtype=np.float32)
        d[:,:,:]=ff[:,:,:]
    return ff.squeeze()

pipeline.amr2raw = amr2raw