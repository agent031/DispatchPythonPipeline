# -*- coding: utf-8 -*-

# Pythn 2/3 compatibility
from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as pl

import dispatch as di
import dispatch.select   as ds

import time

def _kw_extract(kw,dict):
    """ if keys from dict occur in kw, pop them """
    for key,value in dict.items():
        dict[key]=kw.pop(key,value)
    return kw,dict

def _cmap(v,cmap,verbose=0):
    if cmap is None:
        if verbose: print('cmap was None')
        cmap='viridis' if v.min() >= 0.0 else 'coolwarm'
        if verbose: print('cmap is',cmap)
    elif cmap=='default':
        cmap='viridis'
    return cmap

def imshow(f,colorbar=None,origin='lower',interpolation='nearest',
           cmap=None,verbose=0,update=False,**kw):
    """
        imshow with default options and updateable colorbar.  Example use:

            import matplotlib.pyplot as pl
            import dispatch.select   as ds
            import dispatch.graphics as dg
            ...
            s=dispatch.snapshot(1)
            dg.imshow(ds.unigrid_plane(s,iv=0))
            cb=pl.colorbar()
            dg.imshow(ds.unigrid_plane(s,iv=1),cb)

        The second call updates the colorbar from the 1st call
    """
    if f is None: return
    if verbose>0:
        print ('min:',f.min(),' max:',f.max())
    cmap=_cmap(f,cmap)
    im=pl.imshow(np.transpose(f),origin=origin,interpolation=interpolation,cmap=cmap,**kw)
    if verbose:
        print('cmap:',im.get_cmap().name)
    """ Check if there is an existing colorbar """
    cb=None
    ims=pl.gca().images
    if len(ims)>0:
        for im in ims:
            if type(im.colorbar)==matplotlib.colorbar.Colorbar:
                cb=im.colorbar
    """ If a colorbar exists, update the limits - if not, make one """
    if cb:
        cb.set_clim(f.min(),f.max())
        cb.draw_all()
    else:
        pl.colorbar()

def plot(f,**kw):
    """ plot(f) allows f to be (x,y) tuple """
    if type(f)==tuple:
        x,y=f
        pl.plot(x,y,**kw)
    else:
        pl.plot(f,**kw)

def plot_values_along(pp,pt=[0.0,0.0,0.0],xlim=None,**kw):
    """ Plot values along direction dir={0,1,2}, through point pt=[x,y,z] """
    kv = {'dir':0, 'verbose':0, 'all':False, 'iv':0, 'i4':0, 'var':None, 'eos':None}
    kw,kv = _kw_extract(kw,kv)
    plot(ds.values_along(pp,pt,iv=kv['iv'],dir=kv['dir'],all=kv['all'],xlim=xlim),**kw)

def plot_patch_values_along(pp_in,pt=[0.0,0.0,0.0],hold=False,xlim=None,**kw):
    """ Plot values along direction dir={0,1,2}, through point pt=[x,y,z] """
    kv = {'dir':0, 'verbose':0, 'all':False, 'iv':0, 'i4':0, 'var':None, 'eos':None}
    kw,kv = _kw_extract(kw,kv)
    pp = ds.patches_along(pp_in,pt,dir=kv['dir'],verbose=kv['verbose'],xlim=xlim)
    for p in pp:
        v=ds.values_in(p,pt,iv=kv['iv'],dir=kv['dir'])
        plot(v,label=p.id,**kw)
    pl.legend()

def _rotate(d,e):
    '''Rotate data and extent to landscape format'''
    d=d.transpose()
    e=[e[2],e[3],e[0],e[1]]
    return d,e

def power2d(f,*kw):
    """plot power spectrum of 2D array"""
    ft2=np.abs(np.fft.fft2(f))**2
    m=f.shape
    k=np.meshgrid(range(m[0]),range(m[1]))
    k=np.sqrt(k[0]**2+k[1]**2)
    a=2
    k0=1.0/a**0.5
    k1=1.0*a**0.5
    power=[]
    kk=[]
    while(k1 <= m[0]//2):
        kk.append((k0*k1)**0.5)
        w=np.where((k>k0) & (k <= k1))
        power.append(ft2[w].sum())
        k0=k1
        k1=k1*a
    pl.loglog(kk,power,*kw)

def show_plot(figure_id=None):
    """ raise a figure to the top """
    if figure_id is None:
        fig = pl.gcf()
    else:
        # do this even if figure_id == 0
        fig = pl.figure(num=figure_id)
    pl.show()
    pl.pause(1e-9)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()

def pause(time=1e-6):
    """ replace the normal pause, w/o raising window() """
    pl.draw()
    pl.gcf().canvas.start_event_loop(time)

def image_plane(snap,x=None,y=None,z=0.5,iv=0,grids=False,cbar=None,zoom=None,
                cmap=None,title=None,verbose=1):
    o=np.array(snap.cartesian.origin)
    s=np.array(snap.cartesian.size)
    c=o+0.5*s
    if x:
        i=0
        x=x*snap.cartesian.size[i]
        c=c[[1,2]]
        s=s[[1,2]]
    elif y:
        i=1
        y=y*snap.cartesian.size[i]
        c=c[[0,2]]
        s=s[[0,2]]
    elif z:
        i=2
        z=z*snap.cartesian.size[i]
        c=c[[0,1]]
        s=s[[0,1]]
    xyz=(x,y,z)
    sdir=['x','y','z']
    labels=[('y','z'),('z','x'),('y','z')]
    pp=snap.patches_in(x=x,y=y,z=z)
    pl.clf()
    ll={}
    p=pp[0]
    f=p.plane(x=x,y=y,z=z,iv=iv)
    fmin=f.min()
    fmax=f.max()
    e=p.extent[i]
    emin=np.array((e[0],e[2]))
    emax=np.array((e[1],e[3]))
    for p in pp:
        im=p.plane(x=x,y=y,z=z,iv=iv)
        fmin=min(fmin,im.min())
        fmax=max(fmax,im.max())
        e=p.extent[i]
        emin=np.minimum(emin,np.array((e[0],e[2])))
        emax=np.maximum(emax,np.array((e[1],e[3])))
        ll[p.id]=(im,e,p.level)
    cmap=_cmap(np.array([fmin,fmax]),cmap,verbose=verbose)
    for id in sorted(ll.keys()):
        im,e,level=ll[id]
        pl.imshow(im.T,origin='lower',extent=e,vmin=fmin,vmax=fmax,cmap=cmap)
    pl.colorbar()
    pl.xlim(emin[0],emax[0])
    pl.ylim(emin[1],emax[1])
    if title:
        pl.title(title)
    else:
        pl.title('{}={:.3f}'.format(sdir[i],xyz[i]))
    pl.xlabel(labels[i][0])
    pl.ylabel(labels[i][1])
    if grids:
        for p in pp:
            e=p.extent[i]
            x=[e[0],e[0],e[1],e[1],e[0]]
            y=[e[2],e[3],e[3],e[2],e[2]]
            pl.plot(x,y,color='gray')
    if zoom:
        w=zoom*s*0.5
        pl.xlim(c[0]-w[0],c[0]+w[0])
        pl.ylim(c[1]-w[1],c[1]+w[1])
    pl.tight_layout()

def pdf_d(iout,run='',data='../data',iv='d',i4=0,nbin=100,xlim=[-4,3],lnd=False):
    """ Plot the PDF of the density """
    s = di.snapshot(iout,run=run,data=data)
    n = nbin
    bins = np.linspace(xlim[0],xlim[1],n+1)
    htot = 0.0
    i = 0
    for p in s.patches:
        i += 1
        if i%1000==0:
            print('{:.1f}%'.format(i/len(s.patches)*100.0))
        d = p.var(iv,i4=i4)
        if lnd:
            logd = d/np.log(10.)
        else:
            logd = np.log10(d)
        h,e = np.histogram(logd,bins=bins)
        dv=np.product(p.ds)
        htot += h*dv
    pl.hist(bins[0:n],bins=bins,weights=htot,log=True,density=True)
    pl.title('t={:.3f}'.format(s.time))
    return bins,htot

def _axis(x,y,z):
    if not z is None:
        axis=2
        s=z
    if not y is None:
        axis=1
        s=y
    if not x is None:
        axis=0
        s=x
    return axis,s

def _title(sn,x=None,y=None,z=0.5,iv=None,title=None):
    if title:
        pl.title(title)
    else:
        axis,s=_axis(x,y,z)
        ax=['x','y','z'][axis]
        if sn.time<0.01:
            text='{},  {}={:.5f},  t={:.4e}'.format(iv,ax,s,sn.time)
        else:
            text='{},  {}={:.5f},  t={:.4f}'.format(iv,ax,s,sn.time)
        pl.title(text)
    pl.tight_layout()

def unigrid_plane(sn,iv='d',z=0.0,x=None,y=None,title=None,cmap=None,i4=0,**kw):
    """
    keywords passed on to dispatch.select.unigrid_plane(),
    and to dispatch.graphics.imshow()
    """
    import dispatch.select as dse
    if not hasattr(sn,'patches'):
        print('snapshot has no patches')
        return
    ff,pp=dse.unigrid_plane(sn,patches=True,iv=iv,x=x,y=y,z=z,i4=i4,**kw)
    cmap=_cmap(ff,cmap)
    imshow(ff,cmap=cmap,**kw)  
    _title(sn,x,y,z,iv,title)

def outline_extent(e,color='grey'):
    xl=[e[0],e[0],e[1],e[1],e[0]]
    yl=[e[2],e[3],e[3],e[2],e[2]]
    pl.plot(xl,yl,color=color)

def amr_plane(sn,iv='d',z=0.0,x=None,y=None,log=False,to=None,
        xmin=None,xmax=None,ymin=None,ymax=None,lmin=None,lmax=None,
        xlim=None,ylim=None,center=None,width=None,axis=None,
        cmap=None,vmin=None,vmax=None,zero=False,title=None,
        mesh=False,ident=False,bbox=dict(facecolor='grey',alpha=1),
        label_format='%.6f',mark=None,marker='o',verbose=0,**kw):
    """ 
         cmap: color map name (e.g. 'coolwarm')
        x,y,z: select plane by specifying one
       center: 3-tuple zoom center
        width: zoom width
         axis: view axis
    vmin/vmax: colorscale end points
    xmin/xmax: view limits in x (deprecated)
    ymin/ymax: view limits in y (deprecated)
    xlim/ylim: view limits (xmin,xmax), (ymin,ymax)
          log: force log10 values
         zero: symmetric colorscale around value=0.0
        title: text title
         mesh: display patch boundaries
        ident: print the patch ID at lower left corner
         bbox: bounding box of ident text
      verbose: noise level
    """
    if not hasattr(sn,'patches'):
        print('snapshot has no patches')
        return
    def contains(p,s,axis):
        return abs(s[axis]-p.position[axis]) <= p.size[axis]/2
    def outline(e):
        xl=[e[0],e[0],e[1],e[1],e[0]]
        yl=[e[2],e[3],e[3],e[2],e[2]]
        pl.plot(xl,yl,color='grey')
    def interpolate(p,iv,s,axis=2,log=False,verbose=0,**kw):
        n=p.n[axis]
        ds=p.size[axis]/n
        s0=p.position[axis]-p.size[axis]/2+p.ds[axis]/2
        w=(s-s0)/ds+1e-5
        i=int(w)
        i=max(0,min(i,n-2))
        w=w-i
        w=min(max(w,0),1)
        f=p.var(iv,**kw)
        if log:
            f=np.log(f)
        if axis==2:
            f=(1-w)*f[:,:,i]+w*f[:,:,i+1]
        elif axis==1:
            f=(1-w)*f[:,i,:]+w*f[:,i+1,:]
        elif axis==0:
            f=(1-w)*f[i,:,:]+w*f[i+1,:,:]
        if log:
            f=np.exp(f)
        if verbose>1:
            print('id: {:3d}   i: {:2d}   w: {:6.3f}  l: {}  p: {}  max: {:.2e}'.format(p.id,i,w,p.level,p.position,f.max()))
        return f
    if verbose:
        start=time.time()
    if axis is None:
        if x is not None:
            axis=0
        if y is not None:
            axis=1
        if z is not None:
            axis=2
    if width is None:
        width=1
    if np.isscalar(width):
        width=(width,width,width)
    if center is None:
        if x is not None:
            center=[x,0.,0.]
        if y is not None:
            center=[0.,y,0.]
        if z is not None:
            center=[0.,0.,z]
    if axis==0:
        xmin=center[1]-width[0]/2
        xmax=center[1]+width[0]/2
        ymin=center[2]-width[1]/2
        ymax=center[2]+width[1]/2
        x=center[0]
        y=None
        z=None
    if axis==1:
        xmin=center[0]-width[0]/2
        xmax=center[0]+width[0]/2
        ymin=center[2]-width[1]/2
        ymax=center[2]+width[1]/2
        x=None
        y=center[1]
        z=None
    if axis==2:
        xmin=center[0]-width[0]/2
        xmax=center[0]+width[0]/2
        ymin=center[1]-width[1]/2
        ymax=center[1]+width[1]/2
        x=None
        y=None
        z=center[2]
    if xlim:
        xmin=xlim[0]
        xmax=xlim[1]
    if ylim:
        ymin=ylim[0]
        ymax=ylim[1]
    axis,s=_axis(x,y,z)
    if not lmin:
        lmin=1
    if not lmax:
        lmax=30
    
    le=np.zeros((3))
    ue=np.zeros((3))
    if axis==0:
        le[0]=x
        ue[0]=x
        le[1]=center[1]-width[0]/2
        ue[1]=center[1]+width[0]/2
        le[2]=center[2]-width[1]/2
        ue[2]=center[2]+width[1]/2
    if axis==1:
        le[1]=y
        ue[1]=y
        le[0]=center[0]-width[0]/2
        ue[0]=center[0]+width[0]/2
        le[2]=center[2]-width[1]/2
        ue[2]=center[2]+width[1]/2
    if axis==2:
        le[2]=z
        ue[2]=z
        le[0]=center[0]-width[0]/2
        ue[0]=center[0]+width[0]/2
        le[1]=center[1]-width[1]/2
        ue[1]=center[1]+width[1]/2
    pp=[p for p in sn.patches if p.contains(le,ue,lmax=lmax,lmin=lmin)]
    
    pp=sort_by_level(pp)
    vmin1=1e20
    vmax1=-vmin1
    for p in pp:
        d=interpolate(p,iv,s,axis,**kw)
        vmin1=min(vmin1,d.min())
        vmax1=max(vmax1,d.max())
        if verbose>1:
            print('id: {:3d}  axis:{}  dmin: {:10.3e}  dmax: {:10.3e}'.format(p.id,axis,d.min(),d.max()))
    if vmin is None: vmin=vmin1
    if vmax is None: vmax=vmax1
    if zero:
        vmax = max(abs(vmax),abs(vmin))
        vmin = -vmax
    cmap=_cmap(np.array([vmin,vmax]),cmap,verbose=verbose)
    vv=np.array([vmin,vmax])
    if log:
        vmin=np.log10(vmin)
        vmax=np.log10(vmax)
    # set extent, with optional rescaling
    ex=_rescale(pp[0].extent,sn,center,to=to)
    xmn,xmx,ymn,ymx=ex[axis]
    for p in pp:
        ex=_rescale(p.extent,sn,center,to=to)
        e=ex[axis]
        xmn=min(xmn,e[0])
        xmx=max(xmx,e[1])
        ymn=min(xmn,e[2])
        ymx=max(xmx,e[3])
    if xmin:
        xmn=_recenter(sn,axis,center,to,x=xmin)
    if xmax:
        xmx=_recenter(sn,axis,center,to,x=xmax)
    if ymin:
        ymn=_recenter(sn,axis,center,to,y=ymin)
    if ymax:
        ymx=_recenter(sn,axis,center,to,y=ymax)
    if axis==0:
        s=x
        label=('y','z')
    if axis==1:
        s=y
        label=('x','z')
    if axis==2:
        s=z
        label=('x','y')
    for p in pp:
        ex=_rescale(p.extent,sn,center,to=to)
        e=ex[axis]
        d=interpolate(p,iv,s,axis,verbose=verbose)
        if log:
            d=np.log10(interpolate(p,iv,s,axis,log,verbose=verbose))
        pl.imshow(d.transpose(),extent=e,
              vmin=vmin,vmax=vmax,origin='lower',cmap=cmap)
        #print(p.id,(p.position-center)/0.5**lmax)
        if to=='au':
            pl.xlabel(label[0]+' [AU]')
            pl.ylabel(label[1]+' [AU]')
        elif to=='pc':
            pl.xlabel(label[0]+' [pc]')
            pl.ylabel(label[1]+' [pc]')
        else:
            pl.xlabel(label[0])
            pl.ylabel(label[1])
        x0,x1,y0,y1=e
        if mesh:
            outline(e)
        if ident:
            x0,x1,y0,y1=e
            if x0>=xmn and x0<=xmx and y0>=ymn and y0<=ymx:
                pl.text(x0,y0,'{}'.format(p.id),bbox=bbox)
    pl.xlim(xmn,xmx)
    pl.ylim(ymn,ymx)
    pl.colorbar()
    gc=pl.gca()
    # gc.set_xticks(gc.get_xticks()[::2])
    gc.set_aspect('equal',adjustable='datalim')
    gc.set_box_aspect(1)
    if to is None:
        gc.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(label_format))
    _title(sn,x,y,z,iv,title)
    if verbose:
        print('{} patches, {:.2e} sec'.format(len(pp),time.time()-start))
    if mark is not None:
        pl.plot(mark[0],mark[1],marker=marker,**kw)

def _recenter(sn,axis,center=None,to=None,x=None,y=None):
    if to=='au':
        c=sn.scaling.l/sn.cgs.au
    elif to=='pc':
        c=sn.scaling.l/sn.cgs.pc
    else:
        if x:
            return x
        else:
            return y
    if center is None:
        center=[0,0,0]
    if axis==0:
        if x:
            v=x-center[1]
        if y:
            v=y-center[2]
    if axis==1:
        if x:
            v=x-center[0]
        if y:
            v=y-center[2]
    if axis==2:
        if x:
            v=x-center[0]
        if y:
            v=y-center[1]
    return v*c

def _rescale(extent,sn,center=None,to=None):
    e=np.copy(extent)
    if to=='au':
        c=sn.scaling.l/sn.cgs.au
    elif to=='pc':
        c=sn.scaling.l/sn.cgs.pc
    else:
        return e
    if center is not None:
        e[0][0:2] -= center[1]
        e[0][2:4] -= center[2]
        e[1][0:2] -= center[0]
        e[1][2:4] -= center[2]
        e[2][0:2] -= center[0]
        e[2][2:4] -= center[1]
    return e*c

def sort_by_level(pp):
    ii=np.argsort([p.level for p in pp])
    return [pp[i] for i in ii]

def plot_pdf(pdf,**kwargs):
    """ Plot the output of the dispatch.pdf() procedure """
    pl.hist(pdf.bins,bins=pdf.bins,weights=pdf.counts,**kwargs)
    return pdf.time

def pdf(io,run='',data='../data',log=True,**kwargs):
    """ Shortcut of dispatch.graphics.plot_pdf(dispatch.pdf(io,run,data)) """
    tmp=di.pdf(io,run,data,**kwargs)
    plot_pdf(tmp,log=log,**kwargs)

def outline(p,nb,ps='o',color='red',ROI=False):
    """ Outline of nbor patch nb, projected onto
        the xy-plane of patch
    """
    xhat=np.matmul(p.erot,nb.erot[0])[0:2]
    yhat=np.matmul(p.erot,nb.erot[1])[0:2]
    dx=xhat*nb.size[0]
    dy=yhat*nb.size[1]
    if ROI:
        dx*=p.gn[0]/p.n[0]
        dy*=p.gn[1]/p.n[1]
    p0=np.matmul(p.erot,nb.position)[0:2]  # position of the nbor, expressed in local (x,y) coordinates
    xy=np.copy(p0)
    xy=xy-dx/2-dy/2; pp=[xy]
    xy=xy+dy; pp.append(xy)
    xy=xy+dx; pp.append(xy)
    xy=xy-dy; pp.append(xy)
    xy=xy-dx; pp.append(xy)
    pp=np.array(pp).transpose()
    pl.plot(pp[0],pp[1],color=color)
    pl.plot(p0[0],p0[1],ps,color=color)
    
def show_nbors(sn,p=None,size=6,all=None):
    """ Show outlines of nbor patches, by default
        only at the same layer radius
    """
    if p is None:
        p=sn.patches[0]                             # default patch
    pl.figure(figsize=(size,size))                  # setup square figure
    rp=np.sum(p.position*p.erot[2])                 # layer radius
    outline(p,p,'s',ROI=True)                       # outline of patch ROI
    for id in p.nbor_ids:                           # loop over nbor IDs
        nb=sn.patches[id-1]                         # nbor patch
        x=np.sum(nb.position*p.erot[0])             # x-position of center
        y=np.sum(nb.position*p.erot[1])             # y-position of center
        rnb=np.sum(nb.position*nb.erot[2])          # neighbor layer radius
        xx=np.sum(p.erot[0]*nb.erot[0])             # length of nbor x-hat
        if abs(rp-rnb)<1 or all:                    # show only the same layer
            if xx>0.99:                             # same face
                outline(p,nb,color='grey')          # grey outline
            else:                                   # different face
                outline(p,nb,'+',color='lightblue') # light-blue outline
    # make plot isotropic
    xl=pl.xlim(); wx=xl[1]-xl[0]
    yl=pl.ylim(); wy=yl[1]-yl[0]
    xm=(xl[0]+xl[1])/2
    ym=(yl[0]+yl[1])/2
    w=max(wx,wy)/2
    pl.xlim(xm-w,xm+w)
    pl.ylim(ym-w,ym+w)
    if all:
        pl.title('all nbors of patch {}'.format(p.id))
    else:
        pl.title('layer nbors of patch {}'.format(p.id))

def oplot(x,y=None,color=None,linewidth=5,lw=None,opacity=0.01):
    """ For plotting bundles of lines, making a shading variation plot.
        Example:
        
        from numpy.random import normal as rnd
        x=numpy.linspace(0,10)
        for i in range(100):
            oplot(x,sin(x+rnd()*0.01)+rnd()*0.1)
        _
        """
    if lw is None:
        lw=linewidth
    if color=='green' or color=='g':
        c=[0.0,0.8,0.0,opacity]
    elif color=='blue' or color=='b':
        c=[0.0,0.8,0.8,opacity]
    elif color=='red' or color=='r':
        c=[1,0.5,0.0,opacity]
    else:
        c=[0.0,0.0,0.0,opacity]
    if y is None:
        pl.plot(x,color=c,linewidth=lw)
    else:
        pl.plot(x,y,color=c,linewidth=lw)
