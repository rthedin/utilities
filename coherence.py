import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import repeat

from mmctools.helper_functions import calc_spectra

# Collection of general functions related to coherence


def calc_coherece_2signals_par(ds, varnameSweep, sweep, othervar, zref, sep, strname=None,
                               interval='120min', window_length='10min', window='hamming',
                               normal=None, crosscoherence=False, nCores=36):
    
    # Get list of dictionaries for parallel function. Some of these values might be None if single plane
    if varnameSweep == 'x':
        if strname == 'vert':
            s1_list=[{'ds':ds, 'x':sweep[i], 'y':othervar, 'z':zref}     for i in range(len(sweep))]
            s2_list=[{'ds':ds, 'x':sweep[i], 'y':othervar, 'z':zref+sep} for i in range(len(sweep))]
        elif strname == 'long':
            print(f'NOT TESTED')
            s1_list=[{'ds':ds, 'x':sweep[i]+sep, 'y':othervar, 'z':zref} for i in range(len(sweep))]
            s2_list=[{'ds':ds, 'x':sweep[i]+sep, 'y':othervar, 'z':zref} for i in range(len(sweep))]
        else:
            raise ValueError (f'Sweep in x for lateral and longitudinal separation is not implemented.')
    elif varnameSweep == 'y':
        if strname == 'lat':
            s1_list=[{'ds':ds, 'x':othervar, 'y':sweep[i],     'z':zref} for i in range(len(sweep))]
            s2_list=[{'ds':ds, 'x':othervar, 'y':sweep[i]+sep, 'z':zref} for i in range(len(sweep))]
        elif strname == 'vert':
            s1_list=[{'ds':ds, 'x':othervar, 'y':sweep[i],     'z':zref}     for i in range(len(sweep))]
            s2_list=[{'ds':ds, 'x':othervar, 'y':sweep[i],     'z':zref+sep} for i in range(len(sweep))]
        else:
            raise ValueError (f'Sweep in y for longitudinal separation is not supported')
    elif varnameSweep == 'z':   
        raise NotImplemented
    else:
        raise ValueError (f'Variable to sweep can only be x, y, or z. Received {varSweep}.')


    with Pool() as p: #p = Pool()
        ds_ = p.starmap(calc_coherence_2signals, zip(s1_list,                 # s1
                                                     s2_list,                 # s2
                                                     repeat(strname),         # strname
                                                     repeat(interval),        # interval
                                                     repeat(window_length),   # window_length
                                                     repeat(window),          # window
                                                     repeat(normal),          # normal
                                                     repeat(crosscoherence),  # crosscoherence
                                                    )
                                                 )

    # ds_ is a tuple since it has the coherence and the psd resuts returned by calc_coherence_2signals
    coh_var = [d[0] for d in ds_]

    # Add the varying coordinate
    coh_var = xr.concat(coh_var, dim=varnameSweep)
    coh_var[varnameSweep]=sweep

    return coh_var 



def calc_coherence_2signals(s1,s2, strname=None, interval='120min', window_length='10min', window='hamming', normal=None, crosscoherence=False):
    '''
    Calculates the correrence between two signals
    
    Parameters
    ==========
    s1, s2: 
        DataArray with datetime as coordinate 
        
    or,
    s1, s2: dictionary
        Dicts with full Dataset and specified coordinates and
        variable
    
    strname: str
        String name for the variable. Options: 'vert', 'lat', 'lon'
        Checks if the points passed satisfies the array asked. This is just for
        naming the returned array
        
    '''
    
    if isinstance(s1, dict):
        if not isinstance(s2,dict):
            raise ValueError('Both series need to be given in the same format')
        try:
            # Let's first see if any of the values are None. That happens when a single plane is present
            # If that is the case, let's force it to fall on the ifs below that does not look for that dimension
            try:
                if s1['x'] is None:
                    # s2 is also none
                    assert s2['x'] is None
                    normal = 'xNormal'
            except KeyError:
                # 'x' does not exist. so this is fine
                pass
            try:
                if s1['y'] is None:
                    assert s2['y'] is None
                    # s2['y'] should also be None
                    normal = 'yNormal'
            except KeyError:
                # 'y' does not exist, so this is fine
                pass

            if normal=='zNormal':
                sig1 = s1['ds'].sel(x=s1['x'], y=s1['y'], drop=True, method='nearest', tolerance=1e-3)#.squeeze()
                sig2 = s2['ds'].sel(x=s2['x'], y=s2['y'], drop=True, method='nearest', tolerance=1e-3)#.squeeze()
            elif normal=='xNormal': 
                sig1 = s1['ds'].sel(z=s1['z'], y=s1['y'], drop=True, method='nearest', tolerance=1e-3)
                sig2 = s2['ds'].sel(z=s2['z'], y=s2['y'], drop=True, method='nearest', tolerance=1e-3)
            elif normal=='yNormal': 
                sig1 = s1['ds'].sel(z=s1['z'], x=s1['x'], drop=True, method='nearest', tolerance=1e-3)#.squeeze()
                sig2 = s2['ds'].sel(z=s2['z'], x=s2['x'], drop=True, method='nearest', tolerance=1e-3)#.squeeze()
            else:
                # Most calls to this func coming from *par functions will fall here
                sig1 = s1['ds'].sel(x=s1['x'], y=s1['y'], z=s1['z'], drop=True, method='nearest', tolerance=1e-3)
                sig2 = s2['ds'].sel(x=s2['x'], y=s2['y'], z=s2['z'], drop=True, method='nearest', tolerance=1e-3)

        except KeyError:
            print(f"ERROR: A value asked for does not exist.                                      ")
            if normal=='zNormal':
                print(f"       For the 1st signal, asked coordinate is x1={s1['x']}, y1={s1['y']}.",
                               f"Closest x is {s1['ds'].x.sel(x=s1['x'],method='nearest').values}; ",
                               f"closest y is {s1['ds'].y.sel(y=s1['y'],method='nearest').values}")
                print(f"       For the 2nd signal, asked coordinate is x2={s2['x']}, y2={s2['y']}.",
                               f"Closest x is {s2['ds'].x.sel(x=s2['x'],method='nearest').values};",
                               f"closest y is {s2['ds'].y.sel(y=s2['y'],method='nearest').values}")
            elif normal=='xNormal': 
                print(f"       For the 1st signal, asked coordinate is y1={s1['y']}, z1={s1['z']}.",
                               f"Closest y is {s1['ds'].y.sel(y=s1['y'],method='nearest').values}; ",
                               f"closest z is {s1['ds'].z.sel(z=s1['z'],method='nearest').values}")
                print(f"       For the 2nd signal, asked coordinate is y2={s2['y']}, z2={s2['z']}.",
                               f"Closest y is {s2['ds'].y.sel(y=s2['y'],method='nearest').values};",
                               f"closest z is {s2['ds'].z.sel(z=s2['z'],method='nearest').values}")
            elif normal=='yNormal': 
                print(f"       For the 1st signal, asked coordinate is x1={s1['x']}, z1={s1['z']}.",
                               f"Closest x is {s1['ds'].x.sel(x=s1['x'],method='nearest').values}; ",
                               f"closest z is {s1['ds'].z.sel(z=s1['z'],method='nearest').values}")
                print(f"       For the 2nd signal, asked coordinate is x2={s2['x']}, z2={s2['z']}.",
                               f"Closest x is {s2['ds'].x.sel(x=s2['x'],method='nearest').values};",
                               f"closest z is {s2['ds'].z.sel(z=s2['z'],method='nearest').values}")
            raise


    if strname == 'vert':
        try:
            if s1['x'] != s2['x']:
                raise ValueError(f"Requested vertical separation but points have different x value: {s1['x']} and {s2['x']}")
        except KeyError as e:
            if str(e) == "'x'":
                pass
            else:
                raise

        if s1['y'] != s2['y']:
            raise ValueError(f"Requested vertical separation but points have different y value: {s1['y']} and {s2['y']}")

    elif strname == 'lat':
        try:
            if s1['x'] != s2['x']:
                raise ValueError(f"Requested lateral separation but points have different x value: {s1['x']} and {s2['x']}")
        except KeyError as e:
            if str(e) == "'x'":
                pass
            else:
                raise

        if s1['z'] != s2['z']:
            raise ValueError(f"Requested lateral separation but points have different z value: {s1['z']} and {s2['z']}")

    elif strname == 'lon':
        if s1['y'] != s2['y']:
            raise ValueError(f"Requested longitudinal separation but points have different y value: {s1['y']} and {s2['y']}")

        try:  # Maybe z doesn't exist, that is, the ds has been sliced before
            if s1['z'] != s2['z']:
                raise ValueError(f"Requested longitudinal separation but points have different z value: {s1['z']} and {s2['z']}")
        except KeyError:
            pass

    else:
        raise ValueError(f'`strname` needs to be given either as "vert", "lat", or "lon".')
        
            
    spectraTimes = pd.date_range(start=sig1.datetime[0].values, end=sig1.datetime[-1].values, freq='10min')
    
    try:
        signals = xr.merge([ sig1['up'].to_dataset(name='u1'), sig1['vp'].to_dataset(name='v1'), sig1['wp'].to_dataset(name='w1'),
                             sig2['up'].to_dataset(name='u2'), sig2['vp'].to_dataset(name='v2'), sig2['wp'].to_dataset(name='w2') ])
    except xr.MergeError:
        #print(f'Got MergeError exception on the xr.merge. Trying with `override` option. This should not occur if you are ')
        #print( 'actually using the normal planes (i.e. lat and long coherence on znormal, vertical on {x,y}normal.')
        signals = xr.merge([ sig1['up'].to_dataset(name='u1'), sig1['vp'].to_dataset(name='v1'), sig1['wp'].to_dataset(name='w1'),
                             sig2['up'].to_dataset(name='u2'), sig2['vp'].to_dataset(name='v2'), sig2['wp'].to_dataset(name='w2') ],
                             compat='override')

    psd = calc_spectra(signals,
                       var_oi=['u1','u2','v1','v2','w1','w2'],
                       xvar_oi=[('u1','u2'),('v1','v2'),('w1','w2')],
                       spectra_dim='datetime',
                       tstart=spectraTimes[0], # first time instant of nyserda
                       #average_dim='station',
                       #level_dim='height',
                       window=window,
                       interval=interval,
                       window_length=window_length,
                       window_overlap_pct=0.5)

    mscoh = xr.merge([ (abs(psd['u1u2'])**2/(psd['u1']*psd['u2'])).to_dataset(name=f'mscoh_{strname}sep_u1u2'),
                       (abs(psd['v1v2'])**2/(psd['v1']*psd['v2'])).to_dataset(name=f'mscoh_{strname}sep_v1v2'),
                       (abs(psd['w1w2'])**2/(psd['w1']*psd['w2'])).to_dataset(name=f'mscoh_{strname}sep_w1w2') ])
    
    cocoh = xr.merge([ ( psd['u1u2'].real/(psd['u1']*psd['u2'])**0.5 ).to_dataset(name=f'cocoh_{strname}sep_u1u2'),
                       ( psd['v1v2'].real/(psd['v1']*psd['v2'])**0.5 ).to_dataset(name=f'cocoh_{strname}sep_v1v2'),
                       ( psd['w1w2'].real/(psd['w1']*psd['w2'])**0.5 ).to_dataset(name=f'cocoh_{strname}sep_w1w2') ])
    
    qucoh = xr.merge([ ((psd['u1u2']/((psd['u1']*psd['u2'])**0.5)).imag).to_dataset(name=f'qucoh_{strname}sep_u1u2'),
                       ((psd['v1v2']/((psd['v1']*psd['v2'])**0.5)).imag).to_dataset(name=f'qucoh_{strname}sep_v1v2'),
                       ((psd['w1w2']/((psd['w1']*psd['w2'])**0.5)).imag).to_dataset(name=f'qucoh_{strname}sep_w1w2') ])
   
    qucoh_ =xr.merge([ ( psd['u1u2'].imag/(psd['u1']*psd['u2'])**0.5 ).to_dataset(name=f'qucohnew_{strname}sep_u1u2'),
                       ( psd['v1v2'].imag/(psd['v1']*psd['v2'])**0.5 ).to_dataset(name=f'qucohnew_{strname}sep_v1v2'),
                       ( psd['w1w2'].imag/(psd['w1']*psd['w2'])**0.5 ).to_dataset(name=f'qucohnew_{strname}sep_w1w2') ])

    xpsd  = xr.merge([ (psd['u1u2']).to_dataset(name=f'xpsd_{strname}sep_u1u2'),
                       (psd['v1v2']).to_dataset(name=f'xpsd_{strname}sep_v1v2'),
                       (psd['w1w2']).to_dataset(name=f'xpsd_{strname}sep_w1w2') ])

    psd_ =xr.merge([ (psd['u1']).to_dataset(name=f'psd_{strname}sep_u1'),
                    (psd['v1']).to_dataset(name=f'psd_{strname}sep_v1'),
                    (psd['w1']).to_dataset(name=f'psd_{strname}sep_w1'),
                    (psd['v2']).to_dataset(name=f'psd_{strname}sep_u2'),
                    (psd['v2']).to_dataset(name=f'psd_{strname}sep_v2'),
                    (psd['w2']).to_dataset(name=f'psd_{strname}sep_w2') ])

    radius =xr.merge([ ( (psd['u1u2'].real**2+psd['u1u2'].imag**2)**0.5/(psd['u1']*psd['u2'])**0.5 ).to_dataset(name=f'radius_{strname}sep_u1u2'),
                       ( (psd['v1v2'].real**2+psd['v1v2'].imag**2)**0.5/(psd['v1']*psd['v2'])**0.5 ).to_dataset(name=f'radius_{strname}sep_v1v2'),
                       ( (psd['w1w2'].real**2+psd['w1w2'].imag**2)**0.5/(psd['w1']*psd['w2'])**0.5 ).to_dataset(name=f'radius_{strname}sep_w1w2') ])

    return xr.merge([mscoh,cocoh,qucoh,qucoh_,xpsd, psd_, radius]), psd




def plotCoherence(coh_sep,
                  sep_list,
                  meandim,
                  umean=None,
                  xaxis='freq',
                  xscale='linear',
                  qoi=['ms','co','qu'],
                  fig=None, axs=None,
                  showplot=True,  
                  a=None, b=None, B=None,
                  xlim=None, ylim=None,
                  resetColors=False, icolor=0,
                  labelPrefix='',
                  **kwargs):
    '''
    Plot mscoh, co-coherence, quad-coherence, and radius coherence from coh_sep dataset(s)
    
    coh_sep: xr.Dataset, or list of xr.Dataset
        Dataset or list of datasets containing all the coherence values
        One of the coordinates should be sep_x, sep_y, or sep_z
    sep_list: array of floats
        List of separation distances to plot
    meandir: str
        Direction to compute the mean. Should be 'x', 'y', or 'z'
    umean: float
        Mean velocity used to compute IEC coherence. IEC curves are skipped is umean is not specified
        If umean is not specified, the xaxis has to be `freq`.
    xaxis: str ('freq', 'redfreq', 'wavenumb', 'redwave'; default 'freq')
        Quantity to use on the xaxis
    xscale: str
        scale of the x axis (e.g. 'linear', 'log') 
    qoi: str, array of string
        What quantities to plot. Options: 'ms','co','qu','r' (or 'all' instead).
    fig, axs: matplotlib figure and axis
        If want to plot on top of an existing figure and axis
    a, b, B: float
        Values for a and b (or B) for Davenport's model
    xlim, ylim: tuple
        Limits for the axis
    resetColors: bool
        Whether or not to reset the color loops when giving a list of datasets to be plotted.
        This option is helpful when used with ls and alpha given as kwargs
    icolor: int
        Index of the starting color loop for reset color. Changed internally. User should not change this.
    labelPrefix: str
        Label prefix used in the legend for current dataset or list of dataset. Useful if the plan is
        to return axs/fig and re-use them on more datasets. If plotting vertical and lateral separations,
        the legend title is removed, so this might be a good option. E.g. labelPrefix='sep z'
    **kwargs:
        Plot arguments. E.g. ls='--', alpha=0.3

    '''

    import matplotlib.colors as mcolors

    # If the user pass a list, then call the function separately for all items
    if isinstance(coh_sep, list):
        # If this is the first call and there is a list of datasets to plot, we plot the first dataset
        # with fig and axs None, then use the returned fig and axs for the subsequent ones. However, if
        # this is already a call giving fig and axs and is also a list of datasets, we should not clear
        # the fig and axs.

        # Adjust the first color to be used. It will only be reseted once per list of datasets
        if axs is not None and resetColors:
            icolor = -len(axs.flatten()[1].lines)
            resetColors = False
        else:
            icolor = 0

        for d in range(len(coh_sep)):
            fig, axs = plotCoherence(coh_sep[d], sep_list, meandim, umean, xaxis, xscale, qoi,
                                     fig=fig, axs=axs, showplot=False,
                                     a=a, b=b, B=B, xlim=xlim, ylim=ylim, resetColors=resetColors, icolor=icolor,
                                     labelPrefix=labelPrefix, **kwargs)

        return fig, axs

        
   
    # Check what xaxis was requested
    if xaxis not in {'freq','redfreq','wavenumb','redwave'}:
        raise ValueError (f'The argument xaxis can only take `freq` `redfreq`, `wavenumb`, or `redwave`.')

    # Check what plots were requested (mscoh, cocoh, quad, radius)
    if isinstance(qoi,str):
        # Single quantity requested (or all)
        if qoi == 'all':
            qoi = ['ms','co','qu','r']
        else:
            qoi = [qoi]
    nqoi = len(qoi)



    # Get auxiliary arrays
    sepcoord = [coord for coord in list(coh_sep.coords.keys()) if coord.startswith('sep')][0]
    if  sepcoord[-1] == 'x':
        cohsepstr = 'lon'
    elif sepcoord[-1] == 'y':
        cohsepstr = 'lat'
    elif sepcoord[-1] == 'z':
        cohsepstr = 'vert'
    else:
        raise ValueError (f'A coordinate named sep_x, sep_y, or sep_z must exist. Stopping')

    # Let's check if all the separations exist in the dataset before plotting
    if not set(sep_list).issubset(set(coh_sep[sepcoord].values)):
        raise ValueError (f'Not all separation distances requested to plot are available. Stopping. '\
                          f'Available ones: {coh_sep[sepcoord].values}')



    # Initialize figure and axis if not passed
    newAxs = False
    if fig is None and axs is None: 
        fig, axs = plt.subplots(nqoi,3,figsize=(16,nqoi*3), sharey=True, sharex=True, gridspec_kw = {'wspace':0.08, 'hspace':0.08})
        #fig, axs = plt.subplots(nqoi,3,figsize=(18,nqoi*3), sharey=True, sharex=True, gridspec_kw = {'wspace':0.08, 'hspace':0.15})
        axs = np.atleast_2d(axs) # Make axs 2-D even when len(qoi)==1
        newAxs = True

    # Let's figure out how many curves are on each plot so we can pick the next color
    # Note that axs with IEC curve have an extra line, so we will not use those
    colors = list(mcolors.TABLEAU_COLORS)
    ncurves = len(axs.flatten()[1].lines)

    for c, sep in enumerate(sep_list):
        row=0

        # Label to be printed (more info depending on the separation direction; verbose on purpose)
        # If no attributes are present on the datasets, then just print the separation
        if 'zref' not in coh_sep.attrs.keys():
            label = f'{sep} m'
        elif cohsepstr == 'vert': 
            z1 = coh_sep.attrs['zref']
            z2 = z1+sep
            label = f'{sep} m, z={z1}, {z2} m'
        elif cohsepstr == 'lat':
            zref = coh_sep.attrs['zref']
            label = f'{sep} m at z={zref} m'
        elif cohsepstr == 'lon':
            zref = coh_sep.attrs['zref']
            label = f'{sep} m at z={zref} m'

        # If a prefix for the labels has been given, add a space at the end if needed and use that
        if labelPrefix != '':
            label = labelPrefix.rstrip() + ' ' + label

        # get the frequency axis
        if xaxis   == 'freq':     f = coh_sep.frequency;                         xlabel = 'freq [Hz]'
        elif xaxis == 'redfreq':  f = coh_sep.frequency * sep / umean;           xlabel = 'reduced freq (f*sep/u) [-]'
        elif xaxis == 'wavenumb': f = coh_sep.frequency * 2*np.pi / umean;       xlabel = 'wave number k (2pi*f/u) [1/m]'
        elif xaxis == 'redwave':  f = coh_sep.frequency * 2*np.pi * sep /umean;  xlabel = 'reduced wave number (k*sep) [-]'

        # get arrays for convenience and readability
        ms_u = coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'mscoh_{cohsepstr}sep_u1u2']
        ms_v = coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'mscoh_{cohsepstr}sep_v1v2']
        ms_w = coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'mscoh_{cohsepstr}sep_w1w2']
        co_u = coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'cocoh_{cohsepstr}sep_u1u2']
        co_v = coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'cocoh_{cohsepstr}sep_v1v2']
        co_w = coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'cocoh_{cohsepstr}sep_w1w2']
        qu_u = coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'qucoh_{cohsepstr}sep_u1u2']
        qu_v = coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'qucoh_{cohsepstr}sep_v1v2']
        qu_w = coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'qucoh_{cohsepstr}sep_w1w2']

        # Get current color. ncurves might have been reset
        currcolor = colors[(c+ncurves+icolor)%len(colors)]

        # ---------------------
        if 'ms' in qoi or 'mscoh' in qoi:
            # mscoh uu
            axs[row,0].plot(f, ms_u, c=currcolor, label=label, **kwargs)
            if umean is not None:
                axs[row,0].plot(f, davenportExpCoh(coh_sep.frequency,u=umean,delta=sep,Lc='defaultu',a=a,b=b,B=B), c=currcolor, ls='--', alpha=0.7)
            # mscoh vv
            axs[row,1].plot(f, ms_v, c=currcolor, label=label, **kwargs)
            # mscoh vv
            axs[row,2].plot(f, ms_w, c=currcolor, label=label, **kwargs)
            # Set titles
            if c==0 and newAxs:
                axs[row,0].text(0.98, 0.97, f'mscoh $\gamma^2_{{uu, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,0].transAxes, fontsize=14)
                axs[row,1].text(0.98, 0.97, f'mscoh $\gamma^2_{{vv, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,1].transAxes, fontsize=14)
                axs[row,2].text(0.98, 0.97, f'mscoh $\gamma^2_{{ww, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,2].transAxes, fontsize=14)
            row = row+1
        # ---------------------

        # ---------------------
        if 'co' in qoi or 'cocoh' in qoi:
            # co-coh uu
            axs[row,0].plot(f, co_u, c=currcolor, label=label, **kwargs)
            if umean is not None:
                axs[row,0].plot(f, davenportExpCoh(coh_sep.frequency,u=umean,delta=sep,Lc='defaultu',a=a, b=b, B=B), c=currcolor, ls='--', alpha=0.7)
            # co-coh vv
            axs[row,1].plot(f, co_v, c=currcolor, label=label, **kwargs)
            # co-coh ww
            axs[row,2].plot(f, co_w, c=currcolor, label=label, **kwargs)
            # Set titles
            if c==0 and newAxs:
                axs[row,0].text(0.98, 0.97, f'co-coh $\gamma_{{uu, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,0].transAxes, fontsize=14)
                axs[row,1].text(0.98, 0.97, f'co-coh $\gamma_{{vv, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,1].transAxes, fontsize=14)
                axs[row,2].text(0.98, 0.97, f'co-coh $\gamma_{{ww, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,2].transAxes, fontsize=14)
            row = row+1
        # ---------------------

        # ---------------------
        if 'qu' in qoi or 'quadcoh' in qoi:
            # quad-coh uu
            axs[row,0].plot(f, qu_u, c=currcolor, label=label, **kwargs)
            # quad-coh vv
            axs[row,1].plot(f, qu_v, c=currcolor, label=label, **kwargs)
            # quad-coh ww
            axs[row,2].plot(f, qu_w, c=currcolor, label=label, **kwargs)
            # Set titles
            if c==0 and newAxs:
                axs[row,0].text(0.98, 0.97, f'quad-coh $\\rho_{{uu, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,0].transAxes, fontsize=14)
                axs[row,1].text(0.98, 0.97, f'quad-coh $\\rho_{{vv, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,1].transAxes, fontsize=14)
                axs[row,2].text(0.98, 0.97, f'quad-coh $\\rho_{{ww, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,2].transAxes, fontsize=14)
            row = row+1
        # ---------------------

        # ---------------------
        if 'r' in qoi or 'radiuscoh' in qoi or 'rcoh' in qoi:
            # radius coh uu
            axs[row,0].plot(f, (co_u**2+qu_u**2)**0.5, c=currcolor, label=label, **kwargs)
            if umean is not None:
                axs[row,0].plot(f, davenportExpCoh(coh_sep.frequency,u=umean,delta=sep,Lc='defaultu',a=a, b=b, B=B), c=currcolor, ls='--', alpha=0.7)
            # radius coh vv
            axs[row,1].plot(f, (co_v**2+qu_v**2)**0.5, c=currcolor, label=label, **kwargs)
            # radius coh ww
            axs[row,2].plot(f, (co_w**2+qu_w**2)**0.5, c=currcolor, label=label, **kwargs)
            # Set titles
            if c==0 and newAxs:
                axs[row,0].text(0.98, 0.95, f'radius coh $R_{{uu, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,0].transAxes, fontsize=14)
                axs[row,1].text(0.98, 0.95, f'radius coh $R_{{vv, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,1].transAxes, fontsize=14)
                axs[row,2].text(0.98, 0.95, f'radius coh $R_{{ww, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,2].transAxes, fontsize=14)
            row = row+1
        # ---------------------


    for ax in axs.flatten():
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_xscale(xscale)
        ax.grid(True)

    # The title on the legend only makes sense if we're plotting the same separation. If we get the axs and plot another separation
    # on top, then the title can be misleading. So, if this is the second time plotting (given axs, fig), let's check against the 
    # prior legend title. If the same, we go forward. If not, we do not add a legend title
    title = f'{cohsepstr} sep'
    if not newAxs:
        title_curr = axs[0,-1].get_legend().get_title().get_text()
        if title != title_curr:
            title = ''
        
    axs[0,-1].legend(title=title, fontsize=12, title_fontsize=13, loc='upper left', bbox_to_anchor=(1,1))
    for ax in axs[-1,:]:  ax.set_xlabel(xlabel, fontsize=14)
    for ax in axs[:,0]:   ax.set_ylabel('coherence', fontsize=14)
    #plt.show()

    # for wesc figure
    #for ax in axs.flatten():
    #    ax.set_ylim([-0.2, 1])
    #    ax.set_xlim([0, 0.14])
    #    ax.set_xticks([0,0.02,0.04,0.06,0.08,0.10,0.12,0.14, 0.15])
    #    ax.set_xticklabels(['0','0.02','0.04','0.06','0.08','0.10','0.12','0.14',''])
    #    ax.set_yticks([-0.25,0,0.25, 0.5,0.75,1])
    #    ax.set_yticklabels(['-0.25','0','0.25', '0.50','0.75','1'])

    # Let's show the plot if adding curves to the axis
    if not newAxs and showplot:
        print('Showing the fig')
        display(fig)
    #elif not showplot:
    #    print(f'Not showing the fig. newaxs={newAxs}, showplot={showplot}')
    #    plt.close(fig)

    return fig, axs


def IECCoherence(f, Umeanhub, delta, component, cohexp=0, z=1, hubheight=80, mode='sameValues'):
    '''
    Calculate Kaimal coherence model given separation delta and mean wind speed U.
    NOT defined for longitudinal separation, only vertical and lateral.
    
    
    Adds Solari term by setting cohexp, and mean height of two points
    
    Parameters:
    ===========
    component: str
        Turbulence component. Options are 'streamwise'/'u', 'lateral'/'crossstream'/'v', 'vertical'/'w'
        
    hubheight: float
        Hub height. Specify if not 80. Used for Lc calculation
    '''
    
    if   component in ['streamwise','u','u1']:            comp=1
    elif component in ['lateral','crossstream','v','u2']: comp=2
    elif component in ['vertical','w','u3']:              comp=3
    else: raise ValueError('Unknown `component` specification. See docstrings')
    
    a=12
    b=0.12
    
    if hubheight< 60:  Lambda1=0.7*hubheight
    else:              Lambda1=42
    
    if mode == 'sameValues':
        if   comp==1: Lc= 8.1*Lambda1
        elif comp==2: Lc= 8.1*Lambda1
        elif comp==3: Lc= 8.1*Lambda1
    elif mode == 'diffValues':
        if   comp==1: Lc= 8.1*Lambda1
        elif comp==2: Lc= 2.7*Lambda1
        elif comp==3: Lc= 0.66*Lambda1
    else:
        raise ValueError
    
    solariTerm = (delta/z)**cohexp
    gamma = np.exp(-a*solariTerm*( (f*delta/Umeanhub)**2 + (b*delta/Lc)**2 )**0.5)
    
    return gamma

def davenportExpCoh(f,u,delta,Lc='defaultu',a=None,b=None,B=None):
    '''
    Computes the Davenport exponential coherence function
    For the second term, give either B (B = b/Lc), or both
    b and Lc

    If no a, b, or B are given, resort to default a=12, b=0.12


    '''

    if Lc == 'defaultu': Lc = 8.1*42
    elif Lc == 'defaultv': Lc = 2.7*42
    elif Lc == 'defaultw': Lc = 0.66*42
    else: raise valueError (f"Lc can only be `defaultu`, `defaultv`, or `defaultw`.")

    if a is None and b is None and B is None:
        a=12
        b=0.12

    if B is not None:
        if b is not None:
            raise ValueError(f"If giving B, b should not be given")


    if b is not None:
        if not isinstance(b,(float,int)):
            raise ValueError(f"b should be a scalar")
        if B is not None:
            raise ValueError(f"If giving b, B should not be given,")

        B=b/Lc

    return np.exp( -a*np.sqrt( (f*delta/u)**2 + (B*delta)**2) )

    


def calcVertCoh(dsx, sep_z_list, zref, interval, window_length, window, ydist, outputPath=None):
    '''
    Calculates vertical coherence
    
    Inputs
    ======
    dsx: xarray dataset
        Array containing perturbation valocity (up,vp,wp) at coordinates ([x,]y,z,datetime)
    sep_z_list: list of scalars
        Vertical separation values to compute coherence
    zref: scalar
        Reference z location (typically 150, 151.5, etc)
    interval: string
        Time interval of the whole dataset (typically '180min')
    window_length: string
        Window length for windowing approach (typically '5min' or '2min')
    window: string
        Windowing algorithm to use (typically 'hanning')
    ydist: scalar
        Lateral distance at which another pair of points will be collected (typically grid resolution)
    outputPath: string
        Path where a zarr file will be saved to
        
    Example call
    ============
    # Location of reference point
    zref = 151
    # Total length of the time series
    interval='180min'
    # Window parameters
    window_length = '2min'
    window = 'hanning'
    # At every ydist laterally (in y), get another pair of points
    ydist = 2.5
    # Vertical separation values of interest
    sep_z_list = [5, 10, 15, 25, 40, 80, 120]
    # At every ydist laterally (in y), get another pair of points
    ydist = 2.5
    
    coh_zsep = calcVertCoh(ds_cohu, sep_z_list, zref, interval, window_length, window, ydist, outpath)
    
    '''
    
    # ------------------------------ SETTINGS FOR VERTICAL SEPARATION COHERENCE
    # --------------------------------------------------- USING X-NORMAL PLANES
    # At every ydist laterally (in y), let's get another pair of points
    y_loc_list = np.arange(dsx.y.min(), dsx.y.max()-ydist, ydist)

    # Planes to loop on (if exists; will not exist in turbsim data)
    if 'x' in list(dsx.coords.keys()):
        xplanes = dsx.x.values
    else:
        xplanes = [None]
    # -------------------------------------------------------------------------


    if os.path.isdir(os.path.join(str(outputPath),f'coh_zsep_xnormal_{window_length}.zarr')):
        coh_zsep_xnormal = xr.open_zarr(os.path.join(outputPath,f'coh_zsep_xnormal_{window_length}.zarr'))
        return coh_zsep_xnormal
 
    # On each x-normal plane, loop through the separation list and the pairs of points, accumulating the results into a single dataset
    coh_zsep_xnormal = []
    for xplane in xplanes:
        coh_zsep = []
        for sep_z in sep_z_list:
            if xplane is None:
                dsx_currentxplane = dsx
            else:
                dsx_currentxplane = dsx.sel(x=xplane)
                
            print(f'Computing the vertical coherence on plane x={xplane} between (y={y_loc_list[0]}--{y_loc_list[-1]},z={zref})',
                  f'and (y={y_loc_list[0]}--{y_loc_list[-1]},z={zref+sep_z}), for a vertical separation of {sep_z} m')
            
            coh_y=[]
            for y in y_loc_list:
                # Get the two signals
                s1={'ds':dsx_currentxplane, 'z':zref, 'y':y}
                s2={**s1, 'z':zref+sep_z}

                mycoh, mypsd = calc_coherence_2signals(s1,s2, strname='vert', interval=interval, window_length=window_length, window=window, normal='xNormal')
                mycoh = mycoh.expand_dims('y').assign_coords({'y':[y]})
                coh_y.append(mycoh)

            coh_y = xr.concat(coh_y, dim='y')
            coh_zsep.append(coh_y.expand_dims('sep_z').assign_coords({'sep_z':[sep_z]}))
        coh_zsep = xr.concat(coh_zsep, dim='sep_z')
        coh_zsep_xnormal.append(coh_zsep.expand_dims('xplane').assign_coords({'xplane':[xplane]}))
    coh_zsep_xnormal = xr.concat(coh_zsep_xnormal, dim='xplane')

    # Get rid of the single xplane level for single xplanes (TurbSim data, for instance)
    coh_zsep_xnormal = coh_zsep_xnormal.squeeze(drop=True)
    
    if outputPath is not None:
        coh_zsep_xnormal.to_zarr(os.path.join(outputPath,f'coh_zsep_xnormal_{window_length}.zarr'))
    
    return coh_zsep_xnormal




def calcLatCoh(dsx, sep_y_list, yref, zref, interval, window_length, window, ydist, outputPath=None):
    '''
    Calculates lateral coherence
    
    Inputs
    ======
    dsx: xarray dataset
        Array containing perturbation valocity (up,vp,wp) at coordinates ([x,]y,z,datetime)
    sep_y_list: list of scalars
        Lateral separation values to compute coherence
    yref: scalar
        Reference y location (typically 0, even ymin=-100)
    zref: scalar
        Reference z location (typically 150, 151.5, etc)
    interval: string
        Time interval of the whole dataset (typically '180min')
    window_length: string
        Window length for windowing approach (typically '5min' or '2min')
    window: string
        Windowing algorithm to use (typically 'hanning')
    ydist: scalar
        Lateral distance at which another pair of points will be collected (typically grid resolution)
    outputPath: string
        Path where a zarr file will be saved to
        
    Example call
    ============
    # Location of reference point
    yref = 0
    zref = 151
    # Total length of the time series
    interval='180min'
    # Window parameters
    window_length = '2min'
    window = 'hanning'
    # At every ydist laterally (in y), get another pair of points
    ydist = 2.5
    # Lateral separation values of interest
    sep_y_list = [5, 10, 15, 25, 40, 80, 120]
    # At every ydist laterally (in y), get another pair of points
    ydist = 2.5
    
    coh_ysep = calcLatCoh(ds_cohu, sep_y_list, yref, zref, interval, window_length, window, ydist, outpath)
    
    '''
    
    # ------------------------------- SETTINGS FOR LATERAL SEPARATION COHERENCE
    # --------------------------------------------------- USING X-NORMAL PLANES
    # At every ydist laterally (in y), get another pair of points
    y_loc_list = np.arange(dsx.y.min(), dsx.y.max()-max(sep_y_list), ydist)

    # Planes to loop on (if exists; will not exist in turbsim data)
    if 'x' in list(dsx.coords.keys()):
        xplanes = dsx.x.values
    else:
        xplanes = [None]
    # -------------------------------------------------------------------------


    if os.path.isdir(os.path.join(str(outputPath),f'coh_ysep_xnormal_{window_length}_ydist{ydist}.zarr')):
        coh_ysep_xnormal = xr.open_zarr(os.path.join(outputPath,f'coh_ysep_xnormal_{window_length}_ydist{ydist}.zarr'))
        return coh_ysep_xnormal
    
    # On each x-normal plane, loop through the separation list and the pairs of points, accumulating the results into a single dataset
    coh_ysep_xnormal = []
    for xplane in xplanes:
        coh_ysep = []
        for sep_y in sep_y_list:
            if xplane is None:
                dsx_currentxplane = dsx
            else:
                dsx_currentxplane = dsx.sel(x=xplane)
                

            info  = f'Computing the lateral coherence'
            if xplane is not None:
                info += f' on plane x={xplane}'
            info += f'between (y={yref+y_loc_list[0]}--{yref+y_loc_list[-1]},z={zref}),'
            info += f' and (y={yref+y_loc_list[0]+sep_y}--{yref+y_loc_list[-1]+sep_y},z={zref}),'
            info += f'for a lateral separation of {sep_y} m'
            print(info)
            
            coh_y=[]
            for y in y_loc_list:
                # Get the two signals
                s1={'ds':dsx_currentxplane, 'z':zref, 'y':yref+y}
                s2={**s1, 'y':yref+y+sep_y}

                mycoh, mypsd = calc_coherence_2signals(s1,s2, strname='lat', interval=interval, window_length=window_length, window=window, normal='xNormal')
                mycoh = mycoh.expand_dims('y').assign_coords({'y':[y]})
                coh_y.append(mycoh)

            coh_y = xr.concat(coh_y, dim='y')
            coh_ysep.append(coh_y.expand_dims('sep_y').assign_coords({'sep_y':[sep_y]}))
        coh_ysep = xr.concat(coh_ysep, dim='sep_y')
        coh_ysep_xnormal.append(coh_ysep.expand_dims('xplane').assign_coords({'xplane':[xplane]}))
    coh_ysep_xnormal = xr.concat(coh_ysep_xnormal, dim='xplane')

    # Get rid of the single xplane level for single xplanes (TurbSim data, for instance)
    coh_ysep_xnormal = coh_ysep_xnormal.squeeze(drop=True)
    
    if outputPath is not None:
        coh_ysep_xnormal.to_zarr(os.path.join(outputPath,f'coh_ysep_xnormal_{window_length}_ydist{ydist}.zarr'))

    return coh_ysep_xnormal





def calcLongCoh_from_ynormal_par(dsy, sep_x_list, zref, interval, window_length, window, xdist, outputPath=None):
    '''
    Calculates longitudinal coherence from y-normal planes. Y-normal planes are assumed to be in the along-wind
    direction and assumed to vary in dimension x.
    
    The dataset dsy should have coordinated x, z, datetime. Optionally, it can contain the coordinate y.
    If the coordinate y is present, it is assumed that these are different along-wind planes and another
    loop on _all_ these planes is performed. It no coordinate y is present, it is assumed that data on a 
    single y-normal plane is given.
    
    '''
    
    raise NotImplementedError (f'This function has not been implemented yet.')
    
    
    
def calcLongCoh_from_znormal_par(dsz, sep_x_list, zref, interval, window_length, window, xdist, ydist, outputPath=None):
    ''' 
    Calculates longitudinal coherence from z-normal planes. Assumes grid-aligned wind.
    
    The dataset dsz should have coordinated x, y, z, datetime. A plane at z=zref should exist and 
    other planes should exist at the requested separation list
    
    '''
    
    raise NotImplementedError (f'This function has not been implemented yet.')





def calcLatCoh_from_xnormal_par(dsx, sep_y_list, zref, interval, window_length, window, ydist, outputPath=None):
    '''
    Calculates lateral coherence from x-normal planes. X-normal planes are assumed to be in the cross-wind
    direction and assumed to vary in dimension y.
    
    The dataset dsx should have coordinated y, z, datetime. Optionally, it can contain the coordinate x.
    If the coordinate x is present, it is assumed that these are different cross-wind planes and another
    loop on _all_ these planes is performed. It no coordinate x is present, it is assumed that data on a 
    single x-normal plane is given.
    
    '''
    
    # Output filename (even if unused)
    zarrfilename = f'coh_ysep_xnormal_zref{zref}_{window_length}_ydist{ydist}.zarr'
    
    if os.path.isdir(os.path.join(str(outputPath),zarrfilename)):
        print(f'File {zarrfilename} found. Loading it.')
        coh_ysep = xr.open_zarr(os.path.join(outputPath,zarrfilename))
        return coh_ysep
    

    
    #if __name__ ==  '__main__': 
    if True:
        # ------------------------------- SETTINGS FOR LATERAL SEPARATION COHERENCE
        # --------------------------------------------------- USING X-NORMAL PLANES
        # At every ydist laterally (in y), get another pair of points
        y_loc_list = np.arange(dsx.y.min(), dsx.y.max()-max(sep_y_list), ydist)

        # Planes to loop on (if exists; will not exist in turbsim data, or single-plane data)
        if 'x' in list(dsx.coords.keys()):
            xplanes = dsx.x.values
            x = xplanes
        else:
            xplanes = [None]
            x = 'x'
        # -------------------------------------------------------------------------
        

        # On each x-normal plane, loop through the separation list and the pairs of points, accumulating the results into a single dataset
        coh_ysep = []
        for sep_y in sep_y_list:
            info = f'Computing lateral coherence with a lateral separation of {sep_y} m '
            if x == 'x':
                info += 'for the single x-normal plane available.     '
                print(info)

            coh_xy=[]
            for xplane in xplanes:
                if xplane is not None:
                    info += 'for x-normal plane at x = {xplane} m.             '
                    print(info)

                sweep = y_loc_list
                print(f'   Between ({x},y,{zref}) and ({x},y,{zref}) m,   y = {sweep[0]}, {sweep[1]}, ..., {sweep[-2]}, {sweep[-1]} m.', end='\r')
                coh_y = calc_coherece_2signals_par(dsx, varnameSweep='y', sweep=sweep,
                                                   othervar=xplane, zref=zref, sep=sep_y,
                                                   strname='lat',
                                                   interval=interval, window_length=window_length, window=window,
                                                   nCores=36)

                coh_xy.append(coh_y.expand_dims('x').assign_coords({'x':[xplane]}))
            coh_xy = xr.concat(coh_xy, dim='x')

            coh_ysep.append(coh_xy.expand_dims('sep_y').assign_coords({'sep_y':[sep_y]}))
        coh_ysep = xr.concat(coh_ysep, dim='sep_y')

        # Get rid of the single xplane level for single xplanes (TurbSim data, for instance)
        coh_ysep = coh_ysep.squeeze(drop=True)

        # Set attributes
        _set_attrs(coh_ysep, zref, interval, window_length, window, ydist)

        if outputPath is not None:
            print(f'Saving {os.path.join(outputPath,zarrfilename)}...       ')
            coh_ysep.to_zarr(os.path.join(outputPath,zarrfilename))
            
        print(f'Done                                                                               ')
        return coh_ysep
    
    
    
    
    
def calcLatCoh_from_znormal_par(dsz, sep_y_list, zref, interval, window_length, window, xdist, ydist, outputPath=None):
    ''' 
    Calculates lateral coherence from z-normal planes. Assumes grid-aligned wind.
    
    The dataset dsz should have coordinated x, y, z, datetime. A plane at z=zref should exist and 
    other planes should exist at the requested separation list
    
    '''
    
    raise NotImplementedError (f'This function has not been implemented yet.')




def calcVertCoh_from_xnormal_par(dsx, sep_z_list, zref, interval, window_length, window, ydist, outputPath=None, normalstr='xnormal'):
    '''
    
    
    Example input:
    
    # ------------------------------ SETTINGS FOR VERTICAL SEPARATION COHERENCE
    # --------------------------------------------------- USING X-NORMAL PLANES
    # Vertical separation values of interest
    sep_z_list = [-80, -40, 40, 80]
    # Height of reference point
    zref = 150
    # Total length of the time series
    interval='120min'
    # Window parameters
    window_length = '3min'
    window = 'hanning'

    # At every ydist laterally (in y), let's get another pair of points
    ydist = 300
    # -------------------------------------------------------------------------

    coh_zsep = calcVertCoh_from_xnormal_par(dsx, sep_z_list, zref, interval, window_length, window, ydist, outputPath=None)
    
    '''

    # Output filename (even if unused)
    zarrfilename = f'coh_zsep_{normalstr}_zref{zref}_{window_length}_ydist{ydist}.zarr'
    
    if os.path.isdir(os.path.join(str(outputPath),zarrfilename)):
        print(f'File {zarrfilename} found. Loading it.')
        coh_zsep = xr.open_zarr(os.path.join(outputPath,zarrfilename))
        return coh_zsep
    

    #if __name__ ==  '__main__': 
    if True:
        
        # ------------------------------ SETTINGS FOR VERTICAL SEPARATION COHERENCE
        # --------------------------------------------------- USING X-NORMAL PLANES
        # At every ydist laterally (in y), get another pair of vertical points
        y_loc_list = np.arange(dsx.y.min(), dsx.y.max()-ydist, ydist)
        
        # Planes to loop on (if exists; will not exist in turbsim data, or single-plane data)
        if 'x' in list(dsx.coords.keys()):
            xplanes = dsx.x.values
            x = xplanes
        else:
            xplanes = [None]
            x = 'x'
        # -------------------------------------------------------------------------
        
        coh_zsep = []
        for sep_z in sep_z_list:
            info = f'Computing vertical coherence with a vertical separation of {sep_z} m '
            if x == 'x':
                info += 'for the single x-normal plane available.     '
                print(info)
                
            coh_xy=[]
            for xplane in xplanes:
                if xplane is not None:
                    info += 'for x-normal plane at x = {xplane} m.          '
                    print(info)


                sweep = y_loc_list
                print(f'   Between ({x},y,{zref}) and ({x},y,{zref+sep_z}) m,   y = {sweep[0]}, {sweep[1]}, ..., {sweep[-2]}, {sweep[-1]} m.', end='\r')

                coh_y = calc_coherece_2signals_par(dsx, varnameSweep='y', sweep=sweep,
                                                   othervar=xplane, zref=zref, sep=sep_z,
                                                   strname='vert',
                                                   interval=interval, window_length=window_length, window=window,
                                                   nCores=36)

                coh_xy.append(coh_y.expand_dims('x').assign_coords({'x':[xplane]}))

            coh_xy = xr.concat(coh_xy, dim='x')
            coh_zsep.append(coh_xy.expand_dims('sep_z').assign_coords({'sep_z':[sep_z]}))
        coh_zsep = xr.concat(coh_zsep, dim='sep_z')
        
        
        # Set attributes
        _set_attrs(coh_zsep, zref, interval, window_length, window, ydist)

        if outputPath is not None:
            print(f'Saving {os.path.join(outputPath,zarrfilename)}...       ')
            coh_zsep.to_zarr(os.path.join(outputPath,zarrfilename))
            
        print(f'Done                                                                               ')
        return coh_zsep
    

    
    
    
    
    
def calcVertCoh_from_ynormal_par(dsy, sep_z_list, zref, interval, window_length, window, xdist, outputPath=None):
    '''
    For generality.
    
    Example input:
    
    # ------------------------------ SETTINGS FOR VERTICAL SEPARATION COHERENCE
    # --------------------------------------------------- USING Y-NORMAL PLANES
    # Vertical separation values of interest
    sep_z_list = [-80, -40, 40, 80]
    # Height of reference point
    zref = 150
    # Total length of the time series
    interval='120min'
    # Window parameters
    window_length = '3min'
    window = 'hanning'

    # At every xdist longitudinally (in x), let's get another pair of points
    xdist = 800
    # -------------------------------------------------------------------------

    coh_zsep = calcVertCoh_from_ynormal_par(dsy, sep_z_list, zref, interval, window_length, window, xdist, outputPath=None)

    '''
    dsx = dsy.rename({'x':'y'})
    ydist = xdist
    
    print("WARNING: This was weirdly slow/hanging with the hurricane case. I suspect it was because of the zarr chunks.")
    print("         If this works just fine on a different dataset, that is likely the reason. If not, I'm not sure what the reason is")
    
    return calcVertCoh_from_xnormal_par(dsx, sep_z_list, zref, interval, window_length, window, ydist, outputPath=outputPath, normalstr='ynormal')







def calcVertCoh_from_znormal_par(dsz, sep_z_list, zref, interval, window_length, window, xdist, ydist, outputPath=None):
    '''
    Calculates vertical coherence from pairs of z-normal planes. No assumption of the wind direction is
    made.
    
    The dataset dsx should have coordinated x, y, z, datetime. A plane at z=zref should exist and 
    other planes should exist at the requested separation list
    
    Example input:
    
    # ------------------------------ SETTINGS FOR VERTICAL SEPARATION COHERENCE
    # --------------------------------------------------- USING Z-NORMAL PLANES
    # Vertical separation values of interest
    sep_z_list = [-80, -40, 40, 80]
    # Height of reference point
    zref = 145
    # Total length of the time series
    interval='120min'
    # Window parameters
    window_length = '3min'
    window = 'hanning'

    # At every xdist laterally (in x), let's get another pair of points
    xdist = 40

    # At every ydist laterally (in y), let's get another pair of points
    ydist = 100
    # -------------------------------------------------------------------------

    coh_zsep = calcVertCoh_from_znormal_par(dsz, sep_z_list, zref, interval, window_length, window, xdist, ydist, outputPath=None)

    '''
    
    # Output filename (even if unused)
    zarrfilename = f'coh_zsep_znormal_zref{zref}_{window_length}_ydist{ydist}_xdist{xdist}.zarr'
    
    if os.path.isdir(os.path.join(str(outputPath),zarrfilename)):
        print(f'File {zarrfilename} found. Loading it.')
        coh_zsep = xr.open_zarr(os.path.join(outputPath,zarrfilename))
        return coh_zsep
    

    #if __name__ ==  '__main__': 
    if True:
        
        # ------------------------------ SETTINGS FOR VERTICAL SEPARATION COHERENCE
        # --------------------------------------------------- USING Z-NORMAL PLANES
        # At every xdist laterally (in x), let's get another pair of points
        x_loc_list = np.arange(dsz.x.min(), dsz.x.max()-xdist, xdist)
        # At every ydist laterally (in y), get another pair of points
        y_loc_list = np.arange(dsz.y.min(), dsz.y.max()-ydist, ydist)
        # -------------------------------------------------------------------------
        
        coh_zsep = []
        for sep_z in sep_z_list:
            print(f'Computing vertical coherence with a vertical separation of {sep_z} m        ')
            coh_xy=[]
            for y in y_loc_list:

                sweep = x_loc_list
                print(f'   Between (x,{y},{zref}) and (x,{y},{zref+sep_z}) m,   x = {sweep[0]}, {sweep[1]}, ..., {sweep[-2]}, {sweep[-1]} m.', end='\r')

                coh_x = calc_coherece_2signals_par(dsz, varnameSweep='x', sweep=sweep,
                                                   othervar=y, zref=zref, sep=sep_z,
                                                   strname='vert',
                                                   interval=interval, window_length=window_length, window=window,
                                                   nCores=36)

                coh_xy.append(coh_x.expand_dims('y').assign_coords({'y':[y]}))

            coh_xy = xr.concat(coh_xy, dim='y')
            coh_zsep.append(coh_xy.expand_dims('sep_z').assign_coords({'sep_z':[sep_z]}))
        coh_zsep = xr.concat(coh_zsep, dim='sep_z')
        
        # Set attributes
        _set_attrs(coh_zsep, zref, interval, window_length, window, ydist, xdist)
        
        if outputPath is not None:
            print(f'Saving {os.path.join(outputPath,zarrfilename)}...')
            coh_zsep.to_zarr(os.path.join(outputPath,zarrfilename))
            
        print(f'Done')
        return coh_zsep




def _set_attrs(coh, zref, interval, window_length, window, ydist, xdist=None):
    coh.attrs['zref'] = zref
    coh.attrs['interval'] = interval
    coh.attrs['window_length'] = window_length
    coh.attrs['window'] = window
    coh.attrs['ydist'] = ydist
    if xdist is not None:
        coh.attrs['xdist'] = xdist

