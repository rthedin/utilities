import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt

from mmctools.helper_functions import calc_spectra

# Collection of general functions related to coherence


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
            if normal=='zNormal':
                sig1 = s1['ds'].sel(x=s1['x'], y=s1['y'], drop=True, method='nearest', tolerance=1e-3)
                sig2 = s2['ds'].sel(x=s2['x'], y=s2['y'], drop=True, method='nearest', tolerance=1e-3)
            elif normal=='xNormal': 
                sig1 = s1['ds'].sel(z=s1['z'], y=s1['y'], drop=True, method='nearest', tolerance=1e-3)
                sig2 = s2['ds'].sel(z=s2['z'], y=s2['y'], drop=True, method='nearest', tolerance=1e-3)
            elif normal=='yNormal': 
                sig1 = s1['ds'].sel(z=s1['z'], x=s1['x'], drop=True, method='nearest', tolerance=1e-3)
                sig2 = s2['ds'].sel(z=s2['z'], x=s2['x'], drop=True, method='nearest', tolerance=1e-3)
            else:
                print('Assuming probe data with all `height`, `x`, and `y` passed.')
                # probe data
                sig1 = s1['ds'].sel(height=s1['height'], x=s1['x'], y=s1['y'], drop=True, method='nearest', tolerance=1e-3)
                sig2 = s2['ds'].sel(height=s2['height'], x=s2['x'], y=s2['y'], drop=True, method='nearest', tolerance=1e-3)
        except KeyError:
            print(f"ERROR: A value asked for does not exist.")
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
        if s1['y'] != s2['y']:
            raise ValueError(f"Requested vertical separation but points have different y value: {s1['y']} and {s2['y']}")
    elif strname == 'lat':
        if s1['z'] != s2['z']:
            raise ValueError(f"Requested lateral separation but points have different z value: {s1['z']} and {s2['z']}")
    elif strname == 'lon':
        if s1['y'] != s2['y']:
            raise ValueError(f"Requested longitudinal separation but points have different y value: {s1['y']} and {s2['y']}")
    else:
        raise ValueError(f'`strname` needs to be given either as "vert", "lat", or "lon".')
        
            
    spectraTimes = pd.date_range(start=sig1.datetime[0].values, end=sig1.datetime[-1].values, freq='10min')
    
    signals = xr.merge([ sig1['up'].to_dataset(name='u1'), sig1['vp'].to_dataset(name='v1'), sig1['wp'].to_dataset(name='w1'),
                         sig2['up'].to_dataset(name='u2'), sig2['vp'].to_dataset(name='v2'), sig2['wp'].to_dataset(name='w2') ])

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




def plotCoherence(coh_sep, sep_list, meandim, umean=None, xaxis='freq', qoi=['ms','co','qu'], **kwargs):
    '''
    Plot mscoh, co-coherence, and quad-coherence from coh_sep dataset
    
    coh_sep: xr.Dataset
        Dataset containing all the coherence values
        One of the coordinates should be sep_x, sep_y, or sep_z
    sep_list: array of floats
        List of separation distances to plot
    meandir: str
        Direction to compute the mean. Should be 'x', 'y', or 'z'
    umean: float
        Mean velocity used to compute IEC coherence. IEC curves are skipped is umean is not specified
    xaxis: str ('freq', or 'redfreq'; default 'freq')
        Quantity to use on the xaxis
    qoi: str, array of string
        What quantities to plot. Options: 'ms','co','qu','r' (or 'all' instead).
    a, b, B: float
        Values for a and b (or B) for Davenport's model
    
    '''

    import matplotlib.colors as mcolors
    
    # Check what xaxis was requested
    if xaxis not in {'freq','redfreq'}:
        raise ValueError (f'The argument xaxis can only take `freq` or `redfreq`.')

    # Check what plots were requested (mscoh, cocoh, quad, radius)
    if isinstance(qoi,str):
        # Single quantity requested (or all)
        if qoi == 'all':
            qoi = ['ms','co','qu','r']
        else:
            qoi = [qoi]
    nqoi = len(qoi)



    # Get auxiliary arrays
    colors = list(mcolors.TABLEAU_COLORS)
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
        raise ValueError (f'Not all separation distances requested to plot are available. Stopping')


    if 'ylim' in kwargs:
        ylim = kwargs['ylim']
    else:
        ylim = None
    if 'xlim' in kwargs:
        xlim = kwargs['xlim']
    else:
        xlim = None

        
    fig, axs = plt.subplots(nqoi,3,figsize=(17,nqoi*3), sharey=True, sharex=True, gridspec_kw = {'wspace':0.1, 'hspace':0.1})

    for c, sep in enumerate(sep_list):
        row=0

        # get the frequency axis
        if xaxis == 'freq': f = coh_sep.frequency
        else:               f = coh_sep.frequency * sep / umean

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

        # ---------------------
        if 'ms' in qoi or 'mscoh' in qoi:
            # mscoh uu
            axs[row,0].plot(f, ms_u, c=colors[c%len(colors)], label=f'{sep}')
            if umean is not None:
                axs[row,0].plot(f, davenportExpCoh(coh_sep.frequency,u=umean,delta=sep,Lc='defaultu',**kwargs), c=colors[c%len(colors)], ls='--', alpha=0.7)
            # mscoh vv
            axs[row,1].plot(f, ms_v, c=colors[c%len(colors)], label=f'{sep}')
            # mscoh vv
            axs[row,2].plot(f, ms_w, c=colors[c%len(colors)], label=f'{sep}')
            # Set titles
            if c==0:
                axs[row,0].text(0.98, 0.97, f'mscoh $\gamma^2_{{uu, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,0].transAxes, fontsize=14)
                axs[row,1].text(0.98, 0.97, f'mscoh $\gamma^2_{{vv, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,1].transAxes, fontsize=14)
                axs[row,2].text(0.98, 0.97, f'mscoh $\gamma^2_{{ww, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,2].transAxes, fontsize=14)
            row = row+1
        # ---------------------

        # ---------------------
        if 'co' in qoi or 'cocoh' in qoi:
            # co-coh uu
            axs[row,0].plot(f, co_u, c=colors[c%len(colors)], label=f'{sep}')
            if umean is not None:
                axs[row,0].plot(f, davenportExpCoh(coh_sep.frequency,u=umean,delta=sep,Lc='defaultu',**kwargs), c=colors[c%len(colors)], ls='--', alpha=0.7)
            # co-coh vv
            axs[row,1].plot(f, co_v, c=colors[c%len(colors)], label=f'{sep}')
            # co-coh ww
            axs[row,2].plot(f, co_w, c=colors[c%len(colors)], label=f'{sep}')
            # Set titles
            if c==0:
                axs[row,0].text(0.98, 0.97, f'co-coh $\gamma^2_{{uu, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,0].transAxes, fontsize=14)
                axs[row,1].text(0.98, 0.97, f'co-coh $\gamma^2_{{vv, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,1].transAxes, fontsize=14)
                axs[row,2].text(0.98, 0.97, f'co-coh $\gamma^2_{{ww, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,2].transAxes, fontsize=14)
            row = row+1
        # ---------------------

        # ---------------------
        if 'qu' in qoi or 'quadcoh' in qoi:
            # quad-coh uu
            axs[row,0].plot(f, qu_u, c=colors[c%len(colors)], label=f'{sep}')
            # quad-coh vv
            axs[row,1].plot(f, qu_v, c=colors[c%len(colors)], label=f'{sep}')
            # quad-coh ww
            axs[row,2].plot(f, qu_w, c=colors[c%len(colors)], label=f'{sep}')
            # Set titles
            if c==0:
                axs[row,0].text(0.98, 0.97, f'quad-coh $\\rho^2_{{uu, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,0].transAxes, fontsize=14)
                axs[row,1].text(0.98, 0.97, f'quad-coh $\\rho^2_{{vv, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,1].transAxes, fontsize=14)
                axs[row,2].text(0.98, 0.97, f'quad-coh $\\rho^2_{{ww, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,2].transAxes, fontsize=14)
            row = row+1
        # ---------------------

        # ---------------------
        if 'r' in qoi or 'radiuscoh' in qoi or 'rcoh' in qoi:
            # radius coh uu
            axs[row,0].plot(f, (co_u**2+qu_u**2)**0.5, c=colors[c%len(colors)], label=f'{sep}')
            # radius coh vv
            axs[row,1].plot(f, (co_v**2+qu_v**2)**0.5, c=colors[c%len(colors)], label=f'{sep}')
            # radius coh ww
            axs[row,2].plot(f, (co_w**2+qu_w**2)**0.5, c=colors[c%len(colors)], label=f'{sep}')
            # Set titles
            if c==0:
                axs[row,0].text(0.98, 0.95, f'radius coh $R_{{uu, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,0].transAxes, fontsize=14)
                axs[row,1].text(0.98, 0.95, f'radius coh $R_{{vv, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,1].transAxes, fontsize=14)
                axs[row,2].text(0.98, 0.95, f'radius coh $R_{{ww, {cohsepstr}}}$', va='top', ha='right', transform=axs[row,2].transAxes, fontsize=14)
            row = row+1
        # ---------------------


    if xaxis == 'freq': xlabel = 'freq [Hz]'
    else:               xlabel = 'reduced freq (f*d/u) [-]'
    for ax in axs.flatten():
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid()
    axs[0,-1].legend(title=f'{sepcoord} (m)', fontsize=12, title_fontsize=13, loc='upper left', bbox_to_anchor=(1,1))
    for ax in axs[-1,:]:  ax.set_xlabel(xlabel, fontsize=14)
    for ax in axs[:,0]:   ax.set_ylabel('coherence', fontsize=14)
    #fig.tight_layout()


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

def davenportExpCoh(f,u,delta,Lc='defaultu',a=None,b=None,B=None, **kwargs):
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


    if os.path.isdir(os.path.join(str(outputPath),f'coh_ysep_xnormal_{window_length}.zarr')):
        coh_ysep_xnormal = xr.open_zarr(os.path.join(outputPath,f'coh_ysep_xnormal_{window_length}.zarr'))
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
                
            print(f'Computing the lateral coherence on plane x={xplane} between (y={yref+y_loc_list[0]}--{yref+y_loc_list[-1]},z={zref})',
                  f'and (y={yref+y_loc_list[0]+sep_y}--{yref+y_loc_list[-1]+sep_y},z={zref}), for a lateral separation of {sep_y} m')
            
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
        coh_ysep_xnormal.to_zarr(os.path.join(outputPath,f'coh_ysep_xnormal_{window_length}.zarr'))

    return coh_ysep_xnormal




