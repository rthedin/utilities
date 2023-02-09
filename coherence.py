import numpy as np
import pandas as pd
import xarray as xr
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
    

    return xr.merge([mscoh,cocoh,qucoh]), psd




def plotCoherence(coh_sep, sep_list, meandim, umean=None, **kwargs):
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
    a, b: float
        Values for a and B for Davenport's model
    
    '''

    import matplotlib.colors as mcolors
    
    # Get auxiliary arrays
    colors = list(mcolors.TABLEAU_COLORS)
    f = coh_sep.frequency
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

        
    fig, axs = plt.subplots(3,3,figsize=(16,9), sharey=True, sharex=True)

    for c, sep in enumerate(sep_list):
        # mscoh uu
        axs[0,0].plot(f, coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'mscoh_{cohsepstr}sep_u1u2'], c=colors[c%len(colors)], label=f'{sepcoord} = {sep}')
        if umean is not None:
            #axs[0,0].plot(f, IECCoherence(f,umean,delta=sep,component='u',mode='sameValues'), c=colors[c%len(colors)], ls='--', alpha=0.7)
            axs[0,0].plot(f, davenportExpCoh(f,u=umean,delta=sep,Lc='defaultu',**kwargs), c=colors[c%len(colors)], ls='--', alpha=0.7)
        # co-coh uu
        axs[0,1].plot(f, coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'cocoh_{cohsepstr}sep_u1u2'], c=colors[c%len(colors)], label=f'{sepcoord} = {sep}')
        if umean is not None:
            #axs[0,1].plot(f, IECCoherence(f,umean,delta=sep,component='u',mode='sameValues'), c=colors[c%len(colors)], ls='--', alpha=0.7)
            axs[0,1].plot(f, davenportExpCoh(f,u=umean,delta=sep,Lc='defaultu',**kwargs), c=colors[c%len(colors)], ls='--', alpha=0.7)
        # quad-coh uu
        axs[0,2].plot(f, coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'qucoh_{cohsepstr}sep_u1u2'], c=colors[c%len(colors)], label=f'{sepcoord} = {sep}')

        # ---------------------
        # mscoh vv
        axs[1,0].plot(f, coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'mscoh_{cohsepstr}sep_v1v2'], c=colors[c%len(colors)], label=f'{sepcoord} = {sep}')
        # co-coh vv
        axs[1,1].plot(f, coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'cocoh_{cohsepstr}sep_v1v2'], c=colors[c%len(colors)], label=f'{sepcoord} = {sep}')
        # quad-coh vv
        axs[1,2].plot(f, coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'qucoh_{cohsepstr}sep_v1v2'], c=colors[c%len(colors)], label=f'{sepcoord} = {sep}')

        # ---------------------
        # mscoh vv
        axs[2,0].plot(f, coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'mscoh_{cohsepstr}sep_w1w2'], c=colors[c%len(colors)], label=f'{sepcoord} = {sep}')
        # co-coh vv
        axs[2,1].plot(f, coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'cocoh_{cohsepstr}sep_w1w2'], c=colors[c%len(colors)], label=f'{sepcoord} = {sep}')
        # quad-coh vv
        axs[2,2].plot(f, coh_sep.mean(dim=meandim).sel({sepcoord:sep})[f'qucoh_{cohsepstr}sep_w1w2'], c=colors[c%len(colors)], label=f'{sepcoord} = {sep}')


    axs[0,0].set_title(f'mscoh $\gamma^2_{{uu, {cohsepstr}}}$', fontsize=13)
    axs[0,1].set_title(f'co-coh $\gamma_{{uu, {cohsepstr}}}$',  fontsize=13)
    axs[0,2].set_title(f'quad-coh $\\rho_{{uu, {cohsepstr}}}$', fontsize=13)

    axs[1,0].set_title(f'mscoh $\gamma^2_{{vv, {cohsepstr}}}$', fontsize=13)
    axs[1,1].set_title(f'co-coh $\gamma_{{vv, {cohsepstr}}}$',  fontsize=13)
    axs[1,2].set_title(f'quad-coh $\\rho_{{vv, {cohsepstr}}}$', fontsize=13)

    axs[2,0].set_title(f'mscoh $\gamma^2_{{ww, {cohsepstr}}}$', fontsize=13)
    axs[2,1].set_title(f'co-coh $\gamma_{{ww, {cohsepstr}}}$',  fontsize=13)
    axs[2,2].set_title(f'quad-coh $\\rho_{{ww, {cohsepstr}}}$', fontsize=13)


    for ax in axs.flatten():
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid()
    for ax in axs[:,-1]:  ax.legend(fontsize=13)
    for ax in axs[-1,:]:  ax.set_xlabel('freq [Hz]', fontsize=14)
    for ax in axs[:,0]:   ax.set_ylabel('coherence', fontsize=14)
    fig.tight_layout()


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

    


