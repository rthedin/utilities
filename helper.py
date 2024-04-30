# Collections of functions that help perform routine tasks on jupyter notebooks.
# Some of them have been copied from old notebooks, some have been adapted to my
# workflow. Putting them here for convenience-- no need to copy and paste them
# onto new notebooks
#
# Add to notebooks using
#   sys.path.append(os.path.abspath('/home/rthedin/utilities/'))
#   from helper import addScalebar, myupdraftscale

# Regis Thedin
# Oct 18, 2021
#

import numpy as np
import pandas as pd
import xarray as xr
import os, sys
import datetime
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#from mmctools.helper_functions import covariance, calc_wind, calc_spectra
str_var_common = {
    "time"  : "time",
    "height": "height",
    "wdir"  : "wdir",
    "TKE"   : "TKE",
    "TI"    : "TI",
    "TI_TKE":"TI_TKE",
    "u*"    : "calculated_u*",
    "sigma_u_over_u*":"sigma_u_over_calc_u*",       # uu/u*
    "sigma_v_over_u*":"sigma_v_over_calc_u*",       # vv/u*
    "sigma_w_over_u*":"sigma_w_over_calc_u*",       # ww/u*
    "sigma_v_over_sigma_u":"sigma_v_over_sigma_u",  # vv/uu
    "sigma_w_over_sigma_u":"sigma_w_over_sigma_u",  # ww/uu
    "Phi_m": "Phi_m",  # non-dim shear
}
str_var_sowfa = {
    "wspd" : "wspd",
    "uv"   : "uv",
    "vw"   : "vw",
    "uw"   : "uw",
    "uu"   : "uu",
    "vv"   : "vv",
    "ww"   : "ww",
    "u"    : "u",
    "v"    : "v",
    "w"    : "w",
    # it will crash if asking for ustar here. ustar is not an output from sowfa
    **str_var_common
}
str_var_amr = {
    "wspd" : "hvelmag",
    "uv"   : "u'v'_r",
    "vw"   : "v'w'_r",
    "uw"   : "u'w'_r",
    "uu"   : "u'u'_r",
    "vv"   : "v'v'_r",
    "ww"   : "w'w'_r",
    "u"    : "u",
    "v"    : "v",
    "w"    : "w",
    "ustar": "ustar",  # from LES
    **str_var_common
}

def cosd(a): return np.cos(np.deg2rad(a))
def sind(a): return np.sin(np.deg2rad(a))

def myupdraftscale(vmin=-1, vmax=1, thresh=0.85):
    '''
    Custom colormap for updraft with threshold. Threshold constant at 0.85 m/s

    Instructions:
        vmin=-1; vmax=1
        ax.pcolormesh(xx, yy, val, vmin=vmin, vmax=vmax, cmap=myupdraftscale(vmin=vmin,vmax=vmax))
    '''
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Discretize colormap
    nColorsTot = int( (vmax-vmin)*100 ) # 200
    nColors_075above  = max(0, int( nColorsTot*((vmax-thresh)/vmax) ))
    nColors_0_075     = max(0, int( nColorsTot*(thresh/vmax) ))
    try:
        nColors_m075_0    = int( nColorsTot*(thresh/(-vmin)) )
    except ZeroDivisionError:
        nColors_m075_0 = 0
    try:
        nColors_m075below = int( nColorsTot*((vmin+thresh)/vmin) )
    except ZeroDivisionError:
        nColors_m075below = 0
    
    # Yellow part, > 0.75
    myYelo = plt.cm.autumn_r(np.linspace(0, 0.6, nColors_075above))
    myGree = plt.cm.Greens(np.linspace(0.2, 1, nColors_075above))
    myPurp = plt.cm.Purples(np.linspace(0.2, 1, nColors_075above))
    # Mid positive part, between 0, +0.75
    myRd_pos = plt.cm.RdBu_r(np.linspace(0.5, 1, nColors_0_075))
    # Mid negative part, between -0.75, 0
    myBu_neg = plt.cm.RdBu_r(np.linspace(0, 0.5, nColors_m075_0))
    # Light blue part, <-0.75
    myBu = plt.cm.cool_r(np.linspace(0.4, 1, nColors_m075below))
    
    # Another totally different one, colorbind-safe. orange-blue
    myOra_pos = plt.cm.Oranges(np.linspace(0, 1, nColors_0_075))
    myBlue  = plt.cm.Blues(np.linspace(0.2, 1, nColors_075above))
    concatColors = np.vstack((myBu, myBu_neg, myOra_pos, myBlue))


    #concatColors = np.vstack((myBu, myBu_neg, myRd_pos, myGree)) # red-green for >0 values
    #cmap = LinearSegmentedColormap.from_list('updraft',np.vstack((myBu, myBu_neg, myRd_pos, myYelo)))
    cmap = LinearSegmentedColormap.from_list('updraft',concatColors)
    
    return cmap

def interpolate_to_heights(df,heights, timedim='datetime'):
    """
    Interpolate data in dataframe to specified heights
    and return a new dataframe
    """
    from scipy.interpolate import interp1d

    if isinstance(df, xr.Dataset):
        return df.interp(height=heights)

    # If single height is asked
    if isinstance(heights, (float,int)):
        heights=[heights]
    
    # Unstack to single height index (= most time-consuming operation)
    unstacked = df.unstack(level=timedim)
    # Interpolate to specified heights
    f = interp1d(unstacked.index,unstacked,axis=0,fill_value='extrapolate')
    for hgt in heights:
        unstacked.loc[hgt] = f(hgt)
    # Restack and set index
    df_out = unstacked.loc[heights].stack().reset_index().set_index([timedim,'height']).sort_index()
    return df_out



# from https://stackoverflow.com/questions/48068938/set-new-index-for-pandas-dataframe-interpolating
def interpDataframe(df, new_index):
    """
    Return a new DataFrame with all columns values interpolated
    to the new_index values.
    Example: interpDataframe(df, np.arange(0,10) )
    """
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out



# from https://github.com/a2e-mmc/assessment/blob/study/coupling_comparison/studies/coupling_comparison/helpers.py
def reindex_if_needed(df,dt=None):
    """
    Check whether timestamps are equidistant with step dt (in seconds). If dt is not
    specified,  dt is equal to the minimal timestep in the dataframe. If timestamps
    are not equidistant, interpolate to equidistant time grid with step dt.
    """
    dts = np.diff(df.index.get_level_values(0).unique())/pd.to_timedelta(1,'s')

    # If dt not specified, take dt as the minimal timestep
    if dt is None:
        dt = np.min(dts)

    if not np.allclose(dts,dt):
        # df is missing some timestamps, which will cause a problem when computing spectra.
        # therefore, we first reindex the dataframe
        start = df.index.levels[0][0]
        end   = df.index.levels[0][-1]
        new_index = pd.date_range(start,end,freq=pd.to_timedelta(dt,'s'),name='datetime')
        return df.unstack().reindex(new_index).interpolate(method='index').stack()
    else:
        return df



# from https://github.com/a2e-mmc/assessment/blob/study/coupling_comparison/studies/coupling_comparison/helpers.py
def calc_stats(df,offset='10min'):
    """
    Calculate statistics for a given data frame
    and return a new dataframe
    """
    from mmctools.helper_functions import covariance
    # calculate statistical quantities on unstacked 
    unstacked = df.unstack()
    stats = unstacked.resample(offset).mean().stack()
    # - calculate variances
    stats['uu'] = unstacked['u'].resample(offset).var().stack()
    stats['vv'] = unstacked['v'].resample(offset).var().stack()
    stats['ww'] = unstacked['w'].resample(offset).var().stack()
    # - calculate covariances
    stats['uv'] = covariance(unstacked['u'], unstacked['v'], offset, resample=True).stack()
    stats['vw'] = covariance(unstacked['v'], unstacked['w'], offset, resample=True).stack()
    stats['uw'] = covariance(unstacked['u'], unstacked['w'], offset, resample=True).stack()
    if 'theta' in unstacked.keys():
        stats['thetaw'] = covariance(unstacked['theta'], unstacked['w'], offset, resample=True).stack()
    return stats



# based on https://github.com/a2e-mmc/assessment/blob/study/coupling_comparison/studies/coupling_comparison/helpers.py
def calc_QOIs(df, code='sowfa'):
    """
    Calculate derived quantities (IN PLACE)
    """
    if code == 'sowfa':
        str_var = str_var_sowfa
    elif code == 'amr' or code == 'amrwind':
        str_var = str_var_amr
    else:
        raise ValueError("Code options: 'sowfa', 'amr'")

    from mmctools.helper_functions import calc_wind 

    if str_var['wspd'] not in df.keys() and str_var['wdir'] not in df.keys():
        df[str_var['wspd']],df[str_var['wdir']] = calc_wind(df,u=str_var['u'],v=str_var['v'])
    elif str_var['wdir'] not in df.keys():
        _, df[str_var['wdir']] = calc_wind(df, u=str_var['u'], v=str_var['v'])

    df[str_var['u*']] = (df[str_var['uw']]**2 + df[str_var['vw']]**2)**0.25
    df[str_var['TKE']] = 0.5*(df[str_var['uu']] + df[str_var['vv']] + df[str_var['ww']])
    ang = np.arctan2(df[str_var['v']],df[str_var['u']])
    df[str_var['TI']] = df[str_var['uu']]*np.cos(ang)**2 + 2*df[str_var['uv']]*np.sin(ang)*np.cos(ang) + df[str_var['vv']]*np.sin(ang)**2
    df[str_var['TI']] = np.sqrt(df[str_var['TI']]) / df[str_var['wspd']]

    # TI as typical equations, based on TKE (same as AMR-Wind's TI TKE)
    df[str_var['TI_TKE']] = np.sqrt((df[str_var['uu']]+df[str_var['vv']]+df[str_var['ww']])/3.0)/np.sqrt(df[str_var['u']]**2 + df[str_var['v']]**2)


    # Let's also compute the non-dimension variances
    df[str_var['sigma_u_over_u*']] = (df[str_var['uu']])**0.5/df[str_var['u*']]
    df[str_var['sigma_v_over_u*']] = (df[str_var['vv']])**0.5/df[str_var['u*']]
    df[str_var['sigma_w_over_u*']] = (df[str_var['ww']])**0.5/df[str_var['u*']]
    df[str_var['sigma_v_over_sigma_u']] = (df[str_var['vv']])**0.5 / (df[str_var['uu']])**0.5
    df[str_var['sigma_w_over_sigma_u']] = (df[str_var['ww']])**0.5 / (df[str_var['uu']])**0.5


def calc_nondimshear(ds, code='amr'):
    '''
    Caculates Phi, the non-dimensional shear parameter (in place)
    '''
    if code == 'sowfa':
        str_var = str_var_sowfa
    elif code == 'amr' or code == 'amrwind':
        str_var = str_var_amr
    else:
        raise ValueError("Code options: 'sowfa', 'amr'")

    # Doing finite diff on the log space for accuracy
    dUdz = np.empty((len(ds[str_var['time']]),len(ds[str_var['height']]),));  dUdz[:,:] = np.nan
    dU = ds[str_var['wspd']  ].isel({str_var['height']:slice(1,None)}).values - ds[str_var['wspd']  ].isel({str_var['height']:slice(None,-1)}).values
    dZ = ds[str_var['height']].isel({str_var['height']:slice(1,None)}).values - ds[str_var['height']].isel({str_var['height']:slice(None,-1)}).values
    dUdz[:,1:] = dU/dZ

    logz = np.log(ds[str_var['height']])
    dlogz =  logz.isel({str_var['height']:slice(1,None)}).values - logz.isel({str_var['height']:slice(None,-1)}).values
    dU = ds[str_var['wspd']].isel({str_var['height']:slice(1,None)}).values - ds[str_var['wspd']].isel({str_var['height']:slice(None,-1)}).values

    dUdz = np.empty((len(ds[str_var['time']]),len(ds[str_var['height']]),));  dUdz[:,:] = np.nan
    dUdz[:,1:] = dU/dlogz * (1/ ds['height'].isel(height=slice(1,None)).values)
    kappa = 0.41
    ds[str_var['Phi_m']] = ((str_var['time'],str_var['height']), kappa*dUdz/ds[str_var['ustar']].values)
    ds[str_var['Phi_m']] = ds[str_var['Phi_m']]*ds[str_var['height']]



def calc_veer(ds, between_height=[40, 250], code='amr'):
    '''
    Calculate wind veer in degrees/m between the heights specified by `between_height`
    '''
    if code == 'sowfa':
        str_var = str_var_sowfa
    elif code == 'amr' or code == 'amrwind':
        str_var = str_var_amr
    else:
        raise ValueError("Code options: 'sowfa', 'amr'")

    # Calculate veer
    wdir = ds.sel(height=slice(between_height[0],between_height[1]))[str_var['wdir']]
    veer = wdir.polyfit(dim=str_var['height'],deg=1)
    ds['veer_deg_per_m'] = (('time'),  veer.sel(degree=1)['polyfit_coefficients'].data )

    return ds

# adapted from https://stackoverflow.com/questions/20105364
def density_scatter( x , y, ax = None, fig=None, sort = True, bins = 20, colorbar=True, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize 
    from scipy.interpolate import interpn

    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    if(colorbar):
        norm = Normalize(vmin = np.min(z), vmax = np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax, orientation='horizontal')
        cbar.ax.set_xticklabels([])
        cbar.ax.set_xlabel('density of points')
    return ax



def calc_spectra_chunks(ds, times, interval, window_length, spectra_dim,level_dim,  average_dim=None, window='hamming', window_overlap_pct=None, var_oi=None, xvar_oi=None):
    """
    Calculate spectra for a given number of times and return a new dataset
    """
    from mmctools.helper_functions import calc_spectra

    dflist = []
    for tstart in times:
        spectra = calc_spectra(ds,
                      var_oi=var_oi,
                      xvar_oi=xvar_oi,
                      spectra_dim=spectra_dim,
                      average_dim=average_dim,
                      level_dim=level_dim,
                      window=window,
                      tstart=pd.to_datetime(tstart),
                      interval=interval,
                      window_length=window_length,
                      window_overlap_pct=window_overlap_pct
                      #level=[0,1]
                     )
        spectra['time'] = pd.to_datetime(tstart)
        spectra = spectra.expand_dims('time').assign_coords({'time':[pd.to_datetime(tstart)]})
        dflist.append(spectra)

    # When we give data, say, from 01:00:00 to 03:59:59 and ask for '60min' interval, the last
    # 1-hour chunk will have ever so slightly different frequencies, resulting in double the total
    # amount of frequencies. Here we take the freqencies of the first chunk and make sure all of
    # the other chunks have the very same frequency. If not, but they're really close, replace the
    # frequencies of the current chunk with that of the first.
    freq = dflist[0].frequency
    for i in range(len(dflist)):
        if np.array_equal(freq, dflist[i].frequency):
            pass
        else:
            if np.allclose(freq, dflist[i].frequency, rtol=1e-10):
                dflist[i] = dflist[i].assign_coords({"frequency":freq})
            else:
                print(f'WARNING: The spectra list related to different chunks have different',\
                       'frequencies that are not within numerical tolerance. The resulting',\
                       'concatenated spectra might have NaNs on the last frequency. Check it.')



    df_spectra = xr.concat(dflist,dim='time')
    return df_spectra


def calc_streamwise (df, dfwdir, height=None, extrapolateHeights=True, showCoriolis=False, refheight=80, heightvar=None, dateref=None):
    '''
    Calculate streamwise and cross stream velocity components.
    
    By default, calculates the streamwise and cross stream of every height using the wdir
    at that height, which means the cross-stream component will be close to zero at every
    height (no Coriolis will be apparent). Alternatively, the option showCorilios uses a
    reference height to calculate the wdir. This way, Coriolis will be apparent and the
    cross-stream component will only be close to zero at the reference height.
    
    Parameters
    ==========
    df : Dataset
        Dataset containing the u, v, w with at least datetime coordinate
    dfwdir : Dataset, DataArray, or dataframe
        Dataset containing planar-average wind direction
    height: scalar
        If the input dataset only exists in one height (e.g. from VTKs),
        specify the height in this variable
    extrapolateHeights: bool
        Whether or not to extrapolate the wdir from heights given in dfwdir
        onto the heights asked in df. Useful if vertical slices that contain 
        all heights. Only used if showCoriolis is False. True by default.
    showCoriolis: bool
        Whether of not to use a single wdir at `refheight` to rotate the flow
    refheight: scalar
        Referente height to use if `showCoriolis` is True
    heightvar: str
        Name of the variable representing the height. If "height" of "z", no
        need to be specified. But depending on the vtk read, it could be "y".
    dateref: datetime64
        If df dataset has datetime array, this is the dataref needed to align
        the wdir-containing dataset dfwdir and the main dataset df.
      
    '''
    
    # Datasets, DataArrays, or dataframes
    if not isinstance(dfwdir,xr.Dataset):
        if isinstance(dfwdir,pd.DataFrame):
            dfwdir = dfwdir.to_xarray()
        elif isinstance(dfwdir,xr.DataArray):
            dfwdir = dfwdir.to_dataset()
        else:
            raise ValueError(f'unsupported type: {type(dfwdir)}')
            
    
    # Drop variables from dfwdir
    varsToDrop = set(dfwdir.variables.keys()) - set(['datetime','height','z','wdir','time'])
    dfwdir = dfwdir.drop(list(varsToDrop))

    # Interpolate wdir from planar average into the coordinates of df
    if heightvar is not None:
        heightInterp = df[heightvar]
    elif 'height' in list(df.variables.keys()):
        heightInterp = height = df.height
        heightvar = 'height'
    elif 'z' in list(df.variables.keys()):
        heightInterp = df.z
        heightvar = 'z'
    elif height != None:
        heightInterp = height
        heightvar = 'height'
    else:
        raise NameError("The input dataset does not appear to have a 'height' or 'z' coordinate. Use `height=<scalar>` to specify one.")
        
    if showCoriolis:
        # Force all the interpolation heights to be the reference height
        heightInterp = np.repeat(refheight, len(heightInterp))

    # Get name of the datetime and height variables
    if   'datetime' in list(df.variables.keys()):     dfdatetimevar = 'datetime'
    elif 'time'     in list(df.variables.keys()):     dfdatetimevar = 'time'
    else: raise ValueError (f'Could not find a date-related variable in target dataset')
    if   'datetime' in list(dfwdir.variables.keys()): dfwdirdatetimevar = 'datetime'
    elif 'time'     in list(dfwdir.variables.keys()): dfwdirdatetimevar = 'time'
    else: raise ValueError (f'Could not find a date-related variable in wdir-containing dataset')
    if   'height'   in list(dfwdir.variables.keys()): dfwdirheightvar = 'height'
    elif 'z'        in list(dfwdir.variables.keys()): dfwdirheightvar = 'z'
    else: raise ValueError (f'Could not find a height-related variable in wdir-containing dataset')

    # The df dataset can have its time/datetime coordinates in datetime64 format. If that is the case,
    # we need to convert the dfwdir (planar averages) to conform to that. To do that however, we need
    # a reference point. We do not attempt to get such reference point at first, and simply issue the
    # user a message. After the user specifies the required reference time, we continue the routine.
    if type(df[dfdatetimevar][0].values) == np.datetime64:
        if dateref is None:
            print(f'The array given has time coordinate {dfdatetimevar} of type numpy.datetime64.\n'\
                  f'While the wdir-containing dataset has scalars. To be able to compare both, we '\
                  f'need to convert one of them.\nFor that, we use the reference datatime used to '\
                  f'create the datetime array to make sure both arrays have datetime64 type.')
            print(f"Call the same function giving the reference in the following format: dateref=pd.to_datetime('2000-01-01 00:00:00')")
            raise ValueError
        elif isinstance(dateref, datetime.datetime):
            # Let's change the dfwdir
            dfwdir[dfwdirdatetimevar] = pd.to_datetime(dfwdir[dfwdirdatetimevar], unit='s', origin=dateref).round('0.1S')
        else:
            raise ValueError (f'dateref given but not of datetime format. Stopping')

    if extrapolateHeights:
        wdir_at_same_coords = dfwdir.interp({dfwdirdatetimevar:df[dfdatetimevar], dfwdirheightvar:heightInterp}, kwargs={"fill_value": "extrapolate"})
    else:
        wdir_at_same_coords = dfwdir.interp({dfwdirdatetimevar:df[datetimevar], dfheightvarheightvar:heightInterp})

    # Right now, we might have both time and datetime in the wdir_at_same_coords dataset because of the
    # interpolation. Depends on the variale name. Before combining, remove time coordinate, if not a dimension coordinate
    if dfdatetimevar == 'datetime':
        # If here, datetime also exists on wdir_at_same_coords due to the interp function call above and is
        # a dimension coordinate (that is, a bold dimension). In that case, let's drop the original `time`.
        wdir_at_same_coords = wdir_at_same_coords.drop('time')

    if showCoriolis:
        # Rename for clarity and drop height from coordinates since it no longer varies with height
        wdir_at_same_coords = wdir_at_same_coords.rename({'wdir': f'wdirAt{refheight}'})
        wdir_at_same_coords = wdir_at_same_coords.isel(height=1).squeeze()

        # Add wdir information to main dataset. Maybe the join need to use the df timestamps? (see commented out join=left)
        rotdf = xr.combine_by_coords([df, wdir_at_same_coords])#, join='left')
        wdir = rotdf[f'wdirAt{refheight}']
    else:
        # Add wdir information to main dataset
        rotdf = xr.combine_by_coords([df, wdir_at_same_coords])
        wdir = rotdf['wdir']

    
    # Rotate flowfield
    ustream = rotdf['u']*np.cos(np.deg2rad(270-wdir)) + rotdf['v']*np.sin(np.deg2rad(270-wdir))
    vcross =  rotdf['u']*np.sin(np.deg2rad(270-wdir)) - rotdf['v']*np.cos(np.deg2rad(270-wdir))

    return ustream, vcross, wdir





def addScalebar(ax, size_in_m=5000, label='5 km', lw=2, loc='lower left', color='black', fontsize=14, hideTicks=True):

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm 

    ax.add_artist(AnchoredSizeBar(ax.transData, size=size_in_m, label=label, loc=loc, 
                  pad=0.3, color=color, frameon=False, size_vertical=lw, fontproperties=fm.FontProperties(size=fontsize))
                  )

    if hideTicks:
        ax.set_xticks([])
        ax.set_yticks([])



def addLabels(axs,loc='upper right', fontsize=14, alpha=0.6, pad=6, start='a'):
    '''
    Inputs
    ------
    pad: int
        Distance from corner for label. pad=0 is label touching the corner of the plot
    start: char
        Character in which to start labeling the subplots
    '''

    # Ensure axs is iterable
    axs = np.array(axs)
    
    if   loc == 'upper right':  xy=(1,1); xytext=(-pad,-pad); xloc, yloc = 0.97, 0.97;  ha='right'; va='top'
    elif loc == 'upper left' :  xy=(0,1); xytext=( pad,-pad); xloc, yloc = 0.03, 0.97;  ha='left';  va='top'
    elif loc == 'lower right':  xy=(1,0); xytext=(-pad, pad); xloc, yloc = 0.97, 0.03;  ha='right'; va='bottom'
    elif loc == 'lower left' :  xy=(0,0); xytext=( pad, pad); xloc, yloc = 0.03, 0.03;  ha='left';  va='bottom'
    else:
        raise ValueError('loc not recognized. Stopping.')
    
    labels = list(map(chr, range(ord(start), 123)))
    if len(axs.flatten()) > len(labels):
        # More than a--z, so create 1a, 1b, 1c, 2a, etc. First find the number of numbers number of letters
        nN = np.shape(axs)[0]
        nL = np.shape(axs)[1]
        labels = [str(number) + chr(ord('a') + iletter) for number in range(1, nN + 1) for iletter in range(nL)]

    props = dict(facecolor='white', alpha=alpha, edgecolor='silver', boxstyle='square', pad=0.15)
    
    for i, ax in enumerate(axs.flatten()):
        #ax.text(xloc, yloc, f'$({labels[i]})$', color='black', ha=ha, va=va, transform=ax.transAxes, fontsize=fontsize, bbox=props)
        ax.annotate(f'$({labels[i]})$', xy=xy, color='black', ha=ha, va=va, xycoords='axes fraction',
                    xytext=xytext, textcoords='offset points', fontsize=fontsize, bbox=props)




def savePlanesForAnimation(ds, loopvar='datetime', var='u', varunits='m/s',
                           itime=0, ftime=-1, skiptime=1,
                           x=None, y=None, z=None,
                           vmin=None, vmax=None, cmap='viridis',
                           scalebar=500,
                           figsize=(14,6),
                           path=None,
                           prefix=None,
                           fontsize=14,
                           generateAnimation=False):
    '''
    Save planar data in a loop in images in specified directory for animation purposes
    
    Inputs
    ------
    ds: xr.Dataset
        Dataset with data to be plotted. It needs to have at least one time coordinate 
        and at least other two spatial coordinates
    loopvar: str
        Variable to loop on. Likely a datetime or time
    var: str
        Variable to plot. Potentially 'u', 'v', or 'wspd'
    varunits: str
        String for the variable unit to be ploted
    itime, ftime, skiptime: int or None
        Starting, end, and skip index of the time array to plot. It is likely that the 
        dataset has more time instants than it is desired to plot. It may be desirable
        to have skiptime>10 if your time frequency is 1 s or higher
    x, y, z: scalar of None
        If passing a dataset with more than 2 spatial coordinates, then the 3rd should
        be specified. For example for a dataset with a few horizontal planes, the z of
        interest should be given. Only one of the three variables should be passed
    vmin, vmax: scalar or None
        Min and max of pcolormesh. If one of them is None, then both are considered to
        be None and will be set depending on the variable asked
    cmap: str or None
        Colormap if not viridis for u and RdBu_r for v and w
    scalebar: str or None
        If not none, the size in meters of the scale bar to be shown
    figsize: Tuple
        Figure size for matplotlib
    path: str
        Full path of the animation directory where the png will be saved
    prefix: str
        Prefix of the final png files
    generateAnimation: bool
        Whether or not to automatically generate the video out of the pngs
    '''
    
    
    # Perform checks
    if not isinstance(ds, xr.Dataset):
        raise ValueError (f'ds should be an xarray dataset')
    
    if loopvar not in list(ds.coords):
        raise ValueError (f'Loop var requested {loopvar} is not in dataset. Available coordinates are: {list(ds.coords)}')
             
    compute_wspd = False
    if var not in list(ds.data_vars):
        if var == 'wspd':
            if 'u' not in list(ds.data_vars) or 'v' not in list(ds.data_vars):
                raise ValueError (f'Asked for wspd, but wspd is not available. Cannot compute it either since "u" and "v" are also not available.')
            print(f'The variable "wspd" is not in dataset, but will be computed from "u" and "v". Might take longer.')
            compute_wspd = True
        else:
            raise ValueError (f'Variable requested {var} is not in dataset. Available data variables are: {list(ds.data_vars)}')
        
    if not isinstance(varunits, str):
        raise ValueError (f'Unit of the variable requested {varunits} should be a string')
        
    # Fail-safe
    if itime    is None: itime=0
    if ftime    is None: ftime=-1
    if skiptime is None: skiptime=1
        
    if itime>len(ds[loopvar]):
        raise ValueError (f'Starting index {itime} is smaller than length of data series ({len(ds[loopvar])})')
    
    if ftime == -1:
        ftime = len(ds[loopvar])
    if ftime>len(ds[loopvar]):
        raise ValueError (f'Ending index {ftime} is larger than length of data series ({len(ds[loopvar])})')
                                                                                           
    if sum(arg is not None for arg in [x, y, z]) > 1:
        raise ValueError (f'At most, only one of x, y, or z can be passed.')
        
    if x is not None:
        dir1='y'; dir2='z'
        try:
            ds_ = ds.sel(x=x).squeeze()
        except KeyError:
            raise ValueError (f'Value x={x} does not exist in dataset. Valid values are {ds.x.values}')
    elif y is not None:
        dir1='x'; dir2='z'
        try:
            ds_ = ds.sel(y=y).squeeze()
        except KeyError:
            raise ValueError (f'Value y={y} does not exist in dataset. Valid values are {ds.y.values}')
    elif z is not None:
        dir1='x'; dir2='y'
        try:
            ds_ = ds.sel(z=z).squeeze()
        except KeyError:
            raise ValueError (f'Value z={z} does not exist in dataset. Valid values are {ds.z.values}')
    else:
        availDirs = [c for c in list(ds.coords) if c in ['x','y','z']]
        if len(availDirs) != 2:
            raise ValueError (f'Dataset contains {len(availDirs)} spatial coordinates. Unable to plot planar data. '\
                               'Consider giving the plane of interest as as {x,y,z}=<scalar>')
        dir1, dir2 = availDirs
        ds_ = ds.squeeze()
         
    if vmin is None or vmax is None:
        if var in ['u', 'U', 'wspd', 'windspeed', 'u_', 'streamwise']:
            vmin=2; vmax=14; cmap = 'viridis'
        if var in ['v', 'v_', 'crossstream']:
            vmin=-2; vmax=-2; cmap = 'RdBu_r'
        if var in ['w']:
            vmin=-2; vmax=-2; cmap = 'RdBu_r'
    
    if path is not None:
        if os.path.basename(path) == 'animation':
            if not os.path.isdir(path):
                # Animation path doesn't exist. Next logic will take care of it
                path = os.path.split(path)[0]
        if os.path.basename(path) != 'animation':
            if not os.path.isdir(path):
                raise ValueError (f'Path {path} does not exist')
            path = os.path.join(path, 'animation')
            if not os.path.isdir(path):
                print(f'Creating the directory "animation" inside {path} where output will be saved.')
                os.mkdir(path)
        
    if not isinstance(prefix,str):
        raise ValueError (f'File prefix should be a string')
    else:
        if prefix[-1] == '.': prefix=prefix[:-1]
        prefix = prefix.replace('.','_').replace(' ','_')
        
        
        
        
    # Create plot
    xx, yy = np.meshgrid(ds_[dir1], ds_[dir2], indexing='ij')

    for ifile, idatetime in enumerate(np.arange(itime,ftime,skiptime)):
        print(f'Saving time index {idatetime} out of {ftime}', end='\r')

        fig, ax = plt.subplots(1,1,figsize=figsize)
        
        datetime=ds_.isel({loopvar:idatetime})[loopvar].values

        # Compute wspd if needed
        if compute_wspd:
            ds_to_plot = ( ds_.sel({loopvar:datetime})['u']**2 + ds_.sel({loopvar:datetime})['v']**2 )**0.5   
        else:
            ds_to_plot = ds_.sel({loopvar:datetime})[var]

        cm = ax.pcolormesh(xx, yy, ds_to_plot, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)

        ax.set_aspect('equal')
        cax = fig.add_axes([ax.get_position().x1+0.01,  ax.get_position().y0, 0.017, ax.get_position().y1-ax.get_position().y0])
        cbar = fig.colorbar(cm, cax=cax)
        cbar.set_label(f'{var} [{varunits}]', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

        if np.issubdtype(datetime.dtype, np.datetime64):
            ax.annotate(f'{pd.Timestamp(datetime).hour:02d}:{pd.Timestamp(datetime).minute:02d}:{pd.Timestamp(datetime).second:02d}', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=fontsize)
        else:
            ax.annotate(f'{datetime}', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=fontsize)
        
        if isinstance(scalebar, int):
            addScalebar(ax,size_in_m = scalebar, label=f'{str(scalebar)} m', fontsize=fontsize)

        if path is not None:
            fig.savefig(os.path.join(path,f'{prefix}.{ifile:04d}.png'), facecolor='white', transparent=False, bbox_inches='tight')
        plt.cla()
        plt.close()

    if generateAnimation:
        import subprocess
        # https://github.com/rthedin/utilities/blob/master/generateAnimationFromPNG.sh
        print(f'Done generating the PNG. Creating animation...')
        currpath = os.getcwd()
        os.chdir(path)
        process = subprocess.Popen(["bash", "-c", '. /home/rthedin/utilities/generateAnimationFromPNG.sh; generateAnimationFromPNG'])#, stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL)
        # Unfortunately, ffmpeg output goes to stderr, not stdout.
        process.wait()
        os.chdir(currpath)
    
    print(f'Done.                                           ')


