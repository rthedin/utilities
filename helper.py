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

#from mmctools.helper_functions import covariance, calc_wind, calc_spectra
str_var_common = {
    "TKE"  : "TKE",
    "TI"   : "TI",
    "TI_TKE":"TI_TKE",
    "u*"   : "u*",
}
str_var_sowfa = {
    "wspd" : "wspd",
    "wdir" : "wdir",
    "uv"   : "uv",
    "vw"   : "vw",
    "uw"   : "uw",
    "uu"   : "uu",
    "vv"   : "vv",
    "ww"    : "ww",
    "u"    : "u",
    "v"    : "v",
    "w"    : "w",
    **str_var_common
}
str_var_amr = {
    "wspd" : "hvelmag",
    "wdir" : "wdir",
    "uv"   : "u'v'_r",
    "vw"   : "v'w'_r",
    "uw"   : "u'w'_r",
    "uu"   : "u'u'_r",
    "vv"   : "v'v'_r",
    "ww"   : "w'w'_r",
    "u"    : "u",
    "v"    : "v",
    "w"    : "w",
    **str_var_common
}

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



# from https://github.com/a2e-mmc/assessment/blob/study/coupling_comparison/studies/coupling_comparison/helpers.py
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
    df[str_var['u*']] = (df[str_var['uw']]**2 + df[str_var['vw']]**2)**0.25
    df[str_var['TKE']] = 0.5*(df[str_var['uu']] + df[str_var['vv']] + df[str_var['ww']])
    ang = np.arctan2(df[str_var['v']],df[str_var['u']])
    df[str_var['TI']] = df[str_var['uu']]*np.cos(ang)**2 + 2*df[str_var['uv']]*np.sin(ang)*np.cos(ang) + df[str_var['vv']]*np.sin(ang)**2
    df[str_var['TI']] = np.sqrt(df[str_var['TI']]) / df[str_var['wspd']]
    # TI as typical equations, based on TKE (same as AMR-Wind's TI TKE)
    df[str_var['TI_TKE']] = np.sqrt((df[str_var['uu']]+df[str_var['vv']]+df[str_var['ww']])/3.0)/np.sqrt(df[str_var['u']]**2 + df[str_var['v']]**2)



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
    if 'datetime' in list(df.variables.keys()):     dfdatetimevar = 'datetime'
    if 'time'     in list(df.variables.keys()):     dfdatetimevar = 'time'
    if 'datetime' in list(dfwdir.variables.keys()): dfwdirdatetimevar = 'datetime'
    if 'time'     in list(dfwdir.variables.keys()): dfwdirdatetimevar = 'time'
    if 'height'   in list(dfwdir.variables.keys()): dfwdirheightvar = 'height'
    if 'z'        in list(dfwdir.variables.keys()): dfwdirheightvar = 'z'

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
     

    if showCoriolis:
        # Rename for clarity and drop height from coordinates since it no longer varies with height
        wdir_at_same_coords = wdir_at_same_coords.rename({'wdir': f'wdirAt{refheight}'})
        wdir_at_same_coords = wdir_at_same_coords.isel(height=1).squeeze()

        # Add wdir information to main dataset
        rotdf = xr.combine_by_coords([df, wdir_at_same_coords])
        wdir = rotdf[f'wdirAt{refheight}']
    else:
        # Add wdir information to main dataset
        rotdf = xr.combine_by_coords([df, wdir_at_same_coords])
        wdir = rotdf['wdir']

    
    # Rotate flowfield
    ustream = rotdf['u']*np.cos(np.deg2rad(270-wdir)) + rotdf['v']*np.sin(np.deg2rad(270-wdir))
    vcross =  rotdf['u']*np.sin(np.deg2rad(270-wdir)) - rotdf['v']*np.cos(np.deg2rad(270-wdir))

    return ustream, vcross, wdir





def addScalebar(ax, size_in_m=5000, label='5 km', loc='lower left', color='black', fontsize=14, hideTicks=True):

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm 

    ax.add_artist(AnchoredSizeBar(ax.transData, size=size_in_m, label=label, loc=loc, 
                  pad=0.3, color=color, frameon=False, size_vertical=2, fontproperties=fm.FontProperties(size=fontsize))
                  )

    if hideTicks:
        ax.set_xticks([])
        ax.set_yticks([])


