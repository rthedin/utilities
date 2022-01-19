# Collections of functions that help perform routine tasks on jupyter notebooks.
# Some of them have been copied from old notebooks, some have been adapted to my
# workflow. Putting them here for convenience-- no need to copy and paste them
# onto new notebooks
#
# Regis Thedin
# Oct 18, 2021
#

import numpy as np
import pandas as pd
import xarray as xr
import os, sys
from scipy.interpolate import griddata

from mmctools.helper_functions import covariance, calc_wind, calc_spectra

def interpolate_to_heights(df,heights):
    """
    Interpolate data in dataframe to specified heights
    and return a new dataframe
    """
    from scipy.interpolate import interp1d

    # If single height is asked
    if isinstance(heights, (float,int)):
        heights=[heights]
    
    # Unstack to single height index (= most time-consuming operation)
    unstacked = df.unstack(level='datetime')
    # Interpolate to specified heights
    f = interp1d(unstacked.index,unstacked,axis=0,fill_value='extrapolate')
    for hgt in heights:
        unstacked.loc[hgt] = f(hgt)
    # Restack and set index
    df_out = unstacked.loc[heights].stack().reset_index().set_index(['datetime','height']).sort_index()
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
    stats['thetaw'] = covariance(unstacked['theta'], unstacked['w'], offset, resample=True).stack()
    return stats



# from https://github.com/a2e-mmc/assessment/blob/study/coupling_comparison/studies/coupling_comparison/helpers.py
def calc_QOIs(df):
    """
    Calculate derived quantities (IN PLACE)
    """
    from mmctools.helper_functions import covariance, calc_wind 
    df['wspd'],df['wdir'] = calc_wind(df)
    df['u*'] = (df['uw']**2 + df['vw']**2)**0.25
    df['TKE'] = 0.5*(df['uu'] + df['vv'] + df['ww'])
    ang = np.arctan2(df['v'],df['u'])
    df['TI'] = df['uu']*np.cos(ang)**2 + 2*df['uv']*np.sin(ang)*np.cos(ang) + df['vv']*np.sin(ang)**2
    df['TI'] = np.sqrt(df['TI']) / df['wspd']




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

    df_spectra = xr.concat(dflist,dim='time')
    return df_spectra


def calc_streamwise (df, dfwdir, height=None, extrapolateHeights=True):
    '''
    Calculate streamwise and cross stream velocity components
    
    Parameters
    ==========
    df : Dataset
        Dataset containing the u, v, w with at least datetime coordinate
    dfdir : Dataset, DataArray, or dataframe
        Dataset containing planar-average wind direction
    height: scalar
        If the input dataset only exists in one height (e.g. from VTKs),
        specify the height in this variable
    extrapolateHeights: bool
        Whether or not to extrapolate the wdir from heights given in dfwdir
        onto the heights asked in df. Useful if vertical slices that contain 
        all heights. True by default.
      
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
    if 'height' in list(df.variables.keys()):
        heightInterp = height=df.height
    elif 'z' in list(df.variables.keys()):
        heightInterp = df.z
    elif height != None:
        heightInterp = height
    else:
        raise NameError("The input dataset does not appear to have a 'height' or 'z' coordinate. Use `height=<scalar>` to specify one.")
        
    if extrapolateHeights:
        wdir_at_same_coords = dfwdir.interp(datetime=df.datetime, height=heightInterp, kwargs={"fill_value": "extrapolate"})
    else:
        wdir_at_same_coords = dfwdir.interp(datetime=df.datetime, height=heightInterp)
    
    # Add wdir information to main dataset
    rotdf = xr.combine_by_coords([df, wdir_at_same_coords])
    wdir = rotdf['wdir']
    
    # Rotate flowfield
    ustream = rotdf['u']*np.cos(np.deg2rad(270-wdir)) + rotdf['v']*np.sin(np.deg2rad(270-wdir))
    vcross =  rotdf['u']*np.sin(np.deg2rad(270-wdir)) - rotdf['v']*np.cos(np.deg2rad(270-wdir))

    return ustream, vcross, wdir



def calc_coherence(s1,s2, interval='120min', window_length='10min', window='hamming', normal=None):
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
        
    '''
    
    if isinstance(s1, dict):
        if not isinstance(s2,dict):
            raise ValueError('Both series need to be given in the same format')
        assert isinstance(s1['var'],str)
        try:
            if normal==None:
                # probe data
                sig1 = s1['ds'].sel(height=s1['height'], x=s1['x'], y=s1['y'], drop=True, method='nearest', tolerance=1e-3)[s1['var']]
                sig2 = s2['ds'].sel(height=s2['height'], x=s2['x'], y=s2['y'], drop=True, method='nearest', tolerance=1e-3)[s2['var']]
            elif normal=='xNormal': 
                sig1 = s1['ds'].sel(z=s1['z'], y=s1['y'], drop=True, method='nearest', tolerance=1e-3)[s1['var']]
                sig2 = s2['ds'].sel(z=s2['z'], y=s2['y'], drop=True, method='nearest', tolerance=1e-3)[s2['var']]
            elif normal=='yNormal': 
                sig1 = s1['ds'].sel(z=s1['z'], x=s1['x'], drop=True, method='nearest', tolerance=1e-3)[s1['var']]
                sig2 = s2['ds'].sel(z=s2['z'], x=s2['x'], drop=True, method='nearest', tolerance=1e-3)[s2['var']]
        except KeyError:
            print(f'A value given in the dictionary does not appear to exist')
            raise

            
    spectraTimes = pd.date_range(start=sig1.datetime[0].values, end=sig1.datetime[-1].values, freq='10min')
    
    # Rename signals, following the {u,v,w}{1,2} syntax
    var1 = s1['var'][0]+'1'
    var2 = s2['var'][0]+'2'
    
    signals = xr.merge([sig1.to_dataset(name=var1), sig2.to_dataset(name=var2)])

    psd = calc_spectra(signals,
                       var_oi=[var1,var2],
                       xvar_oi=[(var1,var2)],
                       spectra_dim='datetime',
                       tstart=spectraTimes[0], # first time instant of nyserda
                       #average_dim='station',
                       #level_dim='height',
                       window=window,
                       interval=interval,
                       window_length=window_length,
                       window_overlap_pct=0.5)

    gamma2 = psd[var1+var2]**2/(psd[var1]*psd[var2])

    return gamma2.to_dataset(name=var1+var2), psd










