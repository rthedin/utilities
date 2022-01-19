# Collection of functions to deal with spatial and temporal correlations
#
# Regis Thedin
# Dec 2021

import numpy as np
import xarray as xr
import pandas as pd

def performCorrelationSubDomain (ds, nSubDomainsX, nSubDomainsY, ti, tf, dateref=None, window=1200, inc=600):
    '''
    Perform spatial correlations on subdomains with respect to the central point of each subdomain.
    Additionally, performs the same procedure in windows, useful for changing conditions and/or smoother
    results. This is useful on a periodic-type case in order to get smoother curves (analogous to a
    windowing approach for PSDs)

    Parameters
    ==========
    ds: Dataset
        Dataset with the variables of interest in a plane
    nSubDomainsX, nSubDomainsY: int
        Number of subdivisions of the subdomain
    ti, tf: scalars
        Initial and end time of the total series of interest (seconds)
    dateref: datetime (optional)
        Datetime stamp corresponding to time 0, if datetime outputs are
        needed
    window: scalar
        The entire series are split into windows of this size and
        results are given for each window, in seconds. Default 20 min.
    inc: scalar
        Window marching increment (overlap), in seconds. Default 10 min.
        
        
    Example call:
    v2x2 = performCorrelationSubDomain(ds80m[['u','v']],
                                       nSubDomainsX=2, 
                                       nSubDomainsY=2,
                                       ti=133200, tf=133200+4*3600,
                                       dateref=pd.to_datetime('2010-05-14 12:00:00'),
                                       window=1800, inc=600 )
    '''
    
    assert isinstance(nSubDomainsX, int), "Number of subdomains should be an integer"
    assert isinstance(nSubDomainsY, int), "Number of subdomains should be an integer"
    
    # Get subdomain limits
    xmin = ds.x[0].values;  xmax = ds.x[-1].values
    ymin = ds.y[0].values;  ymax = ds.y[-1].values
    xSubDom = np.linspace(xmin, xmax, nSubDomainsX+1)
    ySubDom = np.linspace(ymin, ymax, nSubDomainsY+1)
    x0SubDom = (xSubDom[1:] + xSubDom[:-1]) / 2
    y0SubDom = (ySubDom[1:] + ySubDom[:-1]) / 2

    vlist = []
    while ti+window <= tf:
        ti_dt = pd.to_datetime(ti, unit='s', origin=dateref)
        tf_dt = pd.to_datetime(ti+window, unit='s', origin=dateref)
        for i, _ in enumerate(xSubDom[:-1]):
            for j, _ in enumerate(ySubDom[:-1]):
                x0 = x0SubDom[i];  y0 = y0SubDom[j]
                dssub = ds.sel(x=slice(xSubDom[i],xSubDom[i+1]-1), y=slice(ySubDom[j],ySubDom[j+1]-1)).copy()

                v_curr = spatialCorrelation2D(dssub, x0=x0, y0=y0, ti=ti, tf=ti+window, dateref=dateref)
                v_curr = v_curr.expand_dims('datetime').assign_coords({'datetime': [ti_dt]})
                vlist.append(v_curr)
        ti = ti + inc

    # concatenate the resulting list
    vsubdom = xr.combine_by_coords(vlist)
    return vsubdom



def spatialCorrelation2D (ds, x0, y0, ti=None, tf=None, dateref=None):
    '''
    Perform spatial correlation on a 2D field, averaging in time
    
    Parameters
    ==========
    ds: Dataset
        Dataset with the variables of interest in a plane
    x0, y0: scalars
        Reference point in which the spatial correlation in related to
    ti, tf: scalars
        Initial and end time of the total series of interest (seconds)
    dateref: datetime (optional)
        Datetime stamp corresponding to time 0, if datetime outputs are
        needed
    '''
    
    # Operations in time. Get subset within the times of interest
    if dateref != None:
        ti = pd.to_datetime(ti, unit='s', origin = dateref)
        tf = pd.to_datetime(tf, unit='s', origin = dateref)
    ds = ds.sel(datetime=slice(ti,tf)).copy()
    times = ds.datetime.values
    
    # find position of (x0, y0)
    iNearest = (abs(ds.x-x0)).argmin().values
    jNearest = (abs(ds.y-y0)).argmin().values
    
    print(f'Performing spatial correlation wrt to point ({ds.isel(x=iNearest).x.values}, ' \
          f'{ds.isel(y=jNearest).y.values}), between {ti} and {tf}.', end='\r', flush=True)
    
    mean = ds.sel(datetime=slice(ti,tf)).mean(dim='datetime')
    vlist=[]
    for i, t in enumerate(times):
        primeField = ds.sel(datetime=t) - mean
        v = primeField*primeField.isel(x=iNearest, y=jNearest)
        vlist.append(v)
    
    finalv = xr.concat(vlist, dim='datetime').mean(dim='datetime')
    finalv = finalv/finalv.isel(x=iNearest, y=jNearest)
    
    return finalv



def averageSubdomains(dsv, nSubDomainsX, nSubDomainsY, ds=None):
    '''
    Receives the output of `performCorrelationSubDomain` and average accross
    the different sub-domains
    
    Parameters
    ==========
    dsv: Dataset
        Dataset containing the spatial correlation to be averaged
    ds: Dataset (optional)
        Original dataset used to compute spatial correlation. Needed for 
        wind direction. Assumed `wdir` exists. If provided, wdir is part
        of the output dataset.
    nSubDomainsX, nSubDomainsY: int
        Number of subdivisions of the subdomain
        
    '''
    
    # get window size for wdir averaging
    window = dsv.datetime[1] - dsv.datetime[0]

    # Get subdomain limits. These are _not_ the overall bounding box.
    res = (dsv.x[1]-dsv.x[0]).values
    xmin = dsv.x.min().values;  xmax = dsv.x.max().values+res
    ymin = dsv.y.min().values;  ymax = dsv.y.max().values+res
    xSubDom = np.linspace(xmin, xmax, nSubDomainsX+1)
    ySubDom = np.linspace(ymin, ymax, nSubDomainsY+1)
    x0SubDom = (xSubDom[1:] + xSubDom[:-1]) / 2
    y0SubDom = (ySubDom[1:] + ySubDom[:-1]) / 2

    avgv = []
    for t, d in enumerate(dsv.datetime):
        if isinstance(ds, xr.Dataset):  wdir = ds.sel(datetime=slice(d,d+window))['wdir'].mean().values
        subsubvavg = []
        subv = dsv.sel(datetime=d)
        for i in range(len(x0SubDom)):
            for j in range(len(y0SubDom)):
                x0 = x0SubDom[i];  y0 = y0SubDom[j]
                subsubv = subv.sel(x=slice(xSubDom[i],xSubDom[i+1]-1), y=slice(ySubDom[j],ySubDom[j+1]-1)) 
                subsubvavg.append( subsubv.assign_coords({'x':subsubv.x-x0, 'y':subsubv.y-y0}).expand_dims('datetime') )
        subsubvavg = xr.concat(subsubvavg, dim='datetime').mean(dim='datetime')
        subsubvavg = subsubvavg.expand_dims('datetime').assign_coords({'datetime': [d.values]})
        if isinstance(ds, xr.Dataset):  subsubvavg = subsubvavg.assign({'wdir': ('datetime', [wdir])})
        avgv.append(subsubvavg)
    avgv = xr.concat(avgv, dim='datetime')

    return avgv



def getSpaceCorrAlongWindDir (ds, dsv, nSubDomainsX, nSubDomainsY, var_oi='uu', wdirvar='wdir', \
                              nSubDomainsSkipN = 0, nSubDomainsSkipS = 0, nSubDomainsSkipW = 0, nSubDomainsSkipE = 0):
    '''
    Calculates the spatial correlation values along the streamwise and cross-stream direction.
    Streamwise direction is determined as along the mean wind direction for intervals defined
    in the correlation DataSet. Cross-stream direction is defined as 90deg from streamwise.
    
    Inputs:
    -------
    ds: xArray DataSet
        Windspeed and wind direction data, as well as grid information for all times. Usually
        the output of VTK-reading functions
    dsv: xArray DataSet
        Spatial correlation data. Output of `performCorrelationSubDomain`
    nSubDomains{X,Y}: int
        Number of sub-domains the underlying dsv dataset was split into. Should be a positive integer
    window: int
        window, in seconds, to get the mean wdir from
    nSubDomainsSkip{N,S,W,E}: int
        Number of subdomains to skip on each North, South, West, or East boundary. Use this option in
        boundary-coupled cases where there is a fetch
        
    Outputs:
    --------
    v_long: np.array
        Correlation along the streamwise direction (`long` for longitudinal)
    v_tran: np.array
        Correlation along the cross-stream direction (`tran` for transverse)
    x_{long,tran}: np.array
        Auxiliary array for proper plotting. `x` represents the direction
    L_{long,tran}: np.array
        Integral time scale for the streamwise and cross-stream direction, respectively
    L_{long,tran)_: np.array:
        Same as above, but computed on only half the domain and up to the point where
        it crosses the corr=0 threshold
        
    '''
    from windtools.common import calc_wind
    import scipy.interpolate as interp

    # Get subdomain limits
    xmin = ds.x.min().values;  xmax = ds.x.max().values
    ymin = ds.y.min().values;  ymax = ds.y.max().values
    xSubDom = np.linspace(xmin, xmax, nSubDomainsX+1);  xSubDom = xSubDom[nSubDomainsSkipW:len(xSubDom)-nSubDomainsSkipE]
    ySubDom = np.linspace(ymin, ymax, nSubDomainsY+1);  ySubDom = ySubDom[nSubDomainsSkipS:len(ySubDom)-nSubDomainsSkipN]
    x0SubDom = (xSubDom[1:] + xSubDom[:-1]) / 2
    y0SubDom = (ySubDom[1:] + ySubDom[:-1]) / 2

    # Get clipped-in-space wspd and wdir values based on subdomains considered (excludes skips)
    Lsubdom = xSubDom[1]-xSubDom[0]
    ds = ds.sel(x=slice(xmin+nSubDomainsSkipW*Lsubdom,xmax-nSubDomainsSkipE*Lsubdom), y=slice(ymin+nSubDomainsSkipS*Lsubdom,ymax-nSubDomainsSkipN*Lsubdom) )

    # Assumes uniform window
    window = dsv.datetime[1] - dsv.datetime[0]

    # get correlations streamwise and cross-stream, as well as int scales
    # (ending in underscore means integral up to the point it crosses 0)
    tv_long = [];  L_long = [];  L_long_ = [];  L_long_5pc = [];  tau_long = [];  tau_long_ = []
    tv_tran = [];  L_tran = [];  L_tran_ = [];  L_tran_5pc = [];  tau_tran = [];  tau_long_5pc = []

    # define the linelong and longtran to the total mean (spatial and temporal)
    tlinelong = np.arange( -0.5*(xSubDom[1]-xSubDom[0]), 0.5*(xSubDom[1]-xSubDom[0]), 10)
    tlinetran = np.arange( -0.5*(ySubDom[1]-ySubDom[0]), 0.5*(ySubDom[1]-ySubDom[0]), 10)

    for t, d in enumerate(dsv.datetime):
        print(f'Processing time {d.values}')
        subv_long = [];  subv_tran = []; 
        subv = dsv.sel(datetime=d)
        wdir = ds.sel(datetime=slice(d,d+window))[wdirvar].mean().values

        for i in range(len(x0SubDom)):
            for j in range(len(y0SubDom)):
                x0 = x0SubDom[i];  y0 = y0SubDom[j]
                #print(f'    Processing subdomain i {i}, j {j}; with center({x0},{y0})')
                subsubv = subv.sel(x=slice(xSubDom[i],xSubDom[i+1]-1), y=slice(ySubDom[j],ySubDom[j+1]-1))

                # Define points along the streamwise direction going through (x0, y0)
                xxlong = np.arange(xSubDom[i], xSubDom[i+1], 10)
                a = -np.tan(np.deg2rad(wdir-90));  b = y0 - a*x0
                yylong = a*xxlong + b

                # Define points along the cross flow directions, going through (x0, y0)
                yytran = np.arange(ySubDom[j], ySubDom[j+1], 10)
                a =  np.tan(np.deg2rad(360-wdir));  b = y0 - a*x0
                xxtran = (yytran -b)/a

                # get the correlations along the two directions
                # Create interpolator based on original map
                xx, yy = np.meshgrid(subsubv.x, subsubv.y, indexing='ij')
                grid = np.stack([xx.ravel(), yy.ravel()], -1)
                values_on_grid = subsubv[var_oi].values.ravel()
                f = interp.RBFInterpolator(grid, values_on_grid, smoothing=0, kernel='cubic', neighbors=100)

                # Use interpolator to get values on the points of interest
                points_oi_long = np.stack([xxlong, yylong], -1)
                points_oi_tran = np.stack([xxtran, yytran], -1)
                v_long = f(points_oi_long)
                v_tran = f(points_oi_tran)

                # Create 1-D array of distance across the long/transverse line
                linelong = np.linspace(0, ((xxlong[-1]-xxlong[0])**2 + (yylong[-1]-yylong[0])**2)**0.5, num=len(v_long) )
                linetran = np.linspace(0, ((xxtran[-1]-xxtran[0])**2 + (yytran[-1]-yytran[0])**2)**0.5, num=len(v_tran) )

                # Center line around the central point being at 0
                linelong = linelong - linelong[np.argmax(v_long)]
                linetran = linetran - linetran[np.argmax(v_tran)]

                # Concatenate results for current time interval      
                subv_long.append(v_long)
                subv_tran.append(v_tran)

        subv_long = np.mean(subv_long, axis=0)
        subv_tran = np.mean(subv_tran, axis=0)

        # Get values onto a common line for plotting
        tv_long.append(np.interp(tlinelong, linelong, subv_long))
        tv_tran.append(np.interp(tlinetran, linetran, subv_tran))

        # Calculate the integral length scale of the chunk-average
        L_long.append(np.trapz(tv_long[t], tlinelong))
        L_tran.append(np.trapz(tv_tran[t], tlinetran))


        # Calculate the integral length scale of the chunk-average. Only getting the right-half for the integral
        v_long = np.split(tv_long[t], 2)[1]
        v_tran = np.split(tv_tran[t], 2)[1]
        x_long = np.split(tlinelong, 2)[1]
        x_tran = np.split(tlinetran, 2)[1]

        assert abs(x_long[0])<0.001, f'The spatial dimention should be 0; it curently is {x_long[0]}'
        assert abs(x_tran[0])<0.001, f'The spatial dimention should be 0; it curently is {x_tran[0]}'
        assert abs(v_long[0]-1)<0.001, f'The correlation of the central point should be 1; it curently is {v_long[0]}'
        #if abs(v_tran[0]-1)>0.001:
        #    # ad-hoc catch of a FINO case for 2x2 sub-domains that was not working
        #    print(f'{v_tran[0:10]}')
        #    print(f'The correlation of central point should be 1; it curently is {v_tran[0]}. Trying to get the next position instead.')
        #    print(f'The value of the correlation in {x_tran[1]} is {v_tran[1]}')
        #    v_tran = v_tran[1:]
        assert abs(v_tran[0]-1)<0.001, f'The correlation of the central point should be 1; it curently is {v_tran[0]}'
        # get instant that crosses zero
        pos0cross_long = np.argmax(v_long<0)
        pos0cross_tran = np.argmax(v_tran<0)
        if pos0cross_long == 0:
            # doesn't cross, so integrating the whole half-curve
            L_long_.append(np.trapz(v_long, x_long))
        else:
            # it crosses 0, so integrate up to that point
            L_long_.append(np.trapz(v_long[:pos0cross_long], x_long[:pos0cross_long]))
        if pos0cross_tran == 0:
            L_tran_.append(np.trapz(v_tran, x_tran))
        else:
            L_tran_.append(np.trapz(v_tran[:pos0cross_tran], x_tran[:pos0cross_tran]))

        # get the position where it's 5%, as opposed to zero cross
        pos5pccross_long = np.argmax(v_long<0.05)
        pos5pccross_tran = np.argmax(v_tran<0.05)
        if pos5pccross_long == 0:
            # doesn't reach 5%, so integrating the shole half-curve
            L_long_5pc.append(np.trapz(v_long, x_long))
        else:
            # it reaches 5%, so integrate up to that point
            L_long_5pc.append(np.trapz(v_long[:pos5pccross_long], x_long[:pos5pccross_long]))
        if pos5pccross_tran == 0:
            L_tran_5pc.append(np.trapz(v_tran, x_tran))
        else:
            L_tran_5pc.append(np.trapz(v_tran[:pos5pccross_tran], x_tran[:pos5pccross_tran]))
            

        # Calculate the integral time scale based on the int length scale and space-mean windspeed
        # We don't know if the dataset given has u_ and v_ or u and v:
        try:
            wspd, _ = calc_wind(ds.sel(datetime=slice(d,d+window)), u='u_',v='v_')
        except AssertionError:
            wspd, _ = calc_wind(ds.sel(datetime=slice(d,d+window)), u='u',v='v')
        wspd = wspd.mean().values
        tau_long.append(np.trapz(tv_long[t], tlinelong)/wspd)
        tau_tran.append(np.trapz(tv_tran[t], tlinetran)/wspd)
        tau_long_.append(L_long_[t]/wspd)
        tau_long_5pc.append(L_long_5pc[t]/wspd)

    tv_long = np.vstack(tv_long)
    tv_tran = np.vstack(tv_tran)

    return {'v_long':tv_long, 'v_tran':tv_tran, 'x_long':tlinelong, 'x_tran':tlinetran, 'L_long':L_long, 'L_tran':L_tran, \
            'tau_long':tau_long, 'tau_tran':tau_tran, 'L_long_':L_long_, 'L_tran_':L_tran_, \
            'tau_long_':tau_long_, 'tau_long_5pc':tau_long_5pc, 'L_long_5pc':L_long_5pc, 'L_tran_5pc':L_tran_5pc }


