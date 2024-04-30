#!/usr/bin/env python3

#SBATCH --job-name=sowfaIC6x6_20min
#SBATCH --output log_sowfa_ic_3x3_20min.log
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --account=car

from itertools import repeat
import sys, time
import xarray as xr
import pandas as pd
import numpy as np
from multiprocessing import Pool, freeze_support

sys.path.append('/home/rthedin/utilities')
from correlation import spatialCorrelation2D, performCorrelationSubDomainPar


def main():
    # --------------- MODIFIABLE SETTINGS ----------------#
    ti=133200
    tf=133200+4*3600
    inc=600
    window=1200
    nSubDomainsX = nSubDomainsY = 3
    fullPathToNc = '/projects/mmc/rthedin/OpenFOAM/rthedin-6/run/offshoreCases/05_fino_sampling_small3x3/processedData/ds_VTKfino_z80m_01Z_05Z.nc'
    output = 'sowfa_ic_corr3x3subdom_20min_01Z_05Z.nc'
    dateref = pd.to_datetime('2010-05-14 12:00:00')
    # ------------ END OF MODIFIABLE SETTINGS ------------#

    timeStartInterval = np.arange(ti,tf-window+1,inc)

    p = Pool()
    vlist = p.starmap(performCorrelationSubDomainPar, zip(repeat(fullPathToNc),
                                                          repeat(nSubDomainsX),
                                                          repeat(nSubDomainsY),
                                                          timeStartInterval,
                                                          repeat(dateref),
                                                          repeat(window),
                                                          repeat(inc)) )

    finalv = xr.combine_by_coords(vlist)
    finalv.to_netcdf(output)


if __name__=="__main__":
    print(f'Starting job at {time.ctime()}')
    freeze_support()
    main()
    print(f'Ending job at   {time.ctime()}')

