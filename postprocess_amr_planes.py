#!/home/rthedin/.conda/envs/ssrs_env_scratch/bin/python

#SBATCH --job-name=pp_planes
#SBATCH --output amr2post_planes.log.%j
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --account=car
#SBATCH --qos=high

# ----------------------------------------------------------------------- #
# postprocess_planes.py                                                   #
#                                                                         #
# Post process AMR-Wind output planes into easy-to-read netcdf files that #
# will be saved as zarr files. This script should be called once per file #
# and its output will be saved in $case/processedData. The resulting zarr #
# files can then be read using SLURM cluster (dask_jobqueue.SLURMCluster) #
# for efficient computations.                                             #
#                                                                         #
# Usage:                                                                  #
# postprocess_planes.py -p <fullpath> -f <samplingfile>                   #
#                                                                         #
# Example usage:                                                          #
#    cd /full/path/to/amrwind/case                                        #
#    path=$(pwd -P)                                                       #
#    cd post_processing                                                   #
#    ls  # see what sampling files are available.                         #
#    sbatch postprocess_planes.py -p $path -f samplingyz40000.nc          #
#    sbatch postprocess_planes.py -p $path -f samplingxz40000.nc          #
#    sbatch postprocess_planes.py -p $path -f samplingxy40000.nc          #
#                                                                         #
# Regis Thedin                                                            #
# Feb 8, 2023                                                             #
# regis.thedin@nrel.gov                                                   #
# ----------------------------------------------------------------------- #

import argparse
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import sys, os, time
from itertools import repeat
from multiprocessing import Pool, freeze_support
from windtools.amrwind.post_processing  import Sampling, addDatetime

def main(samplingplanefile, pppath, outpath, dt, itime, ftime, steptime):

    # -------- CONFIGURE RUN
    samplingplanepath = os.path.join(pppath,samplingplanefile)
    s = Sampling(samplingplanepath)

    group = s.groups
    if len(group) > 1:
        raise ValueError(f'The ncfile {samplingplanefile} contains more than one'\
                         f'group: {s.groups}. Specify -g <groupstring> for single group.')
    else:
        group = group[0]

    s.getGroupProperties_xr(group=group)

    print(f'Group {group} has {s.ndt} saved time steps.')
    if itime==0 and ftime==-1:
        chunkedSaving = False
        outputzarr         = os.path.join(outpath, f'{group}.zarr')
        print(f'  Saving all of them.')
        print(f'  If it seems like you will run out of memory, consider giving itime=0'\
                f'and ftime={int(s.ndt/2)} and executing the post processing in chunks.')
    else:
        chunkedSaving = True
        outputzarr         = os.path.join(outpath, f'{group}_dti{itime}_dtf{ftime}.zarr')
        print(f'  Saving the time steps between {itime} and {ftime}, out of {s.ndt}.')

    if os.path.isdir(outputzarr):
        raise ValueError(f'The following target zarr file already exists: {outputzarr}')

    if ftime > s.ndt:
        print(f'Final time step requested {ftime} is larger than what is available in'\
              f'the array, {s.ndt}. Saving up to {s.ndt} instead.')
        ftime = s.ndt
    if ftime == -1:
        ftime = s.ndt

    # Split all the time steps in arrays of roughly the same size
    chunks =  np.array_split(range(itime,ftime), 32)
    # Now, get the beginning and end of each separate chunk
    itime_list = [i[0]    for i in chunks]
    ftime_list = [i[-1]+1 for i in chunks] 
    print(f'itime_list is {itime_list}')
    print(f'ftime_list is {ftime_list}')

    p = Pool()
    ds_ = p.starmap(s.read_single_group, zip(repeat(group),         # group
                                             itime_list,            # itime
                                             ftime_list,            # ftime
                                             repeat(steptime),      # step
                                             repeat(None),          # outputPath
                                             repeat(['u','v','w']), # var
                                             repeat(True),          # simCompleted
                                             repeat(False),         # verbose
                                            )
                                         )
    # Note: For some unknown reason, after saving the combined array, it is like the
    #       ds_ array was already chunked. So if I request 50 time steps, it will be
    #       chunked as (7, 7, 7, 7, 7, 7, 7, 1) in one of the coordines (7*7+1=50).
    #       The issue appears then we try to save the combined array, which means the
    #       chunk in the same dimension now will be (7, 7, ...7, 1, 7, 7, ...), with
    #       the ds_ chunking just concatenated. When we try to save that to a single
    #       zarr file, it complains because only the last chunk can be different. I
    #       noticed it seems to always split it in 8 chunks, so if we pass a number
    #       of dt that is divisible by 8, then we do not run into this issue. The
    #       number of threads is also locked at 32 since 32%8=0, to avoid further
    #       issues

    print('Starmap call done. Combining array')
    comb_ds_ = xr.combine_by_coords(ds_)  # combine in `samplingtimestep`
    print('Done combining array.')

    if chunkedSaving == True:
        # Compute chunked mean quantities to make it easier to get fluctuating part later
        print('Chunked saving requested, computing mean quantities')
        comb_ds_['umean'] = comb_ds_['u'].mean(dim='samplingtimestep')
        comb_ds_['vmean'] = comb_ds_['v'].mean(dim='samplingtimestep')
        comb_ds_['wmean'] = comb_ds_['w'].mean(dim='samplingtimestep')

    print('Saving combined array to zarr')
    comb_ds_.to_zarr(outputzarr)

    print('Done with reshaping and saving xarray.')
    
    if dt is not None:
        print(f'Now calculating fluctuating part and adding datetime. Using dt = {dt}')

        outputdatetimezarr = os.path.join(outpath, f'{group}_datetime_mean.zarr')
        outputdatetimenc   = os.path.join(outpath, f'{group}_datetime_mean.nc')
        if os.path.isdir(outputdatetimezarr):
            raise ValueError(f'The follwoing target zarr file already exists: {outputdatetimezarr}')

        comb_ds = addDatetime(comb_ds_, dt=dt)
        print(f'Writing {outputdatetimezarr}')
        try:
            comb_ds.to_zarr(outputdatetimezarr)
        except ValueError:
            print(f'Zarr is complaining about uniform chunk sizes. Saving it as netcdf instead')
            comb_ds.to_netcdf(outputdatetimenc)
        print(f'Done writing the datetime/mean output file.')




if __name__ == '__main__':

    # ------------------------------------------------------------------------------
    # -------------------------------- PARSING INPUTS ------------------------------
    # ------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", "-p",   type=str, default=os.getcwd(),
                        help="case full path (default cwd)")
    parser.add_argument("--ncfile", "-f", type=str, default=None,
                        help="netcdf sampling planes")
    parser.add_argument("--dt", "-dt",    type=float, default=None,
                        help="time step for adding datetime to xarray (optional)")
    parser.add_argument("--group", "-g",  type=str, default=None,
                        help="group within netcdf file to be read, if more than one is available")
    parser.add_argument("--itime", "-itime",  type=int, default=0,
                        help="sampling time step to start saving the data")
    parser.add_argument("--ftime", "-ftime",  type=int, default=-1,
                        help="sampling time step to end saving the data")
    parser.add_argument("--steptime", "-step",  type=int, default=1,
                        help="sampling time step increment to save the data")

    args = parser.parse_args()

    # Parse inputs
    path     = args.path
    ncfile   = args.ncfile
    dt       = args.dt
    group    = args.group
    itime    = args.itime
    ftime    = args.ftime
    steptime = args.steptime

    # ------------------------------------------------------------------------------
    # --------------------------- DONE PARSING INPUTS ------------------------------
    # ------------------------------------------------------------------------------

    if path == '.':
        path = os.getcwd()

    # We assume the case path was given, but maybe it was $case/post* or $case/proc*
    # Let's check for that and fix it
    if os.path.basename(path) == 'post_processing' or os.path.basename(path) == 'processedData':
        path = os.path.split(path)[0]

    pppath = os.path.join(path,'post_processing')
    outpath = os.path.join(path,'processedData')

    # ------- PERFORM CHECKS
    if not os.path.exists(path):
        parser.error(f"Path {path} does not exist.")

    if not os.path.exists(pppath):
        raise ValueError (f'Path {pppath} does not exist.')

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if isinstance(ncfile,str):
        if not ncfile.endswith('.nc'):
            raise ValueError (f"Received single string as ncfiles, but it does not end with .nc.")
        if not os.path.isfile(os.path.join(pppath,ncfile)):
            raise ValueError(f"File {ncfile} does not exist.")
    else:
        raise ValueError(f"the ncfile should be a string. Received {ncfile}.")

    if dt is not None and  not isinstance(dt,(float,int)):
        raise ValueError(f'dt should be a scalar.')

    if steptime < 1:
        raise ValueError(f'The time step increment should be >= 1.')

    if itime < 0:
        raise ValueError(f'The initial time step should be >= 0.')

    if ftime != -1:
        if ftime < itime:
            raise ValueError(f'The final time step should be larger than the'\
                             f'initial. Received itime={itime} and ftime={ftime}.')
    if itime!=0 or ftime!=-1:
        if ((ftime-itime)/steptime)%8 !=0:
            raise ValueError(f'Due to the inner workings of zarr and the default'\
                             f'chunking, the amount of times requested should be a'\
                             f'multiple of 8. Received {int((ftime-itime)/steptime)}')
        if dt is not None:
            raise ValueError(f'Performing chunked saving. Computing fluctuating'\
                             f'component is not supported. Skip dt specification.')

    # ------------------------------------------------------------------------------
    # ---------------------------- DONE WITH CHECKS --------------------------------
    # ------------------------------------------------------------------------------

    print(f'Reading {path}/{ncfile}')

    print(f'Starting job at {time.ctime()}')
    freeze_support()
    main(ncfile, pppath, outpath, dt, itime, ftime, steptime)
    print(f'Ending job at   {time.ctime()}')

