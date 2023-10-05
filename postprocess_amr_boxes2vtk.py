#!/home/rthedin/.conda/envs/ssrs_env_scratch/bin/python

#SBATCH --job-name=pp_vtk
#SBATCH --output amr2post_vtk.log.%j
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --account=shellwind
#SBATCH --mem=160G
# #SBATCH --qos=high

# ----------------------------------------------------------------------- #
# postprocess_boxes2vtk.py                                                #
#                                                                         #
# Post process AMR-Wind output boxes for FAST.Farm and save vtk files for #
# each group (each individual sampling box). This script should be called #
# once for each output netcdf file and each group. If group is not given, #
# it expects only one (e.g. low box); otherwise, the script will stop and #
# warn the user. The output vtk is saved in $case/processedData/<group>.  #
#                                                                         #
# Usage:                                                                  #
# postprocess_boxes2vtk.py -p <fullpath> -f <ncfile> [-g <group>]         #
#                                                                         #
# Example usage:                                                          #
#    cd /full/path/to/amrwind/case                                        #
#    path=$(pwd -P)                                                       #
#    cd post_processing                                                   #
#    ls  # see what sampling box files are available.                     #
#    sbatch postprocess_boxes2vtk.py [-J <jobname>] -p $path              #
#                                   -f lowres40000.nc -g Low              #
#    sbatch postprocess_boxes2vtk.py [-J <jobname>] -p $path              #
#                    -f highres40000.nc -g HighT1_inflow0deg              #
#                                                                         #
# Regis Thedin                                                            #
# Mar 13, 2023                                                            #
# regis.thedin@nrel.gov                                                   #
# ----------------------------------------------------------------------- #

import argparse
import numpy as np
import os, time
from itertools import repeat
from multiprocessing import Pool, freeze_support
from windtools.amrwind.post_processing  import Sampling

def main(samplingboxfile, pppath, requestedgroup, outpath, dt, t0, itime, ftime, steptime, offsetz):

    # -------- CONFIGURE RUN
    samplingboxpath = os.path.join(pppath,samplingboxfile)
    s = Sampling(samplingboxpath)

    groups = s.groups

    if requestedgroup is None:
        if len(groups) > 1:
            raise ValueError(f'The ncfile {samplingboxfile} contains more than one '\
                             f'group: {groups}. Specify -g <groupstring> for single group.')
        else:
            group = group[0]
            print(f'No group requested. Saving the only one present, {group}.')
    elif requestedgroup in groups:
        group = requestedgroup
        print(f'Saving VTK related to requested group {group}')
    else:
        raise ValueError(f'Requested group {requestedgroup} is not valid. '\
                         f'Value groups: {groups}')

    s.getGroupProperties_xr(group=group)
    outpathgroup = os.path.join(outpath, group)
    if not os.path.exists(outpathgroup):
        # Due to potential race condition, adding exist_ok flag as well
        os.makedirs(outpathgroup, exist_ok=True)

    print(f'Group {group} has {s.ndt} saved time steps.')
    if itime==0 and ftime==-1:
        print(f'  Saving all of them.')
    else:
        print(f'  Saving the time steps between {itime} and {ftime}, out of {s.ndt}.')

    if ftime > s.ndt:
        print(f'Final time step requested {ftime} is larger than what is available in'\
              f'the array, {s.ndt}. Saving up to {s.ndt} instead.')
        ftime = s.ndt
    if ftime == -1:
        ftime = s.ndt

    # Split all the time steps in arrays of roughly the same size
    chunks =  np.array_split(range(itime,ftime), 36)
    # The lists itime_list and ftime_list below will fail if ftime-itime is less than 36, since a
    # non-homogeneous numpy array will be created. If that is the case, let's issue a warning and 
    # end the program here.
    if ftime-itime < 36:
        raise ValueError(f'The number of boxes is lower than the number of cores. Stopping. A fix '\
                         f'for this error is to provide a new value for the number of nodes in the '\
                         f'2_saveVTK script. For example, for the low box, selection of nNodes_low '\
                         f'needs to be such that (ftime_low-itime_low)/nNodes_low is larger than 36.')

    # Now, get the beginning and end of each separate chunk
    itime_list = [i[0]    for i in chunks]
    ftime_list = [i[-1]+1 for i in chunks] 
    print(f'itime_list is {itime_list}')
    print(f'ftime_list is {ftime_list}')

    p = Pool()
    ds_ = p.starmap(s.to_vtk, zip(repeat(group),         # group
                                  repeat(outpathgroup),  # outputPath
                                  repeat(True),          # verbose
                                  repeat(offsetz),       # offset in z
                                  itime_list,            # itime
                                  ftime_list,            # ftime
                                  repeat(t0),            # t0
                                  repeat(dt)             # dt
                                 )
                              )
    print('Finished.')
    

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
                        help="time step for naming the boxes output")
    parser.add_argument("--initialtime", "-t0", default=None,
                        help="Time step related to first box output (optional)")
    parser.add_argument("--group", "-g",  type=str, default=None,
                        help="group within netcdf file to be read, if more than one is available")
    parser.add_argument("--itime", "-itime",  type=int, default=0,
                        help="sampling time step to start saving the data")
    parser.add_argument("--ftime", "-ftime",  type=int, default=-1,
                        help="sampling time step to end saving the data")
    parser.add_argument("--steptime", "-step",  type=int, default=1,
                        help="sampling time step increment to save the data")
    parser.add_argument("--offsetz", "-offsetz",  type=float, default=None,
                        help="Offset in the x direction, ensuring a point at hub height")

    args = parser.parse_args()

    # Parse inputs
    path     = args.path
    ncfile   = args.ncfile
    dt       = args.dt
    t0       = args.initialtime
    group    = args.group
    itime    = args.itime
    ftime    = args.ftime
    steptime = args.steptime
    offsetz  = args.offsetz

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

    if t0 is not None and  not isinstance(t0,(float,int)):
        raise ValueError(f't0 should be a scalar.')

    if steptime < 1:
        raise ValueError(f'The time step increment should be >= 1.')

    if itime < 0:
        raise ValueError(f'The initial time step should be >= 0.')

    if ftime != -1:
        if ftime < itime:
            raise ValueError(f'The final time step should be larger than the'\
                             f'initial. Received itime={itime} and ftime={ftime}.')

    if offsetz is None:
        print(f'!!! WARNING: no offset in z has been given. Ensure you will have a point at hub height.')
        offsetz=0

    # ------------------------------------------------------------------------------
    # ---------------------------- DONE WITH CHECKS --------------------------------
    # ------------------------------------------------------------------------------

    print(f'Reading {path}/{ncfile}')

    print(f'Starting job at {time.ctime()}')
    freeze_support()
    main(ncfile, pppath, group, outpath, dt, t0, itime, ftime, steptime, offsetz)
    print(f'Ending job at   {time.ctime()}')

