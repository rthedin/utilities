#!/home/rthedin/.conda/envs/ssrs_env_scratch/bin/python

#SBATCH --job-name=pp_vtk
#SBATCH --output amr2post_vtk.log.%j
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --account=shellwind
#SBATCH --mem=160G
# #SBATCH --qos=high

# ------------------------------------------------------------------------------ #
# ff_readTurbineOutput.py                                                        #
#                                                                                #
# Read turbine output in chunks.
#                                                                                #
# Usage:                                                                         #
# ff_readTurbineOutput.py -c <caseobj> -dtop <dt_openfast> -dtpr <dt_processing> #
#                         [-iCondition <icond>, -fCondition <fcond>              #
#                          -iCase <icase> -fCase <fcase>                         #
#                          -iSeed <iseed> -fSeed <fseed>                         #
#                          -iTurbine <iturb> -fTurbine <fturb>]                  #
#                                                                         #
# Example usage:                                                          #
#    sbatch ff_readTurbineOutput.py [-J <jobname>] -p $path              #
#                                   -f lowres40000.nc -g Low              #
#    sbatch postprocess_boxes2vtk.py [-J <jobname>] -p $path              #
#                    -f highres40000.nc -g HighT1_inflow0deg              #
#                                                                         #
# Regis Thedin                                                                   #
# Mar 30, 2023                                                                   #
# regis.thedin@nrel.gov                                                          #
# ------------------------------------------------------------------------------ #

import argparse
import numpy as np
import os, sys, time
from itertools import repeat
from multiprocessing import Pool, freeze_support
sys.path.append(os.path.abspath('/home/rthedin/utilities/'))
from helper_fastfarm import readTurbineOutput

def main(caseobj, dt_openfast, dt_processing, saveOutput, 
         iCondition, fCondition, iCase, fCase, iSeed, fSeed, iTurbine, fTurbine):


    # Split all the time steps in arrays of roughly the same size
    #chunks =  np.array_split(range(iCase,fCase), 36)
    chunks =  np.array_split(range(iCase,fCase), 4)
    # Now, get the beginning and end of each separate chunk
    iCase_list = [i[0]    for i in chunks]
    fCase_list = [i[-1]+1 for i in chunks] 
    print(f'iCase_list is {iCase_list}')
    print(f'fCase_list is {fCase_list}')

    p = Pool()
    ds_ = p.starmap(readTurbineOutput, zip(repeat(caseobj),          # caseobj
                                           repeat(dt_openfast),      # dt_openfast
                                           repeat(dt_processing),    # dt_processing
                                           repeat(False),            # saveOutput
                                           repeat(iCondition),       # iCondition
                                           repeat(fCondition),       # fCondition
                                           iCase_list,               # iCase
                                           fCase_list,               # fCase
                                           repeat(iSeed),            # iSeed
                                           repeat(fSeed),            # fSeed
                                           repeat(iTurbine),         # iTurbine
                                           repeat(fTurbine),         # fTurbine
                                          )
                                       )

    print(f'Done reading all output. Concatenating the arrays')
    comb_ds = xr.combine_by_coords(ds_)
    comb_ds.to_zarr(output)

    print('Finished.')

if __name__ == '__main__':

    # ------------------------------------------------------------------------------
    # -------------------------------- PARSING INPUTS ------------------------------
    # ------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("--caseobj", "-c",   type=str, default=None,
                        help="Case FASTFarmCaseCreation object")
    parser.add_argument("--dt_openfast", "-dtop", type=float, default=None,
                        help="OpenFAST time step")
    parser.add_argument("--dt_processing", "-dtpr", type=float, default=1.0,
                        help="Processing time step")
    parser.add_argument("--saveOutput", "-saveout", type=bool, default=True,
                        help="Whether or not to save the output to a zarr file")
    parser.add_argument("--iCondition", "-iConditions", type=float, default=0,
                        help="Initial condition to read (default: 0)")
    parser.add_argument("--fCondition", "-fConditions", type=float, default=-1,
                        help="Final condition to read (default: -1)")
    parser.add_argument("--iCases", "-iCases",          type=float, default=0,
                        help="Initial case to read (default: 0)")
    parser.add_argument("--fCases", "-fCases",          type=float, default=-1,
                        help="Final case to read (default: -1)")
    parser.add_argument("--iSeeds", "-iSeeds",          type=float, default=0,
                        help="Initial seed to read (default: 0)")
    parser.add_argument("--fSeeds", "-fSeeds",          type=float, default=-1,
                        help="Final seed to read (default: -1)")
    parser.add_argument("--iTurbines", "-iTurbines",    type=float, default=0,
                        help="Initial turbine to read (default: 0)")
    parser.add_argument("--fTurbines", "-fTurbines",    type=float, default=-1,
                        help="Final turbine to read (default: -1)")

    args = parser.parse_args()

    # Parse inputs
    caseobj       = args.c
    dt_openfast   = args.dtop
    dt_processing = args.dtpr
    saveOutput    = args.saveout
    iCondition    = args.iCondition
    fCondition    = args.fCondition
    iCase         = args.iCase
    fCase         = args.fCase
    iSeed         = args.iSeed
    fSeed         = args.fSeed
    iTurbine      = args.iTurbine
    fTurbine      = args.fTurbine

    # ------------------------------------------------------------------------------
    # --------------------------- DONE PARSING INPUTS ------------------------------
    # ------------------------------------------------------------------------------

    # ------- PERFORM CHECKS
    outputzarr = os.path.join(caseobj.path,f'ds_turbineOutput_{dt_processing}s.zarr')

    if os.path.exists(outputzarr):
        raise ValueError ('The requested target zarr store directory exists. Stopping.')

    if dt_openfast is not None and not isinstance(dt_openfast,(float,int)):
        raise ValueError(f'dt_openfast should be a scalar.')
    if dt_processing is not None and not isinstance(dt_processing,(float,int)):
        raise ValueError(f'dt_processing should be a scalar.')

    # ------------------------------------------------------------------------------
    # ---------------------------- DONE WITH CHECKS --------------------------------
    # ------------------------------------------------------------------------------

    print(f'Starting job at {time.ctime()}')
    freeze_support()
    main(caseobj, dt_openfast, dt_processing, saveOutput, iCondition, fCondition, iCase, fCase, iSeed, fSeed, iTurbine, fTurbine)
    print(f'Ending job at   {time.ctime()}')

