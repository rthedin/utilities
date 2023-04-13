# Collections of functions that help perform routine tasks related to FAST.Farm
#
# Add to notebooks using
#   sys.path.append(os.path.abspath('/home/rthedin/utilities/'))
#   from helper_fastfarm import readFFPlanes
#
# Regis Thedin
#

import numpy as np
import pandas as pd
import xarray as xr
import os, sys
from multiprocessing import Pool
from itertools import repeat

from pyFAST.input_output import TurbSimFile, FASTOutputFile, VTKFile, FASTInputFile


def readTurbineOutputPar(caseobj, dt_openfast, dt_processing, saveOutput=True,
                         iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1, iTurbine=0, fTurbine=-1,
                         nCores=36):
    '''
    Inputs
    ------
    
    Zero-indexed initial and final values for conditions, cases, seeds, and turbines. 


    '''


    #from multiprocessing import set_start_method
    #try:
    #    set_start_method("spawn")
    #except RuntimeError:
    #    print(f'Fell into RunTime error on `set_start_method("spawn")`. Continuing..\n')


    if fCondition==-1:
        fCondition = caseobj.nConditions
    if fCondition-iCondition <= 0:
        raise ValueError (f'Final condition to read needs to be larger than initial.')

    if fCase==-1:
        fCase = caseobj.nCases
    if fCase-iCase <= 0:
        raise ValueError (f'Final case to read needs to be larger than initial.')

    if fSeed==-1:
        fSeed = caseobj.nSeeds
    if fSeed-iSeed <= 0:
        raise ValueError (f'Final seed to read needs to be larger than initial.')

    if fTurbine==-1:
        fTurbine = caseobj.nTurbines
    if fTurbine-iTurbine <= 0:
        raise ValueError (f'Final turbine to read needs to be larger than initial.')

    if fCase-iCase < nCores:
        print(f'Total number of cases requested ({fCase-iCase}) is lower than number of cores {nCores}.')
        print(f'Changing the number of cores to {fCase-iCase}.')
        nCores = fCase-iCase

    zarrstore = f'ds_turbineOutput_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}_turb{iTurbine}_{fTurbine}_dt{dt_processing}s.zarr'
    outputzarr = os.path.join(caseobj.path, zarrstore)

    if os.path.isdir(outputzarr) and saveOutput:
       print(f'Output file {zarrstore} exists. Attempting to read it..')
       comb_ds = xr.open_zarr(outputzarr)
       return comb_ds
        

    print(f'Running readTurbineOutput in parallel using {nCores} workers')

    # Split all the cases in arrays of roughly the same size
    chunks =  np.array_split(range(iCase,fCase), nCores)
    # Now, get the beginning and end of each separate chunk
    iCase_list = [i[0]    for i in chunks]
    fCase_list = [i[-1]+1 for i in chunks] 
    print(f'iCase_list is {iCase_list}')
    print(f'fCase_list is {fCase_list}')

    p = Pool()
    ds_ = p.starmap(readTurbineOutput, zip(repeat(caseobj),          # caseobj
                                           repeat(dt_openfast),      # dt_openfast
                                           repeat(dt_processing),    # dt_processing
                                           repeat(False),       # saveOutput
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

    if saveOutput:
        pass
        print('Done concatenating. Saving zarr file.')
        comb_ds.to_zarr(outputzarr)

    print('Finished.')

    return comb_ds

def readTurbineOutput(caseobj, dt_openfast, dt_processing=1, saveOutput=True, 
                      iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1, iTurbine=0, fTurbine=-1):
    '''
    caseobj: FASTFarmCaseSetup object
        Object containing all the case information
    dt_openfast: scalar
        OpenFAST time step
    dt_processing: scalar
        Time step to which the processing will be saved. Default=1
    saveOutput: bool
        Whether or not to save the output to a zarr file
    '''
    
    if fCondition==-1:
        fCondition = caseobj.nConditions
    #else:
    #    fCondition += 1  # The user sets the last desired condition. This if for the np.arange.
    if fCondition-iCondition <= 0:
        raise ValueError (f'Final condition to read needs to be larger than initial.')

    if fCase==-1:
        fCase = caseobj.nCases
    #else:
        #fCase += 1  # The user sets the last desired case. This if for the np.arange.
    if fCase-iCase <= 0:
        raise ValueError (f'Final case to read needs to be larger than initial.')

    if fSeed==-1:
        fSeed = caseobj.nSeeds
    #else:
    #    fSeed += 1 # The user sets the last desired seed. This is for the np.arange
    if fSeed-iSeed <= 0:
        raise ValueError (f'Final seed to read needs to be larger than initial.')

    if fTurbine==-1:
        fTurbine = caseobj.nTurbines
    #else:
    #    fTurbine += 1  # The user sets the last desired turbine. This if for the np.arange.
    if fTurbine-iTurbine <= 0:
        raise ValueError (f'Final turbine to read needs to be larger than initial.')


    zarrstore = f'ds_turbineOutput_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}_turb{iTurbine}_{fTurbine}_dt{dt_processing}s.zarr'
    outputzarr = os.path.join(caseobj.path,zarrstore)
    
    # Read or process turbine output
    if os.path.isdir(outputzarr):
        # Data already processed. Reading output
        turbs = xr.open_zarr(outputzarr) 
    else: 
        # Processed data not saved. Reading it
        dt_ratio = int(dt_processing/dt_openfast)

        turbs_cond = []
        for cond in np.arange(iCondition, fCondition, 1):
            turbs_case = []
            for case in np.arange(iCase, fCase, 1):
                turbs_seed = []
                for seed in np.arange(iSeed, fSeed, 1):
                    turbs_t=[]
                    for t in np.arange(iTurbine, fTurbine, 1):
                        print(f'Processing Condition {cond}, Case {case}, Seed {seed}, turbine {t+1}')
                        ff_file = os.path.join(caseobj.path, caseobj.condDirList[cond], caseobj.caseDirList[case], f'Seed_{seed}', f'FFarm_mod.T{t+1}.outb')
                        df   = FASTOutputFile(ff_file).toDataFrame()
                        ds_t = df.rename(columns={'Time_[s]':'time'}).set_index('time').to_xarray()
                        ds_t = ds_t.isel(time=slice(0,None,dt_ratio))
                        ds_t = ds_t.expand_dims(['cond','case','seed','turbine']).assign_coords({'cond': [caseobj.condDirList[cond]],
                                                                                                 'case':[caseobj.caseDirList[case]],
                                                                                                 'seed':[seed],
                                                                                                 'turbine': [t+1]})
                        turbs_t.append(ds_t)
                    turbs_t = xr.concat(turbs_t,dim='turbine')
                    turbs_seed.append(turbs_t)
                turbs_seed = xr.concat(turbs_seed,dim='seed')
                turbs_case.append(turbs_seed)
            turbs_case = xr.concat(turbs_case,dim='case')
            turbs_cond.append(turbs_case)
        turbs_cond = xr.concat(turbs_cond, dim='cond')

        # Rename variables to get rid of problematic characters ('-','/')
        varlist = list(turbs_cond.keys())
        varlistnew = [i.replace('/','_per_').replace('-','') for i in varlist]
        renameDict = dict(zip(varlist, varlistnew))
        turbs = turbs_cond.rename_vars(renameDict)

        if saveOutput:
            print(f'Saving output {zarrstore}...')
            turbs.to_zarr(outputzarr)
            print(f'Saving output {zarrstore}... Done.')
    
    return turbs






def readFFPlanesPar(caseobj, sliceToRead, verbose=False, saveOutput=True, iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1, nCores=36):

    if fCondition==-1:
        fCondition = caseobj.nConditions
    if fCondition-iCondition <= 0:
        raise ValueError (f'Final condition to read needs to be larger than initial.')

    if fCase==-1:
        fCase = caseobj.nCases
    if fCase-iCase <= 0:
        raise ValueError (f'Final case to read needs to be larger than initial.')

    if fSeed==-1:
        fSeed = caseobj.nSeeds
    if fSeed-iSeed <= 0:
        raise ValueError (f'Final seed to read needs to be larger than initial.')

    if fCase-iCase < nCores:
        print(f'Total number of cases requested ({fCase-iCase}) is lower than number of cores {nCores}.')
        print(f'Changing the number of cores to {fCase-iCase}.')
        nCores = fCase-iCase

    zarrstore = f'ds_{sliceToRead}Slices_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}.zarr'
    outputzarr = os.path.join(caseobj.path, zarrstore)

    if os.path.isdir(outputzarr) and saveOutput:
       print(f'Output file {zarrstore} exists. Attempting to read it..')
       comb_ds = xr.open_zarr(outputzarr)
       return comb_ds
        

    print(f'Running readFFPlanes in parallel using {nCores} workers')

    # Split all the cases in arrays of roughly the same size
    chunks =  np.array_split(range(iCase,fCase), nCores)
    # Now, get the beginning and end of each separate chunk
    iCase_list = [i[0]    for i in chunks]
    fCase_list = [i[-1]+1 for i in chunks] 
    print(f'iCase_list is {iCase_list}')
    print(f'fCase_list is {fCase_list}')

    p = Pool()
    ds_ = p.starmap(readFFPlanes, zip(repeat(caseobj),          # caseobj
                                      repeat(sliceToRead),      # slicesToRead
                                      repeat(verbose),            # verbose
                                      repeat(False),            # saveOutput
                                      repeat(iCondition),       # iCondition
                                      repeat(fCondition),       # fCondition
                                      iCase_list,               # iCase
                                      fCase_list,               # fCase
                                      repeat(iSeed),            # iSeed
                                      repeat(fSeed),            # fSeed
                                     )
                                  )

    print(f'Done reading all output. Concatenating the arrays')
    comb_ds = xr.combine_by_coords(ds_)

    if saveOutput:
        pass
        print('Done concatenating. Saving zarr file.')
        comb_ds.to_zarr(outputzarr)

    print('Finished.')

    return comb_ds









def readFFPlanes(caseobj, slicesToRead=['x','y','z'], verbose=False, saveOutput=True, iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1):
    '''
    Read and process FAST.Farm planes into xarrays.

    '''

    if fCondition==-1:
        fCondition = caseobj.nConditions
    #else:
    #    fCondition += 1  # The user sets the last desired condition. This if for the np.arange.
    if fCondition-iCondition <= 0:
        raise ValueError (f'Final condition to read needs to be larger than initial.')

    if fCase==-1:
        fCase = caseobj.nCases
    #else:
        #fCase += 1  # The user sets the last desired case. This if for the np.arange.
    if fCase-iCase <= 0:
        raise ValueError (f'Final case to read needs to be larger than initial.')

    if fSeed==-1:
        fSeed = caseobj.nSeeds
    #else:
    #    fSeed += 1 # The user sets the last desired seed. This is for the np.arange
    if fSeed-iSeed <= 0:
        raise ValueError (f'Final seed to read needs to be larger than initial.')


    print(f'Requesting to save {slicesToRead} slices')

    #if nConditions is None:
    #    nConditions = caseobj.nConditions
    #else:
    #    if nConditions > caseobj.nConditions:
    #        print(f'WARNING: Requested {nConditions} conditions, but only {caseobj.nConditions} are available. Reading {caseobj.nConditions} conditions')
    #        nConditions = caseobj.nConditions

    #if nCases is None:
    #    nCases = caseobj.nCases
    #else:
    #    if nCases > caseobj.nCases:
    #        print(f'WARNING: Requested {nCases} cases, but only {caseobj.nCases} are available. Reading {caseobj.nCases} cases')
    #        nCases = caseobj.nCases

    #if nSeeds is None:
    #    nSeeds = caseobj.nSeeds
    #else:
    #    if nSeeds > caseobj.nSeeds:
    #        print(f'WARNING: Requested {nSeeds} seeds, but only {caseobj.nSeeds} are available. Reading {caseobj.nSeeds} seeds')
    #        nSeeds = caseobj.nSeeds
    
    # Read all VTK output for each plane and save an nc files for each normal. Load files if present.
    for slices in slicesToRead:
    
        zarrstore = f'ds_{slices}Slices_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}.zarr'
        outputzarr = os.path.join(caseobj.path, zarrstore)

        if os.path.isdir(outputzarr):
            if len(slicesToRead) > 1:
                print(f"!! WARNING: Asked for multiple slices. Returning only the first one, {slices}\n",
                      f"           To load other slices, request `slicesToRead='y'`")
            print(f'Processed output for slice {slices} found. Loading it.')
                
            # Data already processed. Reading output
            Slices = xr.open_zarr(outputzarr)

            return Slices
            
        else:
            
            # This for-loop is due to memory allocation requirements
            #print(f'Processing slices normal in the {slices} direction...')
            Slices_cond = []
            for cond in np.arange(iCondition, fCondition, 1):
                Slices_case = []
                for case in np.arange(iCase, fCase, 1):
                    Slices_seed = []
                    for seed in np.arange(iSeed, fSeed, 1):
                        print(f'Processing {slices} slice: Condition {cond}, Case {case}, Seed {seed}')
                        seedPath = os.path.join(caseobj.path, caseobj.condDirList[cond], caseobj.caseDirList[case], f'Seed_{seed}')

                        # Read FAST.Farm input to determine outputs
                        ff_file = FASTInputFile(os.path.join(seedPath,'FFarm_mod.fstf'))

                        tmax          = ff_file['TMax']
                        NOutDisWindXY = ff_file['NOutDisWindXY']
                        OutDisWindZ   = ff_file['OutDisWindZ']
                        NOutDisWindYZ = ff_file['NOutDisWindYZ']
                        OutDisWindX   = ff_file['OutDisWindX']
                        NOutDisWindXZ = ff_file['NOutDisWindXZ']
                        WrDisDT       = ff_file['WrDisDT']

                        # Determine number of output VTKs
                        nOutputTimes = int(np.floor(tmax/WrDisDT))

                        # Determine number of output digits for reading
                        ndigits = len(str(max(NOutDisWindXY,NOutDisWindXZ,NOutDisWindYZ)))

                        if slices == 'z':
                            # Read Low-res z-planes
                            Slices=[]
                            for zplane in range(NOutDisWindXY):
                                Slices_t=[]
                                for t in range(nOutputTimes):
                                    file = f'FFarm_mod.Low.DisXY{zplane+1:0{ndigits}d}.{t:05d}.vtk'
                                    if verbose: print(f'Reading z plane {zplane} for time step {t}: \t {file}')

                                    vtk = VTKFile(os.path.join(seedPath, 'vtk_ff', file))
                                    ds = readAndCreateDataset(vtk, caseobj, cond=cond, case=case, seed=seed, t=t, WrDisDT=WrDisDT)

                                    Slices_t.append(ds)
                                Slices_t = xr.concat(Slices_t,dim='time')
                                Slices.append(Slices_t)
                            Slices = xr.concat(Slices,dim='z')

                        elif slices == 'y':
                            # Read Low-res y-planes
                            Slices=[]
                            for yplane in range(NOutDisWindXZ):
                                Slices_t=[]
                                for t in range(nOutputTimes):
                                    file = f'FFarm_mod.Low.DisXZ{yplane+1:0{ndigits}d}.{t:05d}.vtk'
                                    if verbose: print(f'Reading y plane {yplane} for time step {t}: \t {file}')

                                    vtk = VTKFile(os.path.join(seedPath, 'vtk_ff', file))
                                    ds = readAndCreateDataset(vtk, caseobj, cond=cond, case=case, seed=seed, t=t, WrDisDT=WrDisDT)

                                    Slices_t.append(ds)
                                Slices_t = xr.concat(Slices_t,dim='time')
                                Slices.append(Slices_t)
                            Slices = xr.concat(Slices,dim='y')

                        elif slices == 'x':
                            # Read Low-res x-planes
                            Slices=[]
                            for xplane in range(NOutDisWindYZ):
                                Slices_t=[]
                                print(f'Processing {slices} slice: Condition {cond}, Case {case}, Seed {seed}, x plane {xplane}')
                                for t in range(nOutputTimes):
                                    file = f'FFarm_mod.Low.DisYZ{xplane+1:0{ndigits}d}.{t:05d}.vtk'
                                    if verbose: print(f'Reading x plane {xplane} for time step {t}: \t {file}')

                                    vtk = VTKFile(os.path.join(seedPath, 'vtk_ff', file))
                                    ds = readAndCreateDataset(vtk, caseobj, cond=cond, case=case, seed=seed, t=t, WrDisDT=WrDisDT)

                                    Slices_t.append(ds)
                                Slices_t = xr.concat(Slices_t,dim='time')
                                Slices.append(Slices_t)
                            Slices = xr.concat(Slices,dim='x')

                        else:
                            raise ValueError(f'Only slices x, y, z are available. Slice {slices} was requested. Stopping.')

                        Slices_seed.append(Slices)
                    Slices_seed = xr.concat(Slices_seed, dim='seed')
                    Slices_case.append(Slices_seed)
                Slices_case = xr.concat(Slices_case, dim='case')
                Slices_cond.append(Slices_case)

            Slices = xr.concat(Slices_cond, dim='cond')

            if saveOutput:
                print(f'Saving {slices} slice file...')
                Slices.to_zarr(outputzarr)

            if len(slicesToRead) == 1:
                # Single slice was requested
                print(f'Since single slice was requested, returning it.')
                return Slices


def readAndCreateDataset(vtk, caseobj, cond=None, case=None, seed=None, t=None, WrDisDT=None):
    
    # Get info from VTK
    x = vtk.xp_grid
    y = vtk.yp_grid
    z = vtk.zp_grid
    u = vtk.point_data_grid['Velocity'][:,:,:,0]
    v = vtk.point_data_grid['Velocity'][:,:,:,1]
    w = vtk.point_data_grid['Velocity'][:,:,:,2]
    
    if t is None and WrDisDT is None:
        t=1
        WrDisDT = 1

    ds = xr.Dataset({
            'u': (['x', 'y', 'z'], u),
            'v': (['x', 'y', 'z'], v),
            'w': (['x', 'y', 'z'], w), },
           coords={
            'x': (['x'], x),
            'y': (['y'], y),
            'z': (['z'], z),
            'time': [t*WrDisDT] },
          )
    
    if cond is not None:  ds = ds.expand_dims('cond').assign_coords({'cond': [caseobj.condDirList[cond]]})
    if case is not None:  ds = ds.expand_dims('case').assign_coords({'case': [caseobj.caseDirList[case]]})
    if seed is not None:  ds = ds.expand_dims('seed').assign_coords({'seed': [seed]})           
        
    return ds        


