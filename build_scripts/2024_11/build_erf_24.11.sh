#!/bin/bash

# Build script for ERF (any version) using Intel oneAPI compilers. Includes stall libraries.
# Regis Thedin, 2024-11-15

module purge
module load PrgEnv-intel/8.5.0
module load intel-oneapi-mkl/2024.0.0-intel
module load intel-oneapi/2023.2.0
module load netcdf-c/4.9.2-intel-oneapi-mpi-intel
module load binutils/2.41

module list

export LD_LIBRARY_PATH=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel:$LD_LIBRARY_PATH
export LD_PRELOAD=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpi_intel.so.12:/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpifort_intel.so.12
export MPICH_VERSION_DISPLAY=1
export MPICH_ENV_DISPLAY=1
export MPICH_OFI_CXI_COUNTER_REPORT=2
export FI_MR_CACHE_MONITOR=memhooks
export FI_CXI_RX_MATCH_MODE=software
export MPICH_SMP_SINGLE_COPY_MODE=NONE

echo $LD_LIBRARY_PATH |tr ':' '\n'

cmake .. \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DCMAKE_C_COMPILER:STRING=mpicc \
    -DCMAKE_CXX_COMPILER:STRING=mpicxx \
    -DCMAKE_Fortran_COMPILER:STRING=mpif90 \
    -DERF_DIM:STRING=3 \
    -DERF_ENABLE_MPI:BOOL=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON \
    -DERF_ENABLE_TESTS:BOOL=ON \
    -DERF_ENABLE_FCOMPARE:BOOL=ON \
    -DERF_ENABLE_DOCUMENTATION:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install

nice make -j48
make install

