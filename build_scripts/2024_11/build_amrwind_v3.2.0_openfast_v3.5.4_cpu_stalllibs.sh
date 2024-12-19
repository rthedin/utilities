#!/bin/bash

# Build script for AMR-Wind on CPU using gcc compilers. Includes stall libraries.
# Compatible with OpenFAST 3.5.4 compiled using gcc.
# Regis Thedin, 2024-11-12

module purge
module load PrgEnv-intel/8.5.0
module load netcdf/4.9.2-intel-oneapi-mpi-intel
module load netlib-scalapack/2.2.0-gcc

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
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DMPI_Fortran_COMPILER=mpifort \
    -DCMAKE_Fortran_COMPILER=ifort \
    -DCMAKE_CXX_COMPILER=icpc \
    -DCMAKE_C_COMPILER=icc \
    -DAMR_WIND_ENABLE_CUDA:BOOL=OFF \
    -DAMR_WIND_ENABLE_MPI:BOOL=ON \
    -DAMR_WIND_ENABLE_OPENMP:BOOL=OFF \
    -DAMR_WIND_TEST_WITH_FCOMPARE:BOOL=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMR_WIND_ENABLE_NETCDF:BOOL=ON \
    -DAMR_WIND_ENABLE_HYPRE:BOOL=OFF \
    -DAMR_WIND_ENABLE_MASA:BOOL=OFF \
    -DAMR_WIND_ENABLE_TESTS:BOOL=ON \
    -DAMR_WIND_ENABLE_ALL_WARNINGS:BOOL=ON \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DAMR_WIND_ENABLE_OPENFAST:BOOL=ON \
    -DOpenFAST_ROOT:PATH=/projects/tcwnd/rthedin/repos/openfast_v3.5.4_2024_11_11/openfast/build_cpu/install \
    -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install

nice make -j48
make install

