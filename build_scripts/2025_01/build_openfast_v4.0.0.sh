#!/bin/bash

# Build script for OpenFAST 4.0.0 using Intel oneAPI compilers.
# Regis Thedin, 2025-01-09

# NOTE: From Andy, there is a bug with OpenMP and intel that should be fixed on.
#       For now, disable OpenMP. 2025-01-09

module purge
module load PrgEnv-intel/8.5.0
module load intel-oneapi-mkl/2024.0.0-intel
module load intel-oneapi
module load binutils
module load hdf5/1.14.3-intel-oneapi-mpi-intel
module load netcdf-c/4.9.2-intel-oneapi-mpi-intel

module list

cmake .. \
    -DCMAKE_Fortran_COMPILER=ifx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_FLAGS=-fPIC \
    -DCMAKE_C_FLAGS=-fPIC \
    -DBUILD_OPENFAST_CPP_API=ON \
    -DBUILD_FASTFARM=ON \
    -DDOUBLE_PRECISION:BOOL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENMP=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install

nice make -j48
make install

