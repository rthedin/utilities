#!/bin/bash

# Build script for ROSCO 2.7 using Intel oneAPI compilers.
# Regis Thedin, 2024-11-12

module purge
module load PrgEnv-intel/8.5.0
module load intel-oneapi-mkl/2024.0.0-intel
module load intel-oneapi
module load binutils
module load hdf5/1.14.3-intel-oneapi-mpi-intel

module list
 
cmake .. \
    -DCMAKE_Fortran_COMPILER=ifx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_FLAGS=-fPIC \
    -DCMAKE_C_FLAGS=-fPIC \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install

nice make
make install

