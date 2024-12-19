#!/bin/bash

# Build script for OpenFAST 3.5.4 using gcc compilers.
# Regis Thedin, 2024-11-11

module purge
module load intel-oneapi-compilers/2023.2.0
module load intel-oneapi-mpi/2021.10.0-intel
module load intel-oneapi-mkl/2023.2.0-intel
module load hdf5/1.14.3-intel-oneapi-mpi-intel
module load yaml-cpp/0.8.0
 
module list

cmake .. \
    -DCMAKE_Fortran_COMPILER=ifort \
    -DCMAKE_CXX_COMPILER=icpc \
    -DCMAKE_C_COMPILER=icc \
    -DCMAKE_CXX_FLAGS=-fPIC \
    -DCMAKE_C_FLAGS=-fPIC \
    -DBUILD_OPENFAST_CPP_API=ON \
    -DBUILD_FASTFARM=ON \
    -DDOUBLE_PRECISION:BOOL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENMP=ON \
    -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install

nice make -j48
make install

