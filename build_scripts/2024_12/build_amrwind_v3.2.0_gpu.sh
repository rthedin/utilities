#!/bin/bash

# Build script for AMR-Wind on GPU.
# Regis Thedin, 2024-11-01

module purge
module load binutils
module load PrgEnv-nvhpc
module load cray-libsci/22.12.1.1
module load cmake
module load cmake/3.27.9
module load cray-python
module load netcdf-fortran/4.6.1-oneapi
module load craype-x86-genoa
module load craype-accel-nvidia90 
 
module list

export MPICH_GPU_SUPPORT_ENABLED=1
export CUDAFLAGS="-L/nopt/nrel/apps/gpu_stack/libraries-gcc/06-24/linux-rhel8-zen4/gcc-12.3.0/hdf5-1.14.3-zoremvtiklvvkbtr43olrq3x546pflxe/lib -I/nopt/nrel/apps/gpu_stack/libraries-gcc/06-24/linux-rhel8-zen4/gcc-12.3.0/hdf5-1.14.3-zoremvtiklvvkbtr43olrq3x546pflxe/include -lhdf5 -lhdf5_hl -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_nvidia90} ${PE_MPICH_GTL_LIBS_nvidia90}"
export CXXFLAGS="-L/nopt/nrel/apps/gpu_stack/libraries-gcc/06-24/linux-rhel8-zen4/gcc-12.3.0/hdf5-1.14.3-zoremvtiklvvkbtr43olrq3x546pflxe/lib -I/nopt/nrel/apps/gpu_stack/libraries-gcc/06-24/linux-rhel8-zen4/gcc-12.3.0/hdf5-1.14.3-zoremvtiklvvkbtr43olrq3x546pflxe/include -lhdf5 -lhdf5_hl -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_nvidia90} ${PE_MPICH_GTL_LIBS_nvidia90}"
 
cmake .. \
    -DAMR_WIND_ENABLE_CUDA=ON \
    -DAMR_WIND_ENABLE_TINY_PROFILE:BOOL=ON \
    -DAMReX_CUDA_ERROR_CAPTURE_THIS:BOOL=ON \
    -DCMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION:BOOL=ON \
    -DCMAKE_CXX_COMPILER:STRING=CC \
    -DCMAKE_C_COMPILER:STRING=cc \
    -DMPI_CXX_COMPILER=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin/mpicxx \
    -DMPI_C_COMPILER=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin/mpicc \
    -DMPI_Fortran_COMPILER=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin/mpifort \
    -DAMReX_DIFFERENT_COMPILER=ON \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DAMR_WIND_ENABLE_CUDA=ON \
    -DAMR_WIND_ENABLE_CUDA:BOOL=ON \
    -DAMR_WIND_ENABLE_OPENFAST:BOOL=OFF \
    -DAMR_WIND_ENABLE_NETCDF:BOOL=ON \
    -DAMR_WIND_ENABLE_HDF5:BOOL=ON \
    -DAMR_WIND_ENABLE_MPI:BOOL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMR_WIND_ENABLE_HYPRE:BOOL=OFF \
    -DAMR_WIND_ENABLE_MASA:BOOL=OFF \
    -DAMR_WIND_ENABLE_TESTS:BOOL=ON \
    -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install
 
make -j 32 amr_wind

