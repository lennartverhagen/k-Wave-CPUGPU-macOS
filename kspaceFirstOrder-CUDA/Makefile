# /**
# * @file      Makefile
# * @author    Jiri Jaros
# *            Faculty of Information Technology
# *            Brno University of Technology 
# * @email     jarosjir@fit.vutbr.cz
# * @comments  Linux makefile for Release version on Ubuntu 18.04
# * 
# * @tool      kspaceFirstOrder 3.6
# * @created   02 December  2014, 12:32 
# * @lastModif 11 February  2020, 16:34
# *
# * @copyright Copyright (C) 2014 - 2020 SC\@FIT Research Group, 
# *            Brno University of Technology, Brno, CZ.
# *
# * This file is part of the C++ extension of the k-Wave Toolbox 
# * (http://www.k-wave.org). 
# *
# * k-Wave is free software: you can redistribute it and/or modify it under the
# * terms of the GNU Lesser General Public License as published by the Free
# * Software Foundation, either version 3 of the License, or (at your option) 
# * any later version.
# *
# * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY 
# * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
# * more details.
# *
# * You should have received a copy of the GNU Lesser General Public License 
# * along with k-Wave.
# * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
# */

################################################################################
#                                                                              #
# The source codes can be compiled ONLY under Linux x64 by GNU g++ 7.0 and     #
# newer, or Intel Compiler icpc 2018 and newer. The newer the compiler, the    #
# more advanced instruction set can be used.                                   #
# We recommend compilation with g++ 8.3 or icpc 2019.                          #
#                                                                              #
# The code also requires CUDA 10.x, but can be compiled with CUDA 9.x as well. #
# The code cannot be compiled with older versions of CUDA since                #
# the incompatibility with the cuda FFT library and __syncwarp() routines.     #
#                                                                              #
# This makefile uses the GNU compiler and SEMI-static linking by default.      #
# This means all libraries but CUDA are linked statically. It is also possible #
# to link the code dynamically or statically                                   #
#                                                                              #
# Necessary libraries:                                                         #
#  - HDF5 version 1.8.x and newer                                              #
#                                                                              #
# How to compile libraries                                                     #
#  - CUDA : download from "https://developer.nvidia.com/cuda-toolkit-archive"  #
#           and install the compiler and libraries using the provided          #
#           installers.                                                        #
#  - HDF5 : download from                                                      #
#           "https://www.hdfgroup.org/downloads/hdf5/source-code/"             #
#           run configure script with these parameters:                        #
#             --enable-hl --enable-static --enable-shared                      #
#                                                                              #
# How to compile the code                                                      #
#  1. Select the compiler for the host code (GNU or Intel).                    #
#  2. Select one of the linking possibilities.                                 #
#  3. Select CPU architecture (native is default).                             #
#  4. Load necessary software modules, or set the library paths manually.      #
#  5. make -j                                                                  #
#                                                                              #
################################################################################


################################################################################
#         Set following flags based on your compiler and library paths         #
################################################################################

# Select compiler
# GNU is default due to Intel 2018's compatibility issues with Ubuntu 18.04
 COMPILER = GNU
#COMPILER = Intel

# SEMI static lining is default since it is expected the binary will run on the 
# same system. 
# Everything will be linked statically, may not work on all GPUs
#LINKING = STATIC
# Everything will be linked dynamically
#LINKING = DYNAMIC
# Everything but CUDA will be linked statically
LINKING = SEMI

# Set up paths: If using modules, the paths are set up automatically,
#               otherwise, set paths manually
CUDA_DIR = $(CUDA_HOME)
HDF5_DIR = $(EBROOTHDF5)
ZLIB_DIR = $(EBROOTZLIB)
SZIP_DIR = $(EBROOTSZIP)

# Select CPU architecture (what instruction set to be used).
# The native architecture will compile and optimize the code for the underlying
# processor.

 CPU_ARCH = native
#CPU_ARCH = AVX
#CPU_ARCH = AVX2
#CPU_ARCH = AVX512

############################### Common flags ###################################
# Git hash of release 1.3
GIT_HASH       = -D__KWAVE_GIT_HASH__=\"468dc31c2842a7df5f2a07c3a13c16c9b0b2b770\"

# Replace tabs by spaces
.RECIPEPREFIX += 

# What CUDA GPU architectures to include in the binary
CUDA_ARCH = --generate-code arch=compute_30,code=sm_30 \
            --generate-code arch=compute_32,code=sm_32 \
            --generate-code arch=compute_35,code=sm_35 \
            --generate-code arch=compute_37,code=sm_37 \
            --generate-code arch=compute_50,code=sm_50 \
            --generate-code arch=compute_52,code=sm_52 \
            --generate-code arch=compute_53,code=sm_53 \
            --generate-code arch=compute_60,code=sm_60 \
            --generate-code arch=compute_61,code=sm_61 \
            --generate-code arch=compute_62,code=sm_62 \
            --generate-code arch=compute_70,code=sm_70 \
            --generate-code arch=compute_72,code=sm_72 \
            --generate-code arch=compute_75,code=sm_75

# What libraries to link and how
ifeq ($(LINKING), STATIC)
  LDLIBS = $(HDF5_DIR)/lib/libhdf5_hl.a         \
           $(HDF5_DIR)/lib/libhdf5.a            \
           $(CUDA_DIR)/lib64/libcufft_static.a  \
           $(CUDA_DIR)/lib64/libculibos.a       \
           $(CUDA_DIR)/lib64/libcudart_static.a \
           $(ZLIB_DIR)/lib/libz.a               \
           $(SZIP_DIR)/lib/libsz.a              \
           -ldl

else ifeq ($(LINKING), DYNAMIC)
  LDLIBS = -lhdf5 -lhdf5_hl -lz -lcufft

else ifeq ($(LINKING), SEMI)
  LDLIBS = $(HDF5_DIR)/lib/libhdf5_hl.a \
           $(HDF5_DIR)/lib/libhdf5.a    \
           $(ZLIB_DIR)/lib/libz.a       \
           $(SZIP_DIR)/lib/libsz.a      \
           -lcufft                      \
           -ldl
endif

############################## NVCC + GNU g++ ##################################
ifeq ($(COMPILER), GNU)
  # C++ compiler for CUDA
  CXX       = nvcc

  # C++ standard
  CPP_STD   = -std=c++11

  # Enable OpenMP
  OPENMP    = -fopenmp

  # Set CPU architecture
  # Sandy Bridge, Ivy Bridge
  ifeq ($(CPU_ARCH), AVX)
    CPU_FLAGS = -m64 -mavx

  # Haswell, Broadwell
  else ifeq ($(CPU_ARCH), AVX2)
    CPU_FLAGS = -m64 -mavx2

  # Skylake-X, Ice Lake, Cannon Lake
  else ifeq ($(CPU_ARCH), AVX512)
    CPU_FLAGS = -m64 -mavx512f

  # Maximum performance for this CPU
  else
    CPU_FLAGS = -m64 -march=native -mtune=native
  endif

  # Use maximum optimization
  CPU_OPT   = -O3 -ffast-math -fassociative-math
  # Use maximum optimization
  GPU_OPT   = -O3

  # CPU Debug flags
  CPU_DEBUG = 
  # Debug flags
  GPU_DEBUG = 
  # Profile flags
  PROFILE   = 
  # C++ warning flags
  WARNING   = -Wall

  # Add include directories
  INCLUDES  = -I$(HDF5_DIR)/include -I.
  # Add library directories
  LIB_PATHS = -L$(HDF5_DIR)/lib -L$(CUDA_DIR)/lib64

  # Set compiler flags and header files directories
  CXXFLAGS  = -Xcompiler="$(CPU_FLAGS) $(CPU_OPT) $(OPENMP)  \
                          $(CPU_DEBUG) $(PROFILE) $(WARNING)"\
              $(GPU_OPT) $(CPP_STD) $(GPU_DEBUG) \
              $(GIT_HASH)                        \
              $(INCLUDES)                        \
              --device-c --restrict

  # Set linker flags and library files directories
  LDFLAGS   = -Xcompiler="$(OPENMP)" \
              -Xlinker="-rpath,$(HDF5_DIR)/lib:$(CUDA_DIR)/lib64" \
              -std=c++11             \
               $(LIB_PATHS)
endif

############################ NVCC + Intel icpc #################################
ifeq ($(COMPILER), Intel)
  # C++ compiler for CUDA
  CXX       = nvcc

  # C++ standard
  CPP_STD   = -std=c++11

  # Enable OpenMP
  OPENMP    = -qopenmp

  # Set CPU architecture
  # Sandy Bridge, Ivy Bridge
  ifeq ($(CPU_ARCH), AVX)
    CPU_FLAGS = -m64 -xAVX

  # Haswell, Broadwell
  else ifeq ($(CPU_ARCH), AVX2)
    CPU_FLAGS = -m64 -xCORE-AVX2

  # Skylake-X, Ice Lake, Cannon Lake
  else ifeq ($(CPU_ARCH), AVX512)
    CPU_FLAGS = -m64 -xCORE-AVX512

  # Maximum performance for this CPU
  else
    CPU_FLAGS = -m64 -xhost
  endif

  # Use maximum optimization
  CPU_OPT   = -Ofast
  # Use maximum optimization
  GPU_OPT   = -O3

  # CPU Debug flags
  CPU_DEBUG = 
  # Debug flags
  GPU_DEBUG = 
  # Profile flags
  PROFILE   = 
  # C++ warning flags
  WARNING   = -Wall

  # Add include directories
  INCLUDES  = -I$(HDF5_DIR)/include -I.
  # Add library directories
  LIB_PATHS = -L$(HDF5_DIR)/lib -L$(CUDA_DIR)/lib64

  # Set compiler flags and header files directories
  CXXFLAGS  = -Xcompiler="$(CPU_FLAGS) $(CPU_OPT) $(OPENMP)   \
                          $(CPU_DEBUG) $(PROFILE) $(WARNING)" \
              $(GPU_OPT) $(CPP_STD) $(GPU_DEBUG) \
              $(GIT_HASH)                        \
              $(INCLUDES)                        \
              --device-c --restrict -ccbin=icpc

  # Set linker flags and library files directories
  ifneq ($(LINKING), DYNAMIC)
    LDFLAGS = -Xcompiler="$(OPENMP) -static-intel -qopenmp-link=static"
  else
    LDFLAGS = -Xcompiler="$(OPENMP)"
  endif

  LDFLAGS  += -std=c++11 -ccbin=icpc \
              -Xlinker="-rpath,$(HDF5_DIR)/lib:$(CUDA_DIR)/lib64" \
              $(LIB_PATHS)
endif

################################### Build ######################################
# Target binary name
TARGET       = kspaceFirstOrder-CUDA

# Units to be compiled
DEPENDENCIES = main.o                                   \
               Containers/MatrixContainer.o             \
               Containers/CudaMatrixContainer.o         \
               Containers/OutputStreamContainer.o       \
               Hdf5/Hdf5File.o                          \
               Hdf5/Hdf5FileHeader.o                    \
               KSpaceSolver/KSpaceFirstOrderSolver.o    \
               KSpaceSolver/SolverCudaKernels.o         \
               Logger/Logger.o                          \
               MatrixClasses/BaseFloatMatrix.o          \
               MatrixClasses/BaseIndexMatrix.o          \
               MatrixClasses/CufftComplexMatrix.o       \
               MatrixClasses/ComplexMatrix.o            \
               MatrixClasses/IndexMatrix.o              \
               MatrixClasses/RealMatrix.o               \
               MatrixClasses/TransposeCudaKernels.o     \
               OutputStreams/BaseOutputStream.o         \
               OutputStreams/IndexOutputStream.o        \
               OutputStreams/CuboidOutputStream.o       \
               OutputStreams/WholeDomainOutputStream.o  \
               OutputStreams/OutputStreamsCudaKernels.o \
               Parameters/CommandLineParameters.o       \
               Parameters/Parameters.o                  \
               Parameters/CudaParameters.o              \
               Parameters/CudaDeviceConstants.o

# Build target
all: $(TARGET)

# Link target
$(TARGET): $(DEPENDENCIES)
  $(CXX) $(LDFLAGS) $(DEPENDENCIES) $(LDLIBS) -o $@

# Compile CPU units
%.o: %.cpp
  $(CXX) $(CXXFLAGS) -o $@ -c $<

# Compile CUDA units
%.o: %.cu
  $(CXX) $(CXXFLAGS) $(CUDA_ARCH) -o $@ -c $<

# Clean repository
.PHONY: clean
clean:
  rm -f $(DEPENDENCIES) $(TARGET)
