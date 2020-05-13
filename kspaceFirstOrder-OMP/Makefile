# /**
# * @file      Makefile
# * @author    Jiri Jaros
# *            Faculty of Information Technology
# *            Brno University of Technology
# * @email     jarosjir@fit.vutbr.cz
# * @comments  Linux makefile for Release version on Ubuntu 18.04
# *
# * @tool      kspaceFirstOrder 2.17
# * @created   04 September 2017, 12:50
# * @lastModif 26 February  2020, 16:20
# *
# * @copyright Copyright (C) 2017 - 2020 SC\@FIT Research Group,
# *            Brno University of Technology, Brno, CZ.
# *
# * This file has been modified to support macOS by Lennart Verhagen,
# * Donders Institute, Radboud University, Nijmegen, the Netherlands
# *
# * This file is part of the C++ extension of the k-Wave Toolbox
# * (http://www.k-wave.org).
# *
# * This file is part of the k-Wave. k-Wave is free software: you can
# * redistribute it and/or modify it under the terms of the GNU Lesser General
# * Public License as published by the Free Software Foundation, either version
# * 3 of the License, or (at your option) any later version.
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
# The source codes can be compiled ONLY under Linux x64 by GNU g++ 6.0 and     #
# newer, or Intel Compiler icpc 2018 and newer. The newer the compiler, the    #
# more advanced instruction set can be used.                                   #
# We recommend compilation with g++ 8.3 or icpc 2019.                          #
#                                                                              #
# This makefile uses the GNU compiler and static linking by default.           #
#                                                                              #
# Necessary libraries:                                                         #
#  - FFTW 3.3.x and newer, or Intel MKL 2018 and newer                         #
#  - HDF5 version 1.8.x and newer                                              #
#                                                                              #
# How to compile libraries                                                     #
#  - FFTW : download from "http://www.fftw.org/"                               #
#           run configure script with following parameters:                    #
#             --enable-float --enable-avx --enable-openmp                      #
#               or                                                             #
#             --enable-float --enable-avx2 --enable-openmp                     #
#  - MKL  : Only if not using FFTW. Dowload from                               #
#           "http://software.intel.com/en-us/intel-mkl"                        #
#  - HDF5 : download from                                                      #
#           "https://www.hdfgroup.org/downloads/hdf5/source-code/"             #
#           run configure script with these parameters:                        #
#             --enable-hl --enable-static --enable-shared                      #
#                                                                              #
# How to compile the code                                                      #
#  1. Select the compiler (GNU + FFTW or Intel + MKL).                         #
#  2. Select one of the linking possibilities.                                 #
#  3. Select CPU architecture (native binary is default).                      #
#  4. Load necessary software modules, or set the library paths manually.      #
#  5. make -j                                                                  #
#                                                                              #
################################################################################


################################################################################
#         Set following flags based on your compiler and library paths         #
################################################################################

# Select compiler. GNU is default but Intel may be faster.
COMPILER = GNU
#COMPILER = Intel

# Static linking is default on linux, dynamic on macOS
LINKING = DYNAMIC
#LINKING = STATIC

# Set up paths: If using modules, the paths are set up automatically,
#               otherwise, set paths manually

# hard-coded default paths
MKL_DIR  = /usr/local
#FFT_DIR  = /usr/local/fftw
#HDF5_DIR = /usr/local/hdf5
#ZLIB_DIR = /usr/local/zlib
#SZIP_DIR = /usr/local/szip

# homebrew installations
FFT_DIR  = $(shell brew --prefix fftw)
HDF5_DIR = $(shell brew --prefix hdf5)
ZLIB_DIR = $(shell brew --prefix zlib)
SZIP_DIR = $(shell brew --prefix szip)

# Using EasyBuild / lmod module manager
#MKL_DIR  = $(EBROOTMKL)
#FFT_DIR  = $(EBROOTFFTW)
#HDF5_DIR = $(EBROOTHDF5)
#ZLIB_DIR = $(EBROOTZLIB)
#SZIP_DIR = $(EBROOTSZIP)

# Select CPU architecture (what instruction set to be used).
# The libraries such as FFTW, HDF5 and MKL are to be compiled under the same
# architecture, e.g., if you want to use AVX in k-Wave, compile FFTW with
# --enable-avx
# The native architecture will compile and optimize the code for the underlying
# processor.
# Fat binary, available only for Intel, includes all AVX, AVX2 and AVX512 sets.

CPU_ARCH = native
#CPU_ARCH = AVX
#CPU_ARCH = AVX2
#CPU_ARCH = AVX512
#CPU_ARCH = FAT_BIN

############################### Common flags ###################################
# Git hash of release 1.3
GIT_HASH       = -D__KWAVE_GIT_HASH__=\"0ba023063e3f29685e1e346f56883378d961f9f1\"

# Replace tabs by spaces
.RECIPEPREFIX +=

################################ GNU g++ + FFTW ################################
ifeq ($(COMPILER), GNU)
  # Compiler name
  #CXX			= g++
  CXX       = g++-9

  # C++ standard
  CPP_STD   = -std=c++11

  # Enable OpenMP
  OPENMP    = -fopenmp
	# when using clang
  #OPENMP 		= -Xpreprocessor -fopenmp -lomp

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

  # Default is native - the max performance for this CPU
  else
    CPU_FLAGS = -m64 -march=native -mtune=native
  endif

  # Use maximum optimization
  OPT       = -O3 -ffast-math -fassociative-math

  # Debug flags
  DEBUG     =
  # Profile flags
  PROFILE   =
  # C++ warning flags
  WARNING   = -Wall

  # Add include directories
  INCLUDES  = -I$(HDF5_DIR)/include -I$(FFT_DIR)/include -I.
  # Add library directories
  LIB_PATHS = -L$(HDF5_DIR)/lib -L$(FFT_DIR)/lib

  # Set compiler flags and header files directories
  CXXFLAGS  = $(CPU_FLAGS) $(OPT) $(DEBUG) $(WARNING) $(PROFILE) \
              $(OPENMP) $(CPP_STD)                               \
              $(GIT_HASH)                                        \
              $(INCLUDES)

  # Set linker flags and library files directories
  ifeq ($(LINKING), STATIC)
        # Static link
        LDFLAGS = $(CPU_FLAGS) $(DEBUG) $(WARNING) $(PROFILE) \
                  $(OPENMP) $(CPP_STD)                        \
                  $(LIB_PATHS)                                \
                  -static

        LDLIBS  = $(FFT_DIR)/lib/libfftw3f.a     \
                  $(FFT_DIR)/lib/libfftw3f_omp.a \
                  $(HDF5_DIR)/lib/libhdf5_hl.a   \
                  $(HDF5_DIR)/lib/libhdf5.a      \
                  $(ZLIB_DIR)/lib/libz.a         \
                  $(SZIP_DIR)/lib/libsz.a        \
                  -lm -lmvec                     \
                  -ldl  # We need this for HDF5-1.8.11 and newer

    else
        # Dynamic link with runtime paths
        LDFLAGS = $(CPU_FLAGS) $(OPT) $(DEBUG) $(WARNING) $(PROFILE) \
                   $(OPENMP) $(CPP_STD)                              \
                   $(LIB_PATHS)                                      \
                   -Wl,-rpath,$(HDF5_DIR)/lib:$(FFT_DIR)/lib

        LDLIBS  = -lfftw3f -lfftw3f_omp -lhdf5 -lhdf5_hl -lm -lz
  endif
endif


########################### Intel Compiler + MKL ###############################
ifeq ($(COMPILER), Intel)
  # Compiler name
  CXX       = icpc

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

  # AMD-AVX, Intel's AVX, AVX2 and AVX-512
  else ifeq ($(CPU_ARCH), FAT_BIN)
    CPU_FLAGS = -m64 -mavx -axCORE-AVX2,CORE-AVX512

  # Defautl is native - the max performance for this CPU
  else
    CPU_FLAGS = -m64 -xhost
  endif

  # Use maximum optimization
  OPT       = -Ofast -ipo

  # Debug flags
  DEBUG     =
  # Profile flags
  PROFILE   =
  # C++ warning flags
  WARNING   = -Wall

  # Add include directories
  INCLUDES  = -I$(HDF5_DIR)/include -I.
  # Add library directories
  LIB_PATHS = -L$(HDF5_DIR)/lib


  # Set compiler flags and header files directories
  CXXFLAGS  = $(CPU_FLAGS) $(OPT) $(DEBUG) $(WARNING) $(PROFILE) \
              $(OPENMP) $(CPP_STD)                               \
              $(GIT_HASH)                                        \
              $(INCLUDES)

  # Set linker flags
  ifeq ($(LINKING), STATIC)
        LDFLAGS = $(CPU_FLAGS) $(OPT) $(DEBUG) $(WARNING) $(PROFILE) \
                  $(OPENMP) $(CPP_STD)                               \
                  $(LIB_PATHS)                                       \
                  -mkl=parallel -static-intel -qopenmp-link=static

        LDLIBS  = $(HDF5_DIR)/lib/libhdf5_hl.a  \
                  $(HDF5_DIR)/lib/libhdf5.a     \
                  $(ZLIB_DIR)/lib/libz.a        \
                  $(SZIP_DIR)/lib/libsz.a       \
                  -lm -ldl

  else
        # Dynamic link with runtime paths
        LDFLAGS = $(CPU_FLAGS) $(OPT) $(DEBUG) $(WARNING) $(PROFILE) \
                  $(OPENMP) $(CPP_STD)                               \
                  $(LIB_PATHS)                                       \
                  -Wl,-rpath,$(HDF5_DIR)/lib                         \
                  -mkl=parallel

        LDLIBS  = -lhdf5 -lhdf5_hl -lz -lsz

  endif
endif

################################### Build ######################################
# Target binary name
TARGET       = kspaceFirstOrder-OMP

# Units to be compiled
DEPENDENCIES = main.o                                  \
               Containers/MatrixContainer.o            \
               Containers/OutputStreamContainer.o      \
               Hdf5/Hdf5File.o                         \
               Hdf5/Hdf5FileHeader.o                   \
               KSpaceSolver/KSpaceFirstOrderSolver.o   \
               Logger/Logger.o                         \
               MatrixClasses/BaseFloatMatrix.o         \
               MatrixClasses/BaseIndexMatrix.o         \
               MatrixClasses/ComplexMatrix.o           \
               MatrixClasses/FftwComplexMatrix.o       \
               MatrixClasses/FftwRealMatrix.o          \
               MatrixClasses/IndexMatrix.o             \
               MatrixClasses/RealMatrix.o              \
               OutputStreams/BaseOutputStream.o        \
               OutputStreams/IndexOutputStream.o       \
               OutputStreams/CuboidOutputStream.o      \
               OutputStreams/WholeDomainOutputStream.o \
               Parameters/CommandLineParameters.o      \
               Parameters/Parameters.o

# Build target
all: $(TARGET)

# Link target
$(TARGET): $(DEPENDENCIES)
	$(CXX) $(LDFLAGS) $(DEPENDENCIES) $(LDLIBS) -o $@

# Compile units
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# Clean repository
.PHONY: clean
clean:
	rm -f $(DEPENDENCIES) $(TARGET)
