## Overview

k-Wave is an open source MATLAB toolbox designed for the time-domain simulation
of propagating acoustic waves in 1D, 2D, or 3D. The toolbox has a wide range of
functionality, but at its heart is an advanced numerical model that can account
for both linear or nonlinear wave propagation, an arbitrary distribution of
heterogeneous material parameters, and power law acoustic absorption.
See the k-Wave website (http://www.k-wave.org) for further details.

This project is a part of the k-Wave toolbox accelerating 2D/3D simulations
using an optimized C++ implementation to run small to moderate grid sizes (e.g.,
128x128 to 10,000x10,000 in 2D or 64x64x64 to 512x512x512 in 3D) on systems
with shared memory. 2D simulations can be carried out in both normal and
axisymmetric coordinate systems.


## Repository structure

    .
    +--Containers    - Matrix and output stream containers
    +--Data          - Small test data
    +--GetoptWin64   - Windows version of the getopt routine
    +--Hdf5          - HDF5 classes (file access)
    +--KSpaceSolver  - Solver classes with all the kernels
    +--Logger        - Logger class for reporting progress and errors
    +--MatrixClasses - Matrix classes holding simulation data
    +--OutputStreams - Output streams for sampling data
    +--Parameters    - Parameters of the simulation
    +--Utils         - Utility routines
    Changelog.md     - Change log
    License.md       - License file
    Makefile         - GNU Makefile
    Readme.md        - Read me
    Doxyfile         - Doxygen documentation file
    header_bg.png    - Doxygen logo
    main.cpp         - Main file of the project


## Compilation

The source codes of `kspaceFirstOrder-OMP` are written using the C++-11 standard
and use the OpenMP 4.0, FFTW 3.3.8 or MKL 11, and HDF5 1.10.x libraries.

There are a variety of different C++ compilers that can be used to compile the
source codes. The minimum requirements are the GNU C++ compiler 6.0 or the Intel
C++ compiler 2018. However, we recommend using either the GNU C++ compiler
version 8.3 and newer, or the Intel C++ compiler version 2019 and newer. Please
note that Visual Studio compilers do not support the OpenMP 4.0 standard and
cannot be used to compile this code. Also be aware the Intel compiler 2018 has
an MKL bug which produces incorrect results when AVX2 is enabled. The codes can
be compiled on 64-bit Linux and Windows. 32-bit systems are not supported due to
the memory requirements even for small simulations.

This section describes the compilation procedure using GNU and Intel compilers
on Linux. Windows users are encouraged to download the Visual Studio 2017
project and compile it using the Intel Compiler from within Visual Studio.

Before compiling the code, it is necessary to install a C++ compiler and several
libraries. The GNU C/C++ compiler is usually part of Linux distributions and
distributed as open source. It can be downloaded from the GNU website
(http://gcc.gnu.org/) if necessary.
The Intel compiler can be downloaded from the Intel website
(https://software.intel.com/en-us/parallel-studio-xe). This package also
includes the Intel MKL (Math Kernel Library) library that contains the fast
Fourier transform (FFT). The Intel compiler is only free for non-commercial and
open-source use.

The code also relies on several libraries that are to be installed before
compiling:

 1. HDF5 library - Mandatory I/O library, version 1.8.x,
         https://portal.hdfgroup.org/display/support/HDF5+1.8.21,
         or version 1.10.x.
         https://www.hdfgroup.org/downloads/hdf5/source-code/.
 1. FFTW library - Optional library for FFT, version 3.3.x,
         http://www.fftw.org/.
 1. MKL library  - Optional library for FFT, version 2018 or higher
         http://software.intel.com/en-us/intel-composer-xe/.

Although it is possible to use any combination of the FFT library and the 
compiler, the best performance is observed when using the GNU compiler and FFTW,
or the Intel compiler and Intel MKL.


### The HDF5 library installation procedure

 1. Download the 64-bit HDF5 library
    https://www.hdfgroup.org/downloads/hdf5/source-code/). Please keep in mind
    that versions 1.10.x may not be fully compatible with older versions of
    MATLAB, especially when compression is enabled. In such a case, please
    download version 1.8.x
    https://portal.hdfgroup.org/display/support/HDF5+1.8.21.

 2. Configure the HDF5 distribution. Enable the high-level library and specify
    an installation folder by typing:
    ```bash
    ./configure --enable-hl --prefix=folder_to_install
    ```
 3. Make the HDF5 library by typing:
    ```bash
    make -j
    ```
 4. Install the HDF5 library by typing:
    ```bash
    make install
    ```


### The FFTW library installation procedure

 1. Download the FFTW library package for your platform,
    http://www.fftw.org/download.html.

 2. Configure the FFTW distribution. Enable OpenMP support, the desired SIMD
    instruction set, single precision floating point arithmetic, and specify an
    installation folder:
    ```bash
    ./configure --enable-single --enable-avx --enable-openmp --enable-shared \
                --prefix=folder_to_install
    ```

    If you intend to use the FFTW library (and the C++ code) only on a selected
    machine and want to get the best possible performance, you may also add
    processor specific optimizations and AVX2 or AVX-512 instructions set. Note,
    the compiled binary code is not likely to be portable on different CPUs.
    The AVX version will work on Intel Sandy Bridge and newer,
    AVX2 on Intel Haswell and newer, AVX-512 on Skylake SP and newer processors.
    ```bash
    ./configure --enable-single --enable-avx2 --enable-openmp  --enable-shared \
                --with-gcc-arch=native --prefix=folder_to_install
    ```
   
    More information about the installation and customization can be found at
    http://www.fftw.org/fftw3_doc/Installation-and-Customization.html.

 3. Make the FFTW library by typing:
    ```bash
    make -j
    ```
 4. Install the FFTW library by typing:
    ```bash
    make install
    ```


### The Intel Compiler and MKL installation procedure

 1. Download the Intel Composer XE package for your platform
    http://software.intel.com/en-us/intel-compilers.

 2. Run the installation script and follow the procedure by typing:
    ```bash
    ./install.sh
    ```


### Compiling the C++ code on Linux

After the libraries and the compiler have been installed, you are ready to
compile the `kspaceFirstOrder-OMP` code.

 1. Open Makefile.
    
 2. The Makefile supports code compilation under the GNU compiler with FFTW, or
    the Intel compiler with MKL. Uncomment the desired compiler by removing the
    character `#`.
    GNU is default since it doesn't need installation but Intel may be faster.
    ```bash
     COMPILER = GNU
    #COMPILER = Intel
    ```

 3. Select how to link the libraries. Static linking is preferred as it may be
    a bit faster, however, on some systems (e.g, HPC clusters) it may be better
    to use dynamic linking and use the system specific libraries at runtime.
    ```bash
     LINKING = STATIC
    #LINKING = DYNAMIC
    ```

 4. Set installation paths of the libraries (an example is shown below). Zlib
    and Szip may be required if the compression is switched on. If using
    EasyBuild and Lmod (Environment Module System) to manage your software,
    please load appropriate modules before running make. The makefile will set
    the paths automatically.
    ```bash
    MKL_DIR  = $(EBROOTMKL)
    FFT_DIR  = $(EBROOTFFTW)
    HDF5_DIR = $(EBROOTHDF5)
    ZLIB_DIR = $(EBROOTZLIB)
    SZIP_DIR = $(EBROOTSZIP)
    ```

 5. Select the instruction set and the CPU architecture.
    For users who will only use the binary on the same machine as compiled,
    the best choice is `CPU_ARCH = native`. If you are about to run the same
    binary on different machines or you want to cross-compile the code, you are
    free to use any of the possible choices, where AVX is the most general but
    slowest and AVX512 is the most recent instruction set and (most likely)
    the fastest. The fat binary compiles the code for all architectures,
    however, can only be used with the Intel compiler.
    ```bash
     CPU_ARCH = native
    #CPU_ARCH = AVX
    #CPU_ARCH = AVX2
    #CPU_ARCH = AVX512
    #CPU_ARCH = FAT_BIN
    ```

 6. Close the makefile and compile the source code by typing:
    ```bash
    make -j
    ```
    If you want to clean the distribution, type:
    ```bash
    make clean
    ```

## Usage

The C++ codes offers a lot of parameters and output flags to be used. For more
information, please type:

```bash
./kspaceFirstOrder-OMP --help
```
