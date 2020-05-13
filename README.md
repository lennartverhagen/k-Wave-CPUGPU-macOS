# k-Wave-CPUGPU-macOS

Copyright (C) 2012 - 2020 SC\@FIT Research Group,
Brno University of Technology, Brno, CZ.

This file is part of the C++ extension of the k-Wave Toolbox
(http://www.k-wave.org).

Modifications: macOS compatibility for OMP by Lennart Verhagen, Donders Institute, Radboud University, Nijmegen, the Netherlands


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


## macOS compatibility

The c++ CPU optimised code supporting OpenMP has been modified from the linux distribution to be compatible with macOS. The c++ code to modify has been identified simply by running `grep -r linux .`. Modification can be found by running `grep -r macOS .` and `grep -r APPLE .`. Further modifications have been make in the Makefile. These modifications have only been tested on macOS Catalina 10.15 for the GNU compiler with the FFTW library. Currently, only the CPU OMP source code has been modified. The GPU CUDA code requires NVIDIA graphics cards which are only available on macOS through external cards (eGPU) and some benevolent hacking. For NVIDIA eGPU on macOS, follow this [video](https://youtu.be/JjL_50ZNaKY) and these [instructions](https://theunlockr.com/how-to-use-nvidia-cards-with-your-mac-egpu/) making use of [these scripts](https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/paged/1/).


## macOS installation
### basics
* download the k-Wave CPU/GPU [source code](http://www.k-wave.org/download.php)
* read the `Readme.md` (but keep in mind this is for linux distributions)
* we will make use of **homebrew** to install the required packages and libraries on macOS. Check out [https://brew.sh](https://brew.sh) for instructions
* below we will provide summary instructions, with more elaborate and alternative instructions further down
* all these instructions should be run on the command-line terminal

### install required packages and libraries
* install gcc compilers: `brew install gcc`
* install support for OpenMP: `brew install libomp`
* install open message packaging: `brew install open-mpi`
* install compression libraries from zlib `brew install zlib`
* install compression libraries from szip `brew install szip`
* install file format support for HDF5: `brew install hdf5`
* install open-source FFT library: `brew install fftw`

### compile k-Wave c++ source code
* go to the source code dir `cd kspaceFirstOrder-OMP`
* compile: `make -j`
* test the compiled binary: ` ./kspaceFirstOrder-OMP --help`
* copy the kspaceFirstOrder-OMP executable to the `binaries` folder of the main k-Wave toolbox dir
* to remove the compilation and binary executable from the source code dir, run `make clean`

### details: GNU compilers
* in macOS gcc and g++ link to the clang compiler. Run `gcc --version` to confirm. To use the *true* gcc compilers, you need to explicitly specify the version of the gcc compiler in the Makefile, as `gcc-9`.
* run `which gcc-9` to confirm the installation location
* run `gcc-9 --version` to confirm the homebrew installation version
* optional: force the system-wide adoption of this new gcc compiler (instead of clang) by adding an alias to .bash_profile or .zshrc `alias gcc='gcc-9'` and restart the terminal
* in the above commands `gcc` can be replaced by `g++` to use the latter compiler
* to use OpenMP with the clang compiler, install the OpenMP library with `brew install libomp` and replace the compiler flag `-fopenmp` with `-Xpreprocessor -fopenmp -lomp`

### details: HDF5
* download the HDF5 v1.12.0 [source code](https://www.hdfgroup.org/downloads/hdf5/source-code/)
* unpack to `~/code/hdf5-1.12.0`, or your prefered place. Then `cd` into directory.
* then `./configure --enable-hl --prefix=/usr/local/hdf5`
* possible configuration for backwards compatibility: `--with-default-api-version=v110`
* Then run: `make -j` and `sudo make install`
* Please update the Makefile: comment out the line `HDF5_DIR = $(shell brew --prefix hdf5)`, and uncomment the line `HDF5_DIR = /usr/local/hdf5`

### details: FFTW
* download the FFTW [source code](http://www.fftw.org/download.html)
* configure FFTW compilation: `./configure CC=gcc-9 --enable-single --enable-avx512 --enable-openmp  --enable-shared  --prefix=/usr/local/fftw`
* then `make -j` and `sudo make install`
* Please update the Makefile: comment out the line `FFT_DIR  = $(shell brew --prefix fftw)`, and uncomment the line `FFT_DIR  = /usr/local/fftw`

### details: module management on macOS
* installation of module management is **optional**
* first install lmod, see [here](https://lmod.readthedocs.io)
* for this, install lua: `brew install lua`
* then `brew install luarocks`, `luarocks install luaposix`, and `luarocks install luafilesystem`
* set some definitions for luarocks: `LUAROCKS_PREFIX=/usr/local
export LUA_PATH="$LUAROCKS_PREFIX/share/lua/5.1/?.lua;$LUAROCKS_PREFIX/share/lua/5.1/?/init.lua;;"
export LUA_CPATH="$LUAROCKS_PREFIX/lib/lua/5.1/?.so;;"`
* then `brew install pkg-config`
* and `brew install lmod`
* source in '.zshrc': `source /usr/local/opt/lmod/init/zsh`, and similar for '.bash_profile' or your own preferred shell profile.
* now for EasyBuild, follow [these instructions](https://easybuild.readthedocs.io/en/latest/Installation.html):
*
`EASYBUILD_PREFIX=$HOME/.local/easybuild`  
`# download script`  
`curl -O https://raw.githubusercontent.com/easybuilders/easybuild-framework/develop/easybuild/scripts/bootstrap_eb.py`  
`# bootstrap EasyBuild`  
`python bootstrap_eb.py $EASYBUILD_PREFIX`  
`# update $MODULEPATH, and load the EasyBuild module`  
`module use $EASYBUILD_PREFIX/modules/all`  
`module load EasyBuild`  
