# acousticFieldPropagator-OMP

## Overview

[k-Wave](http://www.k-wave.org "k-Wave homepage") is an open source MATLAB toolbox designed for the time-domain simulation of propagating acoustic waves in 1D, 2D, or 3D. It includes a solver to rapidly calculate acoustic field produced by an array of transducers.

This code is a part of the toolbox accelerating the acoustic field calculation in 3D by implementing the kernels in C++, exploiting shared memory parallelism (vector units + threads) using OpenMP. It allows you to run small to moderate simulations given the sufficient amount of system memory (~62 GB with the 2000x2000x2000 extended grid size). For more information please see the [related article](https://doi.org/10.1121/1.5021245 "Rapid calculation of acoustic fields from arbitrary continuous-wave sources").

Compiled binaries are [available for download](http://www.k-wave.org/download "k-Wave download page") for both Windows and GNU/Linux 64-bit platform.

For details about licensing and citing consult the [LICENSE.md](LICENSE.md) file.


## Building the binaries, documentation

### Source code compilation

This project uses [CMake build system](https://cmake.org/) and supports Intel, GCC, LLVM and MSCV compilers. You are expected to be familiar with the used toolchain, target platform, and the CMake build system. All the required tools have to be installed on your system.

#### Library dependencies

The following 3-rd party libraries are required to build and run the binaries:

+ [HDF5](https://www.hdfgroup.org/downloads/hdf5/), the HDF5 data format is used for data exchange (interface)
+ [Intel MKL](https://software.intel.com/en-us/mkl) _**or**_ [FFTW](http://fftw.org/) to perform fast Fourier transforms (FFTs)

Optionally, the code can be built so that these dependencies are part of the resulting binaries (linked statically). Based on your toolchain and platform choice, additional libraries (like OpenMP runtime) may not be linked statically and need to be present at runtime.

**Please note that the FFTW build can only be distributed under the terms of the GNU General Public License 3.0 ([in this repository](Licenses/gpl.txt) or [online](https://www.gnu.org/licenses/gpl-3.0.html)).**

#### Project-specific CMake options

+ `USE_FFTW` (_boolean_): Use FFTW library instead of Intel MKL for computing FFTs (`false` by default)
+ `STATIC_LINKING` (_boolean_): Try to link the required libraries statically, to compile redistributable libraries (`false` by default)
+ `HDF5_ROOT` (_path_): Hint for the CMake where to look for the HDF5 library
+ `IMKL_ROOT` (_path_): Hint for the CMake where to look for the Intel MKL library (used only when `USE_FFTW` is false)
+ `FFTW3f_ROOT` (_path_): Hint for the CMake where to look for the FFTW library (used only when `USE_FFTW` is true)
+ `KWAVE_VERSION` (_string_): To print the associated k-Wave release version to the output header (used for building k-Wave releases, empty by default)

#### Building procedure examples

In the examples below, the sources are expected to be inside the `acousticFieldPropagator-OMP` folder. The binaries are built **off-tree** inside a _separate_ folder named `acousticFieldPropagator-OMP-build`.

+ Compilation on GNU/Linux, GCC or LLVM toolchain, dynamic linking, FFTW backend
```sh
mkdir acousticFieldPropagator-OMP-build
cd acousticFieldPropagator-OMP-build
cmake ../acousticFieldPropagator-OMP -DCMAKE_BUILD_TYPE=Release -DUSE_FFTW=on
make -j
```
+ Compilation on Windows, Visual Studio 2017 with Intel C++ 2019 compiler, Intel MKL backend
```cmd
mkdir acousticFieldPropagator-OMP-build
cd acousticFieldPropagator-OMP-build
cmake ..\acousticFieldPropagator-OMP -G "Visual Studio 15 2017 Win64" -T "Intel C++ Compiler 19.0" -DIMKL_ROOT="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl" -DHDF5_ROOT="C:\Program Files\HDF_Group\HDF5\1.10.6\cmake"
cmake --build . --config Release
```

You may need to tweak the paths and versions according to your needs. You can also change the compilers (using `CXX` environment variable or `-T` CMake parameter on Windows), generators, or customize the build using standard CMake variables (such as `CMAKE_CXX_FLAGS`). Please refer to the [CMake documentation](https://cmake.org/documentation/) for further inquiries.

### Source documentation

To build the source code documentation you need to have [Doxygen](http://doxygen.nl) installed on your system. It should be picked up by the CMake build system during the configuration step. You can generate the documentation by altering the build command in the procedure above, i.e. by running
```sh
make docs
```
on GNU/Linux platform when using *Unix Makefiles* CMake generator, or, more generally (for all platforms) by running
```cmd
cmake --build . --target docs
```
in the build directory (`acousticFieldPropagator-OMP-build` in the example above).

The resulting documentation is in the html format and can be found inside `Doxygen/html` directory. You can browse it by opening `Doxygen/html/index.html` with your web browser.

If you are using repository sources, please make sure you have set up *filedoc* in your repository properly and the file headers are filled up (see [Contributing](#contributing) below). Alternatively, run `.gitproject/header_hook.py` in the repository root (requires _python 3.x_ with _pygit2 1.x.x_ package). If you downloaded the sources in an archive, the file headers should be ready.


## Running the binaries

Compiled binaries are typically run by the MATLAB k-Wave scripts directly, namely through `acousticFieldPropagatorC.m` wrapper function. Please refer to its documentation in case you are a MATLAB user. If you want to use the binaries directly, write your own scripts, or experiment with them directly, the interface is described below.

### Command-line arguments

The `acousticFieldPropagator-OMP` binary has a command line interface (no graphical one), accepting (and requiring) a few arguments to run. The available arguments are:
+ `-i <input_file>` _(required)_ specifying the input file of the simulation,
+ `-o <output_file>` _(required)_ specifying the output file to use,
+ `-t <num_of_threads>` modifying the number of threads to use (number of logical processors available is used by default),
+ `-c` to output a single complex (4 dimensional) dataset `pressure_out` **instead of** `amp_out` and `phase_out` (see [Output file format](output-file-format) below),
+ `-h` or `--help` to display a help message about the parameters and exit.

To run the simulation kernel on GNU/Linux with the provided example data input file one would execute
```sh
./acousticFieldPropagator-OMP -i <source_directory>/Data/input_data_96_64_64.h5 -o output_data.h5
```

### Input file format

All the information required for the actual simulation are passed inside a HDF5 file. This input file is expected to contain several datasets in its root:

| Dataset name | Description                                                                                                                                                                                                                    | Units     |
|:------------:| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |:---------:|
| `amp_in`     | 3-dimensional cartesian matrix containing pressure amplitude of the source elements in the domain                                                                                                                              | \[Pa\]    |
| `c0`         | medium sound speed (scalar value)                                                                                                                                                                                              | \[m/s\]   |
| `dx`         | grid spacing (referring to `amp_in` and `phase_in` datasets, grid is considered uniform)                                                                                                                                       |           |
| `phase_in`   | 3-dimensional cartesian matrix conaining pressure phase of the source elements in the domain (matrix size must match the `amp_in` dataset)<br/>or, alternatively, a scalar value if the phase is the same for the whole domain | \[rad\]   |
| `sz_ex`      | 3-element vector specifying the extended domain size used for computation to prevent wrapping the waves back into the domain                                                                                                   |           |
| `t`          | time at which the pressure field is calculated (scalar value)                                                                                                                                                                  | \[s\]     |
| `w0`         | angular frequency of the source (scalar value)                                                                                                                                                                                 | \[rad/s\] |

__All the present datasets *must contain SIMPLE dataspace* as defined by HDF5, whether or not the data is scalar, one- or multi-dimensional.__

Please note that the time `t` needs to be sufficient enough for the waves to cover whole domain, and, as a consequence, `sz_ex` needs to be chosen so that no waves wraps back into the input grid. Memory requirements are directly dependent on the `sz_ex` value. This is a consequence of the fact that the computational domain is periodic due to the employed method.

Consult the MATLAB function `acousticFieldPropagator` on how to specify the `t1` and `sz_ex` datasets. These values are calculated both in the MATLAB function and MATLAB C++ wrapper automatically.

Additionally, the file *may* contain the following *file attributes* (attributes of the *root HDF5 group*):

| Attribute name   | Description                                             | Expected value |
|:----------------:| ------------------------------------------------------- | -------------- |
| file_description | Description of the file content, i. e. simulation setup | N/A            |
| file_type        | Type of the file                                        | `afp_input`    |
| major_version    | Major file format version                               | `1`            |
| minor_version    | Minor file format version                               | `0`            |

The `file_description` attribute is optional and when present its content is copied over to the output file. If the other attributes are omitted a warning message is printed to the error output. If they are present they must contain the expected value.

__All the present attributes *must contain SCALAR dataspace* as defined by HDF5 and the content, including the version numbers, *must be stored in a fixed-length string format*.__

### Output file format

The resulting file contains the following datasets:

| Dataset name   | Description                                                                                                                                                                                                                                                                                        | Units   |
|:--------------:| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:-------:|
| `amp_out`      | 3-dimensional cartesian matrix containing resulting pressure amplitude in the domain                                                                                                                                                                                                               | \[Pa\]  |
| `phase_out`    | 3-dimensional cartesian matrix containing resulting pressure phase in the domain                                                                                                                                                                                                                   | \[rad\] |
| `pressure_out` | 4-dimensional matrix containing resulting pressure in a complex form, with 3-dimensional real and imaginary components "glued"<br/>in the 4-th dimension (with a length of 2), this dataset **replaces** the above two when the `-c` [command line argument](#command-line-arguments) is specified | \[Pa\]  |

Output matrix sizes correspond to the input domain size, thus is the same as for the `amp_in` dataset. Additionally, the output file contains the following attributes in a string format, attached to the file root group:
+ `created_by`, `creation_date`, `file_type`, `major_version`, `minor_version`,

    where `file_type` is `afp_output` and file version `1` and `0` respectively,
+ `file_description`,

    with the description copied over from the input file if present (it is omitted otherwise),
+ `host_names`, `number_of_cpu_cores`, `data_loading_phase_execution_time`, `pre-processing_phase_execution_time`, `simulation_phase_execution_time`, `post-processing_phase_execution_time`, `data_storing_phase_execution_time`, `total_execution_time`, `peak_core_memory_in_use`, `total_memory_in_use`

    (information and statistics) with the semantics equivalent to the other k-Wave accelerated C++ codes.

All the attributes are in a human-readable format along with their units wherever applicable.


## Contributing

For developers with access to the source repository, please
+ follow the coding style,
+ make sure the code is properly documented,
+ setup *filedoc* for automatic file header generation (install `.gitproject/clean_filter.py` as a clean filter and `.gitproject/header_hook.py` as a `post-checkout` and `post-commit` hook, requires _python 3.x_ with _pygit2 1.x.x_ package)
+ make sure there is a placeholder for automatically generated documentation in every new C++ file or header,
+ insert your contact information and affiliation into the `CONTRIBUTORS.json` file with your first commit.

If you don't have repository access, please send requested patches to the authors.
