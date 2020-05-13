cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

cmake_policy(SET CMP0074 NEW)
project(acousticFieldPropagator-OMP CXX C)
set(CMAKE_CXX_STANDARD_REQUIRED "on")
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_INCLUDE_CURRENT_DIR "on")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

#
# Properties that needs to be set when building the project
#
# - build with FFTW instead of MKL
set(USE_FFTW false CACHE BOOL "Use FFTW library instead of Intel MKL")
if (USE_FFTW)
    add_definitions("-DUSE_FFTWF")
endif ()
# - statically link if possible
set(STATIC_LINKING false CACHE BOOL "Link libraries statically if possible")
# - set kWave release version if desired
set(KWAVE_VERSION "" CACHE STRING "kWave release version number to be included in the header")
if (KWAVE_VERSION)
    add_definitions("-DKWAVE_VERSION")
endif ()

#
# Platform dependent settings
#
# - _USE_MATH_DEFINES needs to be defined since those are non-standard for the Windows build
# - set Windows to not handle C asynchronous exceptions by C++ catch statements
if (WIN32)
    add_definitions("/D_USE_MATH_DEFINES")
    add_compile_options("/EHsc")
endif ()
# - prepend library suffixes to correctly link HDF5 libraries on *NIX platforms when linking statically
if (UNIX AND STATIC_LINKING)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif ()

#
# OpenMP
#
# - OpenMP support is insufficient with MSVC
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message(WARNING "OpenMP support is disabled for MSVC compiler")
else ()
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # If linking statically with an Intel compiler, try to enforce static linking of the Intel OpenMP
    # runtime library. This may become deprecated in the future and is already impossible on Windows.
    if (UNIX AND STATIC_LINKING AND CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -qopenmp-link=static")
    endif ()
endif ()

#
# Hierarchical Data File 5
#
# - component selection
if (STATIC_LINKING)
    set(HDF5_SEARCH_TYPE "static")
else ()
    set(HDF5_SEARCH_TYPE "shared")
endif ()
# - package search
find_package(HDF5 COMPONENTS "C" ${HDF5_SEARCH_TYPE} NO_MODULE QUIET)
if (HDF5_FOUND)
    # - library pick
    if (STATIC_LINKING)
        set(HDF5_LIBRARIES ${HDF5_C_STATIC_LIBRARY})
    else ()
        set(HDF5_LIBRARIES ${HDF5_C_SHARED_LIBRARY})
    endif ()
    include_directories(${HDF5_INCLUDE_DIR})
else ()
    # - fallback, we can handle this under UNIX systems still, but not so great
    if (STATIC_LINKING AND WIN32)
        message(FATAL_ERROR "Install HDF5 library built with CMake support to enable static linking")
    else ()
        message("HDF5 library with CMake support not found, falling back to module provided with CMake")
    endif ()
    if (STATIC_LINKING)
        message(WARNING "Static linking may not work properly for HDF5 library, install HDF5 library built with "
                "CMake support to enable static linking")
    endif ()
    # - package search
    find_package(HDF5 COMPONENTS "C" REQUIRED)
    include_directories(${HDF5_INCLUDE_DIRS})
    add_definitions(${HDF5_DEFINITIONS})
endif ()
# - required definition
if (WIN32 AND NOT STATIC_LINKING)
    add_definitions("/DH5_BUILT_AS_DYNAMIC_LIB")
endif ()

#
# Fast Fourier Transform
#
if (USE_FFTW)
    # - decide if OpenMP implementation should be used
    if (NOT WIN32)
        set(FFTW3f_USE_OPENMP "on")
    endif ()
    # - package search
    find_package(FFTW3f REQUIRED)
    include_directories(${FFTW3f_INCLUDE_DIRS})
    add_definitions(${FFTW3f_DEFINITIONS})
else () # using Intel MKL as FFT backend
    # - handle static linking
    if (STATIC_LINKING)
        set(IMKL_USE_STATIC_LIBRARIES "on")
    endif ()
    # - package search
    find_package(IMKL REQUIRED)
    include_directories(${IMKL_INCLUDE_DIRS})
    add_definitions(${IMKL_DEFINITIONS})
endif ()

#
# Project content
#
# - HDF5 IO wrappers
aux_source_directory("Hdf5Io" hdf5io_src)
add_library(hdf5io OBJECT ${hdf5io_src})
set(objects ${objects} $<TARGET_OBJECTS:hdf5io>)

# - computational kernels
aux_source_directory("Kernels" kernels_src)
add_library(kernels OBJECT ${kernels_src})
set(objects ${objects} $<TARGET_OBJECTS:kernels>)

# - matrix with FFT support
if (USE_FFTW)
    add_library(fftmatrix OBJECT "Types/FftMatrixFftwf.cpp")
else () # using Intel MKL as FFT backend
    add_library(fftmatrix OBJECT "Types/FftMatrixMkl.cpp")
endif ()
set(objects ${objects} $<TARGET_OBJECTS:fftmatrix>)

# - argument parsing
add_library(arguments OBJECT "Utils/ArgumentParser.cpp")
set(objects ${objects} $<TARGET_OBJECTS:arguments>)

# - hostname
if (WIN32)
    add_library(hostname OBJECT "Utils/HostnameWin.cpp")
    set(libraries ${libraries} "ws2_32")
else ()
    add_library(hostname OBJECT "Utils/HostnameLin.cpp")
endif ()
set(objects ${objects} $<TARGET_OBJECTS:hostname>)

# - memory statistics
if (WIN32)
    add_library(memory OBJECT "Utils/MemoryWin.cpp")
else ()
    add_library(memory OBJECT "Utils/MemoryLin.cpp")
endif ()
set(objects ${objects} $<TARGET_OBJECTS:memory>)

# - main module
add_library(main OBJECT "main.cpp")
set(objects ${objects} $<TARGET_OBJECTS:main>)

# - version module
include("Version/CMakeLists.txt")
set(objects ${objects} $<TARGET_OBJECTS:VERSION_OBJECTS>)

# Link everything together
add_executable(${PROJECT_NAME} ${objects})
target_link_libraries(${PROJECT_NAME} ${HDF5_LIBRARIES} ${IMKL_LIBRARIES} ${FFTW3f_LIBRARIES} ${libraries})

# Doxygen documentation "docs" target
find_package(Doxygen)
if (DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME "Acoustic Field Propagator")
    set(DOXYGEN_PROJECT_LOGO "header_bg.png")
    set(DOXYGEN_OUTPUT_DIRECTORY "Doxygen")
    set(DOXYGEN_TAB_SIZE "2")
    set(DOXYGEN_TOC_INCLUDE_HEADINGS "0")
    set(DOXYGEN_BUILTIN_STL_SUPPORT "YES")
    set(DOXYGEN_PROJECT_NUMBER ${PROJECT_GIT_REVISION})
    set(DOXYGEN_FILE_PATTERNS "*.cpp" "*.h")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "README.md")
    doxygen_add_docs("docs" ${CMAKE_CURRENT_SOURCE_DIR} "README.md" "LICENSE.md" "CHANGELOG.md")
    message("You can now build the Doxygen documentation by specifying '--target docs' along with '--build .' in the cmake command")
else ()
    message(WARNING "Doxygen not found, you won't be able to build the Doxygen documentation")
endif ()
