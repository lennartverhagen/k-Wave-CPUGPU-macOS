# Try to find FFTW3f library
# Uses pkg-config to get hints on library locations.
# Once done, this will define:
#   FFTW3f_FOUND
#   FFTW3f_INCLUDE_DIRS
#   FFTW3f_LIBRARIES
#   FFTW3f_DEFINITIONS
#
# You can also set FFTW3f_ROOT variable to hint the search for the FFTW library. Additionally, if you want to use
# an OpenMP version of the library, set FFTW3f_USE_OPENMP CMake variable.

include(FindPackageHandleStandardArgs)
find_package(PkgConfig QUIET)

pkg_check_modules(PC_FFTW3f QUIET "fftw3f")

find_path(FFTW3f_INCLUDE_DIR "fftw3.h"
        HINTS ${PC_FFTW3f_INCLUDEDIR} ${PC_FFTW3f_INCLUDE_DIRS}
        PATHS ${FFTW3f_ROOT}
        PATH_SUFFIXES "fftw" "fftw3" "include")

set(ORIGINAL_CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_FIND_LIBRARY_PREFIXES})
set(CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_FIND_LIBRARY_PREFIXES} "lib")
find_library(FFTW3f_LIBRARY NAMES "fftw3f" "fftw3f-3"
        HINTS ${PC_FFTW3f_LIBDIR} ${PC_FFTW3f_LIBRARY_DIRS}
        PATHS ${FFTW3f_ROOT}
        PATH_SUFFIXES "lib")

if (FFTW3f_USE_OPENMP)
    find_library(FFTW3f_OMP_LIBRARY NAMES "fftw3f_omp" "fftw3f-3_omp"
            HINTS ${PC_FFTW3f_LIBDIR} ${PC_FFTW3f_LIBRARY_DIRS}
            PATHS ${FFTW3f_ROOT}
            PATH_SUFFIXES "lib")

    find_package_handle_standard_args("FFTW3f" DEFAULT_MSG FFTW3f_INCLUDE_DIR FFTW3f_LIBRARY FFTW3f_OMP_LIBRARY)
else ()
    find_package_handle_standard_args("FFTW3f" DEFAULT_MSG FFTW3f_INCLUDE_DIR FFTW3f_LIBRARY)
endif ()

mark_as_advanced(FFTW3f_INCLUDE_DIR FFTW3f_LIBRARY FFTW3f_OMP_LIBRARY)
set(CMAKE_FIND_LIBRARY_PREFIXES ${ORIGINAL_CMAKE_FIND_LIBRARY_PREFIXES})

set(FFTW3f_INCLUDE_DIRS ${FFTW3f_INCLUDE_DIR})
set(FFTW3f_LIBRARIES ${FFTW3f_OMP_LIBRARY} ${FFTW3f_LIBRARY})
set(FFTW3f_DEFINITIONS ${PC_FFTW3f_CFLAGS_OTHER})
