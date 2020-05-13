# Try to find MKL library, focuses on core functionality with OpenMP
# Once done, this will define:
#   IMKL_FOUND
#   IMKL_INCLUDE_DIRS
#   IMKL_LIBRARIES
#   IMKL_DEFINITIONS
#
# You can also set IMKL_ROOT variable to hint the search for the MKL library.
#
# OpenMP support is picked based on the compiler and currently used OpenMP runtime.
#
# If you wish to link the library statically, set IMKL_USE_STATIC_LIBRARIES CMake variable.

include(FindPackageHandleStandardArgs)

find_path(IMKL_ROOT "include/mkl.h" PATHS ${IMKL_ROOT} PATH_SUFFIXES "mkl")
if (IMKL_FIND_REQUIRED AND NOT IMKL_ROOT)
    message(FATAL_ERROR "Intel MKL root directory was not found")
endif ()
message("Using MKL root directory: ${IMKL_ROOT}")

find_path(IMKL_INCLUDE_DIR "mkl.h" HINTS ${IMKL_ROOT} PATH_SUFFIXES "include" NO_DEFAULT_PATH)

# handle library naming
set(IMKL_LIBRARY_NAME_PREFIX "")
set(IMKL_LIBRARY_NAME_SUFFIX "")
if (WIN32)
    if (IMKL_USE_STATIC_LIBRARIES)
        set(IMKL_LIBRARY_NAME_SUFFIX ".lib")
    else ()
        set(IMKL_LIBRARY_NAME_SUFFIX "_dll.lib")
    endif ()
else ()
    if (IMKL_USE_STATIC_LIBRARIES)
        set(IMKL_LIBRARY_NAME_PREFIX "lib")
        set(IMKL_LIBRARY_NAME_SUFFIX ".a")
    endif ()
endif ()

# search for the libraries
find_library(IMKL_ILP64 NAMES "${IMKL_LIBRARY_NAME_PREFIX}mkl_intel_ilp64${IMKL_LIBRARY_NAME_SUFFIX}"
        HINTS "${IMKL_ROOT}/lib/intel64" NO_DEFAULT_PATH)
find_library(IMKL_CORE NAMES "${IMKL_LIBRARY_NAME_PREFIX}mkl_core${IMKL_LIBRARY_NAME_SUFFIX}"
        HINTS "${IMKL_ROOT}/lib/intel64" NO_DEFAULT_PATH)

# handle OpenMP support
if (CMAKE_C_COMPILER_ID MATCHES "GNU" AND OpenMP_FOUND)
    # the GNU OpenMP is already in use, keep it
    set(IMKL_THREAD_MODEL "gnu_thread")
elseif (CMAKE_C_COMPILER_ID MATCHES "Intel" AND OpenMP_FOUND)
    # the Intel OpenMP is already in use, keep it
    set(IMKL_THREAD_MODEL "intel_thread")
else ()
    # will try to find and use Intel OpenMP runtime library
    set(CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_FIND_LIBRARY_PREFIXES} "lib")
    find_library(IMKL_OPENMP_LIB NAMES "iomp5md")
    if (IMKL_OPENMP_LIB)
        set(IMKL_THREAD_MODEL "intel_thread")
    else ()
        # try GNU OpenMP runtime library instead
        find_library(IMKL_OPENMP_LIB NAMES "gomp")
        if (IMKL_OPENMP_LIB)
            set(IMKL_THREAD_MODEL "gnu_thread")
        else ()
            set(IMKL_THREAD_MODEL "sequential")
            set(IMKL_OPENMP_LIB "")
        endif ()
    endif ()
    mark_as_advanced(IMKL_OPENMP_LIB)
endif ()

message("Intel MKL thread model \"${IMKL_THREAD_MODEL}\" selected")
find_library(IMKL_THREAD NAMES "${IMKL_LIBRARY_NAME_PREFIX}mkl_${IMKL_THREAD_MODEL}${IMKL_LIBRARY_NAME_SUFFIX}"
        HINTS "${IMKL_ROOT}/lib/intel64" NO_DEFAULT_PATH)


find_package_handle_standard_args("IMKL" "Intel MKL library not found"
        IMKL_INCLUDE_DIR IMKL_ILP64 IMKL_CORE IMKL_THREAD)

mark_as_advanced(IMKL_INCLUDE_DIR IMKL_ILP64 IMKL_CORE IMKL_THREAD IMKL_THREAD_MODEL IMKL_LIBRARY_NAME_PREFIX
        IMKL_LIBRARY_NAME_SUFFIX)

set(IMKL_INCLUDE_DIRS ${IMKL_INCLUDE_DIR})
set(IMKL_LIBRARIES ${IMKL_ILP64} ${IMKL_CORE} ${IMKL_THREAD})
set(IMKL_DEFINITIONS "-DMKL_ILP64")

# handle dependency linker flags
if (CMAKE_C_COMPILER_ID MATCHES "GNU" AND NOT IMKL_USE_STATIC_LIBRARIES)
    set(IMKL_LIBRARIES "-Wl,--no-as-needed" ${IMKL_LIBRARIES} "-Wl,--as-needed")
elseif (UNIX AND IMKL_USE_STATIC_LIBRARIES)
    set(IMKL_LIBRARIES "-Wl,--start-group" ${IMKL_LIBRARIES} "-Wl,--end-group")
endif ()

# append additional required libraries
set(IMKL_LIBRARIES ${IMKL_LIBRARIES} ${IMKL_OPENMP_LIB})

#if (UNIX)
#    set(IMKL_LIBRARIES ${IMKL_LIBRARIES} "pthread" "m" "dl")
#endif ()
