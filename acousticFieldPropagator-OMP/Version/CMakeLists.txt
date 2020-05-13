# Create an object file containing version information

# Define required variables
set(PROJECT_GIT_REVISION "")
set(PROJECT_GIT_STATUS "status not available")

#  - git information
find_package(Git QUIET)
if (GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} "describe" "--always" "HEAD" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE git_result OUTPUT_VARIABLE git_output)
    if (NOT git_result) # result is 0, thus success
        string(STRIP "${git_output}" PROJECT_GIT_REVISION)
        execute_process(COMMAND ${GIT_EXECUTABLE} "status" "-s" "-uno" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE git_result OUTPUT_VARIABLE git_output)
        if (git_output)
            execute_process(COMMAND ${GIT_EXECUTABLE} "diff" "--shortstat" "HEAD" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE git_result OUTPUT_VARIABLE git_output)
            string(STRIP "${git_output}" PROJECT_GIT_STATUS)
        else ()
            set(PROJECT_GIT_STATUS "clean")
        endif ()
    endif ()
endif ()

#  - fallback for version - git was not found or the source is not a part of the repository
if (NOT PROJECT_GIT_REVISION)
    file(STRINGS "Version/Version.h" version_comment REGEX "^ [*] @version .*$")
    list(LENGTH version_comment version_comment_len)
    if (version_comment_len EQUAL 1)
        string(REGEX MATCH "^ [*] @version (.*)$" version_comment "${version_comment}")
        string(STRIP "${CMAKE_MATCH_1}" PROJECT_GIT_REVISION)
    endif ()
endif ()

if (NOT PROJECT_GIT_REVISION)
    message(FATAL_ERROR "Unable to obtain a valid revision string, please re-download the sources")
endif ()

#  - parse the version from the revision string
string(REGEX MATCH "^(v([0-9]+)[.]([0-9]+)[.]([0-9]+)(-([^0-9][^-]*))?(-([0-9]+)-)?)?([a-z0-9]+)?$" PROJECT_GIT_REVISION "${PROJECT_GIT_REVISION}")

if (NOT PROJECT_GIT_REVISION)
    message(FATAL_ERROR "Unknown revision format")
endif ()

set(PROJECT_VERSION_MAJOR "${CMAKE_MATCH_2}")
if (NOT PROJECT_VERSION_MAJOR)
    set(PROJECT_VERSION_MAJOR 0)
endif ()
set(PROJECT_VERSION_MINOR "${CMAKE_MATCH_3}")
if (NOT PROJECT_VERSION_MINOR)
    set(PROJECT_VERSION_MINOR 0)
endif ()
set(PROJECT_VERSION_PATCH "${CMAKE_MATCH_4}")
if (NOT PROJECT_VERSION_PATCH)
    set(PROJECT_VERSION_PATCH 0)
endif ()
set(PROJECT_VERSION_HASH "${CMAKE_MATCH_9}")

#  - build date
string(TIMESTAMP PROJECT_BUILD_DATE_TIME "%Y-%m-%d %H:%M:%S UTC" UTC)

# Preprocess the template
configure_file("Version/Version.cpp.in" "${PROJECT_BINARY_DIR}/Version.cpp" @ONLY)

add_library(VERSION_OBJECTS OBJECT "${PROJECT_BINARY_DIR}/Version.cpp")

# We want to run this preprocessing every time the project is built
add_custom_target(RerunCmake ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR})
add_dependencies(VERSION_OBJECTS RerunCmake)
