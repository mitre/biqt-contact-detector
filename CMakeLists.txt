# #######################################################################
# NOTICE
#
# This software (or technical data) was produced for the U.S. Government
# under contract, and is subject to the Rights in Data-General Clause
# 52.227-14, Alt. IV (DEC 2007).
#
# Copyright 2020 The MITRE Corporation. All Rights Reserved.
# ####################################################################### 

cmake_minimum_required(VERSION 3.22)
project(BIQTContactDetector)
set(CMAKE_CXX_STANDARD 11)

OPTION(BUILD_SHARED_LIBS "Builds shared libraries for certain dependencies. Recommended: ON"  ON)
OPTION(BUILD_STATIC_LIBS "Builds static libraries for certain dependencies. Recommended: OFF" OFF)

if(WIN32)
	message(FATAL_ERROR "Windows is not a supported build target for this provider. Please build in linux.")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "-g -fPIC")

if(DEFINED ENV{BIQT_HOME})
	if(BIQT_HOME)
		message(WARNING "Both -DBIQT_HOME and a BIQT_HOME environment variable are defined. Using -DBIQT_HOME (${BIQT_HOME}).")
	else()
	  	set(BIQT_HOME "$ENV{BIQT_HOME}")
  	endif()  
endif()

if(NOT BIQT_HOME)
	message(FATAL_ERROR "BIQT_HOME must be defined as an environment variable or passed to CMake as a parameter using -DBIQT_HOME.")
else()
	set(TO_CMAKE_PATH ${BIQT_HOME} BIQT_HOME)
endif()

if(CMAKE_INSTALL_PREFIX)
	# Ignore CMAKE_INSTALL_PREFIX, but let the user know about it.
	message(WARNING "Overriding provided CMAKE_INSTALL_PREFIX with ${BIQT_HOME}/providers.")
endif()

set(CMAKE_INSTALL_PREFIX "${BIQT_HOME}/providers")

find_package(jsoncpp REQUIRED)
find_library(BIQTAPI NAMES libbiqtapi biqtapi HINTS ${BIQT_HOME}/lib ${BIQT_HOME}/lib64 ${BIQT_HOME}/bin ${BIQT_HOME}/../../lib ${BIQT_HOME}/../../lib64 ${BIQT_HOME}/../../bin)

if(NOT BIQTAPI)
	message(FATAL_ERROR "Failed to find library: biqtapi.")
endif()

find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
# Add a PATHS hint for pyenv.
find_package(pybind11 REQUIRED PATHS ${Python3_EXECUTABLE}/../../versions/${Python3_VERSION}/share/cmake)

include_directories(. "include" ${BIQT_HOME}/include ${BIQT_HOME}/../../include ${Python3_INCLUDE_DIRS})

# Create the provider library 
file(GLOB SOURCE_FILES "src/cxx/*.cpp")

# Determine if the cosmetic file named COSMETIC_LENS_MODEL_FILE has already been retrieved and placed into the source directory. If it hasn't,
# retrieve it from the web.
#   LOCAL_COSMETIC_LENS_MODEL_PATH        The path to a model file that should be installed.
#   TARGET_COSMETIC_LENS_MODEL_FILE_NAME  The name of the model expected by the provider (e.g., efficientnet-v2l-CDM.ckpt).
#   TARGET_COSMETIC_LENS_MODEL_PATH       The expected location of the model file within the source tree.
#   TARGET_COSMETIC_LENS_MODEL_FILE_PATH  The combination of the TARGET_COSMETIC_LENS_MODEL_PATH and TARGET_COSMETIC_LENS_MODEL_FILE_NAME.
set(TARGET_COSMETIC_LENS_MODEL_FILE_NAME "efficientnet-v2l-CDM.ckpt")
set(TARGET_COSMETIC_LENS_MODEL_PATH      "${CMAKE_CURRENT_SOURCE_DIR}/config/models/")
set(TARGET_COSMETIC_LENS_MODEL_FILE_PATH "${TARGET_COSMETIC_LENS_MODEL_PATH}/${TARGET_COSMETIC_LENS_MODEL_FILE_NAME}")
set(LOCAL_COSMETIC_LENS_MODEL_PATH       "${TARGET_COSMETIC_LENS_MODEL_FILE_PATH}" CACHE FILEPATH "The path to a model file that should be installed.")
set(DOWNLOAD_MODELS OFF CACHE BOOL "Force download of model.")
if(DOWNLOAD_MODELS)
	message(STATUS "Models will be downloaded (forced).")
elseif(EXISTS "${LOCAL_COSMETIC_LENS_MODEL_PATH}")
	message(STATUS "Found a local cosmetic lens model at ${LOCAL_COSMETIC_LENS_MODEL_PATH}.")
	configure_file("${LOCAL_COSMETIC_LENS_MODEL_PATH}" "${TARGET_COSMETIC_LENS_MODEL_FILE_PATH}" COPYONLY)
else()
	message(STATUS "A local cosmetic lens model could not be found. It will be downloaded.")
	set(DOWNLOAD_MODELS ON)
endif()

if(DOWNLOAD_MODELS)
	set(REMOTE_COSMETIC_MODEL_URL "https://github.com/mitre/biqt-contact-detector/releases/download/v24.11/efficientnet-v2l-CDM.ckpt")
	file(DOWNLOAD "${REMOTE_COSMETIC_MODEL_URL}" "${TARGET_COSMETIC_LENS_MODEL_FILE_PATH}" SHOW_PROGRESS STATUS DOWNLOAD_STATUS_COSMETIC_MODEL)
	list(GET DOWNLOAD_STATUS_COSMETIC_MODEL 0 DOWNLOAD_FAILED)
	if(DOWNLOAD_FAILED)
		message(FATAL_ERROR "Download ${REMOTE_COSMETIC_MODEL_URL} failed with response: ${DOWNLOAD_STATUS_COSMETIC_MODEL}!")
	endif()
endif()

add_library(BIQTContactDetector SHARED ${SOURCE_FILES} ${EXTRA_FILES})

# numpy likes to have the libpython3 library loaded when being called from cxx.
get_filename_component(PYLIB_NAME "${Python3_LIBRARY_RELEASE}" NAME)
add_definitions(-DPYTHON_LIB="${PYLIB_NAME}")
target_link_libraries(BIQTContactDetector jsoncpp_lib ${BIQTAPI} pybind11::embed ${Python3_LIBRARIES} dl)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/descriptor.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/python/inference DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src/python)

# Set installation rules
install(TARGETS BIQTContactDetector DESTINATION "${CMAKE_PROJECT_NAME}")
install(FILES descriptor.json DESTINATION "${CMAKE_PROJECT_NAME}")
install(DIRECTORY config DESTINATION "${CMAKE_PROJECT_NAME}")
install(DIRECTORY src/python/inference DESTINATION "${CMAKE_PROJECT_NAME}/src/python")
