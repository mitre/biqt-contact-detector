# #######################################################################
# NOTICE
#
# This software (or technical data) was produced for the U.S. Government
# under contract, and is subject to the Rights in Data-General Clause
# 52.227-14, Alt. IV (DEC 2007).
#
# Copyright 2020 The MITRE Corporation. All Rights Reserved.
# ####################################################################### 

cmake_minimum_required(VERSION 3.1)
project(BIQTContactDetector)
set(CMAKE_CXX_STANDARD 11)

OPTION(BUILD_SHARED_LIBS "Builds shared libraries for certain dependencies. Recommended: ON" ON)
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

find_library(JSONCPP NAMES libjsoncpp jsoncpp HINTS ${BIQT_HOME}/lib ${BIQT_HOME}/lib64 ${BIQT_HOME}/bin ${BIQT_HOME}/../../lib ${BIQT_HOME}/../../lib64 ${BIQT_HOME}/../../bin)
find_library(BIQTAPI NAMES libbiqtapi biqtapi HINTS ${BIQT_HOME}/lib ${BIQT_HOME}/lib64 ${BIQT_HOME}/bin ${BIQT_HOME}/../../lib ${BIQT_HOME}/../../lib64 ${BIQT_HOME}/../../bin)

if(NOT JSONCPP)
	message(FATAL_ERROR "Failed to find library: jsoncpp.")
endif()

if(NOT BIQTAPI)
	message(FATAL_ERROR "Failed to find library: biqtapi.")
endif()

find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 REQUIRED)

include_directories(. "include" ${BIQT_HOME}/include ${BIQT_HOME}/../../include ${Python3_INCLUDE_DIRS})

# Create the provider library 
file(GLOB SOURCE_FILES "src/cxx/*.cpp")

# Model Variables
set(COSMETIC_LENS_MODEL_FILE "binary-cosmetic-contact-lens-model.hdf5")
set(COSMETIC_LENS_MODEL_DEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/config/models")
set(COSMETIC_LENS_MODEL_DEST_FILE "${COSMETIC_LENS_MODEL_DEST_DIR}/${COSMETIC_LENS_MODEL_FILE}")
set(LOCAL_COSMETIC_LENS_MODEL_FILE ${COSMETIC_LENS_MODEL_DEST_FILE} CACHE FILEPATH "Location of local cosmetic contact lens model.")

# Check if model files already exist locally. If not, turn on download models
if(NOT EXISTS ${LOCAL_COSMETIC_LENS_MODEL_FILE})
	set(DOWNLOAD_MODELS ON CACHE BOOL "Download models from remote repo.")
else()
	set(DOWNLOAD_MODELS OFF CACHE BOOL "Download models from remote repo.")
endif()

# Handle model file installation
if(DOWNLOAD_MODELS)
	message(STATUS "Attempting to download model files...")
	set(COSMETIC_LENS_MODEL_URL "https://github.com/mitre/biqt-contact-detector/releases/download/v0.1/${COSMETIC_LENS_MODEL_FILE}")
	file(DOWNLOAD "${COSMETIC_LENS_MODEL_URL}" "${CMAKE_CURRENT_SOURCE_DIR}/config/models/${COSMETIC_LENS_MODEL_FILE}" SHOW_PROGRESS STATUS DOWNLOAD_STATUS_COSMETIC_MODEL)
	list(GET DOWNLOAD_STATUS_COSMETIC_MODEL 0 DOWNLOAD_FAILED)
	message(STATUS "${DOWNLOAD_FAILED} ${DOWNLOAD_STATUS_COSMETIC_MODEL}")
	if(DOWNLOAD_FAILED)
		message(FATAL_ERROR "Download ${COSMETIC_LENS_MODEL_FILE} failed with response: ${DOWNLOAD_STATUS_COSMETIC_MODEL}!")
	endif()
else()
	message(STATUS "Attempting to use local model files...")
	if(NOT LOCAL_COSMETIC_LENS_MODEL_FILE)
		message(FATAL_ERROR "Must provide a value for LOCAL_COSMETIC_LENS_MODEL_FILE if DOWNLOAD_MODELS is not on. Provide it using -DLOCAL_COSMETIC_LENS_MODEL_FILE")
	endif()

# Soft Lens (NYI)
#	if(EXISTS ${LOCAL_SOFT_LENS_MODEL_FILE})
#		if(NOT LOCAL_SOFT_LENS_MODEL_FILE EQUAL SOFT_LENS_MODEL_DEST_FILE)
#			file(COPY ${LOCAL_SOFT_LENS_MODEL_FILE} DESTINATION ${SOFT_LENS_MODEL_DEST_DIR})
#		endif()
#	endif()

	if(EXISTS ${LOCAL_COSMETIC_LENS_MODEL_FILE})
		if(NOT LOCAL_COSMETIC_LENS_MODEL_FILE EQUAL COSMETIC_LENS_MODEL_DEST_FILE)
			file(COPY ${LOCAL_COSMETIC_LENS_MODEL_FILE} DESTINATION ${COSMETIC_LENS_MODEL_DEST_DIR})
		endif()
	else()
		message(FATAL_ERROR "LOCAL_COSMETIC_LENS_MODEL_FILE ${LOCAL_COSMETIC_LENS_MODEL_FILE} does not exist. A cosmetic lens model must be provided.")
	endif()
endif()

if(EXISTS ${LOCAL_COSMETIC_LENS_MODEL_FILE})
	message(STATUS "Detected local cosmetic lens model ${LOCAL_COSMETIC_LENS_MODEL_FILE}")
endif()

add_compile_definitions(IS_DUAL_NETWORK=false)

add_library(BIQTContactDetector SHARED ${SOURCE_FILES} ${EXTRA_FILES})

target_link_libraries(BIQTContactDetector ${JSONCPP} ${BIQTAPI} pybind11::embed ${Python3_LIBRARIES} dl)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/descriptor.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/python/inference DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src/python)

# Set installation rules
install(TARGETS BIQTContactDetector DESTINATION "${CMAKE_PROJECT_NAME}")
install(FILES descriptor.json DESTINATION "${CMAKE_PROJECT_NAME}")
install(DIRECTORY config DESTINATION "${CMAKE_PROJECT_NAME}")
install(DIRECTORY src/python/inference DESTINATION "${CMAKE_PROJECT_NAME}/src/python")
