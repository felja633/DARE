cmake_minimum_required(VERSION 3.5)
project(pcl_util_project)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

# PCL
find_package(pybind11 REQUIRED)
find_package(PCL 1.7 REQUIRED COMPONENTS common io surface features)
find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

get_property(the_include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#message(STATUS "${the_include_dirs}")

list(APPEND new_include_dirs "")
foreach(element ${the_include_dirs})
  if(NOT ${element} STREQUAL "/usr/include/python2.7")
#  message(STATUS "dir='${new_include_dirs}'")
    list(APPEND new_include_dirs ${element})
  endif()
endforeach()

message(STATUS "${new_include_dirs}")

string(REPLACE "/usr/include/python2.7;" "" new_include_dirsads ${the_include_dirs})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES ${new_include_dirs})

get_property(the_include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#message(STATUS "${the_include_dirs}")


#add_library(pcl_utils ${LIBS})
#pybind11_add_module(pcl_utils pcl_utils.cpp)

#add_subdirectory(pybind11)
pybind11_add_module(pcl_utils SHARED pcl_utils.cpp)

list(APPEND LIBS ${PCL_LIBRARIES})
target_link_libraries(pcl_utils PRIVATE ${LIBS})

#include_directories(${PCL_INCLUDE_DIRS})
#target_link_libraries(pcl_utils ${LIBS})
#get_property(the_include_dirs DIRECTORY foo PROPERTY INCLUDE_DIRECTORIES)
#string(REPLACE "/usr/include/python2.7/" "" new_include_dirs ${the_include_dirs})
#set_property(DIRECTORY

#add_library(pcl_utils MODULE pcl_utils.cpp)
#target_link_libraries(dummy PRIVATE pybind11::module ${LIBS})
#set_target_properties(pcl_utils PROPERTIES PREFIX "")
#set_target_properties(pcl_utils PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
   #                                      SUFFIX "${PYTHON_MODULE_EXTENSION}")
# Target

#add_library(pcl_utils SHARED pcl_utils.cpp)
#target_link_libraries(pcl_utils PRIVATE ${LIBS})
#set_target_properties(pcl_utils PROPERTIES PREFIX "")
