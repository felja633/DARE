cmake_minimum_required(VERSION 3.5)
project(math_utils_project)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -fopenmp")
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR})

pybind11_add_module(math_utils SHARED math_utils.cpp)


