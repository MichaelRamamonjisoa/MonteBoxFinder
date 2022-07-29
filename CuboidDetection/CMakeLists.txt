cmake_minimum_required(VERSION 3.15)
project(plane_detector)
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(src)

find_package(Eigen3 REQUIRED)

# CGAL
set(CGAL_DIR CGAL-5.2.2)
find_package(CGAL 5.2.2 REQUIRED)

#set(JSON_DIR jsoncpp)
#find_package(JSONCPP)

find_package(PkgConfig REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

MESSAGE( [Main] " EIGEN3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}")
# EIGEN3_INCLUDE_DIRS = /usr/include/eigen3
# MESSAGE( [Main] " JSONCPP_DIR = ${JSONCPP_DIR}")

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CGAL_DIR})
#include_directories(${JSONCPP_DIR})

#pkg_check_modules(JSONCPP jsoncpp)

add_subdirectory(${plane_detector_SOURCE_DIR}/include)

add_executable(ransac_plane_detector main.cpp value_color.h)
target_link_libraries(ransac_plane_detector CGAL::CGAL lib_src cxxopts jsoncpp)#${JSONCPP_LIBRARIES})
