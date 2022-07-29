include_directories(${plane_detector_SOURCE_DIR}/include)
add_library(lib_src cuboids.cpp pointclouds.cpp)
find_package(CGAL REQUIRED)

# JSONCPP
add_library(jsoncpp jsoncpp.cpp)

#link_libraries(${JSONCPP_LIBRARIES})

target_link_libraries(lib_src ${JSONCPP_LIBRARIES} CGAL::CGAL)