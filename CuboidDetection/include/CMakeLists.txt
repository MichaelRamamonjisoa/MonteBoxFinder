add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_sources(cxxopts INTERFACE "cxxopts.hpp")
