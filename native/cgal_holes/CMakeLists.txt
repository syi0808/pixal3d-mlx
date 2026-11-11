cmake_minimum_required(VERSION 3.22)
project(pixal_cgal_holes LANGUAGES CXX)

find_package(CGAL REQUIRED)
if(CGAL_VERSION VERSION_LESS 5.6)
    message(FATAL_ERROR "CGAL 5.6 or newer is required")
endif()
find_package(Eigen3 REQUIRED NO_MODULE)
include(CGAL_Eigen3_support)

add_executable(pixal-cgal-holes main.cpp)
target_compile_features(pixal-cgal-holes PRIVATE cxx_std_20)
target_link_libraries(pixal-cgal-holes PRIVATE CGAL::CGAL CGAL::Eigen3_support)
