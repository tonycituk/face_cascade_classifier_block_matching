cmake_minimum_required(VERSION 4.0)
project(cascade_classifier)

set(CMAKE_CXX_STANDARD 20)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cascade_classifier main.cpp)

target_link_libraries(cascade_classifier ${OpenCV_LIBS})
