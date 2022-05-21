cmake_minimum_required(VERSION 3.5)

project(autofocus LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(src)
