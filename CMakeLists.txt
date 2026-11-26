cmake_minimum_required(VERSION 3.16)
project(Computer-Stablized-Ground-Effect-Vehicle-UAV CXX)

cmake_policy(SET CMP0135 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.13 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
)
FetchContent_MakeAvailable(eigen)

set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(control_module src/bindings.cpp)
target_include_directories(control_module PRIVATE "${eigen_SOURCE_DIR}")

set_target_properties(control_module PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/simulation"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/simulation"
)

target_link_options(control_module PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic)