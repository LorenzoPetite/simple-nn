project("NeuralNet")

cmake_minimum_required(VERSION 3.9)

set(MATPLOPLIB_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/matploplibcpp)

add_executable(train src/train.cpp ${MATPLOPLIB_CPP_ROOT}/matplotlibcpp.h)

set_property(TARGET train PROPERTY CXX_STANDARD 17)
set_property(TARGET train PROPERTY CXX_STANDARD_REQUIRED ON)

message(STATUS "Trying to include Eigen library")
set(EIGEN3_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Eigen3 3.2.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "EIGEN: " ${EIGEN3_VERSION} " (internal)")

find_package(PythonLibs 2.7 REQUIRED)
include_directories(${MATPLOPLIB_CPP_ROOT})
message(STATUS "Python: " ${PYTHONLIBS_VERSION_STRING})

include_directories(${PYTHON_INCLUDE_DIRS})

set(MNIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mnist)
find_package(MNIST REQUIRED)
	if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
else()
	message(STATUS "MNIST_DATA_DIR: " ${MNIST_DATA_DIR})
endif(NOT MNIST_FOUND)
include_directories(${MNIST_INCLUDE_DIR})
target_compile_definitions(train PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

#set(CMAKE_CXX_FLAGS "-g -Wall")
set(CMAKE_BUILD_TYPE Debug)

#message("Your C++ compiler supports these C++ features:")
#foreach(i ${CMAKE_CXX_COMPILE_FEATURES})
  #message("${i}")
#endforeach()

add_subdirectory(src)

target_link_libraries(train Net ${PYTHON_LIBRARIES})



