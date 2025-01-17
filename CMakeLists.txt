cmake_minimum_required(VERSION 3.10)
project(LinearRegressionExample)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(main src/main.cpp src/linear_regression.cpp src/perceptron.cpp)

# Include directories for headers
target_include_directories(main PRIVATE include include/ML-LITE)

