cmake_minimum_required(VERSION 3.10)
project(LinearRegressionExample)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(main src/main.cpp src/linear_regression.cpp src/perceptron.cpp)

# Include directories for headers
target_include_directories(main PRIVATE include include/ML-LITE)

# Testing
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable(test_ml_lite test/test_perceptron.cpp src/perceptron.cpp)
target_link_libraries(test_ml_lite ${GTEST_LIBRARIES} pthread)

add_test(NAME PerceptronTests COMMAND test_ml_lite)
