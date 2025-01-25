cmake_minimum_required(VERSION 3.10)
project(LinearRegressionExample)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Main executable
add_executable(main src/main.cpp src/linear_regression.cpp src/perceptron.cpp src/k_means.cpp)

# Include directories for the main executable
target_include_directories(main PRIVATE include)

# Enable testing
enable_testing()

# Find GTest
find_package(GTest REQUIRED)
if(GTest_FOUND)
    message(STATUS "GTest found: ${GTEST_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "GTest not found")
endif()

# Test executable
add_executable(test_ml_lite test/test_perceptron.cpp src/perceptron.cpp src/k_means.cpp)

# Include directories for the test executable
target_include_directories(test_ml_lite PRIVATE include)

# Link GTest and pthread for the test target
target_link_libraries(test_ml_lite PRIVATE GTest::GTest GTest::Main pthread)

# Add tests
add_test(NAME PerceptronTests COMMAND test_ml_lite)
