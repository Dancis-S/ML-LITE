cmake_minimum_required(VERSION 3.10)
project(ML-LITE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Main executable
add_executable(main 
    src/main.cpp 
    src/linear_regression.cpp 
    src/perceptron.cpp 
    src/k_means.cpp 
    src/decision_tree.cpp
 "test/test_decision_tree.cpp" "test/test_k_means.cpp")

target_include_directories(main PRIVATE include)

# Enable testing
enable_testing()

# Find and link GTest if available
find_package(GTest)
if(GTEST_FOUND)
    message(STATUS "GTest found: ${GTEST_INCLUDE_DIRS}")

    add_executable(test_ml_lite 
        test/test_perceptron.cpp 
        src/perceptron.cpp 
        src/k_means.cpp 
        src/decision_tree.cpp
     "test/test_decision_tree.cpp" "test/test_k_means.cpp")

    target_include_directories(test_ml_lite PRIVATE include)
    target_link_libraries(test_ml_lite PRIVATE GTest::GTest GTest::Main pthread)
    add_test(NAME PerceptronTests COMMAND test_ml_lite)
else()
    message(STATUS "GTest not found, skipping test target")
endif()

