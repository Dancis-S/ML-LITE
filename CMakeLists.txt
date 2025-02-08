cmake_minimum_required(VERSION 3.10)
project(ML-LITE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Collect source files from subdirectories
set(SUPERVISED_SOURCES
    src/supervised/linear_regression.cpp
    src/supervised/perceptron.cpp
    src/supervised/decision_tree.cpp
    src/supervised/logistic_regression.cpp
    src/supervised/support_vector_machine.cpp
)

set(UNSUPERVISED_SOURCES
    src/unsupervised/k_means.cpp
)

set(OTHER_SOURCES
    src/utils/utils.cpp
)

set(TEST_SOURCES
    test/test_perceptron.cpp
    test/test_decision_tree.cpp
    test/test_k_means.cpp
    test/test_utils.cpp
)

# Main executable
add_executable(main 
    main.cpp 
    ${OTHER_SOURCES}
    ${SUPERVISED_SOURCES}
    ${UNSUPERVISED_SOURCES}
)

target_include_directories(main PRIVATE include)

# Enable testing
enable_testing()

# Find and link GTest if available
find_package(GTest)
if(GTEST_FOUND)
    message(STATUS "GTest found: ${GTEST_INCLUDE_DIRS}")

    add_executable(test_ml_lite 
        ${OTHER_SOURCES}
        ${TEST_SOURCES}
        ${SUPERVISED_SOURCES}
        ${UNSUPERVISED_SOURCES}
    )

    target_include_directories(test_ml_lite PRIVATE include)
    target_link_libraries(test_ml_lite PRIVATE GTest::GTest GTest::Main pthread)
    add_test(NAME PerceptronTests COMMAND test_ml_lite)
else()
    message(STATUS "GTest not found, skipping test target")
endif()


