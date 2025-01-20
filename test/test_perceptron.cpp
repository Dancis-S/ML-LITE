#include <gtest/gtest.h>
#include "ML-LITE/perceptron.h"
#include <Eigen/Dense>

// Test for default initialization
TEST(PerceptronTests, DefaultInitialization) {
  Perceptron p;
  EXPECT_EQ(p.getBias(), 1.0);  // Default bias is initialized to 1.0
  EXPECT_EQ(p.getWeights().size(), 0);  // Weights should be uninitialized
}

// Test the fit function with a linearly separable dataset
TEST(PerceptronTests, FitOnSeparableData) {
  Perceptron p;

  // Input: 2D linearly separable data
  Eigen::MatrixXd input(4, 2);
  input << 0, 0,
    0, 1,
    1, 0,
    1, 1;

  // Target: Labels for OR function
  Eigen::VectorXd target(4);
  target << -1, -1, -1, 1;

  // Train the perceptron
  p.fit(input, target, 0.1);

  // Check that predictions match the target
  for (int i = 0; i < input.rows(); ++i) {
    EXPECT_EQ(p.predict(input.row(i)), target(i));
  }
}

TEST(PerceptronTests, PredictAfterTraining) {
  Perceptron p;

  // Input: 2D linearly separable data
  Eigen::MatrixXd input(4, 2);
  input << 0, 0,
    0, 1,
    1, 0,
    1, 1;

  // Target: Labels for AND function
  Eigen::VectorXd target(4);
  target << -1, -1, -1, 1;

  // Train the perceptron
  p.fit(input, target, 0.1);

  // Test individual predictions using proper Eigen initialization
  Eigen::VectorXd test1(2);
  test1 << 0, 0;
  EXPECT_EQ(p.predict(test1), -1);

  Eigen::VectorXd test2(2);
  test2 << 1, 1;
  EXPECT_EQ(p.predict(test2), 1);
}

// Test handling of an empty dataset
TEST(PerceptronTests, FitWithEmptyData) {
  Perceptron p;

  // Empty input and target
  Eigen::MatrixXd input(0, 0);
  Eigen::VectorXd target(0);

  // Fit should not throw an exception
  EXPECT_NO_THROW(p.fit(input, target, 0.1));
}

// Test edge case: Single data point
TEST(PerceptronTests, FitWithSingleDataPoint) {
  Perceptron p;

  // Single data point
  Eigen::MatrixXd input(1, 2);
  input << 1, 1;

  Eigen::VectorXd target(1);
  target << 1;

  // Train the perceptron
  p.fit(input, target, 0.1);

  // Prediction should match the target
  EXPECT_EQ(p.predict(input.row(0)), target(0));
}

// Test evaluate (when implemented)
TEST(PerceptronTests, EvaluateAccuracy) {
  Perceptron p;

  // Input: Simple 2D dataset
  Eigen::MatrixXd input(4, 2);
  input << 0, 0,
    0, 1,
    1, 0,
    1, 1;

  // Target: Labels for OR function
  Eigen::VectorXd target(4);
  target << -1, -1, -1, 1;

  // Train the perceptron
  p.fit(input, target, 0.1);

  // Evaluate accuracy (stub currently returns 0.0)
  EXPECT_NEAR(p.evaluate(input), 1.0, 0.1);  // Expect accuracy close to 1.0
}
