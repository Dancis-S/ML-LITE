#include "linear_regression.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
  // Define a small dataset: y = 2x + 3
  Eigen::MatrixXd input(3, 1); // 3 rows, 1 feature
  input << 1, 2, 3;            // x values

  Eigen::MatrixXd target(3, 1); // y values
  target << 5, 7, 9;            // Corresponding y values for y = 2x + 3

  // Initialize the LinearRegression model
  LinearRegression model;

  // Train the model
  double learning_rate = 0.01;
  int epochs = 1000;
  model.fit(input, target, learning_rate, epochs);

  // Print the learned weights and bias
  std::cout << "Trained Weights:\n" << model.getWeights() << std::endl;
  std::cout << "Trained Bias:\n" << model.getBias() << std::endl;

  // Test the model on new input
  Eigen::MatrixXd test_input(1, 1);
  test_input << 4; // Predict y for x = 4
  Eigen::MatrixXd prediction = model.predict(test_input);

  std::cout << "Prediction for input 4:\n" << prediction << std::endl;

  return 0;
}
