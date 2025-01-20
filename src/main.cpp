#include "ML-LITE/linear_regression.h"
#include "ML-LITE/perceptron.h"
#include <Eigen/Dense>
#include <iostream>


void test_perceptron() {
  Eigen::MatrixXd input(4, 2);
  input << 1, 2,
    2, 4,
    1, 5,
    2, 6;

  Eigen::VectorXd target(4);
  target << -1, -1, 1, 1;

  Perceptron model;
  model.fit(input, target, 0.1);

  // Test case 1
  Eigen::VectorXd test1(2);
  test1 << 1, 1;

  // Test case 1
  Eigen::VectorXd test2(2);
  test2 << 5, 15;

  int predicted = model.predict(test1);
  std::cout << "Model classes as " << predicted << std::endl;
  std::cout << "Model classes (3,6) as  " << model.predict(test2) << std::endl;
}

void test_linear_regression() {
   //Testing for y = 2x + 3
  Eigen::MatrixXd input(5, 1); // 3 rows, 1 feature
  input << 1, 2, 3, 4, 5;      // x values

  Eigen::MatrixXd target(5, 1); // y values
  target << 5, 7, 9, 11, 13;    // Corresponding y values for y = 2x + 3

  // Initialize the LinearRegression model
  LinearRegression model;

  // Train the model
  double learning_rate = 0.005;
  int epochs = 10000;
  model.fit(input, target, learning_rate, epochs);

  // Print the learned weights and bias
  std::cout << "Trained Weights:\n" << model.getWeights() << std::endl;
  std::cout << "Trained Bias:\n" << model.getBias() << std::endl;

  // Test the model on new input
  Eigen::MatrixXd test_input(1, 1);
  test_input << 10; // Predict y for x = 4
  Eigen::MatrixXd prediction = model.predict(test_input);

  std::cout << "Prediction for input 10:\n" << prediction << std::endl;
}

int main() {
  std::cout << "Test functions called here" << std::endl;
  return 0;
}
