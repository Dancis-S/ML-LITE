// This will be the lin reg .cpp file
#include "linear_regression.h"
#include <iostream>

LinearRegression::LinearRegression() : weights_(), bias_(0.0) {
  // I'll set weights dynamically during fit
}

void LinearRegression::fit(const Eigen::MatrixXd &input,
                           const Eigen::MatrixXd &target, double learning_rate,
                           int epochs) {
  // Weight and Bias init
  bias_ = 0.0;
  weights_ = Eigen::MatrixXd::Random(input.cols(), 1) * 0.5;

  // Linear Reg logic here. Predict, Error, Gradients, Updates.
  while (epochs--) {
    // Predict: y = Xw + b
    Eigen::MatrixXd prediction = (input * weights_) + bias_;

    // Error: (y - target)
    Eigen::MatrixXd error = prediction - target;

    // Calculate grad for weights (and bias) and update
    double bias_gradient = error.sum() / error.rows();
    bias_ = bias_ - (learning_rate * bias_gradient);

    Eigen::MatrixXd gradients = (input.transpose() * error) / input.rows();
    weights_ = weights_ - (learning_rate * gradients);
  }
}
