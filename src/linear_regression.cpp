#include "ML-LITE/linear_regression.h"

LinearRegression::LinearRegression() : weights_(), bias_(0.0) {}
LinearRegression::~LinearRegression() {}

void LinearRegression::fit(const Eigen::MatrixXd &input,
                           const Eigen::MatrixXd &target, double learning_rate,
                           int epochs) {
  // Weight and Bias init
  bias_ = 0.0;
  weights_ = Eigen::MatrixXd::Random(input.cols(), 1) * 0.5;

  // Linear Reg logic here. Predict, Error, Gradients, Updates.
  while (epochs--) {
    // Predict: y = Xw + b
    Eigen::MatrixXd prediction =
        (input * weights_) + Eigen::VectorXd::Constant(input.rows(), bias_);

    // Error: (y - target)
    Eigen::MatrixXd error = prediction - target;

    // Calculate grad for weights (and bias) and update
    double bias_gradient = error.sum() / error.rows();
    bias_ = bias_ - (learning_rate * bias_gradient);

    Eigen::MatrixXd gradients = (input.transpose() * error) / input.rows();
    weights_ = weights_ - (learning_rate * gradients);
  }
}

Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd &input) {
  return (input * weights_) + Eigen::VectorXd::Constant(input.rows(), bias_);
}

double LinearRegression::evaluate(const Eigen::MatrixXd& input, const Eigen::VectorXd& target) {
  if (input.rows() != target.rows()) {
    throw std::invalid_argument("Rows in Input and Target don't match!");
  }

  double squared_error = 0;
  Eigen::VectorXd predicted_values = predict(input);

  for (int i = 0; i < target.size(); ++i) {
    double diff = target[i] - predicted_values[i];
    squared_error += diff;
  }
  
  return (squared_error / target.size());
}

// Getters
double LinearRegression::getBias() const { return bias_; }

Eigen::VectorXd LinearRegression::getWeights() const { return weights_; }
