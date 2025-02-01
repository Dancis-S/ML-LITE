#include "ML-LITE/perceptron.h"

Perceptron::Perceptron() : weights_(), bias_(1.0) {}
Perceptron::~Perceptron() {}

void Perceptron::fit(const Eigen::MatrixXd &input, const Eigen::VectorXd &target,
                     double learning_rate= 0.05) {
  // define the starting weights
  weights_ = Eigen::VectorXd::Random(input.cols(), 1) * 0.5;

  int epochs = 10000; // Just to prevent the alg getting stuck in non separable
  bool converged = false;

  while (epochs-- && !converged) {
    converged = true;
    for (int i = 0; i < input.rows(); i++) {
      double output = input.row(i) * weights_ + bias_;

      int predicted = (output < 0) ? -1 : 1;

      // Handle wrong prediction udpates
      if (predicted != target(i)) {
        converged = false;
        Eigen::VectorXd difference = learning_rate * target(i) * input.row(i).transpose();
        weights_ = weights_ + difference;
        bias_ += (learning_rate * target(i));
      }
    }
  }
}

int Perceptron::predict(const Eigen::VectorXd &input) {
  double output = (input.dot(weights_)) + bias_;

  int predicted = (output < 0) ? -1 : 1;
  return predicted;
}

double Perceptron::accuracy(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target) {
  double total_count = input.rows();
  double correct_count = 0;

  for (int i = 0; i < input.rows(); i++) {
    Eigen::VectorXd current_row = input.row(i).transpose();
    int assigned_class = predict(current_row);

    if (assigned_class == target.coeff(i)) {
      correct_count++;
    }
  }
  return correct_count / total_count;
}

Eigen::VectorXd Perceptron::getWeights() const { return weights_; }

double Perceptron::getBias() const { return bias_; }
