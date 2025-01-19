#include "ML-LITE/perceptron.h"

Perceptron::Perceptron() : weights_(), bias_(1.0) {}
Perceptron::~Perceptron() {}

void Perceptron::fit(const Eigen::MatrixXd &input, const Eigen::VectorXd &target,
                     double learning_rate= 0.1) {
  // define the starting weights
  weights_ = Eigen::VectorXd::Random(input.cols(), 1) * 0.5;

  int epochs = 1000; // Just to prevent the alg getting stuck in non separable
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

double Perceptron::evaluate(const Eigen::MatrixXd &input) { return 0.0; }

Eigen::VectorXd Perceptron::getWeights() const { return weights_; }

double Perceptron::getBias() const { return bias_; }
