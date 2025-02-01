#ifndef PERCEPTRON
#define PERCEPTRON

#include <Eigen/Dense>

class Perceptron {
public:
  Perceptron();
  ~Perceptron();

  void fit(Eigen::MatrixXd const &input, Eigen::VectorXd const &target,
           double learning_rate);

  int predict(Eigen::VectorXd const &input);

  double accuracy(Eigen::MatrixXd const &input, Eigen::MatrixXd const &target);

  Eigen::VectorXd getWeights() const;
  double getBias() const;

protected:
  Eigen::VectorXd weights_;
  double bias_;
};

#endif
