#ifndef LINEAR_CLASSIFIER
#define LINEAR_CLASSIFIER

#include <Eigen/Dense>

class Perceptron {
public:
  Perceptron();
  ~Perceptron();

  void fit(Eigen::MatrixXd const &input, Eigen::VectorXd const &target,
           double learning_rate);

  int predict(Eigen::VectorXd const &input);

  double evaluate(Eigen::MatrixXd const &input);

  Eigen::VectorXd getWeights() const;
  double getBias() const;

protected:
  Eigen::VectorXd weights_;
  double bias_;
};

#endif
