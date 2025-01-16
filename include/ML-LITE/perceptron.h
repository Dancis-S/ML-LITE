#IFNDEF LINEAR_CLASSIFIER
#DEF LINEAR_CLASSIFIER

#include <Eigen/Dense>

class Perceptron {
public:
  Perceptron();
  ~Perceptron();

  void fit(Eigen::MatrixXd const &input, Eigen::MatrixXd const &target,
           double learning_rate);

  Eigen::VectorXd predict(Eigen::MatrixXd const &input,
                          Eigen::MatrixXd const &target);

  double evaluate(Eigen::MatrixXd const &input);

  Eigen::VectorXd getWeights() const;
  double getBias() const;

protected:
  Eigen::VectorXd weights_;
  double bias_;
};

#END
