#ifndef LINEAR_REGRESSION
#define LINEAR_REGRESSION

#include <Eigen/Dense>
#include <string>

class LinearRegression {
public:
  LinearRegression();
  ~LinearRegression();

  // Getters
  Eigen::VectorXd getWeights() const;
  double getBias() const;

  void fit(const Eigen::MatrixXd &matrix, const Eigen::MatrixXd &target,
           double learning_rate, int epochs);

  Eigen::VectorXd predict(const Eigen::MatrixXd &matrix);

  double evaluate(const Eigen::MatrixXd &input, const Eigen::VectorXd &target);

private:
  double bias_;
  Eigen::VectorXd weights_;
};

#endif
