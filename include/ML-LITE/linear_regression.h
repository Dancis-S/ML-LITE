#ifndef LINEAR_REGRESSION
#define LINEAR_REGRESSION

#include "model.h"
#include <Eigen/Dense>
#include <string>

class LinearRegression : public Model {
public:
  LinearRegression();
  ~LinearRegression() override;

  // Overriden virtual methods from model
  void fit(const Eigen::MatrixXd &input, const Eigen::MatrixXd target, double learning_rate, int epochs ) override;
  Eigen::VectorXd predict(const Eigen::MatrixXd &trix) const override;
  double evaluate(const Eigen::MatrixXd &trix) const override;

private:
  double bias_;
  Eigen::VectorXd weights_;
}
