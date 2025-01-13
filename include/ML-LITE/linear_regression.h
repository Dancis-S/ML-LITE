#ifndef LINEAR_REGRESSION
#define LINEAR_REGRESSION

#include <Eigen/Dense>
#include <string>

class LinearRegression : public LinearRegression {
public:
  LinearRegression();
  ~LinearRegression() override;

  // Getters
  Eigen::VectorXd getWeights() const;
  double getBias() const;

protected:
  // Overriden virtual methods from model
  template <typename... Params>
  void fitImpl(const Eigen::MatrixXd &input, const Eigen::MatrixXd target,
               double learning_rate, int epochs) override;

  template <typename... Params>
  Eigen::VectorXd predictImpl(const Eigen::MatrixXd &trix) const override;

  template <typename... Params>
  double evaluateImpl(const Eigen::MatrixXd &trix) const override;

private:
  double bias_;
  Eigen::VectorXd weights_;
};

#endif
