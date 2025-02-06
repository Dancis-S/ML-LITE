#ifndef LOGISTIC_REGRESSION
#define LOGISTIC_REGRESSION

#include <Eigen/Dense>

class LogisticRegression {
public:
	LogisticRegression();

	void fit(const Eigen::MatrixXd& input, const Eigen::VectorXd& target,
						double learning_rate, int epcohs);
	Eigen::VectorXd predict(const Eigen::MatrixXd& input);
	double evaluate(const Eigen::MatrixXd& input, const Eigen::VectorXd& target);

private:
	Eigen::VectorXd weights_;
	double bias_;
};

#endif