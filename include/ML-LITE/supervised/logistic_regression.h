#ifndef LOGISTIC_REGRESSION
#define LOGISTIC_REGRESSION

#include <Eigen/Dense>

class LogisticRegression {
public:
	LogisticRegression();
	~LogisticRegression();

	void fit(const Eigen::MatrixXd& input, const Eigen::VectorXd& target);
	Eigen::VectorXd predict(const Eigen::MatrixXd& input);
	double evaluate(const Eigen::MatrixXd& input, const Eigen::VectorXd& target);

private:
	Eigen::VectorXd weights_;
	double bias_;
};

#endif