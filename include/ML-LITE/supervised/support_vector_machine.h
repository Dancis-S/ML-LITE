#ifndef SUPPORT_VECTOR_MACHINE
#define SUPPORT_VECTOR_MACHINE

#include <Eigen/Dense>

class SVM {
public:
	SVM();

	void fit(Eigen::MatrixXd& input, Eigen::VectorXd& target);
	Eigen::VectorXd predict(Eigen::MatrixXd& input);
	double evaluate(Eigen::MatrixXd& input, Eigen::VectorXd& target);

	Eigen::VectorXd getWeights();
	double getBias();

private:
	Eigen::VectorXd weights_;
	double bias_;
};

#endif