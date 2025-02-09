#include "ML-LITE/supervised/support_vector_machine.h"

SVM::SVM() : weights_(), bias_(0.0) {

}


void SVM::fit(Eigen::MatrixXd& input, Eigen::VectorXd& target) {
	// Fit the SVM model
}


Eigen::VectorXd SVM::predict(Eigen::MatrixXd& input) {

	return Eigen::VectorXd(0);
}


double SVM::evaluate(Eigen::MatrixXd& input, Eigen::VectorXd& target) {
	return 0.0;
}


Eigen::VectorXd SVM::getWeights() {
	return weights_;
}


double SVM::getBias() {
	return bias_;
}
