#include "ML-LITE/supervised/logistic_regression.h"
#include "ML-LITE/utils/utils.h"
#include <iostream>
#include <cmath>


LogisticRegression::LogisticRegression() : bias_(0.0), weights_() {}

void LogisticRegression::fit(const Eigen::MatrixXd& input, const Eigen::VectorXd& target, double learning_rate = 0.1, int epochs = 1000) {
	weights_ = Eigen::VectorXd::Random(input.cols(), 1) * 0.5;
	int sample_count = input.rows();

	while (--epochs) {
		Eigen::VectorXd errors = Eigen::VectorXd::Zero(weights_.rows());
		double error_sum = 0.0;

		for (int i = 0; i < input.rows(); i++) {
			double value = (input.row(i).dot(weights_)) + bias_;
			double classification_error = Utils::sigmoid(value) - target[i];

			error_sum += classification_error;
			Eigen::VectorXd difference = input.row(i).transpose() * (classification_error);
			errors += difference;
		}

		// update the weights and bias
		weights_ -= (errors / sample_count);
		bias_ -= learning_rate * (error_sum / sample_count);
	}
}


Eigen::VectorXd LogisticRegression::predict(const Eigen::MatrixXd& input) {
	if (input.cols() != weights_.rows()) {
		throw std::invalid_argument("Input columns and weight rows don't match!");
	}
	Eigen::VectorXd computed = (input * weights_) + Eigen::VectorXd::Constant(input.rows(), bias_);
	return computed.unaryExpr(&Utils::sigmoid);
}


double LogisticRegression::evaluate(const Eigen::MatrixXd& input, const Eigen::VectorXd& target) {
	// We evaluate using binary cross-entropy
	if (input.rows() != target.rows()) {
		throw std::invalid_argument("Row counts in input and target don't match!");
	}

	Eigen::VectorXd predictions = predict(input);
	int sample_count = target.rows();
	double error_sum = 0;

	for (int i = 0; i < sample_count; ++i) {
		double current_loss = (target[i] * std::log(predictions[i]))  + 
													( (1 - target[i]) * std::log(1 - predictions[i]));
		error_sum += current_loss;
	}
	error_sum = -error_sum / sample_count;
	return error_sum;
}


Eigen::VectorXd LogisticRegression::getWeights() {
	return weights_;
}


double LogisticRegression::getBias() {
	return bias_;
}
