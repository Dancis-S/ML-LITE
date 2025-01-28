#include "ML-LITE/decision_tree.h"
#include <iostream>
#include <unordered_map>
#include <cmath>

DecisionTree::DecisionTree() : root() {
}


double gini(std::vector<int>& target) {
	// Get the counts for each of the groups
	std::unordered_map<int, int> group_counts;
	size_t total_count = target.size();

	for (size_t i = 0; i < total_count; i++) {
		group_counts[target[i]]++;
	}

	// Calculate the gini impurity of the set G(s) = 1 - sum(for all (pi)^2) 
	double gini = 0;

	for (const auto& [key, value] : group_counts) {
		if (value > 0) {
			double proportion = static_cast<double>(value) / total_count;
			gini += std::pow(proportion, 2);
		}
	}
	gini = 1 - gini;
	return gini;
}


// Function that given the input and target will return best col to split on.
int find_best_feature(Eigen::MatrixXd& input, Eigen::MatrixXd& target) {
	int best_feature = 0;
	double best_score = std::numeric_limits<double>::infinity();

	for (int i = 0; i < input.cols(); i++) {
		double total = input.col(i).sum();
		double mean = total / input.rows();

		std::vector<int> accepted;  // Greater than or equal to mean
		std::vector<int> rejected;  // Less than mean

		for (int k = 0; k < input.rows(); k++) {
			double value = input(k, i);
			if (value < mean) {
				rejected.push_back(target[k]);
			}
			else {
				accepted.push_back(target[k]);
			}
		}
		double feature_score =
			(gini(accepted) * (accepted.size() / target.size())) +
				(gini(rejected) * (rejected.size() / target.size()));
		
		if (feature_score < best_score) {
			best_feature = i;
			best_score = feature_score;
		}
	}

	return best_feature;
}


void DecisionTree::fit(Eigen::MatrixXd& input, Eigen::VectorXd& target) {
	// We need to select the best feature to split using the Gini Impurity
	std::cout << "Fitting Code Here" << std::endl;
}


int DecisionTree::predict(Eigen::VectorXd& input) {
	std::cout << "Calculate which class it belongs to" << std::endl;
	return 0;
}
