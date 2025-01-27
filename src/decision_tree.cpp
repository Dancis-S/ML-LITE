#include "ML-LITE/decision_tree.h"
#include <iostream>
#include <unordered_map>
#include <cmath>

DecisionTree::DecisionTree() {
}

void DecisionTree::fit(Eigen::MatrixXd& input, Eigen::VectorXd& target) {
	// We need to select the best feature to split using the Gini Impurity
	std::cout << "Fitting Code Here" << std::endl;
}

double DecisionTree::gini(Eigen::VectorXd &target) {
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

int DecisionTree::predict(Eigen::VectorXd& input) {
	std::cout << "Calculate which class it belongs to" << std::endl;
	return 0;
}
