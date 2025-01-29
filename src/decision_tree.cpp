#include "ML-LITE/decision_tree.h"
#include <iostream>
#include <unordered_map>
#include <cmath>
#include <stack>
#include <numeric>

DecisionTree::DecisionTree() : root_() {
}

DecisionTree::~DecisionTree() {
	DecisionTree::delete_tree(root_);
}


double DecisionTree::gini(std::vector<int>& target) {
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
std::pair<int, double> DecisionTree::find_best_feature_and_threshold(Eigen::MatrixXd& input, 
											Eigen::VectorXd& target, std::vector<int> &indexes) {
	int best_feature = 0;
	double threshold = 0;
	double best_score = std::numeric_limits<double>::infinity();

	for (int i = 0; i < input.cols(); i++) {

		double total = 0; 
		for (const auto& val : indexes) {
			total += input(val, i);
		}
		double mean = total / indexes.size();

		std::vector<int> accepted;  // Greater than or equal to mean
		std::vector<int> rejected;  // Less than mean

		for (int k = 0; k < indexes.size(); k++) {
			int index = indexes[k];
			double value = input(index, i);
			if (value < mean) {
				rejected.push_back(target[index]);
			}
			else {
				accepted.push_back(target[index]);
			}
		}

		if (accepted.empty() || rejected.empty()) continue;  // skip empty subsets

		// Calculate gini impur for given feature and see how it fairs
		double weight_accepted = static_cast<double>(accepted.size()) / indexes.size();
		double weight_rejected = static_cast<double>(rejected.size()) / indexes.size();

		double feature_score = (DecisionTree::gini(accepted) * weight_accepted) +
			(DecisionTree::gini(rejected) * weight_rejected);
		
		if (feature_score < best_score) {
			best_feature = i;
			best_score = feature_score;
			threshold = mean;
		}
	}
	return std::make_pair(best_feature, threshold);
}


Node* DecisionTree::build(Eigen::MatrixXd& input, Eigen::VectorXd& target, std::vector<int>& indexes) {
	// First check whether we are a leaf node (gini == 0)
	std::vector<int> current_targets;

	for (const auto& val : indexes) {
		current_targets.push_back(target[val]);
	}

	double current_gini = DecisionTree::gini(current_targets);

	if (current_gini == 0) {
		Node* node = new  Node{ -1, -1, nullptr, nullptr, true, static_cast<int>(target[indexes[0]])};
		return node;
	}

	// Otherwise we are not a leaf and need to find the next feature to split on!
	std::pair feature_threshold = DecisionTree::find_best_feature_and_threshold(input, target, indexes);
	int split_on = feature_threshold.first;
	double threshold = feature_threshold.second;

	std::vector<int> left;
	std::vector<int> right;

	for (const auto& val : indexes) {
		if (input(val, split_on) < threshold) {
			left.push_back(val);
		}
		else {
			right.push_back(val);
		}
	}

	Node* left_child = DecisionTree::build(input, target, left);
	Node* right_child = DecisionTree::build(input, target, right);

	Node* current = new Node{ split_on, threshold, left_child, right_child, false, -1 };
	return current;
}


void DecisionTree::fit(Eigen::MatrixXd& input, Eigen::VectorXd& target) {
	// Check that we dont have a homogeneous group as input and that rows counts match
	std::vector<int> std_target(target.data(), target.data() + target.size());
	double root_gini = DecisionTree::gini(std_target);

	if (input.rows() != target.rows()) {
		throw std::invalid_argument("Rows in Input and Target don't match!");
	}
	else if (root_gini == 0) {
		throw std::invalid_argument("Input data contains only 1 class!");
	}

	std::vector<int> all_indexes(input.rows());
	std::iota(all_indexes.begin(), all_indexes.end(), 0);

	root_ = DecisionTree::build(input, target, all_indexes);
}


int DecisionTree::predict(Eigen::VectorXd& input) {
	std::cout << "Calculate which class it belongs to" << std::endl;
	return 0;
}


void DecisionTree::delete_tree(Node* node) {
	// Implement post order traversal where we delete the node

}
