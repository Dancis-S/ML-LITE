#ifndef DECISION_TREE
#define DECISION_TREE

#include <Eigen/Dense>
#include <vector>

struct Node {
	int feature_index;
	double threshold;
	Node* left = nullptr;
	Node* right = nullptr;
	double prediction = 0.0;
	bool is_leaf = false;
};

class DecisionTree {
public: 
	DecisionTree();
	void fit(Eigen::MatrixXd& input, Eigen::VectorXd& target);
	double gini(Eigen::VectorXd& target);
	int predict(Eigen::VectorXd& input);

private: 
	Node root;
};

#endif
