#ifndef DECISION_TREE
#define DECISION_TREE

#include <Eigen/Dense>
#include <vector>

struct Node {
	int feature_index;
	double threshold;
	Node* left = nullptr;
	Node* right = nullptr;
	bool is_leaf = false;
	int predicted = -1;
};

class DecisionTree {
public: 
	DecisionTree();
	~DecisionTree();
	void fit(Eigen::MatrixXd& input, Eigen::VectorXd& target);
	
	int predict(Eigen::VectorXd& input);

private: 
	static Node* build(Eigen::MatrixXd& input, Eigen::VectorXd& target, std::vector<int>& indexes);
	static double gini(std::vector<int>& target);
	static std::pair<int, double> find_best_feature_and_threshold(Eigen::MatrixXd& input,
		Eigen::VectorXd& target, std::vector<int>& indexes);
	void delete_tree(Node* node);
	Node* root_;
};

#endif
