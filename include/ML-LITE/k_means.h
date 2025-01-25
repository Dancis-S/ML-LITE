#ifndef K_MEANS
#define K_MEANS

#include <Eigen/Dense>
#include <vector>

class KMeans {
public:
	KMeans(int n_clusters);

	Eigen::VectorXd fit_predict(Eigen::MatrixXd& input);
	void fit(const Eigen::MatrixXd& input, int max_iters);

	Eigen::MatrixXd getClusters() const;
	int getClusterCount() const;

protected:
	int n_cluster_;
	Eigen::MatrixXd clusters_;
};
#endif