#ifndef K_MEANS
#define K_MEANS

#include <Eigen/Dense>

class KMeans {
public:
	KMeans(int n_clusters);
	~KMeans();

	void fit(const Eigen::MatrixXd& input);

protected:
	int n_cluster_;
	Eigen::MatrixXd clusters_;
};
#endif