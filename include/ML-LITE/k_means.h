#ifndef K_MEANS
#define K_MEANS

#include <Eigen/Dense>

class KMeans {
public:
	KMeans();
	~KMeans();

	void fit(Eigen::MatrixXd& input, int clusters);

protected:
	Eigen::MatrixXd clusters_;
};



#endif