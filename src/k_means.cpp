#include "ML-LITE/k_means.h"
#include <vector>

KMeans::KMeans(int n_clusters) : 
	clusters_(), n_cluster_(n_clusters)
		{
	
}

void KMeans::fit(const Eigen::MatrixXd& input) {
	if (input.rows() < n_cluster_) {
		throw std::invalid_argument("Number of data points must be greater than or equal to the number of clusters.");
	}

	// Set select first x points as cluster heads
	clusters_ = input.topRows(n_cluster_);

	std::vector<double> assigned_cluster(input.rows(), -1);
	std::vector<int> cluster_count(n_cluster_, 0);

	// Calculate the eulerian distances
	for (size_t i = 0; i < input.rows(); i++) {
		double distance = std::numeric_limits<double>::infinity();

		for (int k = 0; k < n_cluster_; k++) {
			// calculate who belongs where
		}
	}

	// Update the cluster points
	Eigen::MatrixXd new_clusters = Eigen::MatrixXd::Zero(clusters_.rows(), clusters_.cols());

	for (size_t i = 0; i < input.rows(); i++) {
		int assigned = assigned_cluster[i];
		new_clusters.row(assigned) += input.row(i);
	}
	for (int i = 0; i < n_cluster_; i++) {
		new_clusters /= cluster_count[i];
	}
}

