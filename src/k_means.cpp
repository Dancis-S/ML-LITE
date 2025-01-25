#include "ML-LITE/k_means.h"
#include <vector>
#include <iostream>

KMeans::KMeans(int n_clusters) : 
	clusters_(), n_cluster_(n_clusters)
		{
}

void KMeans::fit(const Eigen::MatrixXd& input, int max_iters) {
	if (input.rows() < n_cluster_) {
		throw std::invalid_argument("Number of data points must be greater than or equal to the number of clusters.");
	}
	
	// Set select first x points as cluster heads
	clusters_ = input.topRows(n_cluster_);
	
	while (max_iters) {
	std::vector<double> assigned_cluster(input.rows(), -1);
	std::vector<int> cluster_count(n_cluster_, 0);

		// Calculate the eulerian distances
		for (size_t i = 0; i < input.rows(); i++) {
			double min_distance = std::numeric_limits<double>::infinity();
			int ass_class = 0;
			for (int k = 0; k < n_cluster_; k++) {
				double distance = (input.row(i) - clusters_.row(k)).norm();
			
				if (distance < min_distance) {
					min_distance = distance;
					ass_class = k;
				}
			}
			assigned_cluster[i] = ass_class; // Map the class its assigend to
			cluster_count[ass_class]++;  // Increment the cluster count for the assigned class
		}

		// Update the cluster points
		Eigen::MatrixXd new_clusters = Eigen::MatrixXd::Zero(clusters_.rows(), clusters_.cols());

		for (size_t i = 0; i < input.rows(); i++) {
			int assigned = assigned_cluster[i];
			new_clusters.row(assigned) += input.row(i);
		}
		for (int i = 0; i < n_cluster_; i++) {
			if (cluster_count[i] > 0) {
				new_clusters.row(i) /= cluster_count[i];
			}
		}
		// assign the updated clusters
		if ((new_clusters - clusters_).norm() < 1e-6) break;
		clusters_ = new_clusters;
		max_iters--;
	}
}

Eigen::VectorXd KMeans::fit_predict(Eigen::MatrixXd& input) {
	Eigen::VectorXd assigned_class = Eigen::VectorXd::Constant(input.rows(), -1); // Using -1 as invalid for debug
	
	for (size_t i = 0; i < input.rows(); i++) {
		double min_distance = std::numeric_limits<double>::infinity();
		int ass_class = -1;

		for (int k = 0; k < n_cluster_; k++) {
			double distance = (input.row(i) - clusters_.row(k)).norm();
			if (distance < min_distance) {
				min_distance = distance;
				ass_class = k;
			}
		}
		assigned_class[i] = ass_class; // Map the class its assigend to
	}
	return assigned_class;
}


Eigen::MatrixXd KMeans::getClusters() const {
	return clusters_;
}

int KMeans::getClusterCount() const {
	return n_cluster_;
}

