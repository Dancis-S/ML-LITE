#include "ML-LITE/utils/utils.h"
#include <cmath>

namespace Utils {

	double sigmoid(double value) {
		return (1.0 / (1.0 + std::pow(std::exp(1.0), -(value))));
	}

}

