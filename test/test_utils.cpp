#include <gtest/gtest.h>
#include "ML-LITE/utils/utils.h"
#include <cmath>

TEST(sigmoid_test, handles_zero_test) {
	EXPECT_NEAR(Utils::sigmoid(0), 0.5, 1e-6);
}