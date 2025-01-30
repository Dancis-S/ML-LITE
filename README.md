# ML-LITE: From Equations to Execution

## Overview

ML-LITE is a long-term personal project focused on building machine learning models from scratch in C++. It serves as a way to explore machine learning concepts, deepen understanding, and improve C++ programming skills. The project evolves over time as new models and techniques are added, making it a continuous learning journey in both machine learning and software engineering.

## Features

✅ Implemented Models:

- **Supervised Learning**
  - Linear Regression (Gradient Descent)
  - Perceptron (Binary Classification)
  - Decision Tree (Gini Impurity-based Splitting)
- **Unsupervised Learning**
  - K-Means Clustering

🚀 Upcoming Models:

- Logistic Regression
- Support Vector Machine (SVM)
- Principal Component Analysis (PCA)
- Gaussian Mixture Models (GMM)
- Random Forests & Gradient Boosting

## Installation

To use ML-LITE, clone the repository and build using CMake:

```bash
git clone https://github.com/YOUR_GITHUB/ml-lite.git
cd ml-lite
mkdir build && cd build
cmake ..
make
```

### Eigen Dependency

This project uses the Eigen library for numerical operations. **Eigen is not included in this repository**, so you must ensure it is available in the `include` directory. You can install Eigen manually:

```bash  
sudo apt install libeigen3-dev  # For Ubuntu/Debian  
yay -S eigen  # For Arch Linux  
brew install eigen  # For macOS  
```

Or download it from [Eigen's official website](https://eigen.tuxfamily.org/) and place the `Eigen` folder inside `include/`.

## Usage

The framework is designed for easy integration and experimentation. The `main.cpp` file includes functions such as `test_perceptron()` that demonstrate how to use the models.

To run an example from `main.cpp`, modify and execute:

```cpp  
#include "ML-LITE/perceptron.h"

int main() {  
    test_perceptron();  
    return 0;  
}  
```

### Compile and Run

```bash  
cd build  
cmake ..  
make  
./ml-lite  
```

## License

MIT License
