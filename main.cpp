#include "ML-LITE/supervised/linear_regression.h"
#include "ML-LITE/supervised/perceptron.h"
#include "ML-LITE/supervised/decision_tree.h"
#include "ML-LITE/supervised/logistic_regression.h"
#include "ML-LITE/unsupervised/k_means.h"
#include "ML-LITE/utils/utils.h"

#include <Eigen/Dense>
#include <iostream>


// List of pre-defined functions using the model feel free to play with
void test_perceptron();
void test_linear_regression();
void test_k_means();
void test_decision_tree();
void test_logistic_regression();

int main() {
  test_logistic_regression();
  std::cout << "\n" << "!!!!==== Test Ended ====!!!!" << std::endl;
  return 0;
}

// ===================== Functions below ===============================
// ====== Feel free to edit them and test the models as you wish =======

void test_perceptron() {
  Eigen::MatrixXd input(4, 2);
  input << 1, 2,
    2, 4,
    1, 5,
    2, 6;

  Eigen::VectorXd target(4);
  target << -1, -1, 1, 1;

  Perceptron model;
  model.fit(input, target, 0.1);

  // Test case 1
  Eigen::VectorXd test1(2);
  test1 << 1, 1;

  // Test case 2
  Eigen::VectorXd test2(2);
  test2 << 5, 15;

  // Evualation test case

  int predicted = model.predict(test1);
  std::cout << "Model classes as " << predicted << std::endl;
  std::cout << "Model classes (3,6) as  " << model.predict(test2) << std::endl;
  std::cout << "Evaluation: " << model.accuracy(input, target) * 100 << "%" << std::endl;
}

void test_linear_regression() {
   //Testing for y = 2x + 3
  Eigen::MatrixXd input(5, 1); // 3 rows, 1 feature
  input << 1, 2, 3, 4, 5;      // x values

  Eigen::MatrixXd target(5, 1); // y values
  target << 5, 7, 9, 11, 13;    // Corresponding y values for y = 2x + 3

  // Initialize the LinearRegression model
  LinearRegression model;

  // Train the model
  double learning_rate = 0.005;
  int epochs = 10000;
  model.fit(input, target, learning_rate, epochs);

  // Print the learned weights and bias
  std::cout << "Trained Weights:\n" << model.getWeights() << std::endl;
  std::cout << "Trained Bias:\n" << model.getBias() << std::endl;

  // Test the model on new input
  Eigen::MatrixXd test_input(1, 1);
  test_input << 10; // Predict y for x = 4
  Eigen::MatrixXd prediction = model.predict(test_input);

  std::cout << "Prediction for input 10:\n" << prediction << std::endl;

  std::cout << "Evaluate Input set (MSE): " << model.evaluate(input, target) << std::endl;
}

void test_k_means() {
  KMeans model = KMeans(2);

  Eigen::MatrixXd data(6, 2);
  data << 1, 2,
    1, 3,
    2, 3,
    8, 9,
    9, 10,
    9, 9;

  std::cout << "Startig to fit clusters" << std::endl;
  model.fit(data, 100);
  std::cout << "K-Means Clustering Complete!" << std::endl;
  std::cout << "Cluster counts" << model.getClusterCount() << std::endl;
  std::cout << "Assinged input class\n" << model.fit_predict(data) << "\n";
  std::cout << "\n" << "=================" << "Final Cluster coords:" 
    << "====================" << "\n" << model.getClusters() << std::endl;
}

void test_decision_tree() {
  Eigen::MatrixXd input(4, 2);
  input << 1, 2,
    2, 4,
    1, 5,
    2, 6;

  Eigen::VectorXd target(4);
  target << 1, 1, 2, 2;

  DecisionTree model;
  std::cout << "!! Starting to train Decision Tree !! \n";
  model.fit(input, target);
  std::cout << "!! End of training !!\n" << std::endl;

  Eigen::VectorXd test_case(2);
  test_case << 3, 6;
  std::cout << "Prediction for point (3,6): " << model.predict(test_case) << std::endl;
}

void test_logistic_regression() {
  // Training Data
  Eigen::MatrixXd input(4, 2);
  input << 1, 2,
    2, 4,
    1, 5,
    2, 6;

  Eigen::VectorXd target(4);
  target << 0, 0, 1, 1;

  double learning_rate = 0.1;
  int epochs = 1000;

  // Train Logistic Regression Model
  LogisticRegression model;
  model.fit(input, target, learning_rate, epochs);

  // Test Predictions
  Eigen::MatrixXd test1(2, 2);
  test1 << 1, 1,
    5, 15;

  std::cout << "Predictions for test set 1:\n" << model.predict(test1) << std::endl;

  Eigen::MatrixXd test_input(4, 2);
  test_input << 1, 3,
    3, 6,
    2, 8,
    4, 10;

  Eigen::VectorXd test_target(4);
  test_target << 0, 0, 1, 1;


  // Evaluate Model
  double logloss = model.evaluate(test_input, test_target);
  std::cout << "Evaluation on additional dataset: Logloss = " << logloss << std::endl;
}
