#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <string>
#include <vector>

// ABC for all the models we will implement
class Model {
public:
  virtual ~Model() = default;

  virtual void fit(const Eigen::MatrixXd &trix) = 0;
  virtual Eigen::VectorXd predict(const Eigen::MatrixXd &trix) const = 0;

  virtual double evaluate(const Eigen::MatrixXd &trix) const = 0;

  //virtual void save_model(const std::string &file_path) const {}
  //virtual void load_model(const std::string &file_path) {}
};

#endif
