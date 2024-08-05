#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace osqpsqp
{
  double autoScale(const Eigen::VectorXd& s, const Eigen::VectorXd& q)
  {
    double s_norm2 = s.transpose() * s;
    double y_norm2 = q.transpose() * q;
    double ys = std::abs(q.transpose() * s);
    if(ys == 0.0 || y_norm2 == 0 || s_norm2 == 0) {
      return 1.0;
    } else {
      return y_norm2 / ys;
    }
  }

  Eigen::SparseMatrix<double> convertSparseMatrix(const Eigen::MatrixXd& mat)
  {
    Eigen::SparseMatrix<double> ret;
    ret.resize(mat.rows(), mat.cols());
    for(size_t i = 0; i < ret.rows(); ++i) {
      for(size_t j = 0; j < ret.cols(); ++j) {
        if(mat(i, j) != 0.0) {
          ret.insert(i, j) = mat(i, j);
        }
      }
    }
    return ret;
  }

  class sqpSolverOption
  {
  public:
    int sqp_max_iter = 100;
    double sqp_relative_tolerance = 1e-4;
    double qp_relative_tolerance = 1e-4;
    bool qp_verbose = false;
  };
} // namespace osqpsqp

