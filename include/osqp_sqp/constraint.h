#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

namespace osqpsqp
{
  class ConstraintBase
  {
  public:
    ConstraintBase(const std::vector<double>& tol)
    {
      tol_.resize(tol.size());
      n_constraints_ = tol_.size();
      for(size_t i = 0; i < n_constraints_; ++i) tol_(i) = tol.at(i);
    }
    virtual ~ConstraintBase(){}

    virtual void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values, Eigen::SparseMatrix<double> &jacobian, size_t constraint_idx_head) = 0;
    virtual bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                              Eigen::SparseMatrix<double> &jacobian,
                              const Eigen::VectorXd sqp_lower_bounds, const Eigen::VectorXd sqp_upper_bounds,
                              Eigen::VectorXd &qp_lower_bounds, Eigen::VectorXd &qp_upper_bounds,
                              size_t constraint_idx_head) = 0;

    bool checkJacobian(const Eigen::VectorXd &x, bool verbose = false, double eps=1e-8)
    {
      int n_variables = x.size();
      Eigen::SparseMatrix<double> ana_jac = Eigen::SparseMatrix<double>(n_constraints_, n_variables);
      Eigen::MatrixXd num_jac = Eigen::MatrixXd::Zero(n_constraints_, n_variables);
      ana_jac.setZero();
      num_jac.setZero();

      { // numerical differentiation
        Eigen::SparseMatrix<double> whatever = Eigen::SparseMatrix<double>(n_constraints_, n_variables);
        Eigen::VectorXd f0 = Eigen::VectorXd::Zero(n_constraints_);
        evaluate(x, f0, whatever, 0);

        for(size_t i = 0; i < n_variables; ++i)
          {
            Eigen::VectorXd x_plus = x;
            x_plus(i) += eps;
            Eigen::VectorXd f1 = Eigen::VectorXd::Zero(n_constraints_);
            evaluate(x_plus, f1, whatever, 0);
            num_jac.col(i) = (f1 - f0) * (1 / eps);
          }
      }

      { // analytical differentiation
        Eigen::VectorXd whatever = Eigen::VectorXd::Zero(n_constraints_);
        evaluate(x, whatever, ana_jac, 0);
      }

      double max_diff = (num_jac - ana_jac.toDense()).cwiseAbs().maxCoeff();
      double check_eps = eps * 10;
      bool ok = max_diff < check_eps;
      if(verbose && !ok)
        {
          std::cout << "max_diff: " << max_diff << std::endl;
          std::cout << "check_eps: " << check_eps << std::endl;
          std::cout << "num_jac: " << std::endl << num_jac << std::endl;
          std::cout << "ana_jac: " << std::endl << ana_jac.toDense() << std::endl;
        }
      if(verbose && ok)
        {
          std::cout << "numerical and analytical jacobian is identical" << std::endl;
        }
      return ok;
    }

    size_t getNumberOfConstraints() {return n_constraints_;}

  protected:
    size_t n_constraints_;
    Eigen::VectorXd tol_;
  };

  class InequalityConstraintBase : public ConstraintBase
  {
  public:
    using ConstraintBase::ConstraintBase;
    bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                      Eigen::SparseMatrix<double> &jacobian,
                      const Eigen::VectorXd sqp_lower_bounds, const Eigen::VectorXd sqp_upper_bounds,
                      Eigen::VectorXd &qp_lower_bounds, Eigen::VectorXd &qp_upper_bounds,
                      size_t constraint_idx_head)
    {
      evaluate(x, values, jacobian, constraint_idx_head);
      auto jacobian_sliced = jacobian.middleRows(constraint_idx_head, n_constraints_);
      auto value_sliced = values.middleRows(constraint_idx_head, n_constraints_);

      qp_lower_bounds.segment(constraint_idx_head, n_constraints_) = sqp_lower_bounds.segment(constraint_idx_head, n_constraints_) - value_sliced;
      qp_upper_bounds.segment(constraint_idx_head, n_constraints_) = sqp_upper_bounds.segment(constraint_idx_head, n_constraints_) - value_sliced;

      bool is_feasible = ((sqp_lower_bounds.segment(constraint_idx_head, n_constraints_) - tol_).array() - value_sliced.array()).all()  && (value_sliced.array() <= (sqp_upper_bounds.segment(constraint_idx_head, n_constraints_) + tol_).array()).all();  // sqp_lower_bounds - tol_ <= value_sliced <= sqp_upper_bounds + tol_

      return is_feasible;
    }
  };
}
