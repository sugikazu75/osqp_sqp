#pragma once

#include <Eigen/Dense>
#include <memory>
#include <osqp_sqp/constraint.h>
#include <vector>

namespace osqpsqp
{
  class ConstraintSet
  {
  public:
    ConstraintSet()
    {
      constraints_.resize(0);
    }
    virtual ~ConstraintSet(){}

    void add(std::shared_ptr<ConstraintBase> constraint) {constraints_.push_back(constraint);}
    void clear() {constraints_.clear();}

    bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                      Eigen::SparseMatrix<double> &jacobian,
                      const Eigen::VectorXd sqp_lower_bounds, const Eigen::VectorXd sqp_upper_bounds,
                      Eigen::VectorXd &qp_lower_bounds, Eigen::VectorXd &qp_upper_bounds)
    {
      size_t constraint_idx_head = 0;
      bool is_feasible = true;
      for(auto c : constraints_)
        {
          is_feasible &= c->evaluateFull(x, values,
                                         jacobian,
                                         sqp_lower_bounds, sqp_upper_bounds,
                                         qp_lower_bounds, qp_upper_bounds,
                                         constraint_idx_head);
          constraint_idx_head += c->getNumberOfConstraints();
        }
      return is_feasible;
    }

    bool checkJacobians(const Eigen::VectorXd &x, bool verbose=false, double eps=1e-8)
    {
      bool ok = true;
      for(auto constraint : constraints_)
        {
          ok &= constraint->checkJacobian(x, verbose, eps);
        }
      return ok;
    }

    size_t getNumberOfConstraints()
    {
      size_t n_constraints = 0;
      for(auto constraint : constraints_)
        {
          n_constraints += constraint->getNumberOfConstraints();
        }
      return n_constraints;
    }

  private:
    std::vector<std::shared_ptr<ConstraintBase>> constraints_;
  };
}
