#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>
#include <memory>
#include <string>
#include <chrono>
#include <vector>

namespace osqpsqp
{
  class CostBase
  {
  public:
    CostBase(){}
    ~CostBase(){}

    virtual void evaluate(const Eigen::VectorXd &x, double &value, Eigen::VectorXd &gradient) = 0;
    bool checkGradient(const Eigen::VectorXd &x, bool verbose = false, double eps=1e-8);
  };

  class ConstraintBase
  {
  public:
    ConstraintBase(const std::vector<double>& tol)
    {
      tol_.resize(tol.size());
      n_constraints_ = tol_.size();
      for(size_t i = 0; i < n_constraints_; i++) tol_(i) = tol.at(i);
    }
    virtual ~ConstraintBase(){}

    virtual void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values, Eigen::SparseMatrix<double> &jacobian, size_t constraint_idx_head) = 0;
    virtual bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                              Eigen::SparseMatrix<double> &jacobian,
                              const Eigen::VectorXd sqp_lower_bounds, const Eigen::VectorXd sqp_upper_bounds,
                              Eigen::VectorXd &qp_lower_bounds, Eigen::VectorXd &qp_upper_bounds,
                              size_t constraint_idx_head) = 0;

    bool checkJacobian(const Eigen::VectorXd &x, bool verbose = false, double eps=1e-8);
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
                      Eigen::VectorXd &qp_lower_bounds, Eigen::VectorXd &qp_upper_bounds_,
                      size_t constraint_idx_head) override;
  };

  class ConstraintSet
  {
  public:
    ConstraintSet(){};
    virtual ~ConstraintSet(){}

    void add(std::shared_ptr<ConstraintBase> constraint) {constraints_.push_back(constraint);}
    void clear() {constraints_.clear();}

    bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                      Eigen::SparseMatrix<double> &jacobian,
                      const Eigen::VectorXd sqp_lower_bounds, const Eigen::VectorXd sqp_upper_bounds,
                      Eigen::VectorXd &qp_lower_bounds, Eigen::VectorXd &qp_upper_bounds);

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

  class sqpSolverOption
  {
  public:
    int max_iter = 100;
    double sqp_relative_tolerance_ = 1e-4;
    double qp_relative_tolerance_ = 1e-4;
    bool qp_verbose = false;
  };

  class sqpSolver
  {
  public:
    sqpSolver(int n_variables);
    ~sqpSolver(){};

    void setCost(std::shared_ptr<CostBase> cost) {cost_ = cost;}
    void addConstraint(std::shared_ptr<ConstraintBase> constraint) {cstset_->add(constraint);}
    void setConstraintSet(std::shared_ptr<ConstraintSet> cstset) {cstset_ = cstset_;}
    void setInitialVariable(const Eigen::VectorXd initial_x) {initial_x_ = initial_x;}
    void solve();
    void setLowerBounds(const Eigen::VectorXd lower_bounds) {sqp_lower_bounds_ = lower_bounds;}
    void setUpperBounds(const Eigen::VectorXd upper_bounds) {sqp_upper_bounds_ = upper_bounds;}

    double getCostValue() {return sqp_cost_value_;}
    int getIteration() {return sqp_iter_;}
    Eigen::VectorXd getSolution() {return sqp_solution_;}
    Eigen::VectorXd getConstrainValue() {return sqp_constraint_value_;}
    double getSolveTime() {return sqp_solve_time_;}

    bool checkCostGradient(const Eigen::VectorXd &x, bool verbose=false, double eps=1e-8) {return cost_->checkGradient(x, verbose, eps);}
    bool checkConstraintJacobians(const Eigen::VectorXd &x, bool verbose=false, double eps=1e-8){return cstset_->checkJacobians(x, verbose, eps);}

  private:
    int n_variables_;
    int n_constraints_;
    sqpSolverOption sqp_solver_option_;
    std::shared_ptr<CostBase> cost_;
    std::shared_ptr<ConstraintSet> cstset_;

    double sqp_solve_time_;
    double sqp_cost_value_;
    int sqp_iter_;
    Eigen::VectorXd sqp_solution_;
    Eigen::VectorXd sqp_constraint_value_;
    Eigen::VectorXd sqp_lower_bounds_;
    Eigen::VectorXd sqp_upper_bounds_;

    OsqpEigen::Solver qp_solver_;
    bool qp_first_solve_;
    bool hessian_first_update_;
    Eigen::VectorXd initial_x_;
    Eigen::VectorXd qp_x_prev_;
    Eigen::MatrixXd qp_hessian_;
    Eigen::SparseMatrix<double> qp_hessian_sparse_;
    Eigen::VectorXd qp_gradient_;
    Eigen::MatrixXd qp_constraint_linear_matrix_;
    Eigen::SparseMatrix<double> qp_constraint_linear_matrix_sparse_;
    Eigen::VectorXd qp_lower_bounds_;
    Eigen::VectorXd qp_upper_bounds_;
    Eigen::VectorXd qp_dual_solution_;

    void initSolver(const Eigen::VectorXd& x);
    Eigen::VectorXd QP(const Eigen::VectorXd& x);
    Eigen::MatrixXd updateHessianBFGS(Eigen::MatrixXd h_prev, const Eigen::VectorXd& x, const Eigen::VectorXd& x_prev,
                                      const Eigen::VectorXd& lambda_prev, const Eigen::VectorXd& grad_f, const Eigen::VectorXd& grad_f_prev,
                                      const Eigen::MatrixXd& grad_g, const Eigen::MatrixXd& grad_g_prev);
  };

  double autoScale(const Eigen::VectorXd& s, const Eigen::VectorXd& q)
  {
    double s_norm2 = s.transpose() * s;
    double y_norm2 = q.transpose() * q;
    double ys = std::abs(q.transpose() * s);
    if (ys == 0.0 || y_norm2 == 0 || s_norm2 == 0) {
      return 1.0;
    } else {
      return y_norm2 / ys;
    }
  }

  Eigen::SparseMatrix<double> convertSparseMatrix(const Eigen::MatrixXd& mat)
  {
    Eigen::SparseMatrix<double> ret;
    ret.resize(mat.rows(), mat.cols());
    for (size_t i = 0; i < ret.rows(); ++i) {
      for (size_t j = 0; j < ret.cols(); ++j) {
        if (mat(i, j) != 0.0) {
          ret.insert(i, j) = mat(i, j);
        }
      }
    }
    return ret;
  }

} // namespace osqpsqp

