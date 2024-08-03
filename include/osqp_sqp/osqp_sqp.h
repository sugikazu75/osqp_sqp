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
    CostBase(int n_variables):
      n_variables_(n_variables)
    {}
    ~CostBase(){}

    virtual void evaluate(const Eigen::VectorXd &x, double &value, Eigen::VectorXd &gradient) = 0;
    bool checkGradient(const Eigen::VectorXd &x, double eps=1e-8, bool verbose = false);

  private:
    size_t n_variables_;
  };

  class ConstraintBase
  {
  public:
    ConstraintBase(int n_variables, const std::vector<double>& tol);
    virtual ~ConstraintBase(){}

    virtual void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values, Eigen::SparseMatrix<double> &jacobian, size_t constraint_idx_head) = 0;
    virtual bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                              Eigen::SparseMatrix<double> &jacobian,
                              const Eigen::VectorXd sqp_lower_bounds, const Eigen::VectorXd sqp_upper_bounds,
                              Eigen::VectorXd &qp_lower_bounds, Eigen::VectorXd &qp_upper_bounds,
                              size_t constraint_idx_head) = 0;

    bool checkJacobian(const Eigen::VectorXd &x, double eps=1e-8, bool verbose = false);
    size_t getNumberOfConstraints() {return n_constraints_;}
  protected:
    size_t n_variables_;
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

    void add(std::shared_ptr<ConstraintBase> constraint)
    {
      constraints_.push_back(constraint);
    }

    void setNumberOfVariables(int n_variables) {n_variables_ = n_variables;}

    bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                      Eigen::SparseMatrix<double> &jacobian,
                      const Eigen::VectorXd sqp_lower_bounds, const Eigen::VectorXd sqp_upper_bounds,
                      Eigen::VectorXd &qp_lower_bounds, Eigen::VectorXd &qp_upper_bounds);

    bool checkJacobian(const Eigen::VectorXd &x, double eps=1e-8, bool verbose=true)
    {
      for(auto constraint : constraints_)
        {
          std::cout << "check jacobian" << std::endl;
          constraint->checkJacobian(x, eps, verbose);
        }
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

    int n_variables_;
    std::vector<std::shared_ptr<ConstraintBase>> constraints_;
  };

  class sqpSolverOption
  {
  public:
    int max_iter = 100;
    double sqp_relative_tolerance_ = 1e-4;
    double qp_relative_tolerance_ = 1e-2;

    bool verbose = true;
    bool qp_verbose = false;
  };

  class sqpSolver
  {
  public:
    sqpSolver(int n_variables);
    ~sqpSolver(){};

    void setCost(std::shared_ptr<CostBase> cost) {cost_ = cost;}
    void addConstraint(std::shared_ptr<ConstraintBase> constraint) {cstset_->add(constraint);}
    void setNumverOfVariables(int n_variables) {n_variables_ = n_variables;}
    void setInitialVariable(const Eigen::VectorXd initial_x) {initial_x_ = initial_x;}
    void solve();
    void setLowerBounds(const Eigen::VectorXd lower_bounds) {sqp_lower_bounds_ = lower_bounds;}
    void setUpperBounds(const Eigen::VectorXd upper_bounds) {sqp_upper_bounds_ = upper_bounds;}
    Eigen::VectorXd getLastX() {
      return x_log_.back();
    }

    std::vector<Eigen::VectorXd> getXLog() {
      return x_log_;
    }

    bool checkCostGradient(const Eigen::VectorXd &x, double eps=1e-8, bool verbose=true)
    {
      cost_->checkGradient(x, eps, verbose);
    }
    bool checkConstraintsJacobian(const Eigen::VectorXd &x, double eps=1e-8, bool verbose=true)
    {
      cstset_->checkJacobian(x, eps, verbose);
    }

  private:
    int n_variables_;
    int n_constraints_;
    bool qp_first_solve_;
    OsqpEigen::Solver qp_solver_;
    sqpSolverOption sqp_solver_option_;
    std::shared_ptr<CostBase> cost_;
    std::shared_ptr<ConstraintSet> cstset_;

    Eigen::VectorXd sqp_lower_bounds_;
    Eigen::VectorXd sqp_upper_bounds_;
    Eigen::VectorXd cstset_values_;
    Eigen::SparseMatrix<double> cstset_jacobian_;

    Eigen::VectorXd initial_x_;
    Eigen::VectorXd qp_x_;
    Eigen::MatrixXd qp_hessian_;
    Eigen::SparseMatrix<double> qp_hessian_sparse_;
    Eigen::VectorXd qp_gradient_;
    Eigen::MatrixXd qp_constraint_linear_matrix_;
    Eigen::SparseMatrix<double> qp_constraint_linear_matrix_sparse_;
    Eigen::VectorXd qp_lower_bounds_;
    Eigen::VectorXd qp_upper_bounds_;
    Eigen::VectorXd qp_dual_solution_;
    bool hessian_first_update_;

    std::vector<Eigen::VectorXd> x_log_;

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

