#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>
#include <memory>
#include <string>
#include <chrono>

namespace osqpsqp
{
  class CostBase
  {
  public:
    CostBase(int n_variables):
      n_variables_(n_variables)
    {}
    virtual ~CostBase(){}

    virtual void evaluate(const Eigen::VectorXd &x, double &values, Eigen::VectorXd &gradient);
    bool checkGradient(const Eigen::VectorXd &x, double eps=1e-8, bool verbose = false);

  private:
    int n_variables_;
  };

  class ConstraintBase
  {
  public:
    ConstraintBase(int n_variables, Eigen::VectorXd &tol):
      n_variables_(n_variables),
      tol_(tol)
    {
      n_constraints_ = tol_.size();
    }
    virtual ~ConstraintBase(){}

    virtual void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values, Eigen::SparseMatrix<double> &jacobian) = 0;
    virtual bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                               Eigen::SparseMatrix<double> &jacobian,
                               Eigen::VectorXd &lower, Eigen::VectorXd &upper) = 0;

    bool checkJacobian(const Eigen::VectorXd &x, double eps=1e-8, bool verbose = false);
    int getNumberOfConstraints() {return n_constraints_;}
  private:
    int n_variables_;
    int n_constraints_;
    Eigen::VectorXd tol_;
  };

  class EqualityConstraintBase : public ConstraintBase
  {
  public:
    void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values, Eigen::SparseMatrix<double> &jacobian) override;
    bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                       Eigen::SparseMatrix<double> &jacobian,
                       Eigen::VectorXd &lower, Eigen::VectorXd &upper) override;
  };

  class InequalityConstraintBase : public ConstraintBase
  {
  public:
    void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values, Eigen::SparseMatrix<double> &jacobian) override;
    bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                       Eigen::SparseMatrix<double> &jacobian,
                       Eigen::VectorXd &lower, Eigen::VectorXd &upper) override;
  };

  // class BoxConstraint : public ConstraintBase
  // {
  // public:
  //   BoxConstraint(int n_variables, const std::vector<double> &tol,
  //                 const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
  //                 bool verbose = false):
  //     ConstraintBase(n_variables, tol),
  //     lb_(lb),
  //     ub_(ub)
  //   {}
  //   void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values, Eigen::SparseMatrix<double> &jacobian) override;
  //   bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
  //                      Eigen::SparseMatrix<double> &jacobian,
  //                      Eigen::VectorXd &lower, Eigen::VectorXd &upper) override;
  // private:
  //   Eigen::VectorXd lb_;
  //   Eigen::VectorXd ub_;
  // };

  class ConstraintSet
  {
  public:
    ConstraintSet();
    virtual ~ConstraintSet(){}

    void add(std::shared_ptr<ConstraintBase> constraint)
    {
      constraints_.push_back(constraint);
    }

    bool evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                       Eigen::SparseMatrix<double> &jacobian, Eigen::VectorXd &lower,
                       Eigen::VectorXd &upper);

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

    void setNumverOfVariables(int n_variables) {n_variables_ = n_variables;}
    void setInitialVariable(const Eigen::VectorXd initial_x) {initial_x_ = initial_x;}
    void solve();
    Eigen::VectorXd getLastX() {
      return x_log_.back();
    }

    std::vector<Eigen::VectorXd> getXLog() {
      return x_log_;
    }

  private:
    int n_variables_;
    OsqpEigen::Solver qp_solver_;
    sqpSolverOption sqp_solver_option_;
    std::shared_ptr<CostBase> cost_;
    std::shared_ptr<ConstraintSet> cstset_;
    Eigen::VectorXd cstset_lower_bounds_;
    Eigen::VectorXd cstset_upper_bounds_;
    Eigen::VectorXd cstset_values_;
    Eigen::SparseMatrix<double> cstset_jacobian_;

    Eigen::VectorXd initial_x_;
    Eigen::MatrixXd qp_hessian_;
    Eigen::MatrixXd qp_hessian_sparse_;
    std::vector<Eigen::VectorXd> x_log_;

    void initSolver(const Eigen::VectorXd& x);
    Eigen::VectorXd QP(const Eigen::VectorXd& x);
    Eigen::MatrixXd updateHessianBFGS();
  };

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

