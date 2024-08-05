#pragma once

#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <osqp_sqp/constraint.h>
#include <osqp_sqp/constraint_set.h>
#include <osqp_sqp/utils.h>
#include <chrono>
#include <memory>
#include <vector>

namespace osqpsqp
{
  class sqpSolver
  {
  public:
    sqpSolver(int n_variables):
      n_variables_(n_variables),
      qp_first_solve_(true),
      hessian_first_update_(true)
    {
      initial_x_ = Eigen::VectorXd::Zero(n_variables_);
      qp_hessian_ = Eigen::MatrixXd::Identity(n_variables_, n_variables_);
      qp_gradient_ = Eigen::VectorXd::Zero(n_variables_);
      sqp_log_.resize(0);
      cstset_ = std::make_shared<ConstraintSet>();
    }
    ~sqpSolver(){};

    void setCost(std::shared_ptr<CostBase> cost) {cost_ = cost;}
    void addConstraint(std::shared_ptr<ConstraintBase> constraint) {cstset_->add(constraint);}
    void setConstraintSet(std::shared_ptr<ConstraintSet> cstset) {cstset_ = cstset_;}
    void setInitialVariable(const Eigen::VectorXd initial_x) {initial_x_ = initial_x;}
    void setLowerBounds(const Eigen::VectorXd lower_bounds) {sqp_lower_bounds_ = lower_bounds;}
    void setUpperBounds(const Eigen::VectorXd upper_bounds) {sqp_upper_bounds_ = upper_bounds;}

    double getCostValue() {return sqp_cost_value_;}
    int getIteration() {return sqp_iter_;}
    std::vector<Eigen::VectorXd> getSolutionLog() {return sqp_log_;}
    Eigen::VectorXd getSolution() {return sqp_solution_;}
    Eigen::VectorXd getConstrainValue() {return sqp_constraint_value_;}
    double getSolveTime() {return sqp_solve_time_;}

    bool checkCostGradient(const Eigen::VectorXd &x, bool verbose=false, double eps=1e-8) {return cost_->checkGradient(x, verbose, eps);}
    bool checkConstraintJacobians(const Eigen::VectorXd &x, bool verbose=false, double eps=1e-8){return cstset_->checkJacobians(x, verbose, eps);}

    void solve()
    {
      auto start = std::chrono::high_resolution_clock::now();
      sqp_log_.clear();
      sqp_solution_ = initial_x_;
      initSolver(sqp_solution_);
      for(size_t i = 0; i < sqp_solver_option_.sqp_max_iter; ++i)
        {
          sqp_log_.push_back(sqp_solution_);
          Eigen::VectorXd solution = QP(sqp_solution_);
          sqp_solution_ += solution;
          sqp_iter_ = i + 1;
          if(solution.cwiseAbs().maxCoeff() < sqp_solver_option_.sqp_relative_tolerance)
            break;
        }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      sqp_solve_time_ = duration.count();
    }

  private:
    int n_variables_;
    int n_constraints_;
    sqpSolverOption sqp_solver_option_;
    std::shared_ptr<CostBase> cost_;
    std::shared_ptr<ConstraintSet> cstset_;

    double sqp_solve_time_;
    double sqp_cost_value_;
    int sqp_iter_;
    std::vector<Eigen::VectorXd> sqp_log_;
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

    void initSolver(const Eigen::VectorXd& x)
    {
      qp_solver_.settings()->setWarmStart(true);
      qp_solver_.settings()->setVerbosity(sqp_solver_option_.qp_verbose);
      qp_solver_.settings()->setRelativeTolerance(sqp_solver_option_.qp_relative_tolerance);

      n_constraints_ = cstset_->getNumberOfConstraints();

      sqp_constraint_value_.resize(n_constraints_);
      qp_constraint_linear_matrix_sparse_ = Eigen::SparseMatrix<double>(n_constraints_, n_variables_);
      qp_constraint_linear_matrix_sparse_.setZero();
      qp_lower_bounds_.resize(n_constraints_);
      qp_upper_bounds_.resize(n_constraints_);

      cost_->evaluate(x, sqp_cost_value_, qp_gradient_);
      cstset_->evaluateFull(x, sqp_constraint_value_,
                            qp_constraint_linear_matrix_sparse_,
                            sqp_lower_bounds_, sqp_upper_bounds_,
                            qp_lower_bounds_, qp_upper_bounds_);

      qp_hessian_sparse_ = qp_hessian_.sparseView();
      qp_constraint_linear_matrix_ = qp_constraint_linear_matrix_sparse_.toDense();

      bool ok = true;
      qp_solver_.data()->setNumberOfVariables(n_variables_);
      qp_solver_.data()->setNumberOfConstraints(n_constraints_);
      ok &= qp_solver_.data()->setHessianMatrix(qp_hessian_sparse_);
      ok &= qp_solver_.data()->setGradient(qp_gradient_);
      ok &= qp_solver_.data()->setLinearConstraintsMatrix(qp_constraint_linear_matrix_sparse_);
      ok &= qp_solver_.data()->setLowerBound(qp_lower_bounds_);
      ok &= qp_solver_.data()->setUpperBound(qp_upper_bounds_);

      if(!ok)
        {
          std::cerr << "variable : " << x.transpose() << std::endl;
          throw std::runtime_error("initSolver failed");
        }
      qp_solver_.initSolver();
    }

    Eigen::VectorXd QP(const Eigen::VectorXd& x)
    {
      //update solver
      Eigen::VectorXd qp_gradient_prev = qp_gradient_;
      Eigen::MatrixXd qp_constraint_linear_matrix_prev = qp_constraint_linear_matrix_;

      cost_->evaluate(x, sqp_cost_value_, qp_gradient_);
      cstset_->evaluateFull(x, sqp_constraint_value_,
                            qp_constraint_linear_matrix_sparse_,
                            sqp_lower_bounds_, sqp_upper_bounds_,
                            qp_lower_bounds_, qp_upper_bounds_);
      qp_constraint_linear_matrix_ = qp_constraint_linear_matrix_sparse_.toDense();

      if(qp_first_solve_)
        qp_first_solve_ = false;
      else
        {
          qp_hessian_ = updateHessianBFGS(qp_hessian_, x, qp_x_prev_,
                                          qp_dual_solution_, qp_gradient_, qp_gradient_prev,
                                          qp_constraint_linear_matrix_, qp_constraint_linear_matrix_prev);
        }
      qp_hessian_sparse_ = convertSparseMatrix(qp_hessian_);

      bool ok = true;
      ok &= qp_solver_.updateHessianMatrix(qp_hessian_sparse_); //H
      ok &= qp_solver_.updateGradient(qp_gradient_); //q
      ok &= qp_solver_.updateLinearConstraintsMatrix(qp_constraint_linear_matrix_sparse_); //A
      ok &= qp_solver_.updateBounds(qp_lower_bounds_, qp_upper_bounds_);
      ok &= qp_solver_.solve();

      if(!ok)
        {
          std::cerr << "variable : " << x.transpose() << std::endl;
          std::cerr << "solution : " << qp_solver_.getSolution().transpose() << std::endl;
          throw std::runtime_error("QP failed");
        }

      qp_dual_solution_ = qp_solver_.getDualSolution();
      qp_x_prev_ = x;

      return qp_solver_.getSolution();
    }

    Eigen::MatrixXd updateHessianBFGS(Eigen::MatrixXd h_prev, const Eigen::VectorXd& x, const Eigen::VectorXd& x_prev,
                                      const Eigen::VectorXd& lambda_prev, const Eigen::VectorXd& grad_f, const Eigen::VectorXd& grad_f_prev,
                                      const Eigen::MatrixXd& grad_g, const Eigen::MatrixXd& grad_g_prev)
    {
      Eigen::VectorXd s = x - x_prev;
      Eigen::VectorXd q = (grad_f - grad_f_prev) + (grad_g - grad_g_prev).transpose() * lambda_prev;

      double qs = q.transpose() * s;
      Eigen::VectorXd Hs = h_prev * s;
      double sHs = s.transpose() * h_prev * s;

      if(sHs < 0.0 || hessian_first_update_) {
        h_prev = Eigen::MatrixXd::Identity(n_variables_, n_variables_) * autoScale(s, q);
        Hs = h_prev * s;
        sHs = s.transpose() * h_prev * s;
        hessian_first_update_ = false;
      }

      if(qs < 0.2 * sHs) {
        double update_factor = (1 - 0.2) / (1 - qs / sHs);
        q = update_factor * q + (1 - update_factor) * Hs;
        qs = q.transpose() * s;
      }

      Eigen::MatrixXd h =  h_prev + (q * q.transpose()) / qs - (Hs * Hs.transpose()) / sHs;
      return h;
    }
  };
}
