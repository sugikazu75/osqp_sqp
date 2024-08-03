#include <osqp_sqp/osqp_sqp.h>

namespace osqpsqp
{
  ConstraintBase::ConstraintBase(int n_variables, const std::vector<double>& tol):
    n_variables_(n_variables)
  {
    tol_.resize(tol.size());
    for(size_t i = 0; i < tol.size(); i++) tol_(i) = tol.at(i);
    n_constraints_ = tol_.size();
  }

  bool CostBase::checkGradient(const Eigen::VectorXd &x, double eps, bool verbose)
  {
    Eigen::VectorXd ana_grad = Eigen::VectorXd(n_variables_);
    Eigen::VectorXd num_grad = Eigen::VectorXd(n_variables_);

    { // numerical differentiation
      Eigen::VectorXd whatever = Eigen::VectorXd(n_variables_);
      double f0;
      evaluate(x, f0, whatever);

      for(int i = 0; i < n_variables_; i++)
        {
          Eigen::VectorXd x_plus = x;
          x_plus(i) += eps;
          double f1;
          evaluate(x_plus, f1, whatever);
          num_grad(i) = (f1 - f0) * (1 / eps);
        }
    }

    { // analytical differentiation
      double whatever;
      evaluate(x, whatever, ana_grad);
    }

    double max_diff = (num_grad - ana_grad).cwiseAbs().maxCoeff();
    double check_eps = eps * 10;
    bool ok = max_diff < check_eps;
    if(verbose && !ok)
      {
        std::cout << "max_diff: " << max_diff << std::endl;
        std::cout << "check_eps: " << check_eps << std::endl;
        std::cout << "num_grad: " << num_grad.transpose() << std::endl;
        std::cout << "ana_grad: " << ana_grad.transpose() << std::endl;
      }
    if(verbose && ok)
      {
        std::cout << "numerical and analytical gradient is identical" << std::endl;
      }
    return ok;
  }

  bool ConstraintBase::checkJacobian(const Eigen::VectorXd &x, double eps, bool verbose)
  {
    Eigen::SparseMatrix<double> ana_jac = Eigen::SparseMatrix<double>(n_constraints_, n_variables_);
    Eigen::MatrixXd num_jac = Eigen::MatrixXd::Zero(n_constraints_, n_variables_);
    ana_jac.setZero();
    num_jac.setZero();

  { // numerical differentiation
    Eigen::SparseMatrix<double> whatever = Eigen::SparseMatrix<double>(n_constraints_, n_variables_);
    Eigen::VectorXd f0 = Eigen::VectorXd::Zero(n_constraints_);
    evaluate(x, f0, whatever, 0);

    for (int i = 0; i < n_variables_; i++)
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
  if (verbose && !ok)
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

  bool InequalityConstraintBase::evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
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

    //  sqp_lower_bounds - tol_   < value_sliced  < sqp_upper_bounds + tol_ ?
    bool is_feasible =((sqp_lower_bounds.segment(constraint_idx_head, n_constraints_) - tol_).array() - value_sliced.array()).all()  && (value_sliced.array() <= (sqp_upper_bounds.segment(constraint_idx_head, n_constraints_) + tol_).array()).all();
    return is_feasible;
  }

  bool ConstraintSet::evaluateFull(const Eigen::VectorXd &x, Eigen::VectorXd &values,
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

  sqpSolver::sqpSolver(int n_variables):
    n_variables_(n_variables),
    hessian_first_update_(true)
  {
    qp_first_solve_ = true;
    initial_x_ = Eigen::VectorXd::Zero(n_variables_);
    qp_gradient_ = Eigen::VectorXd::Zero(n_variables_);
    qp_hessian_ = Eigen::MatrixXd::Identity(n_variables_, n_variables_);
    cstset_ = std::make_shared<ConstraintSet>();
    cstset_->setNumberOfVariables(n_variables_);
  }

  void sqpSolver::solve()
  {
    Eigen::VectorXd x = initial_x_;
    initSolver(x);
    std::cout << "init solver done" << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < sqp_solver_option_.max_iter; ++i) {
      x_log_.push_back(x);
      std::cout << "iter:" << i << std::endl;
      Eigen::VectorXd solution = QP(x);
      x += solution;

      if (solution.norm() < sqp_solver_option_.sqp_relative_tolerance_) {
        break;
      }
    }
  }

  void sqpSolver::initSolver(const Eigen::VectorXd& x)
  {
    std::cout << sqp_solver_option_.qp_relative_tolerance_ << std::endl;
    qp_solver_.settings()->setWarmStart(true);
    qp_solver_.settings()->setVerbosity(sqp_solver_option_.qp_verbose);
    qp_solver_.settings()->setRelativeTolerance(sqp_solver_option_.qp_relative_tolerance_);
    std::cout << "qp solver setting" << std::endl;

    double cost_value;
    cost_->evaluate(x, cost_value, qp_gradient_);

    n_constraints_ = cstset_->getNumberOfConstraints();
    qp_constraint_linear_matrix_sparse_ = Eigen::SparseMatrix<double>(n_constraints_, n_variables_);
    qp_lower_bounds_.resize(n_constraints_);
    qp_upper_bounds_.resize(n_constraints_);
    Eigen::VectorXd constraint_value = Eigen::VectorXd::Zero(n_constraints_);
    cstset_->evaluateFull(x, constraint_value,
                          qp_constraint_linear_matrix_sparse_,
                          sqp_lower_bounds_, sqp_upper_bounds_,
                          qp_lower_bounds_, qp_upper_bounds_);
    qp_hessian_sparse_ = qp_hessian_.sparseView();
    qp_constraint_linear_matrix_ = qp_constraint_linear_matrix_sparse_.toDense();

    qp_solver_.data()->setNumberOfVariables(n_variables_);
    qp_solver_.data()->setNumberOfConstraints(cstset_->getNumberOfConstraints());
    bool ok = true;
    ok &= qp_solver_.data()->setHessianMatrix(qp_hessian_sparse_);
    ok &= qp_solver_.data()->setGradient(qp_gradient_);
    ok &= qp_solver_.data()->setLinearConstraintsMatrix(qp_constraint_linear_matrix_sparse_);
    ok &= qp_solver_.data()->setLowerBound(qp_lower_bounds_);
    ok &= qp_solver_.data()->setUpperBound(qp_upper_bounds_);

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "x: " << x.transpose() << std::endl;
    std::cout << "hessian: \n" << qp_hessian_sparse_.toDense() << std::endl;
    std::cout << "gradient: " << qp_gradient_.transpose() << std::endl;
    std::cout << "jacobian\n" << qp_constraint_linear_matrix_sparse_.toDense() << std::endl;
    std::cout << "qp_lower: " << qp_lower_bounds_.transpose() << std::endl;
    std::cout << "qp_upper: " << qp_upper_bounds_.transpose() << std::endl;

    if (!ok)
      {
        std::cerr << "variable : " << x.transpose() << std::endl;
        throw std::runtime_error("initSolver failed");
      }
    qp_solver_.initSolver();
  }

  Eigen::VectorXd sqpSolver::QP(const Eigen::VectorXd& x)
  {
    //update solver
    Eigen::VectorXd qp_gradient_prev = qp_gradient_;
    Eigen::MatrixXd qp_constraint_linear_matrix_prev = qp_constraint_linear_matrix_;

    double cost_value;
    cost_->evaluate(x, cost_value, qp_gradient_);

    Eigen::VectorXd constraint_value = Eigen::VectorXd(n_constraints_);
    cstset_->evaluateFull(x, constraint_value,
                          qp_constraint_linear_matrix_sparse_,
                          sqp_lower_bounds_, sqp_upper_bounds_,
                          qp_lower_bounds_, qp_upper_bounds_);

    qp_constraint_linear_matrix_ = qp_constraint_linear_matrix_sparse_.toDense();

    if(qp_first_solve_)
      qp_first_solve_ = false;
    else
      {
        qp_hessian_ = updateHessianBFGS(qp_hessian_, x, qp_x_,
                                        qp_dual_solution_, qp_gradient_, qp_gradient_prev,
                                        qp_constraint_linear_matrix_, qp_constraint_linear_matrix_prev);
      }
    qp_hessian_sparse_ = convertSparseMatrix(qp_hessian_);

    std::cout << "x: " << x.transpose() << std::endl;
    std::cout << "hessian" << std::endl;
    std::cout << qp_hessian_ << std::endl;
    std::cout << "gradient " << qp_gradient_.transpose() << std::endl;
    std::cout << "jacobian\n" << qp_constraint_linear_matrix_ << std::endl;
    std::cout << "qp_lower: " << qp_lower_bounds_.transpose() << std::endl;
    std::cout << "qp_upper: " << qp_upper_bounds_.transpose() << std::endl;

    bool ok = true;
    ok &= qp_solver_.updateHessianMatrix(qp_hessian_sparse_); //H
    ok &= qp_solver_.updateGradient(qp_gradient_); //q
    ok &= qp_solver_.updateLinearConstraintsMatrix(qp_constraint_linear_matrix_sparse_); //A
    ok &= qp_solver_.updateBounds(qp_lower_bounds_, qp_upper_bounds_);
    ok &= qp_solver_.solve();
    
    if (!ok)
      {
        std::cerr << "variable : " << x.transpose() << std::endl;
        std::cerr << "solution : " << qp_solver_.getSolution().transpose() << std::endl;
        throw std::runtime_error("QP failed");
      }
    
    qp_dual_solution_ = qp_solver_.getDualSolution();
    qp_x_ = x;
    
    std::cout << "solution: " << qp_solver_.getSolution().transpose() << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    return qp_solver_.getSolution();
  }

  Eigen::MatrixXd sqpSolver::updateHessianBFGS(Eigen::MatrixXd h_prev, const Eigen::VectorXd& x, const Eigen::VectorXd& x_prev,
                                               const Eigen::VectorXd& lambda_prev, const Eigen::VectorXd& grad_f, const Eigen::VectorXd& grad_f_prev,
                                               const Eigen::MatrixXd& grad_g, const Eigen::MatrixXd& grad_g_prev)
  {
    Eigen::VectorXd s = x - x_prev;
    Eigen::VectorXd q = (grad_f - grad_f_prev) + (grad_g - grad_g_prev).transpose() * lambda_prev;

    double qs = q.transpose() * s;
    Eigen::VectorXd Hs = h_prev * s;
    double sHs = s.transpose() * h_prev * s;

    if (sHs < 0.0 || hessian_first_update_) {
      h_prev = Eigen::MatrixXd::Identity(n_variables_, n_variables_) * autoScale(s, q);
      Hs = h_prev * s;
      sHs = s.transpose() * h_prev * s;
      hessian_first_update_ = false;
    }

    if (qs < 0.2 * sHs) {
      double update_factor = (1 - 0.2) / (1 - qs / sHs);
      q = update_factor * q + (1 - update_factor) * Hs;
      qs = q.transpose() * s;
    }

    Eigen::MatrixXd h =  h_prev + (q * q.transpose()) / qs - (Hs * Hs.transpose()) / sHs;
    return h;
  }
};
