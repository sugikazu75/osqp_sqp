#include <osqp_sqp/osqp_sqp.h>

namespace osqpsqp
{
  bool ConstraintBase::checkJacobian(const Eigen::VectorXd &x, double eps, bool verbose)
  {
    Eigen::SparseMatrix<double> ana_jac = Eigen::SparseMatrix<double>(n_constraints_, n_variables_);
    Eigen::MatrixXd num_jac = Eigen::MatrixXd::Zero(n_constraints_, n_variables_);
    ana_jac.setZero();
    num_jac.setZero();

  { // numerical differentiation
    Eigen::SparseMatrix<double> whatever = Eigen::SparseMatrix<double>(n_constraints_, n_variables_);
    Eigen::VectorXd f0 = Eigen::VectorXd::Zero(n_constraints_);
    evaluate(x, f0, whatever);

    for (size_t i = 0; i < n_variables_; i++) {
      Eigen::VectorXd x_plus = x;
      x_plus(i) += eps;
      Eigen::VectorXd f1 = Eigen::VectorXd::Zero(n_constraints_);
      evaluate(x_plus, f1, whatever);
      num_jac.col(i) = (f1 - f0) * (1 / eps);
    }
  }

  { // analytical differentiation
    Eigen::VectorXd whatever = Eigen::VectorXd::Zero(n_constraints_);
    evaluate(x, whatever, ana_jac);
  }

  double max_diff = (num_jac - ana_jac.toDense()).cwiseAbs().maxCoeff();
  double check_eps = eps * 10;
  bool ok = max_diff < check_eps;
  if (verbose && !ok) {
    std::cout << "max_diff: " << max_diff << std::endl;
    std::cout << "check_eps: " << check_eps << std::endl;
    std::cout << "num_jac: " << std::endl << num_jac << std::endl;
    std::cout << "ana_jac: " << std::endl << ana_jac.toDense() << std::endl;
  }
  return true;
}

  sqpSolver::sqpSolver(int n_variables):
    n_variables_(n_variables)
  {
    initial_x_ = Eigen::VectorXd::Zero(n_variables_);
    qp_hessian_ = Eigen::MatrixXd::Zero(n_variables_, n_variables_);
    
  }

  void sqpSolver::solve()
  {
    Eigen::VectorXd x = initial_x_;
    initSolver(x);

    for (int i = 0; i < sqp_solver_option_.max_iter; ++i) {
      x_log_.push_back(x);
      Eigen::VectorXd solution = QP(x);
      x += solution;

      if (solution.norm() < sqp_solver_option_.sqp_relative_tolerance_) {
        break;
      }
    }
  }

  void sqpSolver::initSolver(const Eigen::VectorXd& x)
  {
    qp_solver_.settings()->setWarmStart(true);
    qp_solver_.settings()->setVerbosity(sqp_solver_option_.qp_verbose);
    qp_solver_.settings()->setRelativeTolerance(sqp_solver_option_.qp_relative_tolerance_);

    qp_hessian_sparse_ = qp_hessian_.sparseView();

  }

  Eigen::VectorXd sqpSolver::QP(const Eigen::VectorXd& x)
  {

    return qp_solver_.getSolution();
  }
};
