#include <osqp_sqp/osqp_sqp.h>

/*
  x in R^2 x = [x0, x1]

  min sqrt(x1)
  s.t.
  x1 >= 0
  x1 - x0^2 + x0 >= 5/4
  x1 - x0^2 = 0

  ans: 5/4 (x0, x1) = (5/4, 25/16)

*/

class MyCost : public osqpsqp::CostBase
{
public:
  using CostBase::CostBase;
  void evaluate(const Eigen::VectorXd &x, double &value, Eigen::VectorXd &gradient)
  {
    value = std::sqrt(x(1));
    gradient(0) = 0.0;
    gradient(1) = 0.5 / std::sqrt(x(1));
  }
};

class MyConstraints : public osqpsqp::InequalityConstraintBase
{
public:
  using InequalityConstraintBase::InequalityConstraintBase;
  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values, Eigen::SparseMatrix<double> &jacobian, size_t constraint_idx_head)
  {
    values(constraint_idx_head +  0) = x(1);
    values(constraint_idx_head +  1) = x(1) - x(0) * x(0) + x(0);
    values(constraint_idx_head +  2) = x(1) - x(0) * x(0);

    jacobian.coeffRef(constraint_idx_head + 0, 0) = 0;
    jacobian.coeffRef(constraint_idx_head + 1, 0) = -2 * x(0) + 1;
    jacobian.coeffRef(constraint_idx_head + 2, 0) = -2 * x(0);
    jacobian.coeffRef(constraint_idx_head + 0, 1) = 1;
    jacobian.coeffRef(constraint_idx_head + 1, 1) = 1;
    jacobian.coeffRef(constraint_idx_head + 2, 1) = 1;
  }
};

int main()
{
  osqpsqp::sqpSolver solver(2);

  solver.setCost(std::make_shared<MyCost>());
  solver.addConstraint(std::make_shared<MyConstraints>(std::vector<double>{1e-4, 1e-4, 1e-4}));

  Eigen::VectorXd initial_x = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd sqp_lower_bounds = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd sqp_upper_bounds = Eigen::VectorXd::Zero(3);
  initial_x << 4, 5;
  sqp_lower_bounds << 0, 5.0 / 4.0, 0;
  sqp_upper_bounds << INFINITY, INFINITY, 0;

  solver.checkCostGradient(initial_x);
  solver.checkConstraintJacobians(initial_x);
  solver.setInitialVariable(initial_x);
  solver.setLowerBounds(sqp_lower_bounds);
  solver.setUpperBounds(sqp_upper_bounds);

  solver.solve();

  std::cout << "solved with " << solver.getIteration() << " iteration" << std::endl;
  std::cout << "final cost: " << solver.getCostValue() << std::endl;
  std::cout << "final solution: " << solver.getSolution().transpose() << std::endl;
  std::cout << "solve time: " << solver.getSolveTime() << "[us]" << std::endl;
}
