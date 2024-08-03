#include <osqp_sqp/osqp_sqp.h>

/*
  x in R^2 x = [x0, x1]

  min sqrt(x1)
  s.t.
      x1 >= 0
      x1 - (-x0 + 1)^3 >= 0
      x1 - (2x0)^3 >= 0

 ans: 0.54433 (x0, x1) = (1.3, 8/27)

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
    // std::cout << "start evaluate impl" << std::endl;
    values(constraint_idx_head +  0) = x(1);
    values(constraint_idx_head +  1) = x(1) - (-x(0) + 1) * (-x(0) + 1) * (-x(0) + 1);
    values(constraint_idx_head +  2) = x(1) - (2 * x(0)) * (2 * x(0)) * (2 * x(0));

    jacobian.coeffRef(constraint_idx_head + 0, 0) = 0;
    jacobian.coeffRef(constraint_idx_head + 1, 0) = -3 * (-1) * (-x(0) + 1) * (-x(0) + 1);
    jacobian.coeffRef(constraint_idx_head + 2, 0) = -3 * 2    * (2 * x(0)) * (2 * x(0));
    jacobian.coeffRef(constraint_idx_head + 0, 1) = 1;
    jacobian.coeffRef(constraint_idx_head + 1, 1) = 1;
    jacobian.coeffRef(constraint_idx_head + 2, 1) = 1;
    // std::cout << "end evaluate impl" << std::endl;
  }
};


int main()
{
  osqpsqp::sqpSolver solver(2);

  solver.setCost(std::make_shared<MyCost>(2));
  solver.addConstraint(std::make_shared<MyConstraints>(2, std::vector<double>{1e-4, 1e-4, 1e-4}));

  Eigen::VectorXd initial_x = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd sqp_lower_bounds = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd sqp_upper_bounds = Eigen::VectorXd::Zero(3);
  initial_x << 1, 1;
  sqp_lower_bounds << 0, 0, 0;
  sqp_upper_bounds << INFINITY, INFINITY, INFINITY;

  solver.checkCostGradient(initial_x);
  solver.checkConstraintsJacobian(initial_x);
  solver.setInitialVariable(initial_x);
  solver.setLowerBounds(sqp_lower_bounds);
  solver.setUpperBounds(sqp_upper_bounds);

  solver.solve();

  std::vector<Eigen::VectorXd>  x_log = solver.getXLog();
  // for(auto x :x_log)
  //   {
  //     std::cout << x.transpose() << std::endl;
  //     std::cout << std::endl;
  //   }

  std::cout << "final solution: " << solver.getLastX().transpose() << std::endl;
}


  

  
