#pragma once

#include <Eigen/Dense>
#include <iostream>

namespace osqpsqp
{
  class CostBase
  {
  public:
    CostBase(){}
    ~CostBase(){}

    virtual void evaluate(const Eigen::VectorXd &x, double &value, Eigen::VectorXd &gradient) = 0;
    bool checkGradient(const Eigen::VectorXd &x, bool verbose = false, double eps=1e-8)
    {
      int n_variables = x.size();
      Eigen::VectorXd ana_grad = Eigen::VectorXd::Zero(n_variables);
      Eigen::VectorXd num_grad = Eigen::VectorXd::Zero(n_variables);

      { // numerical differentiation
        Eigen::VectorXd whatever = Eigen::VectorXd(n_variables);
        double f0;
        evaluate(x, f0, whatever);

        for(size_t i = 0; i < n_variables; ++i)
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
  };
}
