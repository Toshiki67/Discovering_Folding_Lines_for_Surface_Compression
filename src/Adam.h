#ifndef FREEFORM_ADAM_H
#define FREEFORM_ADAM_H
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <igl/parallel_for.h>

class AdamOptimizer {
public:
    AdamOptimizer(double lr, double beta1, double beta2, double epsilon)
        : lr(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void update(Eigen::MatrixXd& params, const Eigen::MatrixXd& grads) {
        if (m.size() == 0) {
            m = Eigen::MatrixXd::Zero(grads.rows(), grads.cols());
            v = Eigen::MatrixXd::Zero(grads.rows(), grads.cols());
        }
        t++;
        igl::parallel_for(params.rows(), [&](int i) {
            for (int j = 0; j < params.cols(); j++) {
                m(i, j) = beta1 * m(i, j) + (1 - beta1) * grads(i, j);
                v(i, j) = beta2 * v(i, j) + (1 - beta2) * grads(i, j) * grads(i, j);
                double m_hat = m(i, j) / (1 - std::pow(beta1, t));
                double v_hat = v(i, j) / (1 - std::pow(beta2, t));
                params(i, j) -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }, 10000);
    }

private:
    double lr, beta1, beta2, epsilon;
    int t = 0;
    Eigen::MatrixXd m, v;
};

#endif FREEFORM_ADAM_H