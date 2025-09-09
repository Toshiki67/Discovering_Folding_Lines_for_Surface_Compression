#ifndef FREEFORM_OPTIMIZATION_H
#define FREEFORM_OPTIMIZATION_H
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

void matrix_to_vector(const Eigen::MatrixXd &V, Eigen::VectorXd &x);

void two_matrix_to_vector(const Eigen::MatrixXd &V1, const Eigen::MatrixXd &V2, Eigen::VectorXd &x);

void Minimize(Meshes &meshes, int num_iterations);


void Newton(Meshes &meshes, int num_iterations);

#endif FREEFORM_OPTIMIZATION_H
