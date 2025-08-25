#ifndef FREEFORM_OPTIMIZATION_H
#define FREEFORM_OPTIMIZATION_H
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

Eigen::MatrixXd pseudoInverse(const Eigen::SparseMatrix<double> &A, double tol);
void matrix_to_vector(const Eigen::MatrixXd &V, Eigen::VectorXd &x);

void two_matrix_to_vector(const Eigen::MatrixXd &V1, const Eigen::MatrixXd &V2, Eigen::VectorXd &x);


bool Newton_step(Meshes &meshes);

void Newton(Meshes &meshes);

void Minimize(Meshes &meshes, int num_iterations);

void Minimize_sub(Meshes &meshes, int num_iterations);

void Newton(Meshes &meshes, int num_iterations);

#endif FREEFORM_OPTIMIZATION_H
