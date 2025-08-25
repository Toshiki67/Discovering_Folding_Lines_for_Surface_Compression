#include "Uniform.h"
#include <iostream>
#include <igl/local_basis.h>
#include <igl/doublearea.h>
#include <igl/per_face_normals.h>
#include <igl/parallel_for.h>
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>


void Uniform_initialize(Meshes &meshes) {
    // Prepare The Uniform Laplacian 
	// Mesh in (V,F)
	Eigen::SparseMatrix<double> A;

	igl::adjacency_matrix(meshes.F_undeformed, A);
    // sum each row 
	Eigen::SparseVector<double> Asum;
	igl::sum(A,1,Asum);
    // Convert row sums into diagonal of sparse matrix
	Eigen::SparseMatrix<double> Adiag;
    // clear and resize output
    Eigen::SparseMatrix<double> dyn_X(Asum.size(),Asum.size());
    dyn_X.reserve(Asum.size());
    // loop over non-zeros
    for(typename Eigen::SparseVector<double>::InnerIterator it(Asum); it; ++it)
    {
        dyn_X.coeffRef(it.index(),it.index()) += it.value();
    }
    Adiag = Eigen::SparseMatrix<double>(dyn_X);
    // Build uniform laplacian
    meshes.L = A-Adiag;

	meshes.L2 = 2 * meshes.L * meshes.L;
}

void Compute_Quad_Uniform(Meshes &meshes) {
    meshes.Quad_Uni_C = 0.0;
	Eigen::VectorXd X(meshes.V_undeformed.rows()), Y(meshes.V_undeformed.rows()), Z(meshes.V_undeformed.rows());
    igl::parallel_for(meshes.V_undeformed.rows(), [&](int vi) {
        X(vi) = meshes.V_undeformed(vi, 0);
        Y(vi) = meshes.V_undeformed(vi, 1);
        Z(vi) = meshes.V_undeformed(vi, 2);
    },10000);
	meshes.Quad_Uni_C =
		(meshes.L * X).squaredNorm() + 
		(meshes.L * Y).squaredNorm() + 
		(meshes.L * Z).squaredNorm();
    igl::parallel_for(meshes.V_deformed.rows(), [&](int vi) {
        X(vi) = meshes.V_deformed(vi, 0);
        Y(vi) = meshes.V_deformed(vi, 1);
        Z(vi) = meshes.V_deformed(vi, 2);
    },10000);
    meshes.Quad_Uni_C +=
        (meshes.L * X).squaredNorm() +
        (meshes.L * Y).squaredNorm() +
        (meshes.L * Z).squaredNorm();
}

void Compute_Quad_derivatives_Uniform(Meshes &meshes) {
    Eigen::MatrixXd &Quad_Uniform_grad = meshes.Quad_Uniform_grad;
    Quad_Uniform_grad = Eigen::MatrixXd::Zero(meshes.V_undeformed.rows() + meshes.V_deformed.rows(), 3);
	// gradient = 2*||L * x|| * L
	Eigen::VectorXd X(meshes.V_undeformed.rows()), Y(meshes.V_undeformed.rows()), Z(meshes.V_undeformed.rows());
	igl::parallel_for(meshes.V_undeformed.rows(), [&](int vi) {
        X(vi) = meshes.V_undeformed(vi, 0);
        Y(vi) = meshes.V_undeformed(vi, 1);
        Z(vi) = meshes.V_undeformed(vi, 2);
    },10000);
	Eigen::VectorXd grad_X = meshes.L2 * X;
	Eigen::VectorXd grad_Y = meshes.L2 * Y;
	Eigen::VectorXd grad_Z = meshes.L2 * Z;

    igl::parallel_for(meshes.V_undeformed.rows(), [&](int vi) {
        Quad_Uniform_grad(vi, 0) = grad_X(vi);
        Quad_Uniform_grad(vi, 1) = grad_Y(vi);
        Quad_Uniform_grad(vi, 2) = grad_Z(vi);
    },10000);
	

    igl::parallel_for(meshes.V_deformed.rows(), [&](int vi) {
        X(vi) = meshes.V_deformed(vi, 0);
        Y(vi) = meshes.V_deformed(vi, 1);
        Z(vi) = meshes.V_deformed(vi, 2);
    },10000);
    grad_X = meshes.L2 * X;
    grad_Y = meshes.L2 * Y;
    grad_Z = meshes.L2 * Z;
    igl::parallel_for(meshes.V_deformed.rows(), [&](int vi) {
        Quad_Uniform_grad(vi + meshes.V_undeformed.rows(), 0) = grad_X(vi);
        Quad_Uniform_grad(vi + meshes.V_undeformed.rows(), 1) = grad_Y(vi);
        Quad_Uniform_grad(vi + meshes.V_undeformed.rows(), 2) = grad_Z(vi);
    },10000);
}


void Compute_Quad_Uniform_sub(Meshes &meshes) {
    meshes.Quad_Uni_C = 0.0;
	Eigen::VectorXd X(meshes.V_undeformed.rows()), Y(meshes.V_undeformed.rows()), Z(meshes.V_undeformed.rows());
    igl::parallel_for(meshes.V_undeformed.rows(), [&](int vi) {
        X(vi) = meshes.V_undeformed(vi, 0);
        Y(vi) = meshes.V_undeformed(vi, 1);
        Z(vi) = meshes.V_undeformed(vi, 2);
    },10000);
	meshes.Quad_Uni_C =
		(meshes.L * X).squaredNorm() + 
		(meshes.L * Y).squaredNorm() + 
		(meshes.L * Z).squaredNorm();
    igl::parallel_for(meshes.V_deformed.rows(), [&](int vi) {
        X(vi) = meshes.V_deformed(vi, 0);
        Y(vi) = meshes.V_deformed(vi, 1);
        Z(vi) = meshes.V_deformed(vi, 2);
    },10000);
    meshes.Quad_Uni_C +=
        (meshes.L * X).squaredNorm() +
        (meshes.L * Y).squaredNorm() +
        (meshes.L * Z).squaredNorm();
}

void Compute_Quad_derivatives_Uniform_sub(Meshes &meshes) {
    Eigen::MatrixXd &Quad_Uniform_grad = meshes.Quad_Uniform_grad;
    Quad_Uniform_grad = Eigen::MatrixXd::Zero(meshes.V_undeformed.rows() + meshes.V_deformed.rows(), 3);
	// gradient = 2*||L * x|| * L
	Eigen::VectorXd X(meshes.V_undeformed.rows()), Y(meshes.V_undeformed.rows()), Z(meshes.V_undeformed.rows());
	igl::parallel_for(meshes.V_undeformed.rows(), [&](int vi) {
        X(vi) = meshes.V_undeformed(vi, 0);
        Y(vi) = meshes.V_undeformed(vi, 1);
        Z(vi) = meshes.V_undeformed(vi, 2);
    },10000);
	Eigen::VectorXd grad_X = meshes.L2 * X;
	Eigen::VectorXd grad_Y = meshes.L2 * Y;
	Eigen::VectorXd grad_Z = meshes.L2 * Z;

    igl::parallel_for(meshes.V_undeformed.rows(), [&](int vi) {
        Quad_Uniform_grad(vi, 0) = grad_X(vi);
        Quad_Uniform_grad(vi, 1) = grad_Y(vi);
        Quad_Uniform_grad(vi, 2) = grad_Z(vi);
    },10000);
	

    igl::parallel_for(meshes.V_deformed.rows(), [&](int vi) {
        X(vi) = meshes.V_deformed(vi, 0);
        Y(vi) = meshes.V_deformed(vi, 1);
        Z(vi) = meshes.V_deformed(vi, 2);
    },10000);
    grad_X = meshes.L2 * X;
    grad_Y = meshes.L2 * Y;
    grad_Z = meshes.L2 * Z;
    igl::parallel_for(meshes.V_deformed.rows(), [&](int vi) {
        Quad_Uniform_grad(vi + meshes.V_undeformed.rows(), 0) = grad_X(vi);
        Quad_Uniform_grad(vi + meshes.V_undeformed.rows(), 1) = grad_Y(vi);
        Quad_Uniform_grad(vi + meshes.V_undeformed.rows(), 2) = grad_Z(vi);
    },10000);
}
