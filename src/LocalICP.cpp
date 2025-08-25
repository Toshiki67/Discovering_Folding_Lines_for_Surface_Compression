#include "LocalICP.h"
#include <cmath>
#include <iostream>
#include <igl/edge_topology.h>
#include <igl/edge_lengths.h>
#include <igl/parallel_for.h>
#include <omp.h>


void ICP(const Eigen::MatrixXd &V, const Eigen::MatrixXd &V_target, Eigen::MatrixXd &R, Eigen::VectorXd &t, Eigen::VectorXd &V_result){
    Eigen::MatrixXd V_ = V;
    Eigen::Vector3d V_target_mean = V_target.colwise().mean();
    Eigen::MatrixXd V_target_centered = V_target.rowwise() - V_target_mean.transpose();
    {
        Eigen::Vector3d V_mean = V_.colwise().mean();
        Eigen::MatrixXd V_centered = V_.rowwise() - V_mean.transpose();
        Eigen::MatrixXd H = V_target_centered.transpose() * V_centered;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd Vt = svd.matrixV();
        R = U * Vt.transpose();
        if (R.determinant() < 0) {
            U.col(2) *= -1;
            R = U * Vt.transpose();
        }
        t = V_target_mean - R * V_mean;
        Eigen::MatrixXd V_transformed = (R * V_.transpose()).transpose();
        V_transformed = V_transformed.rowwise() + t.transpose();
        double error = (V_target - V_transformed).norm();
        V_ = V_transformed;
        V_result = R * V_result + t;
    }
}

void Compute_Quad_LocalICP(Meshes &meshes) {
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    std::vector<std::vector<int>> &A = meshes.A;
    meshes.V_deformed_target = V_deformed;
    const double delta = meshes.delta;
    meshes.Quad_ICP_C = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < V_deformed.rows(); i++) {
        if (A[i].size() <= 2) {
            continue;
        }
        Eigen::MatrixXd V_undeformed_sub(A[i].size(), 3);
        for (int j = 0; j < A[i].size(); j++) {
            V_undeformed_sub.row(j) = V_undeformed.row(A[i][j]);
        }
        // V_undeformed_sub.row(A[i].size()) = V_undeformed.row(i);
        Eigen::MatrixXd V_deformed_sub(A[i].size() , 3);
        for (int j = 0; j < A[i].size(); j++) {
            V_deformed_sub.row(j) = V_deformed.row(A[i][j]);
        }
        // V_deformed_sub.row(A[i].size()) = V_deformed.row(i);
        Eigen::MatrixXd R;
        Eigen::VectorXd t;
        Eigen::VectorXd Vi_transformed = V_undeformed.row(i);
        ICP(V_undeformed_sub, V_deformed_sub, R, t, Vi_transformed);
        Eigen::Vector3d Vi = V_deformed.row(i);
        meshes.V_deformed_target.row(i) = Vi_transformed;
        double norm = (Vi - Vi_transformed).norm();
        #pragma omp atomic
        meshes.Quad_ICP_C += std::pow(norm, 2)/(std::pow(norm, 2) + delta);
    }
}

void Compute_Quad_derivatives_LocalICP(Meshes &meshes) {
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    std::vector<std::vector<int>> &A = meshes.A;
    const double delta = meshes.delta;
    meshes.Quad_ICP_grad = Eigen::MatrixXd::Zero(V_undeformed.rows() + V_deformed.rows(), 3);
    #pragma omp parallel for
    for (int i = 0; i < V_deformed.rows(); i++) {
        if (A[i].size() <= 2) {
            continue;
        }
        Eigen::Vector3d Vi = V_deformed.row(i);
        Eigen::Vector3d Vi_transformed = meshes.V_deformed_target.row(i);
        double norm = (Vi - Vi_transformed).norm();
        meshes.Quad_ICP_grad.row(i + V_undeformed.rows()) = 2 * norm * delta/std::pow(std::pow(norm, 2) + delta, 2) / norm * (Vi - Vi_transformed);
    }
}