//
// Created by 青木俊樹 on 4/10/24.
//

#include "Closeness.h"
#include <igl/point_mesh_squared_distance.h>
#include <igl/per_face_normals.h>
#include <igl/parallel_for.h>
#include <cmath>
#include <iostream>
#include <omp.h>


void Compute_Closeness(Meshes &meshes) {
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_refer = meshes.V_refer;
    Eigen::MatrixXi &F_refer = meshes.F_refer;
    Eigen::VectorXd &sqrD = meshes.sqrD;
    Eigen::VectorXi &I = meshes.I;
    Eigen::MatrixXd &Cp = meshes.Cp;
    Eigen::MatrixXd &N_refer = meshes.N_refer;
    igl::point_mesh_squared_distance(V_undeformed, V_refer, F_refer, sqrD, I, Cp);

    igl::parallel_for(V_undeformed.rows(), [&](int i) {
        meshes.C(i + meshes.current_num) = (V_undeformed.row(i) - V_refer.row(F_refer(I(i), 0))).dot(N_refer.row(I(i)));
    },10000);
    meshes.current_num += V_undeformed.rows();
}

void Compute_derivatives_Closeness(Meshes &meshes) {
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::VectorXi &I = meshes.I;
    Eigen::MatrixXd &N_refer = meshes.N_refer;
    std::vector<Eigen::Triplet<double>> &tripletList = meshes.tripletList;
    igl::parallel_for(V_undeformed.rows(),  [&](int i) {
        Eigen::Vector3d grad = N_refer.row(I(i));
        tripletList[3 * i + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, i * 3, grad(0));
        tripletList[3 * i + 1 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, i * 3 + 1, grad(1));
        tripletList[3 * i + 2 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, i * 3 + 2, grad(2));
    },10000);
    meshes.current_triplet_num += V_undeformed.rows() * 3;
    meshes.current_num += V_undeformed.rows();
}

void Compute_Quad_Closeness(Meshes &meshes) {
     Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_refer = meshes.V_refer;
    Eigen::MatrixXi &F_refer = meshes.F_refer;
    Eigen::VectorXd &sqrD = meshes.sqrD;
    Eigen::VectorXi &I = meshes.I;
    Eigen::MatrixXd &Cp = meshes.Cp;
    Eigen::MatrixXd &N_refer = meshes.N_refer;
    igl::point_mesh_squared_distance(V_undeformed, V_refer, F_refer, sqrD, I, Cp);
    meshes.Quad_Closeness_Vector = Eigen::VectorXd::Zero(V_undeformed.rows());
    meshes.Quad_Closeness_C = 0.0;
    // igl::parallel_for(V_undeformed.rows(), [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < V_undeformed.rows(); i++) {
        double c = (V_undeformed.row(i) - V_refer.row(F_refer(I(i), 0))).dot(N_refer.row(I(i)));
        #pragma omp atomic
        meshes.Quad_Closeness_C += std::pow(c, 2);
        meshes.Quad_Closeness_Vector(i) = c;
    }
    // },10000);
}

void Compute_Quad_derivatives_Closeness(Meshes &meshes) {
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::VectorXi &I = meshes.I;
    Eigen::MatrixXd &N_refer = meshes.N_refer;
    Eigen::MatrixXd &Quad_Closeness_grad = meshes.Quad_Closeness_grad;
    Quad_Closeness_grad = Eigen::MatrixXd::Zero(V_undeformed.rows()+V_deformed.rows(), 3);
    // igl::parallel_for(V_undeformed.rows(),  [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < V_undeformed.rows(); i++) {
        double c = meshes.Quad_Closeness_Vector(i);
        Eigen::Vector3d grad = 2*c*N_refer.row(I(i));
        Quad_Closeness_grad.row(i) = grad;
    }
    // },10000);
}

void Compute_Newton_Closeness(Meshes &meshes) {
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_refer = meshes.V_refer;
    Eigen::MatrixXi &F_refer = meshes.F_refer;
    Eigen::VectorXd &sqrD = meshes.sqrD;
    Eigen::VectorXi &I = meshes.I;
    Eigen::MatrixXd &Cp = meshes.Cp;
    Eigen::MatrixXd &N_refer = meshes.N_refer;
    igl::point_mesh_squared_distance(V_undeformed, V_refer, F_refer, sqrD, I, Cp);
    meshes.C_Closeness = Eigen::VectorXd::Zero(V_undeformed.rows()*3 + V_undeformed.rows() * 3);
    meshes.energy_Closeness = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < V_undeformed.rows(); i++) {
        double c = (V_undeformed.row(i) - V_refer.row(F_refer(I(i), 0))).dot(N_refer.row(I(i)));
        #pragma omp atomic
        meshes.energy_Closeness += c * c * meshes.weight_closeness;
        for (int j = 0; j < 3; j++) {
            meshes.C_Closeness(i*3 + j) = 2 * c * N_refer(I(i), j) * meshes.weight_closeness;
        }
    }
}

void Compute_Newton_derivatives_Closeness(Meshes &meshes) {
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::VectorXi &I = meshes.I;
    Eigen::MatrixXd &N_refer = meshes.N_refer;
    Eigen::SparseMatrix<double> &dC_dV = meshes.dC_Closeness_dV;
    // dC_dV = Eigen::SparseMatrix<double>(V_undeformed.rows(), V_undeformed.rows() * 3 + V_deformed.rows() * 3);
    std::vector<Eigen::Triplet<double>> tripletList = std::vector<Eigen::Triplet<double>>(V_undeformed.rows() * 3 * 3);
    // igl::parallel_for(V_undeformed.rows(),  [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < V_undeformed.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                tripletList[9 * i + 3*j + k] = Eigen::Triplet<double>(i * 3 + j, i * 3 + k, 2 * N_refer(I(i), k) * N_refer(I(i), j) * meshes.weight_closeness);
            }
        }
    }
    // dC_dV.setFromTriplets(tripletList.begin(), tripletList.end());
    meshes.tripletList_Closeness = tripletList;
    // meshes.current_triplet_num += V_undeformed.rows();
}