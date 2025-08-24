//
// Created by 青木俊樹 on 4/10/24.
//

#include "Isometry.h"
#include <cmath>
#include <iostream>
#include <igl/edge_topology.h>
#include <igl/edge_lengths.h>
#include <igl/parallel_for.h>
#include <omp.h>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

void Compute_Isometry(Meshes &meshes) {
    igl::parallel_for(meshes.uE.rows(), [&](int i) {
        Eigen::Vector3d Vec_undeformed = meshes.V_undeformed.row(meshes.uE(i, 0)) - meshes.V_undeformed.row(meshes.uE(i, 1));
        Eigen::Vector3d Vec_deformed = meshes.V_deformed.row(meshes.uE(i, 0)) - meshes.V_deformed.row(meshes.uE(i, 1));
        meshes.C(i + meshes.current_num) = Vec_undeformed.norm() - Vec_deformed.norm();
    },10000);
    meshes.current_num += meshes.uE.rows();
}

void Compute_derivatives_Isometry(Meshes &meshes) {
    int vertex_num = meshes.V_undeformed.rows() * 3;
    std::vector<Eigen::Triplet<double>> &tripletList = meshes.tripletList;
    igl::parallel_for(meshes.uE.rows(), [&](int i) {
        Eigen::Vector3d Vec_undeformed = meshes.V_undeformed.row(meshes.uE(i, 0)) - meshes.V_undeformed.row(meshes.uE(i, 1));
        double un_norm = Vec_undeformed.norm();
        Eigen::Vector3d Vec_deformed = meshes.V_deformed.row(meshes.uE(i, 0)) - meshes.V_deformed.row(meshes.uE(i, 1));
        double de_norm = Vec_deformed.norm();
        tripletList[12 * i + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, meshes.uE(i, 0) * 3, Vec_undeformed(0)/un_norm);
        tripletList[12 * i + 1 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, meshes.uE(i, 0) * 3 + 1, Vec_undeformed(1)/un_norm);
        tripletList[12 * i + 2 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, meshes.uE(i, 0) * 3 + 2, Vec_undeformed(2)/un_norm);
        tripletList[12 * i + 3 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, meshes.uE(i, 1) * 3, -Vec_undeformed(0)/un_norm);
        tripletList[12 * i + 4 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, meshes.uE(i, 1) * 3 + 1, -Vec_undeformed(1)/un_norm);
        tripletList[12 * i + 5 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, meshes.uE(i, 1) * 3 + 2, -Vec_undeformed(2)/un_norm);
        tripletList[12 * i + 6 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, vertex_num + meshes.uE(i, 0) * 3, -Vec_deformed(0)/de_norm);
        tripletList[12 * i + 7 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, vertex_num + meshes.uE(i, 0) * 3 + 1, -Vec_deformed(1)/de_norm);
        tripletList[12 * i + 8 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, vertex_num + meshes.uE(i, 0) * 3 + 2, -Vec_deformed(2)/de_norm);
        tripletList[12 * i + 9 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, vertex_num + meshes.uE(i, 1) * 3, Vec_deformed(0)/de_norm);
        tripletList[12 * i + 10 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, vertex_num + meshes.uE(i, 1) * 3 + 1, Vec_deformed(1)/de_norm);
        tripletList[12 * i + 11 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, vertex_num + meshes.uE(i, 1) * 3 + 2, Vec_deformed(2)/de_norm);
    },10000);
    meshes.current_triplet_num += 12 * meshes.uE.rows();
    meshes.current_num += meshes.uE.rows();
}

void Compute_Quad_Isometry(Meshes &meshes) {
    meshes.Quad_Isometry_Vector = Eigen::VectorXd::Zero(meshes.uE.rows());
    meshes.Quad_Isometry_C = 0.0;
    Eigen::VectorXd Quad_Vector = Eigen::VectorXd::Zero(meshes.uE.rows());
    // igl::parallel_for(meshes.uE.rows(), [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < meshes.uE.rows(); i++) {
        Eigen::Vector3d Vec_undeformed = meshes.V_undeformed.row(meshes.uE(i, 0)) - meshes.V_undeformed.row(meshes.uE(i, 1));
        Eigen::Vector3d Vec_deformed = meshes.V_deformed.row(meshes.uE(i, 0)) - meshes.V_deformed.row(meshes.uE(i, 1));
        double c = Vec_undeformed.norm() - Vec_deformed.norm();
        // meshes.Quad_Isometry_C += std::pow(c, 2);
        Quad_Vector(i) = std::pow(c, 2);
        #pragma omp atomic
        meshes.Quad_Isometry_C += std::pow(c, 2);
    }
    // },10000);
}

void Compute_Quad_derivatives_Isometry(Meshes &meshes) {
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXd &Quad_Isometry_grad = meshes.Quad_Isometry_grad;
    Quad_Isometry_grad = Eigen::MatrixXd::Zero(V_undeformed.rows()+V_deformed.rows(), 3);

    #pragma omp parallel for
    for (int i = 0; i < meshes.uE.rows(); i++) {
        Eigen::Vector3d Vec_undeformed = meshes.V_undeformed.row(meshes.uE(i, 0)) - meshes.V_undeformed.row(meshes.uE(i, 1));
        double un_norm = Vec_undeformed.norm();
        Eigen::Vector3d Vec_deformed = meshes.V_deformed.row(meshes.uE(i, 0)) - meshes.V_deformed.row(meshes.uE(i, 1));
        double de_norm = Vec_deformed.norm();
        double c = (un_norm - de_norm);
        for (int j = 0; j < 3; j++) {
            #pragma omp atomic
            Quad_Isometry_grad(meshes.uE(i, 0), j) += 2 * c / un_norm * Vec_undeformed(j);
            #pragma omp atomic
            Quad_Isometry_grad(meshes.uE(i, 1), j) -= 2 * c / un_norm * Vec_undeformed(j);
            #pragma omp atomic
            Quad_Isometry_grad(V_undeformed.rows() + meshes.uE(i, 0), j) -= 2 * c / de_norm * Vec_deformed(j);
            #pragma omp atomic
            Quad_Isometry_grad(V_undeformed.rows() + meshes.uE(i, 1), j) += 2 * c / de_norm * Vec_deformed(j);
        }
    }
    std::cout << "Quad_Isometry_grad: " << Quad_Isometry_grad.norm() << std::endl;
}

void Compute_Quad_Isometry_sub(Meshes &meshes) {
    meshes.Quad_Isometry_Vector = Eigen::VectorXd::Zero(meshes.uE.rows());
    meshes.Quad_Isometry_C = 0.0;
    Eigen::VectorXd Quad_Vector = Eigen::VectorXd::Zero(meshes.uE.rows());
    // igl::parallel_for(meshes.uE.rows(), [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < meshes.uE.rows(); i++) {
        Eigen::Vector3d Vec_undeformed = meshes.V_undeformed.row(meshes.uE(i, 0)) - meshes.V_undeformed.row(meshes.uE(i, 1));
        Eigen::Vector3d Vec_deformed = meshes.V_deformed.row(meshes.uE(i, 0)) - meshes.V_deformed.row(meshes.uE(i, 1));
        double c = Vec_undeformed.norm() - Vec_deformed.norm();
        // meshes.Quad_Isometry_C += std::pow(c, 2);
        Quad_Vector(i) = std::pow(c, 2);
        #pragma omp atomic
        meshes.Quad_Isometry_C += std::pow(c, 2);
    }
    // },10000);
}

void Compute_Quad_derivatives_Isometry_sub(Meshes &meshes) {
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXd &Quad_Isometry_grad = meshes.Quad_Isometry_grad;
    Quad_Isometry_grad = Eigen::MatrixXd::Zero(V_undeformed.rows()+V_deformed.rows(), 3);

    #pragma omp parallel for
    for (int i = 0; i < meshes.uE.rows(); i++) {
        Eigen::Vector3d Vec_undeformed = meshes.V_undeformed.row(meshes.uE(i, 0)) - meshes.V_undeformed.row(meshes.uE(i, 1));
        double un_norm = Vec_undeformed.norm();
        Eigen::Vector3d Vec_deformed = meshes.V_deformed.row(meshes.uE(i, 0)) - meshes.V_deformed.row(meshes.uE(i, 1));
        double de_norm = Vec_deformed.norm();
        double c = (un_norm - de_norm);
        for (int j = 0; j < 3; j++) {
            // #pragma omp atomic
            // Quad_Isometry_grad(meshes.uE(i, 0), j) += 2 * c / un_norm * Vec_undeformed(j);
            // #pragma omp atomic
            // Quad_Isometry_grad(meshes.uE(i, 1), j) -= 2 * c / un_norm * Vec_undeformed(j);
            #pragma omp atomic
            Quad_Isometry_grad(V_undeformed.rows() + meshes.uE(i, 0), j) -= 2 * c / de_norm * Vec_deformed(j);
            #pragma omp atomic
            Quad_Isometry_grad(V_undeformed.rows() + meshes.uE(i, 1), j) += 2 * c / de_norm * Vec_deformed(j);
        }
    }
    std::cout << "Quad_Isometry_grad: " << Quad_Isometry_grad.norm() << std::endl;
}

autodiff::dual2nd ComputeIsometry_energy(
    const autodiff::ArrayXdual2nd& x, const autodiff::dual2nd &weight) {
    autodiff::dual2nd V1_undeformed_x = x(0);
    autodiff::dual2nd V1_undeformed_y = x(1);
    autodiff::dual2nd V1_undeformed_z = x(2);
    autodiff::dual2nd V2_undeformed_x = x(3);
    autodiff::dual2nd V2_undeformed_y = x(4);
    autodiff::dual2nd V2_undeformed_z = x(5);
    autodiff::dual2nd V1_deformed_x = x(6);
    autodiff::dual2nd V1_deformed_y = x(7);
    autodiff::dual2nd V1_deformed_z = x(8);
    autodiff::dual2nd V2_deformed_x = x(9);
    autodiff::dual2nd V2_deformed_y = x(10);
    autodiff::dual2nd V2_deformed_z = x(11);
    autodiff::dual2nd Vec_undeformed_x = V1_undeformed_x - V2_undeformed_x;
    autodiff::dual2nd Vec_undeformed_y = V1_undeformed_y - V2_undeformed_y;
    autodiff::dual2nd Vec_undeformed_z = V1_undeformed_z - V2_undeformed_z;
    autodiff::dual2nd Vec_deformed_x = V1_deformed_x - V2_deformed_x;
    autodiff::dual2nd Vec_deformed_y = V1_deformed_y - V2_deformed_y;
    autodiff::dual2nd Vec_deformed_z = V1_deformed_z - V2_deformed_z;
    using namespace autodiff;
    autodiff::dual2nd Vec_undeformed_norm = pow(Vec_undeformed_x * Vec_undeformed_x + Vec_undeformed_y * Vec_undeformed_y + Vec_undeformed_z * Vec_undeformed_z, 0.5);
    autodiff::dual2nd Vec_deformed_norm = pow(Vec_deformed_x * Vec_deformed_x + Vec_deformed_y * Vec_deformed_y + Vec_deformed_z * Vec_deformed_z, 0.5);
    autodiff::dual2nd c = Vec_undeformed_norm - Vec_deformed_norm;
    autodiff::dual2nd energy = c * c * weight;
    return energy;
}


void Compute_Newton_Isometry(Meshes &meshes) {
    meshes.Quad_Isometry_Vector = Eigen::VectorXd::Zero(meshes.uE.rows());
    meshes.Quad_Isometry_C = 0.0;
    // meshes.C_Isometry = Eigen::VectorXd::Zero(meshes.uE.rows());
    meshes.C_Isometry = Eigen::VectorXd::Zero(meshes.V_undeformed.rows()*3 + meshes.V_deformed.rows() * 3);
    // igl::parallel_for(meshes.uE.rows(), [&](int i) {
    meshes.energy_Isometry = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < meshes.uE.rows(); i++) {
        // Eigen::Vector3d Vec_undeformed = meshes.V_undeformed.row(meshes.uE(i, 0)) - meshes.V_undeformed.row(meshes.uE(i, 1));
        // Eigen::Vector3d Vec_deformed = meshes.V_deformed.row(meshes.uE(i, 0)) - meshes.V_deformed.row(meshes.uE(i, 1));
        // double c = Vec_undeformed.norm() - Vec_deformed.norm();
        // #pragma omp atomic
        // meshes.energy_Isometry += c * c * meshes.weight_isometry;
        // double grad_base = 2 * c;
        // Eigen::Vector3d grad_undeformed = grad_base * Vec_undeformed / Vec_undeformed.norm() * meshes.weight_isometry;
        // Eigen::Vector3d grad_deformed = -grad_base * Vec_deformed / Vec_deformed.norm() * meshes.weight_isometry;
        // for (int j = 0; j < 3; j++) {
        //     #pragma omp atomic
        //     meshes.C_Isometry(meshes.uE(i, 0) * 3 + j) += grad_undeformed(j);
        //     #pragma omp atomic
        //     meshes.C_Isometry(meshes.uE(i, 1) * 3 + j) -= grad_undeformed(j);
        //     #pragma omp atomic
        //     meshes.C_Isometry(meshes.uE(i, 0) * 3 + j + meshes.V_undeformed.rows() * 3) += grad_deformed(j);
        //     #pragma omp atomic
        //     meshes.C_Isometry(meshes.uE(i, 1) * 3 + j + meshes.V_undeformed.rows() * 3) -= grad_deformed(j);
        // }
        autodiff::ArrayXdual2nd x(12);
        x(0) = meshes.V_undeformed(meshes.uE(i, 0), 0);
        x(1) = meshes.V_undeformed(meshes.uE(i, 0), 1);
        x(2) = meshes.V_undeformed(meshes.uE(i, 0), 2);
        x(3) = meshes.V_undeformed(meshes.uE(i, 1), 0);
        x(4) = meshes.V_undeformed(meshes.uE(i, 1), 1);
        x(5) = meshes.V_undeformed(meshes.uE(i, 1), 2);
        x(6) = meshes.V_deformed(meshes.uE(i, 0), 0);
        x(7) = meshes.V_deformed(meshes.uE(i, 0), 1);
        x(8) = meshes.V_deformed(meshes.uE(i, 0), 2);
        x(9) = meshes.V_deformed(meshes.uE(i, 1), 0);
        x(10) = meshes.V_deformed(meshes.uE(i, 1), 1);
        x(11) = meshes.V_deformed(meshes.uE(i, 1), 2);
        autodiff::dual2nd weight = meshes.weight_isometry;
        using namespace autodiff;
        autodiff::dual2nd energy;
        Eigen::VectorXd grad = gradient(ComputeIsometry_energy, wrt(x), at(x, weight), energy);
        double energy_double = static_cast<double>(energy);
        #pragma omp atomic
        meshes.energy_Isometry += energy_double;
        for (int j = 0; j < 3; j++) {
            #pragma omp atomic
            meshes.C_Isometry(meshes.uE(i, 0) * 3 + j) += grad(j);
            #pragma omp atomic
            meshes.C_Isometry(meshes.uE(i, 1) * 3 + j) += grad(j + 3);
            #pragma omp atomic
            meshes.C_Isometry(meshes.uE(i, 0) * 3 + j + meshes.V_undeformed.rows() * 3) += grad(j + 6);
            #pragma omp atomic
            meshes.C_Isometry(meshes.uE(i, 1) * 3 + j + meshes.V_undeformed.rows() * 3) += grad(j + 9);
        }
    }
}
int choose(int num, int v1, int v2, int v3, int v4) {
    if (num == 0) {
        return v1;
    }
    else if (num == 1) {
        return v2;
    }
    else if (num == 2) {
        return v3;
    }
    else {
        return v4;
    }
}

void Compute_Newton_derivatives_Isometry(Meshes &meshes) {
    // Eigen::SparseMatrix<double> &dC_dV = meshes.dC_Isometry_dV;
    // dC_dV = Eigen::SparseMatrix<double>(meshes.uE.rows(), meshes.V_undeformed.rows() * 3 + meshes.V_deformed.rows() * 3);
    // initialize triplet list as 4*meshes.uE.rows() * 3
    std::vector<Eigen::Triplet<double>> &tripletList = meshes.tripletList_Isometry;
    tripletList = std::vector<Eigen::Triplet<double>>(12 * 12 * meshes.uE.rows());
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    #pragma omp parallel for
    for (int k = 0; k < meshes.uE.rows(); k++) {
        // Eigen::Vector3d Vec_undeformed = meshes.V_undeformed.row(meshes.uE(k, 0)) - meshes.V_undeformed.row(meshes.uE(k, 1));
        // Eigen::Vector3d Vec_deformed = meshes.V_deformed.row(meshes.uE(k, 0)) - meshes.V_deformed.row(meshes.uE(k, 1));
        // double c = Vec_undeformed.norm() - Vec_deformed.norm();
        // for (int uE_num = 0; uE_num < 2; uE_num++) {
        //     int uE_i = meshes.weight_isometry;
        //     if (uE_num == 1) {
        //         uE_i = -meshes.weight_isometry;
        //     }
        //     for (int i = 0; i < 3; i++) {
        //         for (int j = 0; j < 3; j++) {
        //             if (i == j){
        //                 double ij = 2*Vec_undeformed(j)/Vec_undeformed.norm() * Vec_undeformed(i)/Vec_undeformed.norm() +
        //                     2*c*Vec_undeformed(i)*(-Vec_undeformed(j))/std::pow(Vec_undeformed.norm(), 3) +
        //                     2*c/Vec_undeformed.norm();
        //                 ij = uE_i * ij;
        //                 tripletList[k * 144 + uE_num * 36 + i * 12 + j*4 + 0] = Eigen::Triplet<double>(meshes.uE(k,uE_num)*3 + i, meshes.uE(k,0)*3 + j, ij);
        //                 tripletList[k * 144 + uE_num * 36 + i * 12 + j*4 + 1] = Eigen::Triplet<double>(meshes.uE(k,uE_num)*3 + i, meshes.uE(k,1)*3 + j, -ij);
        //             }
        //             else{
        //                 double ij = 2*Vec_undeformed(j)/Vec_undeformed.norm() * Vec_undeformed(i)/Vec_undeformed.norm() +
        //                 2*c*Vec_undeformed(i)*(-Vec_undeformed(j))/std::pow(Vec_undeformed.norm(), 3);
        //                 ij = uE_i * ij;
        //                 tripletList[k * 144 + uE_num * 36 + i * 12 + j*4 + 0] = Eigen::Triplet<double>(meshes.uE(k,uE_num)*3 + i, meshes.uE(k,0)*3 + j, ij);
        //                 tripletList[k * 144 + uE_num * 36 + i * 12 + j*4 + 1] = Eigen::Triplet<double>(meshes.uE(k,uE_num)*3 + i, meshes.uE(k,1)*3 + j, -ij);
        //             }
        //             double ij = -2*Vec_deformed(j)/Vec_deformed.norm() * Vec_undeformed(i)/Vec_undeformed.norm();
        //             ij = uE_i * ij;
        //             tripletList[k * 144 + uE_num * 36 + i * 12 + j*4 + 2] = Eigen::Triplet<double>(meshes.uE(k,uE_num)*3 + i, V_undeformed.rows() * 3 + meshes.uE(k,0)*3 + j, ij);
        //             tripletList[k * 144 + uE_num * 36 + i * 12 + j*4 + 3] = Eigen::Triplet<double>(meshes.uE(k,uE_num)*3 + i, V_undeformed.rows() * 3 + meshes.uE(k,1)*3 + j, -ij);
        //         }
        //     }
        // }

        // for (int uE_num = 0; uE_num < 2; uE_num++) {
        //     int uE_i = -meshes.weight_isometry;
        //     if (uE_num == 1) {
        //         uE_i = meshes.weight_isometry;
        //     }
        //     for (int i = 0; i < 3; i++) {
        //         for (int j = 0; j < 3; j++) {
        //             if (i == j){
        //                 double ij = -2*Vec_deformed(j)/Vec_deformed.norm() * Vec_deformed(i)/Vec_deformed.norm() +
        //                     2*c*Vec_deformed(i)*(-Vec_deformed(j))/std::pow(Vec_deformed.norm(), 3) +
        //                     2*c/Vec_deformed.norm();
        //                 ij = uE_i * ij;
        //                 tripletList[k * 144 + (uE_num + 2) * 36 + i * 12 + j*4 + 0] = Eigen::Triplet<double>(meshes.V_undeformed.rows() * 3 + meshes.uE(k,uE_num)*3 + i, V_undeformed.rows() * 3 + meshes.uE(k,0)*3 + j, ij);
        //                 tripletList[k * 144 + (uE_num + 2) * 36 + i * 12 + j*4 + 1] = Eigen::Triplet<double>(meshes.V_undeformed.rows() * 3 + meshes.uE(k,uE_num)*3 + i, V_undeformed.rows() * 3 + meshes.uE(k,1)*3 + j, -ij);
        //             }
        //             else{
        //                 double ij = -2*Vec_deformed(j)/Vec_deformed.norm() * Vec_deformed(i)/Vec_deformed.norm() +
        //                     2*c*Vec_deformed(i)*(-Vec_deformed(j))/std::pow(Vec_deformed.norm(), 3);
        //                 ij = uE_i * ij;
        //                 tripletList[k * 144 + (uE_num + 2) * 36 + i * 12 + j*4 + 0] = Eigen::Triplet<double>(meshes.V_undeformed.rows() * 3 + meshes.uE(k,uE_num)*3 + i, V_undeformed.rows() * 3 + meshes.uE(k,0)*3 + j, ij);
        //                 tripletList[k * 144 + (uE_num + 2) * 36 + i * 12 + j*4 + 1] = Eigen::Triplet<double>(meshes.V_undeformed.rows() * 3 + meshes.uE(k,uE_num)*3 + i, V_undeformed.rows() * 3 + meshes.uE(k,1)*3 + j, -ij);
        //             }
        //             double ij = 2*Vec_undeformed(j)/Vec_undeformed.norm() * Vec_undeformed(i)/Vec_undeformed.norm();
        //             ij = uE_i * ij;
        //             tripletList[k * 144 + (uE_num + 2) * 36 + i * 12 + j*4 + 2] = Eigen::Triplet<double>(meshes.V_undeformed.rows() * 3 + meshes.uE(k,uE_num)*3 + i, meshes.uE(k,0)*3 + j, ij);
        //             tripletList[k * 144 + (uE_num + 2) * 36 + i * 12 + j*4 + 3] = Eigen::Triplet<double>(meshes.V_undeformed.rows() * 3 + meshes.uE(k,uE_num)*3 + i, meshes.uE(k,1)*3 + j, -ij);
        //         }
        //     }
        // }
        autodiff::ArrayXdual2nd x(12);

        x(0) = meshes.V_undeformed(meshes.uE(k, 0), 0);
        x(1) = meshes.V_undeformed(meshes.uE(k, 0), 1);
        x(2) = meshes.V_undeformed(meshes.uE(k, 0), 2);
        x(3) = meshes.V_undeformed(meshes.uE(k, 1), 0);
        x(4) = meshes.V_undeformed(meshes.uE(k, 1), 1);
        x(5) = meshes.V_undeformed(meshes.uE(k, 1), 2);
        x(6) = meshes.V_deformed(meshes.uE(k, 0), 0);
        x(7) = meshes.V_deformed(meshes.uE(k, 0), 1);
        x(8) = meshes.V_deformed(meshes.uE(k, 0), 2);
        x(9) = meshes.V_deformed(meshes.uE(k, 1), 0);
        x(10) = meshes.V_deformed(meshes.uE(k, 1), 1);
        x(11) = meshes.V_deformed(meshes.uE(k, 1), 2);
        autodiff::dual2nd weight = meshes.weight_isometry;
        using namespace autodiff;
        autodiff::dual2nd energy;
        autodiff::VectorXdual grad;
        Eigen::MatrixXd H = hessian(ComputeIsometry_energy, wrt(x), at(x, weight), energy, grad);
        int v0_index_undeformed = meshes.uE(k, 0) * 3;
        int v1_index_undeformed = meshes.uE(k, 1) * 3;
        int v0_index_deformed = meshes.uE(k, 0) * 3 + meshes.V_undeformed.rows() * 3;
        int v1_index_deformed = meshes.uE(k, 1) * 3 + meshes.V_undeformed.rows() * 3;
        for (int idx_i = 0; idx_i < 4; idx_i++) {
            int v_row = choose(idx_i, v0_index_undeformed, v1_index_undeformed, v0_index_deformed, v1_index_deformed);
            for (int idx_j = 0; idx_j < 4; idx_j++){
                for (int dim_i = 0; dim_i < 3; dim_i++) {
                    int trip_row = v_row + dim_i;
                    for (int dim_j = 0; dim_j < 3; dim_j++) {
                        int trip_col = choose(idx_j, v0_index_undeformed, v1_index_undeformed, v0_index_deformed, v1_index_deformed) + dim_j;
                        double hessian_value = H(idx_i * 3 + dim_i, idx_j * 3 + dim_j);
                        tripletList[k * 144 + idx_i * 36 + idx_j * 9 + dim_i * 3 + dim_j] = Eigen::Triplet<double>(trip_row, trip_col, hessian_value);
                    }
                }
            }
        }
    }
    // dC_dV.setFromTriplets(tripletList.begin(), tripletList.end());
    // meshes.tripletList_Isometry = tripletList;
    meshes.current_triplet_num += meshes.uE.rows();
}
