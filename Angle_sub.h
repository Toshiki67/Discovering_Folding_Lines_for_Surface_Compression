#ifndef FREEFORM_ANGLESUB_H
#define FREEFORM_ANGLESUB_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

void Compute_Angle_sub(Meshes &meshes);

void Compute_Quad_Angle_sub(Meshes &meshes);

void Compute_Quad_derivatives_Angle_sub(Meshes &meshes);

void Compute_Quad_Angle_sub_sub(Meshes &meshes);

void Compute_Quad_derivatives_Angle_sub_sub(Meshes &meshes);

void Set_selected_edges(Meshes &meshes);


// void Compute_Newton_derivatives_Angle(Meshes &meshes);

// Structure to hold dihedral angle derivatives (to avoid recomputing)
// Structure to hold dihedral angle derivatives (to avoid recomputing)
struct DihedralDerivatives {
    double angle = 0.0;
    // First derivatives (gradients w.r.t. vA, vB, n1, n2)
    Eigen::RowVector3d grad_vA = Eigen::RowVector3d::Zero();
    Eigen::RowVector3d grad_vB = Eigen::RowVector3d::Zero();
    Eigen::RowVector3d grad_n1 = Eigen::RowVector3d::Zero();
    Eigen::RowVector3d grad_n2 = Eigen::RowVector3d::Zero();

    // Second derivatives (Hessian blocks - ANALYTICALLY VERY COMPLEX)
    // Example: H_vA_n1 = d^2(angle) / (dvA dn1)
    Eigen::Matrix3d H_vA_vA = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d H_vA_vB = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d H_vA_n1 = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d H_vA_n2 = Eigen::Matrix3d::Zero();

    Eigen::Matrix3d H_vB_vB = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d H_vB_n1 = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d H_vB_n2 = Eigen::Matrix3d::Zero();

    Eigen::Matrix3d H_n1_n1 = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d H_n1_n2 = Eigen::Matrix3d::Zero();

    Eigen::Matrix3d H_n2_n2 = Eigen::Matrix3d::Zero();

    // We also need the transposed blocks for assembly later if not directly computed
    // e.g. H_vB_vA, H_n1_vA etc. If calculated directly, ensure H_ij = H_ji^T
};


void Compute_Angle_And_Derivatives(
    const Eigen::RowVector3d& V1, const Eigen::RowVector3d& V2, // Edge vertices
    const Eigen::RowVector3d& V3, const Eigen::RowVector3d& V4, // Opposite vertices
    const Eigen::RowVector3d& n1, const Eigen::RowVector3d& n2, // Face normals
    DihedralDerivatives& derivs);

// struct Params_Angle {
//     autodiff::var weight = 0.0;
//     autodiff::var delta = 0.0;
// }

// autodiff::var Compute_Angle(const autodiff::ArrayXvar& x, const autodiff::var & weight, const autodiff::var& delta);
autodiff::dual2nd Compute_Angle(const autodiff::VectorXdual2nd& x, const autodiff::dual & weight, const autodiff::dual& delta);
autodiff::var Compute_Normal_Energy(const autodiff::ArrayXvar& x);

void Compute_Newton_Angle(Meshes &meshes);
int choose_v_num(int num, int v1, int v2, int v3, int v4, int n1, int n2, int n3, int n4);
void Compute_Newton_derivatives_Angle(Meshes &meshes);

#endif FREEFORM_ANGLESUB_H