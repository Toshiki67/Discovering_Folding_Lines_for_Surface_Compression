#include "Constraint.h"
#include "Closeness.h"
#include "Isometry.h"
#include "Boundary.h"
#include "SymmetricDirichlet.h"
#include "Angle_sub.h"
#include "Uniform.h"
#include <iostream>

int isometry = 1;
int closeness = 1;
int boundary = 1;
int symmetric = 1;


void Compute_Quad_Constraints(Meshes &meshes) {
    meshes.Quad_C = 0.0;
    Compute_Quad_Isometry(meshes);
    // std::cout << "Quad_Isometry_C: " << meshes.weight_isometry * meshes.Quad_Isometry_C << std::endl;
    meshes.Quad_C += meshes.weight_isometry * meshes.Quad_Isometry_C;
    Compute_Quad_Closeness(meshes);
    // std::cout << "Quad_Closeness_C: " << meshes.weight_closeness * meshes.Quad_Closeness_C << std::endl;
    meshes.Quad_C += meshes.weight_closeness * meshes.Quad_Closeness_C;
    Compute_Quad_Boundary_Constraints(meshes);
    // std::cout << "Quad_Boundary_C: " << meshes.weight_boundary * meshes.Quad_Boundary_C << std::endl;
    meshes.Quad_C += meshes.weight_boundary * meshes.Quad_Boundary_C;
    Compute_Quad_SymmetricDirichlet(meshes);
    // std::cout << "Quad_Symmetric_C: " << meshes.weight_symmetric * meshes.Quad_Symmetric_C << std::endl;
    meshes.Quad_C += meshes.weight_symmetric * meshes.Quad_Symmetric_C;
    Compute_Quad_Angle_sub(meshes);
    // std::cout << "Quad_Angle_C: " << meshes.weight_angle * meshes.Quad_Angle_C << std::endl;
    meshes.Quad_C += meshes.weight_angle * meshes.Quad_Angle_C;
    Compute_Quad_Uniform(meshes);
    // std::cout << "Quad_Uniform_C: " << meshes.weight_uniform * meshes.Quad_Uni_C << std::endl;
    meshes.Quad_C += meshes.weight_uniform * meshes.Quad_Uni_C;

    // std::cout << "Quad_C: " << meshes.Quad_C << std::endl;
}

void Compute_Quad_derivatives(Meshes &meshes) {
    Compute_Quad_derivatives_Isometry(meshes);
    Compute_Quad_derivatives_Closeness(meshes);
    Compute_Quad_derivatives_Boundary_Constraints(meshes);
    Compute_Quad_derivatives_SymmetricDirichlet(meshes);
    Compute_Quad_derivatives_Angle_sub(meshes);
    Compute_Quad_derivatives_Uniform(meshes);
    meshes.Quad_grad = meshes.weight_isometry * meshes.Quad_Isometry_grad +
                          meshes.weight_closeness * meshes.Quad_Closeness_grad +
                            meshes.weight_boundary * meshes.Quad_Boundary_grad +
                              meshes.weight_symmetric * meshes.Quad_Symmetric_grad +
                                meshes.weight_angle * meshes.Quad_Angle_grad +
                                  meshes.weight_uniform * meshes.Quad_Uniform_grad;
    meshes.Quad_N_grad = meshes.Quad_Angle_sub_N_grad;

    // std::cout << "Grad_Iso: " << meshes.weight_isometry * meshes.Quad_Isometry_grad.norm() << std::endl;
    // std::cout << "Grad_Closeness: " << meshes.weight_closeness * meshes.Quad_Closeness_grad.norm() << std::endl;
    // std::cout << "Grad_Boundary: " << meshes.weight_boundary * meshes.Quad_Boundary_grad.norm() << std::endl;
    // std::cout << "Grad_Symmetric: " << meshes.weight_symmetric * meshes.Quad_Symmetric_grad.norm() << std::endl;
    // std::cout << "Grad_Angle_sub: " << meshes.weight_angle * meshes.Quad_Angle_grad.norm() << std::endl;
    // std::cout << "Grad_Uniform: " << meshes.weight_uniform * meshes.Quad_Uniform_grad.norm() << std::endl;

}



void Compute_Newton_Constraints(Meshes &meshes) {
    Compute_Newton_Isometry(meshes);
    Compute_Newton_Closeness(meshes);
    Compute_Newton_Boundary_Constraints(meshes);
    Compute_Newton_Angle(meshes);
    Compute_Newton_SymmetricDirichlet(meshes);
    Eigen::VectorXd C = Eigen::VectorXd::Zero(meshes.V_undeformed.rows() * 3 + meshes.V_deformed.rows() * 3);
    C = meshes.C_Isometry + meshes.C_Closeness + meshes.C_Boundary + meshes.C_Angle + meshes.C_Symmetric ;
    Eigen::VectorXd N_all = Eigen::VectorXd::Zero(meshes.N_undeformed_opt.rows() * 3 + meshes.N_deformed_opt.rows() * 3);
    N_all << meshes.C_Angle_N;
    meshes.C_all = Eigen::VectorXd::Zero(C.rows() + N_all.rows());
    meshes.C_all << C, N_all;
    meshes.energy_all = meshes.energy_Isometry + meshes.energy_Closeness + meshes.energy_Boundary + meshes.energy_Angle + meshes.energy_Symmetric;
}

void Compute_Newton_derivatives(Meshes &meshes) {
    meshes.current_triplet_num = 0;
    Compute_Newton_derivatives_Isometry(meshes);
    std::cout << "Compute_Newton_derivatives_Isometry" << std::endl;
    Compute_Newton_derivatives_Closeness(meshes);
    std::cout << "Compute_Newton_derivatives_Closeness" << std::endl;
    Compute_Newton_derivatives_Boundary_Constraints(meshes);
    std::cout << "Compute_Newton_derivatives_Boundary_Constraints" << std::endl;
    Compute_Newton_derivatives_Angle(meshes);
    std::cout << "Compute_Newton_derivatives_Angle" << std::endl;
    Compute_Newton_derivatives_SymmetricDirichlet(meshes);
    std::cout << "Compute_Newton_derivatives_SymmetricDirichlet" << std::endl;
    Eigen::SparseMatrix<double> &dC_dV = meshes.dC_dV_all;
    int matrix_cols = meshes.V_deformed.rows() * 3 + meshes.V_undeformed.rows() * 3 + meshes.N_undeformed_opt.rows() * 3 + meshes.N_deformed_opt.rows() * 3;
    dC_dV = Eigen::SparseMatrix<double>(matrix_cols, matrix_cols);

    // first concatenate all triplet lists tripletList_Isometry, tripletList_Closeness, tripletList_Boundary
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(meshes.tripletList_Isometry.size() + meshes.tripletList_Closeness.size() + meshes.tripletList_Boundary.size() + meshes.tripletList_Angle.size());
    tripletList.insert(tripletList.end(), meshes.tripletList_Isometry.begin(), meshes.tripletList_Isometry.end());
    tripletList.insert(tripletList.end(), meshes.tripletList_Closeness.begin(), meshes.tripletList_Closeness.end());
    tripletList.insert(tripletList.end(), meshes.tripletList_Boundary.begin(), meshes.tripletList_Boundary.end());
    tripletList.insert(tripletList.end(), meshes.tripletList_Angle.begin(), meshes.tripletList_Angle.end());
    tripletList.insert(tripletList.end(), meshes.tripletList_Symmetric.begin(), meshes.tripletList_Symmetric.end());
    dC_dV.setFromTriplets(tripletList.begin(), tripletList.end());
    
    std::cout << "dC_dV: " << dC_dV.rows() << " " << dC_dV.cols() << std::endl;
}