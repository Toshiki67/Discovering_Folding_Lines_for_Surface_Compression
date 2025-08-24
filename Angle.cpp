#include "Angle.h"
#include <cmath>
#include <iostream>
#include <igl/edge_topology.h>
#include <igl/edge_lengths.h>
#include <igl/parallel_for.h>
#include <igl/per_face_normals.h>
#include <omp.h>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

void Compute_perpendicular(const Eigen::Vector3d &p, const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                           Eigen::Vector3d &perpendicular) {
    Eigen::Vector3d v1 = p - p1;
    Eigen::Vector3d base = p2 - p1;
    perpendicular = v1 - base * v1.dot(base) / base.squaredNorm();
}

void Compute_Angle(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &N, int f1, int f2,
    const int v1, const int v2, const int v3, const int v4, double &angle) {
    Eigen::Vector3d n1 = N.row(f1);
    Eigen::Vector3d n2 = N.row(f2);
    angle = atan2(n1.cross(n2).dot((V.row(v2) - V.row(v1)).normalized()), n1.dot(n2));
}

void Compute_Quad_Angle(Meshes &meshes) {
    igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed);
    igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed);
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &EF = meshes.EF;
    const double delta = meshes.delta;
    meshes.Quad_Angle_C = 0.0;
    Eigen::VectorXd &Quad_Angle_sub_Vector = meshes.Quad_Angle_sub_Vector;
    Quad_Angle_sub_Vector = Eigen::VectorXd::Zero(meshes.uE.rows());
    // igl::parallel_for(meshes.uE.rows(), [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < meshes.uE.rows(); i++) {
        if (!(EF(i, 0) == -1 || EF(i, 1) == -1)) {
            std::vector<int> half_edges = meshes.uE2E[i];
            const int num_faces = meshes.F_undeformed.rows();
            const int f1 = half_edges[0] % num_faces;
            const int f2 = half_edges[1] % num_faces;
            const int c1 = half_edges[0] / num_faces;
            const int c2 = half_edges[1] / num_faces;
            const int v1 = meshes.F_undeformed(f1, (c1+1)%3);
            const int v2 = meshes.F_undeformed(f1, (c1+2)%3);
            const int v4 = meshes.F_undeformed(f1, c1);
            const int v3 = meshes.F_undeformed(f2, c2);
            double angle_un, angle_de;
            Compute_Angle(V_undeformed, F_undeformed, meshes.N_undeformed, f1, f2, v1, v2, v3, v4, angle_un);
            Compute_Angle(V_deformed, F_deformed, meshes.N_deformed, f1, f2, v1, v2, v3, v4, angle_de);
            double difference = std::pow(angle_un - angle_de, 2);
            #pragma omp atomic
            meshes.Quad_Angle_C += std::pow(difference, 2)/(std::pow(difference, 2) + delta);
            if (difference > 0.01) {
                if (angle_de > angle_un) {
                    Quad_Angle_sub_Vector(i) = 1;
                }
                else {
                    Quad_Angle_sub_Vector(i) = -1;
                }
            }
        }
    }
    // },10000);
}

void Compute_Quad_derivatives_Angle(Meshes &meshes) {
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &EF = meshes.EF;
    const double delta = meshes.delta;
    meshes.Quad_Angle_C = 0.0;
    Eigen::MatrixXd &Quad_Angle_grad = meshes.Quad_Angle_grad;
    Quad_Angle_grad = Eigen::MatrixXd::Zero(V_undeformed.rows()+V_deformed.rows(), 3);
    // igl::parallel_for(meshes.uE.rows(), [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < meshes.uE.rows(); i++) {
        if (!(EF(i, 0) == -1 || EF(i, 1) == -1)) {
            std::vector<int> half_edges = meshes.uE2E[i];
            const int num_faces = meshes.F_undeformed.rows();
            const int f1 = half_edges[0] % num_faces;
            const int f2 = half_edges[1] % num_faces;
            const int c1 = half_edges[0] / num_faces;
            const int c2 = half_edges[1] / num_faces;
            const int v1 = meshes.F_undeformed(f1, (c1+1)%3);
            const int v2 = meshes.F_undeformed(f1, (c1+2)%3);
            const int v4 = meshes.F_undeformed(f1, c1);
            const int v3 = meshes.F_undeformed(f2, c2);
            int index_i = v2;
            int index_j = v1;
            int index_k = v4;
            int index_l = v3;


            double angle_un, angle_de;
            Compute_Angle(V_undeformed, F_undeformed, meshes.N_undeformed, f1, f2, v1, v2, v3, v4, angle_un);
            Compute_Angle(V_deformed, F_deformed, meshes.N_deformed, f1, f2, v1, v2, v3, v4, angle_de);
            double difference = std::pow(angle_un - angle_de, 2);
            Eigen::Vector3d partial_k = Eigen::Vector3d::Zero();
            Eigen::Vector3d partial_l = Eigen::Vector3d::Zero();
            Eigen::Vector3d partial_i = Eigen::Vector3d::Zero();
            Eigen::Vector3d partial_j = Eigen::Vector3d::Zero();

            Eigen::Vector3d perpend_k, perpend_l;
            Compute_perpendicular(V_undeformed.row(index_i), V_undeformed.row(index_j), V_undeformed.row(index_k), perpend_k);
            Compute_perpendicular(V_undeformed.row(index_i), V_undeformed.row(index_j), V_undeformed.row(index_l), perpend_l);
            partial_k = meshes.N_undeformed.row(f1)/perpend_k.norm();
            partial_l = meshes.N_undeformed.row(f2)/perpend_l.norm();

            Eigen::Vector3d j_k = (V_undeformed.row(index_k) - V_undeformed.row(index_j)).normalized();
            Eigen::Vector3d j_i = (V_undeformed.row(index_i) - V_undeformed.row(index_j)).normalized();
            Eigen::Vector3d j_l = (V_undeformed.row(index_l) - V_undeformed.row(index_j)).normalized();
            Eigen::Vector3d i_j = (V_undeformed.row(index_j) - V_undeformed.row(index_i)).normalized();
            Eigen::Vector3d i_k = (V_undeformed.row(index_k) - V_undeformed.row(index_i)).normalized();
            Eigen::Vector3d i_l = (V_undeformed.row(index_l) - V_undeformed.row(index_i)).normalized();

            double cot_jki = j_k.dot(j_i) / j_k.cross(j_i).norm();
            double cot_ijk = i_j.dot(i_k) / i_j.cross(i_k).norm();
            double cot_jil = j_i.dot(j_l) / j_i.cross(j_l).norm();
            double cot_ilj = i_l.dot(i_j) / i_l.cross(i_j).norm();

            partial_i = (-cot_jki)/(cot_jki + cot_ijk) * partial_k + (-cot_jil)/(cot_jil + cot_ilj) * partial_l;
            partial_j = (-cot_ijk)/(cot_jki + cot_ijk) * partial_k + (-cot_ilj)/(cot_jil + cot_ilj) * partial_l;

            partial_k = 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_un - angle_de) * partial_k;
            partial_l = 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_un - angle_de) * partial_l;
            partial_i = 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_un - angle_de) * partial_i;
            partial_j = 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_un - angle_de) * partial_j;
            
            // Quad_Angle_grad.row(index_i) += partial_i;
            // Quad_Angle_grad.row(index_j) += partial_j;
            // Quad_Angle_grad.row(index_k) += partial_k;
            // Quad_Angle_grad.row(index_l) += partial_l;



            partial_k = Eigen::Vector3d::Zero();
            partial_l = Eigen::Vector3d::Zero();
            partial_i = Eigen::Vector3d::Zero();
            partial_j = Eigen::Vector3d::Zero();

            Compute_perpendicular(V_deformed.row(index_i), V_deformed.row(index_j), V_deformed.row(index_k), perpend_k);
            Compute_perpendicular(V_deformed.row(index_i), V_deformed.row(index_j), V_deformed.row(index_l), perpend_l);
            partial_k = meshes.N_deformed.row(f1)/perpend_k.norm();
            partial_l = meshes.N_deformed.row(f2)/perpend_l.norm();

            j_k = (V_deformed.row(index_k) - V_deformed.row(index_j)).normalized();
            j_i = (V_deformed.row(index_i) - V_deformed.row(index_j)).normalized();
            j_l = (V_deformed.row(index_l) - V_deformed.row(index_j)).normalized();
            i_j = (V_deformed.row(index_j) - V_deformed.row(index_i)).normalized();
            i_k = (V_deformed.row(index_k) - V_deformed.row(index_i)).normalized();
            i_l = (V_deformed.row(index_l) - V_deformed.row(index_i)).normalized();

            cot_jki = j_k.dot(j_i) / j_k.cross(j_i).norm();
            cot_ijk = i_j.dot(i_k) / i_j.cross(i_k).norm();
            cot_jil = j_i.dot(j_l) / j_i.cross(j_l).norm();
            cot_ilj = i_l.dot(i_j) / i_l.cross(i_j).norm();

            partial_i = (-cot_jki)/(cot_jki + cot_ijk) * partial_k + (-cot_jil)/(cot_jil + cot_ilj) * partial_l;
            partial_j = (-cot_ijk)/(cot_jki + cot_ijk) * partial_k + (-cot_ilj)/(cot_jil + cot_ilj) * partial_l;

            partial_k = 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_de - angle_un) * partial_k;
            partial_l = 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_de - angle_un) * partial_l;
            partial_i = 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_de - angle_un) * partial_i;
            partial_j = 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_de - angle_un) * partial_j;
            
            // Quad_Angle_grad.row(index_i + V_undeformed.rows()) += partial_i;
            // Quad_Angle_grad.row(index_j + V_undeformed.rows()) += partial_j;
            // Quad_Angle_grad.row(index_k + V_undeformed.rows()) += partial_k;
            // Quad_Angle_grad.row(index_l + V_undeformed.rows()) += partial_l;
            for (int j = 0; j < 3; j++) {
                #pragma omp atomic
                Quad_Angle_grad(index_i + V_undeformed.rows(), j) += partial_i(j);
                #pragma omp atomic
                Quad_Angle_grad(index_j + V_undeformed.rows(), j) += partial_j(j);
                #pragma omp atomic
                Quad_Angle_grad(index_k + V_undeformed.rows(), j) += partial_k(j);
                #pragma omp atomic
                Quad_Angle_grad(index_l + V_undeformed.rows(), j) += partial_l(j);
                
            }
        }
    }
    // },10000);
}


autodiff::dual2nd Compute_Direct_Angle_energy(
    const autodiff::ArrayXdual2nd& x, const autodiff::dual2nd &weight, const autodiff::dual2nd &delta) {
    const autodiff::dual2nd V1_undeformed_x = x(0);
    const autodiff::dual2nd V1_undeformed_y = x(1);
    const autodiff::dual2nd V1_undeformed_z = x(2);
    const autodiff::dual2nd V2_undeformed_x = x(3);
    const autodiff::dual2nd V2_undeformed_y = x(4);
    const autodiff::dual2nd V2_undeformed_z = x(5);
    const autodiff::dual2nd V3_undeformed_x = x(6);
    const autodiff::dual2nd V3_undeformed_y = x(7);
    const autodiff::dual2nd V3_undeformed_z = x(8);
    const autodiff::dual2nd V4_undeformed_x = x(9);
    const autodiff::dual2nd V4_undeformed_y = x(10);
    const autodiff::dual2nd V4_undeformed_z = x(11);

    const autodiff::dual2nd V1_deformed_x = x(12);
    const autodiff::dual2nd V1_deformed_y = x(13);
    const autodiff::dual2nd V1_deformed_z = x(14);
    const autodiff::dual2nd V2_deformed_x = x(15);
    const autodiff::dual2nd V2_deformed_y = x(16);
    const autodiff::dual2nd V2_deformed_z = x(17);
    const autodiff::dual2nd V3_deformed_x = x(18);
    const autodiff::dual2nd V3_deformed_y = x(19);
    const autodiff::dual2nd V3_deformed_z = x(20);
    const autodiff::dual2nd V4_deformed_x = x(21);
    const autodiff::dual2nd V4_deformed_y = x(22);
    const autodiff::dual2nd V4_deformed_z = x(23);
    
    const autodiff::dual2nd V2_V1_undeformed_x = V2_undeformed_x - V1_undeformed_x;
    const autodiff::dual2nd V2_V1_undeformed_y = V2_undeformed_y - V1_undeformed_y;
    const autodiff::dual2nd V2_V1_undeformed_z = V2_undeformed_z - V1_undeformed_z;
    const autodiff::dual2nd V4_V2_undeformed_x = V4_undeformed_x - V2_undeformed_x;
    const autodiff::dual2nd V4_V2_undeformed_y = V4_undeformed_y - V2_undeformed_y;
    const autodiff::dual2nd V4_V2_undeformed_z = V4_undeformed_z - V2_undeformed_z;
    const autodiff::dual2nd n1_x = V2_V1_undeformed_y * V4_V2_undeformed_z - V2_V1_undeformed_z * V4_V2_undeformed_y;
    const autodiff::dual2nd n1_y = V2_V1_undeformed_z * V4_V2_undeformed_x - V2_V1_undeformed_x * V4_V2_undeformed_z;
    const autodiff::dual2nd n1_z = V2_V1_undeformed_x * V4_V2_undeformed_y - V2_V1_undeformed_y * V4_V2_undeformed_x;
    const autodiff::dual2nd n1_norm = pow(n1_x * n1_x + n1_y * n1_y + n1_z * n1_z, 0.5);
    const autodiff::dual2nd n1_x_normalized = n1_x / n1_norm;
    const autodiff::dual2nd n1_y_normalized = n1_y / n1_norm;
    const autodiff::dual2nd n1_z_normalized = n1_z / n1_norm;
    const autodiff::dual2nd V2_V3_undeformed_x = V2_undeformed_x - V3_undeformed_x;
    const autodiff::dual2nd V2_V3_undeformed_y = V2_undeformed_y - V3_undeformed_y;
    const autodiff::dual2nd V2_V3_undeformed_z = V2_undeformed_z - V3_undeformed_z;
    const autodiff::dual2nd V1_V2_undeformed_x = V1_undeformed_x - V2_undeformed_x;
    const autodiff::dual2nd V1_V2_undeformed_y = V1_undeformed_y - V2_undeformed_y;
    const autodiff::dual2nd V1_V2_undeformed_z = V1_undeformed_z - V2_undeformed_z;
    const autodiff::dual2nd n2_x = V2_V3_undeformed_y * V1_V2_undeformed_z - V2_V3_undeformed_z * V1_V2_undeformed_y;
    const autodiff::dual2nd n2_y = V2_V3_undeformed_z * V1_V2_undeformed_x - V2_V3_undeformed_x * V1_V2_undeformed_z;
    const autodiff::dual2nd n2_z = V2_V3_undeformed_x * V1_V2_undeformed_y - V2_V3_undeformed_y * V1_V2_undeformed_x;
    const autodiff::dual2nd n2_norm = pow(n2_x * n2_x + n2_y * n2_y + n2_z * n2_z, 0.5);
    const autodiff::dual2nd n2_x_normalized = n2_x / n2_norm;
    const autodiff::dual2nd n2_y_normalized = n2_y / n2_norm;
    const autodiff::dual2nd n2_z_normalized = n2_z / n2_norm;

    const autodiff::dual2nd un_length_v2_v1 = pow(pow(V2_undeformed_x - V1_undeformed_x, 2) +
        pow(V2_undeformed_y - V1_undeformed_y, 2) +
        pow(V2_undeformed_z - V1_undeformed_z, 2), 0.5);

    const autodiff::dual2nd un_normalized_v2_v1_x = (V2_undeformed_x - V1_undeformed_x) / un_length_v2_v1;
    const autodiff::dual2nd un_normalized_v2_v1_y = (V2_undeformed_y - V1_undeformed_y) / un_length_v2_v1;
    const autodiff::dual2nd un_normalized_v2_v1_z = (V2_undeformed_z - V1_undeformed_z) / un_length_v2_v1;

    const autodiff::dual2nd un_n1_n2_cross_x = n1_y_normalized * n2_z_normalized - n1_z_normalized * n2_y_normalized;
    const autodiff::dual2nd un_n1_n2_cross_y = n1_z_normalized * n2_x_normalized - n1_x_normalized * n2_z_normalized;
    const autodiff::dual2nd un_n1_n2_cross_z = n1_x_normalized * n2_y_normalized - n1_y_normalized * n2_x_normalized;

    const autodiff::dual2nd un_n1_n2_dot = n1_x_normalized * n2_x_normalized + n1_y_normalized * n2_y_normalized + n1_z_normalized * n2_z_normalized;
    const autodiff::dual2nd un_n1_n2_cross_dot_v2_v1 = un_n1_n2_cross_x * un_normalized_v2_v1_x +
        un_n1_n2_cross_y * un_normalized_v2_v1_y +
        un_n1_n2_cross_z * un_normalized_v2_v1_z;

    autodiff::dual2nd un_angle = atan2(un_n1_n2_cross_dot_v2_v1, un_n1_n2_dot);

    
    const autodiff::dual2nd V2_V1_deformed_x = V2_deformed_x - V1_deformed_x;
    const autodiff::dual2nd V2_V1_deformed_y = V2_deformed_y - V1_deformed_y;
    const autodiff::dual2nd V2_V1_deformed_z = V2_deformed_z - V1_deformed_z;
    const autodiff::dual2nd V4_V2_deformed_x = V4_deformed_x - V2_deformed_x;
    const autodiff::dual2nd V4_V2_deformed_y = V4_deformed_y - V2_deformed_y;
    const autodiff::dual2nd V4_V2_deformed_z = V4_deformed_z - V2_deformed_z;
    const autodiff::dual2nd n1_de_x = V2_V1_deformed_y * V4_V2_deformed_z - V2_V1_deformed_z * V4_V2_deformed_y;
    const autodiff::dual2nd n1_de_y = V2_V1_deformed_z * V4_V2_deformed_x - V2_V1_deformed_x * V4_V2_deformed_z;
    const autodiff::dual2nd n1_de_z = V2_V1_deformed_x * V4_V2_deformed_y - V2_V1_deformed_y * V4_V2_deformed_x;
    const autodiff::dual2nd n1_de_norm = pow(n1_de_x * n1_de_x + n1_de_y * n1_de_y + n1_de_z * n1_de_z, 0.5);
    const autodiff::dual2nd n1_de_x_normalized = n1_de_x / n1_de_norm;
    const autodiff::dual2nd n1_de_y_normalized = n1_de_y / n1_de_norm;
    const autodiff::dual2nd n1_de_z_normalized = n1_de_z / n1_de_norm;
    const autodiff::dual2nd V2_V3_deformed_x = V2_deformed_x - V3_deformed_x;
    const autodiff::dual2nd V2_V3_deformed_y = V2_deformed_y - V3_deformed_y;
    const autodiff::dual2nd V2_V3_deformed_z = V2_deformed_z - V3_deformed_z;
    const autodiff::dual2nd V1_V2_deformed_x = V1_deformed_x - V2_deformed_x;
    const autodiff::dual2nd V1_V2_deformed_y = V1_deformed_y - V2_deformed_y;   
    const autodiff::dual2nd V1_V2_deformed_z = V1_deformed_z - V2_deformed_z;   
    const autodiff::dual2nd n2_de_x = V2_V3_deformed_y * V1_V2_deformed_z - V2_V3_deformed_z * V1_V2_deformed_y;
    const autodiff::dual2nd n2_de_y = V2_V3_deformed_z * V1_V2_deformed_x - V2_V3_deformed_x * V1_V2_deformed_z;
    const autodiff::dual2nd n2_de_z = V2_V3_deformed_x * V1_V2_deformed_y - V2_V3_deformed_y * V1_V2_deformed_x;
    const autodiff::dual2nd n2_de_norm = pow(n2_de_x * n2_de_x + n2_de_y * n2_de_y + n2_de_z * n2_de_z, 0.5);
    const autodiff::dual2nd n2_de_x_normalized = n2_de_x / n2_de_norm;
    const autodiff::dual2nd n2_de_y_normalized = n2_de_y / n2_de_norm;
    const autodiff::dual2nd n2_de_z_normalized = n2_de_z / n2_de_norm;

    const autodiff::dual2nd de_length_v2_v1 = pow(pow(V2_deformed_x - V1_deformed_x, 2) +
        pow(V2_deformed_y - V1_deformed_y, 2) +
        pow(V2_deformed_z - V1_deformed_z, 2), 0.5);
    const autodiff::dual2nd de_normalized_v2_v1_x = (V2_deformed_x - V1_deformed_x) / de_length_v2_v1;
    const autodiff::dual2nd de_normalized_v2_v1_y = (V2_deformed_y - V1_deformed_y) / de_length_v2_v1;
    const autodiff::dual2nd de_normalized_v2_v1_z = (V2_deformed_z - V1_deformed_z) / de_length_v2_v1;
    const autodiff::dual2nd de_n1_n2_cross_x = n1_de_y_normalized * n2_de_z_normalized - n1_de_z_normalized * n2_de_y_normalized;
    const autodiff::dual2nd de_n1_n2_cross_y = n1_de_z_normalized * n2_de_x_normalized - n1_de_x_normalized * n2_de_z_normalized;
    const autodiff::dual2nd de_n1_n2_cross_z = n1_de_x_normalized * n2_de_y_normalized - n1_de_y_normalized * n2_de_x_normalized;
    const autodiff::dual2nd de_n1_n2_dot = n1_de_x_normalized * n2_de_x_normalized + n1_de_y_normalized * n2_de_y_normalized + n1_de_z_normalized * n2_de_z_normalized;
    const autodiff::dual2nd de_n1_n2_cross_dot_v2_v1 = de_n1_n2_cross_x * de_normalized_v2_v1_x +
        de_n1_n2_cross_y * de_normalized_v2_v1_y +
        de_n1_n2_cross_z * de_normalized_v2_v1_z;
    autodiff::dual2nd de_angle = atan2(de_n1_n2_cross_dot_v2_v1, de_n1_n2_dot);
    autodiff::dual2nd t = pow((un_angle - de_angle), 2);
    autodiff::dual2nd energy = weight * pow(t, 2)/(pow(t,2) + delta);
    return energy;
}

void Compute_Newton_Direct_Angle(Meshes &meshes) {
    meshes.energy_Angle = 0.0;
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &N_undeformed_opt = meshes.N_undeformed_opt;
    Eigen::MatrixXd &N_deformed_opt = meshes.N_deformed_opt;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &EF = meshes.EF;
    const double delta = meshes.delta;
    meshes.Quad_Angle_C = 0.0;
    Eigen::VectorXd &C_Angle = meshes.C_Angle;
    C_Angle = Eigen::VectorXd::Zero((V_undeformed.rows()+V_deformed.rows()) * 3);
    Eigen::VectorXd &C_Angle_N = meshes.C_Angle_N;
    C_Angle_N = Eigen::VectorXd::Zero((N_undeformed_opt.rows() + N_deformed_opt.rows()) * 3);
    #pragma omp parallel for
    for (int i = 0; i < meshes.uE.rows(); i++) {
        if (!(EF(i, 0) == -1 || EF(i, 1) == -1)) {
            std::vector<int> half_edges = meshes.uE2E[i];
            const int num_faces = meshes.F_undeformed.rows();
            const int f1 = half_edges[0] % num_faces;
            const int f2 = half_edges[1] % num_faces;
            const int c1 = half_edges[0] / num_faces;
            const int c2 = half_edges[1] / num_faces;
            const int v1 = meshes.F_undeformed(f1, (c1+1)%3);
            const int v2 = meshes.F_undeformed(f1, (c1+2)%3);
            const int v4 = meshes.F_undeformed(f1, c1);
            const int v3 = meshes.F_undeformed(f2, c2);
            autodiff::ArrayXdual2nd x(24);
            x(0) = V_undeformed(v1, 0);
            x(1) = V_undeformed(v1, 1);
            x(2) = V_undeformed(v1, 2);
            x(3) = V_undeformed(v2, 0);
            x(4) = V_undeformed(v2, 1);
            x(5) = V_undeformed(v2, 2);
            x(6) = V_undeformed(v3, 0);
            x(7) = V_undeformed(v3, 1);
            x(8) = V_undeformed(v3, 2);
            x(9) = V_undeformed(v4, 0);
            x(10) = V_undeformed(v4, 1);
            x(11) = V_undeformed(v4, 2);
            x(12) = V_deformed(v1, 0);
            x(13) = V_deformed(v1, 1);
            x(14) = V_deformed(v1, 2);
            x(15) = V_deformed(v2, 0);
            x(16) = V_deformed(v2, 1);
            x(17) = V_deformed(v2, 2);
            x(18) = V_deformed(v3, 0);
            x(19) = V_deformed(v3, 1);
            x(20) = V_deformed(v3, 2);
            x(21) = V_deformed(v4, 0);
            x(22) = V_deformed(v4, 1);
            x(23) = V_deformed(v4, 2);
            autodiff::dual2nd weight = meshes.weight_angle_sub * meshes.weight_angle;
            autodiff::dual2nd delta = meshes.delta;
            autodiff::dual2nd energy;
            using namespace autodiff;
            Eigen::VectorXd grad = gradient(Compute_Direct_Angle_energy, wrt(x), at(x, weight, delta), energy);
            double energy_value = static_cast<double>(energy);
            #pragma omp atomic
            meshes.energy_Angle += energy_value;
            for (int j = 0; j < 3; j++){
                #pragma omp atomic
                C_Angle(v1*3 + j) += grad(j);
                #pragma omp atomic
                C_Angle(v2*3 + j) += grad(j+3);
                #pragma omp atomic
                C_Angle(v3*3 + j) += grad(j+6);
                #pragma omp atomic
                C_Angle(v4*3 + j) += grad(j+9);
                #pragma omp atomic
                C_Angle(v1*3 + j + V_undeformed.rows() * 3) += grad(j+12);
                #pragma omp atomic
                C_Angle(v2*3 + j + V_undeformed.rows() * 3) += grad(j+15);
                #pragma omp atomic
                C_Angle(v3*3 + j + V_undeformed.rows() * 3) += grad(j+18);
                #pragma omp atomic
                C_Angle(v4*3 + j + V_undeformed.rows() * 3) += grad(j+21);
            }
        }
    }
}

int choose(int num, int v1_un, int v2_un, int v3_un, int v4_un, int v1_de, int v2_de, int v3_de, int v4_de) {
    if (num == 0) {
        return v1_un;
    }
    else if (num == 1) {
        return v2_un;
    }
    else if (num == 2) {
        return v3_un;
    }
    else if (num == 3) {
        return v4_un;
    }
    else if (num == 4) {
        return v1_de;
    }
    else if (num == 5) {
        return v2_de;
    }
    else if (num == 6) {
        return v3_de;
    }
    else if (num == 7) {
        return v4_de;
    }
}

void Compute_Newton_derivatives_Direct_Angle(Meshes &meshes) {
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &N_undeformed_opt = meshes.N_undeformed_opt;
    Eigen::MatrixXd &N_deformed_opt = meshes.N_deformed_opt;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &EF = meshes.EF;
    std::vector<Eigen::Triplet<double>> &tripletList = meshes.tripletList_Angle;
    tripletList = std::vector<Eigen::Triplet<double>>(24 * 24 * meshes.uE.rows());
    #pragma omp parallel for
    for (int i = 0; i < meshes.uE.rows(); i++) {
        if (!(EF(i, 0) == -1 || EF(i, 1) == -1)) {
            std::vector<int> half_edges = meshes.uE2E[i];
            const int num_faces = meshes.F_undeformed.rows();
            const int f1 = half_edges[0] % num_faces;
            const int f2 = half_edges[1] % num_faces;
            const int c1 = half_edges[0] / num_faces;
            const int c2 = half_edges[1] / num_faces;
            const int v1 = meshes.F_undeformed(f1, (c1+1)%3);
            const int v2 = meshes.F_undeformed(f1, (c1+2)%3);
            const int v4 = meshes.F_undeformed(f1, c1);
            const int v3 = meshes.F_undeformed(f2, c2);
            autodiff::ArrayXdual2nd x(24);
            x(0) = V_undeformed(v1, 0);
            x(1) = V_undeformed(v1, 1);
            x(2) = V_undeformed(v1, 2);
            x(3) = V_undeformed(v2, 0);
            x(4) = V_undeformed(v2, 1);
            x(5) = V_undeformed(v2, 2);
            x(6) = V_undeformed(v3, 0);
            x(7) = V_undeformed(v3, 1);
            x(8) = V_undeformed(v3, 2);
            x(9) = V_undeformed(v4, 0);
            x(10) = V_undeformed(v4, 1);
            x(11) = V_undeformed(v4, 2);
            x(12) = V_deformed(v1, 0);
            x(13) = V_deformed(v1, 1);
            x(14) = V_deformed(v1, 2);
            x(15) = V_deformed(v2, 0);
            x(16) = V_deformed(v2, 1);
            x(17) = V_deformed(v2, 2);
            x(18) = V_deformed(v3, 0);
            x(19) = V_deformed(v3, 1);
            x(20) = V_deformed(v3, 2);
            x(21) = V_deformed(v4, 0);
            x(22) = V_deformed(v4, 1);
            x(23) = V_deformed(v4, 2);
            autodiff::dual2nd weight = meshes.weight_angle_sub * meshes.weight_angle;
            autodiff::dual2nd delta = meshes.delta;
            autodiff::dual2nd energy;
            autodiff::VectorXdual grad;
            using namespace autodiff;
            Eigen::MatrixXd H = hessian(Compute_Direct_Angle_energy, wrt(x), at(x, weight, delta), energy, grad);
            int index_1_undeformed = v1*3;
            int index_2_undeformed = v2*3;
            int index_3_undeformed = v3*3;
            int index_4_undeformed = v4*3;
            int index_1_deformed = v1*3 + V_undeformed.rows() * 3;
            int index_2_deformed = v2*3 + V_undeformed.rows() * 3;
            int index_3_deformed = v3*3 + V_undeformed.rows() * 3;
            int index_4_deformed = v4*3 + V_undeformed.rows() * 3;
            for (int idx_i = 0; idx_i < 8; idx_i++) {
                for (int idx_j = 0; idx_j < 8; idx_j++) {
                    for (int dim_i = 0; dim_i < 3; dim_i++) {
                        for (int dim_j = 0; dim_j < 3; dim_j++) {
                            int trip_row = choose(idx_i, index_1_undeformed, index_2_undeformed, index_3_undeformed, index_4_undeformed,
                                index_1_deformed, index_2_deformed, index_3_deformed, index_4_deformed) + dim_i;
                            int trip_col = choose(idx_j, index_1_undeformed, index_2_undeformed, index_3_undeformed, index_4_undeformed,
                                index_1_deformed, index_2_deformed, index_3_deformed, index_4_deformed) + dim_j;
                            tripletList[i * 24 * 24 + idx_i * 72 + idx_j * 9 + dim_i * 3 + dim_j] = Eigen::Triplet<double>(trip_row, trip_col, H(idx_i * 3 + dim_i, idx_j * 3 + dim_j));
                        }
                    }
                }
            }
        }
    }
}