#include "Angle_sub.h"
#include <cmath>
#include <iostream>
#include <igl/edge_topology.h>
#include <igl/edge_lengths.h>
#include <igl/parallel_for.h>
#include <igl/per_face_normals.h>
#include <omp.h>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

void Compute_Angle_sub(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &N, int f1, int f2,
    const int v1, const int v2, double &angle) {
    Eigen::Vector3d n1 = N.row(f1);
    Eigen::Vector3d n2 = N.row(f2);
    angle = atan2(n1.cross(n2).dot((V.row(v2) - V.row(v1)).normalized()), n1.dot(n2));
}

void Set_selected_edges(Meshes &meshes) {
    meshes.selected_edges.clear();
    for (int i = 0; i < meshes.uE.rows(); i++) {
        if (meshes.Quad_Angle_sub_Vector(i) != 0) {
            int min_v = std::min(meshes.uE(i, 0), meshes.uE(i, 1));
            int max_v = std::max(meshes.uE(i, 0), meshes.uE(i, 1));
            std::pair<int, int> edge = std::make_pair(min_v, max_v);
            meshes.selected_edges.push_back(edge);
        }
    }
}

void Compute_Quad_Angle_sub(Meshes &meshes) {
    igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed);
    igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed);
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &N_undeformed_opt = meshes.N_undeformed_opt;
    Eigen::MatrixXd &N_deformed_opt = meshes.N_deformed_opt;
    // N_undeformed_opt = meshes.N_undeformed;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &EF = meshes.EF;
    const double delta = meshes.delta;
    meshes.Quad_Angle_C = 0.0;
    Eigen::VectorXd &Quad_Angle_sub_Vector = meshes.Quad_Angle_sub_Vector;
    Eigen::VectorXd &Quad_Angle_opt = meshes.Quad_Angle_Vector;
    Quad_Angle_opt = Eigen::VectorXd::Zero(meshes.uE.rows());
    Quad_Angle_sub_Vector = Eigen::VectorXd::Zero(meshes.uE.rows());
    meshes.selected_edges.clear();
    int count = 0;
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
            Compute_Angle_sub(V_undeformed, F_undeformed, N_undeformed_opt, f1, f2, v1, v2, angle_un);
            Compute_Angle_sub(V_deformed, F_deformed, N_deformed_opt, f1, f2, v1, v2, angle_de);
            double difference = std::pow(angle_un - angle_de, 2);
            double v1_v2 = (V_undeformed.row(v2) - V_undeformed.row(v1)).squaredNorm();
            double v1_v2_de = (V_deformed.row(v2) - V_deformed.row(v1)).squaredNorm();
            double energy = std::pow(difference, 2)/(std::pow(difference, 2) + delta) * v1_v2 * v1_v2_de;;
            #pragma omp atomic
            meshes.Quad_Angle_C += energy;

            double real_angle_un, real_angle_de;
            Compute_Angle_sub(V_undeformed, F_undeformed, meshes.N_undeformed, f1, f2, v1, v2, real_angle_un);
            Compute_Angle_sub(V_deformed, F_deformed, meshes.N_deformed, f1, f2, v1, v2, real_angle_de);
            Quad_Angle_opt(i) = difference;
            if (std::pow(real_angle_un - real_angle_de, 2) > 0.1&& difference > 0) {
                if (real_angle_de > real_angle_un) {
                    Quad_Angle_sub_Vector(i) = real_angle_de - real_angle_un;
                }
                else {
                    Quad_Angle_sub_Vector(i) = real_angle_de - real_angle_un;
                }
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < F_undeformed.rows(); i++) {
        double energy = 0;
        energy += std::pow(N_undeformed_opt.row(i).dot(V_undeformed.row(F_undeformed(i, 0)) - V_undeformed.row(F_undeformed(i, 1))), 2);
        energy += std::pow(N_undeformed_opt.row(i).dot(V_undeformed.row(F_undeformed(i, 1)) - V_undeformed.row(F_undeformed(i, 2))), 2);
        energy += std::pow(N_undeformed_opt.row(i).dot(V_undeformed.row(F_undeformed(i, 2)) - V_undeformed.row(F_undeformed(i, 0))), 2);
        energy += std::pow(N_deformed_opt.row(i).squaredNorm() - 1, 2);
        #pragma omp atomic
        meshes.Quad_Angle_C += energy;
    }

    #pragma omp parallel for
    for (int i = 0; i < F_deformed.rows(); i++) {
        double energy = 0;
        energy += std::pow(N_deformed_opt.row(i).dot(V_deformed.row(F_deformed(i, 0)) - V_deformed.row(F_deformed(i, 1))), 2);
        energy += std::pow(N_deformed_opt.row(i).dot(V_deformed.row(F_deformed(i, 1)) - V_deformed.row(F_deformed(i, 2))), 2);
        energy += std::pow(N_deformed_opt.row(i).dot(V_deformed.row(F_deformed(i, 2)) - V_deformed.row(F_deformed(i, 0))), 2);
        energy += std::pow(N_deformed_opt.row(i).squaredNorm() - 1, 2);
        #pragma omp atomic
        meshes.Quad_Angle_C += energy;
    }

}

void Compute_Quad_derivatives_Angle_sub(Meshes &meshes) {
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &N_undeformed_opt = meshes.N_undeformed_opt;
    Eigen::MatrixXd &N_deformed_opt = meshes.N_deformed_opt;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &EF = meshes.EF;
    const double delta = meshes.delta;
    Eigen::MatrixXd &Quad_Angle_grad = meshes.Quad_Angle_grad;
    Quad_Angle_grad = Eigen::MatrixXd::Zero(V_undeformed.rows()+V_deformed.rows(), 3);
    Eigen::MatrixXd &Quad_Angle_sub_N_grad = meshes.Quad_Angle_sub_N_grad;
    Quad_Angle_sub_N_grad = Eigen::MatrixXd::Zero(N_undeformed_opt.rows() + N_deformed_opt.rows(), 3);
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
            Compute_Angle_sub(V_undeformed, F_undeformed, N_undeformed_opt, f1, f2, v1, v2, angle_un);
            Compute_Angle_sub(V_deformed, F_deformed, N_deformed_opt, f1, f2, v1, v2, angle_de);
            // double difference = std::pow(angle_un - angle_de, 2);
            double v1_v2 = (V_undeformed.row(v2) - V_undeformed.row(v1)).squaredNorm();
            double v1_v2_de = (V_deformed.row(v2) - V_deformed.row(v1)).squaredNorm();
            Eigen::RowVector3d v1_v2_vec = (V_undeformed.row(v2) - V_undeformed.row(v1));
            Eigen::RowVector3d v1_v2_vec_de = (V_deformed.row(v2) - V_deformed.row(v1));
            double difference = std::pow(angle_un - angle_de, 2);
            double t = difference * v1_v2 * v1_v2_de;
            Eigen::RowVector3d n1 = N_undeformed_opt.row(f1);
            Eigen::RowVector3d n2 = N_undeformed_opt.row(f2);

            Eigen::RowVector3d V1 = V_undeformed.row(v1);
            Eigen::RowVector3d V2 = V_undeformed.row(v2);

            Eigen::RowVector3d n1_cross_n2 = n1.cross(n2);
            double n1_dot_n2 = n1.dot(n2);
            Eigen::RowVector3d v2_minus_v1 = V2 - V1;
            double l_v2_minus_v1 = v2_minus_v1.norm();
            double x = n1_dot_n2;
            double y = n1_cross_n2.dot(v2_minus_v1.normalized());
            double partial_atan_x = -y / (x * x + y * y);
            double partial_atan_y = x / (x * x + y * y);
            Eigen::RowVector3d partial_x_n1 = n2;
            Eigen::RowVector3d partial_x_n2 = n1;
            Eigen::RowVector3d partial_y_n1 = n2.cross(v2_minus_v1.normalized());
            Eigen::RowVector3d partial_y_n2 = v2_minus_v1.normalized().cross(n1);
            Eigen::RowVector3d partial_x_v1 = Eigen::RowVector3d::Zero();
            Eigen::RowVector3d partial_x_v2 = Eigen::RowVector3d::Zero();
            Eigen::RowVector3d partial_y_v1 = (-n1_cross_n2*l_v2_minus_v1 + n1_cross_n2.dot(v2_minus_v1)*v2_minus_v1/l_v2_minus_v1)/(l_v2_minus_v1*l_v2_minus_v1);
            Eigen::RowVector3d partial_y_v2 = (n1_cross_n2*l_v2_minus_v1 - n1_cross_n2.dot(v2_minus_v1)*v2_minus_v1/l_v2_minus_v1)/(l_v2_minus_v1*l_v2_minus_v1);

            double scale_un = 2 * delta * t / std::pow(std::pow(t, 2) + delta, 2);

            for (int j = 0; j < 3; j++) {
                double grad = (scale_un * v1_v2 * v1_v2_de * 2 * (angle_un - angle_de) * (partial_atan_x * partial_x_n1 + partial_atan_y * partial_y_n1))(j);
                #pragma omp atomic
                Quad_Angle_sub_N_grad(f1, j) += grad;
                grad = (scale_un * v1_v2 * v1_v2_de * 2 * (angle_un - angle_de) * (partial_atan_x * partial_x_n2 + partial_atan_y * partial_y_n2))(j);
                #pragma omp atomic
                Quad_Angle_sub_N_grad(f2, j) += grad;
                grad = (scale_un * v1_v2 * v1_v2_de * 2 * (angle_un - angle_de) * (partial_atan_x * partial_x_v1 + partial_atan_y * partial_y_v1))(j) +
                    scale_un * difference * v1_v2_de * 2 * (-v1_v2_vec)(j);
                #pragma omp atomic
                Quad_Angle_grad(v1, j) += grad;
                grad = (scale_un * v1_v2 * v1_v2_de * 2 * (angle_un - angle_de) * (partial_atan_x * partial_x_v2 + partial_atan_y * partial_y_v2))(j) +
                    scale_un * difference * v1_v2_de * 2 * (v1_v2_vec)(j);
                #pragma omp atomic
                Quad_Angle_grad(v2, j) += grad;
            }



            n1 = N_deformed_opt.row(f1);
            n2 = N_deformed_opt.row(f2);

            V1 = V_deformed.row(v1);
            V2 = V_deformed.row(v2);

            n1_cross_n2 = n1.cross(n2);
            n1_dot_n2 = n1.dot(n2);
            v2_minus_v1 = V2 - V1;
            l_v2_minus_v1 = v2_minus_v1.norm();
            x = n1_dot_n2;
            y = n1_cross_n2.dot(v2_minus_v1.normalized());
            partial_atan_x = -y / (x * x + y * y);
            partial_atan_y = x / (x * x + y * y);
            partial_x_n1 = n2;
            partial_x_n2 = n1;
            partial_y_n1 = n2.cross(v2_minus_v1.normalized());
            partial_y_n2 = v2_minus_v1.normalized().cross(n1);
            partial_x_v1 = Eigen::RowVector3d::Zero();
            partial_x_v2 = Eigen::RowVector3d::Zero();
            partial_y_v1 = (-n1_cross_n2*l_v2_minus_v1 + n1_cross_n2.dot(v2_minus_v1)*v2_minus_v1/l_v2_minus_v1)/(l_v2_minus_v1*l_v2_minus_v1);
            partial_y_v2 = (n1_cross_n2*l_v2_minus_v1 - n1_cross_n2.dot(v2_minus_v1)*v2_minus_v1/l_v2_minus_v1)/(l_v2_minus_v1*l_v2_minus_v1);

            double scale_de = 2 * delta * t / std::pow(std::pow(t, 2) + delta, 2);
            for (int j = 0; j < 3; j++) {
                double grad = (scale_de * v1_v2 * v1_v2_de * 2 * (angle_de - angle_un) * (partial_atan_x * partial_x_n1 + partial_atan_y * partial_y_n1))(j);
                #pragma omp atomic
                Quad_Angle_sub_N_grad(f1 + N_undeformed_opt.rows(), j) += grad;
                grad = (scale_de * v1_v2 * v1_v2_de * 2 * (angle_de - angle_un) * (partial_atan_x * partial_x_n2 + partial_atan_y * partial_y_n2))(j);
                #pragma omp atomic
                Quad_Angle_sub_N_grad(f2 + N_undeformed_opt.rows(), j) += grad;
                grad = (scale_de * v1_v2 * v1_v2_de * 2 * (angle_de - angle_un) * (partial_atan_x * partial_x_v1 + partial_atan_y * partial_y_v1))(j) +
                    scale_de * difference * v1_v2 * 2 * (-v1_v2_vec_de)(j);
                #pragma omp atomic
                Quad_Angle_grad(v1 + V_undeformed.rows(), j) += grad;
                grad = (scale_de * v1_v2 * v1_v2_de * 2 * (angle_de - angle_un) * (partial_atan_x * partial_x_v2 + partial_atan_y * partial_y_v2))(j) +
                    scale_de * difference * v1_v2 * 2 * (v1_v2_vec_de)(j);
                #pragma omp atomic
                Quad_Angle_grad(v2 + V_undeformed.rows(), j) += grad;
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < F_undeformed.rows(); i++) {
        const int v1 = F_undeformed(i, 0);
        const int v2 = F_undeformed(i, 1);
        const int v3 = F_undeformed(i, 2);
        
        double N_v2_v1 = N_undeformed_opt.row(i).dot(V_undeformed.row(v2) - V_undeformed.row(v1));
        double N_v3_v2 = N_undeformed_opt.row(i).dot(V_undeformed.row(v3) - V_undeformed.row(v2));
        double N_v1_v3 = N_undeformed_opt.row(i).dot(V_undeformed.row(v1) - V_undeformed.row(v3));
        
        for (int j = 0; j < 3; j++){
            #pragma omp atomic
            Quad_Angle_grad(v1, j) += (meshes.weight_angle_sub * 2 * (-N_v2_v1) * N_undeformed_opt.row(i) +
                meshes.weight_angle_sub * 2 * (N_v1_v3) * N_undeformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_grad(v2, j) += (meshes.weight_angle_sub * 2 * (N_v2_v1) * N_undeformed_opt.row(i) +
                meshes.weight_angle_sub * 2 * (-N_v3_v2) * N_undeformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_grad(v3, j) += (meshes.weight_angle_sub * 2 * (N_v3_v2) * N_undeformed_opt.row(i) +
                meshes.weight_angle_sub * 2 * (-N_v1_v3) * N_undeformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_N_grad(i, j) += (meshes.weight_angle_sub * 2 * (N_v2_v1) * (V_undeformed.row(v2) - V_undeformed.row(v1)) +
                meshes.weight_angle_sub * 2 * (N_v3_v2) * (V_undeformed.row(v3) - V_undeformed.row(v2)) +
                meshes.weight_angle_sub * 2 * (N_v1_v3) * (V_undeformed.row(v1) - V_undeformed.row(v3)) +
                meshes.weight_angle_sub * 2 * (N_undeformed_opt.row(i).squaredNorm() - 1) * 2 * N_undeformed_opt.row(i))(j);
        }

        N_v2_v1 = N_deformed_opt.row(i).dot(V_deformed.row(v2) - V_deformed.row(v1));
        N_v3_v2 = N_deformed_opt.row(i).dot(V_deformed.row(v3) - V_deformed.row(v2));
        N_v1_v3 = N_deformed_opt.row(i).dot(V_deformed.row(v1) - V_deformed.row(v3));
        for (int j = 0; j < 3; j++){
            double grad = (2 * (-N_v2_v1) * N_deformed_opt.row(i) +
                2 * (N_v1_v3) * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_grad(v1 + V_undeformed.rows(), j) += grad;
            grad = (2 * (N_v2_v1) * N_deformed_opt.row(i) +
                2 * (-N_v3_v2) * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_grad(v2 + V_undeformed.rows(), j) += grad;
            grad = (2 * (N_v3_v2) * N_deformed_opt.row(i) +
                2 * (-N_v1_v3) * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_grad(v3 + V_undeformed.rows(), j) += grad;
            grad = (2 * (N_v2_v1) * (V_deformed.row(v2) - V_deformed.row(v1)) +
                2 * (N_v3_v2) * (V_deformed.row(v3) - V_deformed.row(v2)) +
                2 * (N_v1_v3) * (V_deformed.row(v1) - V_deformed.row(v3)) +
                2 * (N_deformed_opt.row(i).squaredNorm() - 1) * 2 * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_N_grad(i + N_undeformed_opt.rows(), j) += grad;
        }
    }
}








autodiff::var Compute_Angle(const autodiff::ArrayXvar& x){
    const autodiff::var &v_un_de_1_x = x(0);
    const autodiff::var &v_un_de_1_y = x(1);
    const autodiff::var &v_un_de_1_z = x(2);
    const autodiff::var &v_un_de_2_x = x(3);
    const autodiff::var &v_un_de_2_y = x(4);
    const autodiff::var &v_un_de_2_z = x(5);

    const autodiff::var &v_de_1_x = x(6);
    const autodiff::var &v_de_1_y = x(7);
    const autodiff::var &v_de_1_z = x(8);
    const autodiff::var &v_de_2_x = x(9);
    const autodiff::var &v_de_2_y = x(10);
    const autodiff::var &v_de_2_z = x(11);

    const autodiff::var &n_un_1_x = x(12);
    const autodiff::var &n_un_1_y = x(13);
    const autodiff::var &n_un_1_z = x(14);
    const autodiff::var &n_un_2_x = x(15);
    const autodiff::var &n_un_2_y = x(16);
    const autodiff::var &n_un_2_z = x(17);

    const autodiff::var &n_de_1_x = x(18);
    const autodiff::var &n_de_1_y = x(19);
    const autodiff::var &n_de_1_z = x(20);
    const autodiff::var &n_de_2_x = x(21);
    const autodiff::var &n_de_2_y = x(22);
    const autodiff::var &n_de_2_z = x(23);

    const autodiff::var &weight = x(24);
    const autodiff::var &delta = x(25);
    using namespace autodiff;
    using namespace std;

    const autodiff::var un_length_v2_v1 = pow(pow(v_un_de_2_x - v_un_de_1_x, 2) +
        pow(v_un_de_2_y - v_un_de_1_y, 2) +
        pow(v_un_de_2_z - v_un_de_1_z, 2), 0.5);

    const autodiff::var un_normalized_v2_v1_x = (v_un_de_2_x - v_un_de_1_x) / un_length_v2_v1;
    const autodiff::var un_normalized_v2_v1_y = (v_un_de_2_y - v_un_de_1_y) / un_length_v2_v1;
    const autodiff::var un_normalized_v2_v1_z = (v_un_de_2_z - v_un_de_1_z) / un_length_v2_v1;

    const autodiff::var un_n1_n2_cross_x = n_un_1_y * n_un_2_z - n_un_1_z * n_un_2_y;
    const autodiff::var un_n1_n2_cross_y = n_un_1_z * n_un_2_x - n_un_1_x * n_un_2_z;
    const autodiff::var un_n1_n2_cross_z = n_un_1_x * n_un_2_y - n_un_1_y * n_un_2_x;

    const autodiff::var un_n1_n2_dot = n_un_1_x * n_un_2_x + n_un_1_y * n_un_2_y + n_un_1_z * n_un_2_z;
    const autodiff::var un_n1_n2_cross_dot_v2_v1 = un_n1_n2_cross_x * un_normalized_v2_v1_x +
        un_n1_n2_cross_y * un_normalized_v2_v1_y +
        un_n1_n2_cross_z * un_normalized_v2_v1_z;

    autodiff::var un_angle = atan2(un_n1_n2_cross_dot_v2_v1, un_n1_n2_dot);


    const autodiff::var de_length_v2_v1 = pow(pow(v_de_2_x - v_de_1_x, 2) +
        pow(v_de_2_y - v_de_1_y, 2) +
        pow(v_de_2_z - v_de_1_z, 2), 0.5);
    const autodiff::var de_normalized_v2_v1_x = (v_de_2_x - v_de_1_x) / de_length_v2_v1;
    const autodiff::var de_normalized_v2_v1_y = (v_de_2_y - v_de_1_y) / de_length_v2_v1;
    const autodiff::var de_normalized_v2_v1_z = (v_de_2_z - v_de_1_z) / de_length_v2_v1;
    const autodiff::var de_n1_n2_cross_x = n_de_1_y * n_de_2_z - n_de_1_z * n_de_2_y;
    const autodiff::var de_n1_n2_cross_y = n_de_1_z * n_de_2_x - n_de_1_x * n_de_2_z;
    const autodiff::var de_n1_n2_cross_z = n_de_1_x * n_de_2_y - n_de_1_y * n_de_2_x;
    const autodiff::var de_n1_n2_dot = n_de_1_x * n_de_2_x + n_de_1_y * n_de_2_y + n_de_1_z * n_de_2_z;
    const autodiff::var de_n1_n2_cross_dot_v2_v1 = de_n1_n2_cross_x * de_normalized_v2_v1_x +
        de_n1_n2_cross_y * de_normalized_v2_v1_y +
        de_n1_n2_cross_z * de_normalized_v2_v1_z;
    autodiff::var de_angle = atan2(de_n1_n2_cross_dot_v2_v1, de_n1_n2_dot);

    autodiff::var t = pow((un_angle - de_angle), 2);
    autodiff::var energy = weight * pow(t, 2)/(pow(t,2) + delta);
    return energy;
}


autodiff::var Compute_Normal_Energy(const autodiff::ArrayXvar& x){
    const autodiff::var &v_1_x = x(0);
    const autodiff::var &v_1_y = x(1);
    const autodiff::var &v_1_z = x(2);
    const autodiff::var &v_2_x = x(3);
    const autodiff::var &v_2_y = x(4);
    const autodiff::var &v_2_z = x(5);
    const autodiff::var &v_3_x = x(6);
    const autodiff::var &v_3_y = x(7);
    const autodiff::var &v_3_z = x(8);
    const autodiff::var &n_1_x = x(9);
    const autodiff::var &n_1_y = x(10);
    const autodiff::var &n_1_z = x(11);

    const autodiff::var &weight = x(12);

    using namespace autodiff;
    using namespace std;

    const autodiff::var v1_v2_x = v_2_x - v_1_x;
    const autodiff::var v1_v2_y = v_2_y - v_1_y;
    const autodiff::var v1_v2_z = v_2_z - v_1_z;
    const autodiff::var v2_v3_x = v_2_x - v_3_x;
    const autodiff::var v2_v3_y = v_2_y - v_3_y;
    const autodiff::var v2_v3_z = v_2_z - v_3_z;
    const autodiff::var v3_v1_x = v_1_x - v_3_x;
    const autodiff::var v3_v1_y = v_1_y - v_3_y;
    const autodiff::var v3_v1_z = v_1_z - v_3_z;
    const autodiff::var n1_v1_v2_dot_squared = pow(n_1_x * v1_v2_x + n_1_y * v1_v2_y + n_1_z * v1_v2_z, 2);
    const autodiff::var n1_v2_v3_dot_squared = pow(n_1_x * v2_v3_x + n_1_y * v2_v3_y + n_1_z * v2_v3_z, 2);
    const autodiff::var n1_v3_v1_dot_squared = pow(n_1_x * v3_v1_x + n_1_y * v3_v1_y + n_1_z * v3_v1_z, 2);

    const autodiff::var n_normal = pow(n_1_x * n_1_x + n_1_y * n_1_y + n_1_z * n_1_z - 1, 2);
    return weight * (n1_v1_v2_dot_squared + n1_v2_v3_dot_squared + n1_v3_v1_dot_squared + n_normal);
}

void Compute_Newton_Angle(Meshes &meshes){
    meshes.energy_Angle = 0.0;
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &N_undeformed_opt = meshes.N_undeformed_opt;
    Eigen::MatrixXd &N_deformed_opt = meshes.N_deformed_opt;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &EF = meshes.EF;
    const double delta = meshes.delta;
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
            autodiff::ArrayXvar x(26);
            x(0) = V_undeformed(v1, 0);
            x(1) = V_undeformed(v1, 1);
            x(2) = V_undeformed(v1, 2);
            x(3) = V_undeformed(v2, 0);
            x(4) = V_undeformed(v2, 1);
            x(5) = V_undeformed(v2, 2);
            x(6) = V_deformed(v1, 0);
            x(7) = V_deformed(v1, 1);
            x(8) = V_deformed(v1, 2);
            x(9) = V_deformed(v2, 0);
            x(10) = V_deformed(v2, 1);
            x(11) = V_deformed(v2, 2);
            x(12) = N_undeformed_opt(f1, 0);
            x(13) = N_undeformed_opt(f1, 1);
            x(14) = N_undeformed_opt(f1, 2);
            x(15) = N_undeformed_opt(f2, 0);
            x(16) = N_undeformed_opt(f2, 1);
            x(17) = N_undeformed_opt(f2, 2);
            x(18) = N_deformed_opt(f1, 0);
            x(19) = N_deformed_opt(f1, 1);
            x(20) = N_deformed_opt(f1, 2);
            x(21) = N_deformed_opt(f2, 0);
            x(22) = N_deformed_opt(f2, 1);
            x(23) = N_deformed_opt(f2, 2);
            
            x(24) = meshes.weight_angle_sub * meshes.weight_angle;
            x(25) = meshes.delta;
            autodiff::var energy = Compute_Angle(x);
            double energy_double = static_cast<double>(energy);
            #pragma omp atomic
            meshes.energy_Angle += energy_double;
            using namespace autodiff;

            using namespace std;
            Eigen::VectorXd grad = gradient(energy, x);
            #pragma omp atomic
            C_Angle(v1 * 3 + 0) += grad(0);
            #pragma omp atomic
            C_Angle(v1 * 3 + 1) += grad(1);
            #pragma omp atomic
            C_Angle(v1 * 3 + 2) += grad(2);
            #pragma omp atomic
            C_Angle(v2 * 3 + 0) += grad(3);
            #pragma omp atomic
            C_Angle(v2 * 3 + 1) += grad(4);
            #pragma omp atomic
            C_Angle(v2 * 3 + 2) += grad(5);
            #pragma omp atomic
            C_Angle(v1 * 3 + 0 + V_undeformed.rows() * 3) += grad(6);
            #pragma omp atomic
            C_Angle(v1 * 3 + 1 + V_undeformed.rows() * 3) += grad(7);
            #pragma omp atomic
            C_Angle(v1 * 3 + 2 + V_undeformed.rows() * 3) += grad(8);
            #pragma omp atomic
            C_Angle(v2 * 3 + 0 + V_undeformed.rows() * 3) += grad(9);
            #pragma omp atomic
            C_Angle(v2 * 3 + 1 + V_undeformed.rows() * 3) += grad(10);
            #pragma omp atomic
            C_Angle(v2 * 3 + 2 + V_undeformed.rows() * 3) += grad(11);
            #pragma omp atomic
            C_Angle_N(f1 * 3 + 0) += grad(12);
            #pragma omp atomic
            C_Angle_N(f1 * 3 + 1) += grad(13);
            #pragma omp atomic
            C_Angle_N(f1 * 3 + 2) += grad(14);

            #pragma omp atomic
            C_Angle_N(f2 * 3 + 0) += grad(15);
            #pragma omp atomic
            C_Angle_N(f2 * 3 + 1) += grad(16);
            #pragma omp atomic
            C_Angle_N(f2 * 3 + 2) += grad(17);
            #pragma omp atomic
            C_Angle_N(f1 * 3 + 0 + N_undeformed_opt.rows() * 3) += grad(18);
            #pragma omp atomic
            C_Angle_N(f1 * 3 + 1 + N_undeformed_opt.rows() * 3) += grad(19);
            #pragma omp atomic
            C_Angle_N(f1 * 3 + 2 + N_undeformed_opt.rows() * 3) += grad(20);
            #pragma omp atomic
            C_Angle_N(f2 * 3 + 0 + N_undeformed_opt.rows() * 3) += grad(21);
            #pragma omp atomic
            C_Angle_N(f2 * 3 + 1 + N_undeformed_opt.rows() * 3) += grad(22);
            #pragma omp atomic
            C_Angle_N(f2 * 3 + 2 + N_undeformed_opt.rows() * 3) += grad(23);

        }
    }

    #pragma omp parallel for
    for (int i = 0; i < F_undeformed.rows(); i++) {
        const int v1 = F_undeformed(i, 0);
        const int v2 = F_undeformed(i, 1);
        const int v3 = F_undeformed(i, 2);
        autodiff::ArrayXvar normal_un(13);
        normal_un(0) = V_undeformed(v1, 0);
        normal_un(1) = V_undeformed(v1, 1);
        normal_un(2) = V_undeformed(v1, 2);
        normal_un(3) = V_undeformed(v2, 0);
        normal_un(4) = V_undeformed(v2, 1);
        normal_un(5) = V_undeformed(v2, 2);
        normal_un(6) = V_undeformed(v3, 0);
        normal_un(7) = V_undeformed(v3, 1);
        normal_un(8) = V_undeformed(v3, 2);
        normal_un(9) = N_undeformed_opt(i, 0);
        normal_un(10) = N_undeformed_opt(i, 1);
        normal_un(11) = N_undeformed_opt(i, 2);
        normal_un(12) = meshes.weight_angle;
        autodiff::var energy = Compute_Normal_Energy(normal_un);
        double energy_double = static_cast<double>(energy);
        #pragma omp atomic
        meshes.energy_Angle += energy_double;
        using namespace autodiff;
        using namespace std;
        Eigen::VectorXd grad = gradient(energy, normal_un);
        #pragma omp atomic
        C_Angle(v1 * 3 + 0) += grad(0);
        #pragma omp atomic
        C_Angle(v1 * 3 + 1) += grad(1);
        #pragma omp atomic
        C_Angle(v1 * 3 + 2) += grad(2);
        #pragma omp atomic
        C_Angle(v2 * 3 + 0) += grad(3);
        #pragma omp atomic
        C_Angle(v2 * 3 + 1) += grad(4);
        #pragma omp atomic
        C_Angle(v2 * 3 + 2) += grad(5);
        #pragma omp atomic
        C_Angle(v3 * 3 + 0) += grad(6);
        #pragma omp atomic
        C_Angle(v3 * 3 + 1) += grad(7);
        #pragma omp atomic
        C_Angle(v3 * 3 + 2) += grad(8);
        #pragma omp atomic
        C_Angle_N(i * 3 + 0) += grad(9);
        #pragma omp atomic
        C_Angle_N(i * 3 + 1) += grad(10);
        #pragma omp atomic
        C_Angle_N(i * 3 + 2) += grad(11);
    }
    #pragma omp parallel for
    for (int i = 0; i < F_deformed.rows(); i++) {
        const int v1 = F_deformed(i, 0);
        const int v2 = F_deformed(i, 1);
        const int v3 = F_deformed(i, 2);
        autodiff::ArrayXvar normal_de(13);
        normal_de(0) = V_deformed(v1, 0);
        normal_de(1) = V_deformed(v1, 1);
        normal_de(2) = V_deformed(v1, 2);
        normal_de(3) = V_deformed(v2, 0);
        normal_de(4) = V_deformed(v2, 1);
        normal_de(5) = V_deformed(v2, 2);
        normal_de(6) = V_deformed(v3, 0);
        normal_de(7) = V_deformed(v3, 1);
        normal_de(8) = V_deformed(v3, 2);
        normal_de(9) = N_deformed_opt(i, 0);
        normal_de(10) = N_deformed_opt(i, 1);
        normal_de(11) = N_deformed_opt(i, 2);
        normal_de(12) = meshes.weight_angle;
        autodiff::var energy = Compute_Normal_Energy(normal_de);
        double energy_double = static_cast<double>(energy);
        #pragma omp atomic
        meshes.energy_Angle += energy_double;
        using namespace autodiff;
        using namespace std;
        Eigen::VectorXd grad = gradient(energy, normal_de);
        #pragma omp atomic
        C_Angle(v1 * 3 + 0 + V_undeformed.rows() * 3) += grad(0);
        #pragma omp atomic
        C_Angle(v1 * 3 + 1 + V_undeformed.rows() * 3) += grad(1);
        #pragma omp atomic
        C_Angle(v1 * 3 + 2 + V_undeformed.rows() * 3) += grad(2);
        #pragma omp atomic
        C_Angle(v2 * 3 + 0 + V_undeformed.rows() * 3) += grad(3);
        #pragma omp atomic
        C_Angle(v2 * 3 + 1 + V_undeformed.rows() * 3) += grad(4);
        #pragma omp atomic
        C_Angle(v2 * 3 + 2 + V_undeformed.rows() * 3) += grad(5);
        #pragma omp atomic
        C_Angle(v3 * 3 + 0 + V_undeformed.rows() * 3) += grad(6);
        #pragma omp atomic
        C_Angle(v3 * 3 + 1 + V_undeformed.rows() * 3) += grad(7);
        #pragma omp atomic
        C_Angle(v3 * 3 + 2 + V_undeformed.rows() * 3) += grad(8);
        #pragma omp atomic
        C_Angle_N(i * 3 + 0 + N_undeformed_opt.rows() * 3) += grad(9);
        #pragma omp atomic
        C_Angle_N(i * 3 + 1 + N_undeformed_opt.rows() * 3) += grad(10);
        #pragma omp atomic
        C_Angle_N(i * 3 + 2 + N_undeformed_opt.rows() * 3) += grad(11);

    }
}

int choose_v_num(int num, int v1, int v2, int v3, int v4, int n1, int n2, int n3, int n4){
    if (num == 0) {
        return v1;
    } else if (num == 1) {
        return v2;
    } else if (num == 2) {
        return v3;
    } else if (num == 3) {
        return v4;
    } else if (num == 4) {
        return n1;
    } else if (num == 5) {
        return n2;
    } else if (num == 6) {
        return n3;
    } else if (num == 7) {
        return n4;
    }
}


void Compute_Newton_derivatives_Angle(Meshes &meshes){

    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &N_undeformed_opt = meshes.N_undeformed_opt;
    Eigen::MatrixXd &N_deformed_opt = meshes.N_deformed_opt;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &EF = meshes.EF;
    const double delta = meshes.delta;
    std::vector<Eigen::Triplet<double>> &tripletList = meshes.tripletList_Angle;
    tripletList = std::vector<Eigen::Triplet<double>>(24 * 24 * meshes.uE.rows() + 12 * 12 * meshes.F_undeformed.rows() + 12 * 12 * meshes.F_deformed.rows());
    const int V_un_start = 0;
    const int V_de_start = V_undeformed.rows() * 3;
    const int N_un_start = V_undeformed.rows() * 3 + V_deformed.rows() * 3;
    const int N_de_start = V_undeformed.rows() * 3 + V_deformed.rows() * 3 + N_undeformed_opt.rows() * 3;

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
            autodiff::ArrayXvar x(26);
            x(0) = V_undeformed(v1, 0);
            x(1) = V_undeformed(v1, 1);
            x(2) = V_undeformed(v1, 2);
            x(3) = V_undeformed(v2, 0);
            x(4) = V_undeformed(v2, 1);
            x(5) = V_undeformed(v2, 2);
            x(6) = V_deformed(v1, 0);
            x(7) = V_deformed(v1, 1);
            x(8) = V_deformed(v1, 2);
            x(9) = V_deformed(v2, 0);
            x(10) = V_deformed(v2, 1);
            x(11) = V_deformed(v2, 2);
            x(12) = N_undeformed_opt(f1, 0);
            x(13) = N_undeformed_opt(f1, 1);
            x(14) = N_undeformed_opt(f1, 2);
            x(15) = N_undeformed_opt(f2, 0);
            x(16) = N_undeformed_opt(f2, 1);
            x(17) = N_undeformed_opt(f2, 2);
            x(18) = N_deformed_opt(f1, 0);
            x(19) = N_deformed_opt(f1, 1);
            x(20) = N_deformed_opt(f1, 2);
            x(21) = N_deformed_opt(f2, 0);
            x(22) = N_deformed_opt(f2, 1);
            x(23) = N_deformed_opt(f2, 2);
            
            x(24) = meshes.weight_angle_sub * meshes.weight_angle;
            x(25) = meshes.delta;
            autodiff::var energy = Compute_Angle(x);
            using namespace autodiff;

            using namespace std;
            Eigen::VectorXd grad;
            Eigen::MatrixXd H = hessian(energy, x, grad);
            int index_first = i * 24 * 24;
            int v1_un_index = v1 * 3 + V_un_start;
            int v2_un_index = v2 * 3 + V_un_start;
            int v1_de_index = v1 * 3 + V_de_start;
            int v2_de_index = v2 * 3 + V_de_start;
            int f1_un_index = f1 * 3 + N_un_start;
            int f2_un_index = f2 * 3 + N_un_start;
            int f1_de_index = f1 * 3 + N_de_start;
            int f2_de_index = f2 * 3 + N_de_start;
            for (int index_i = 0; index_i < 8; index_i++) {
                int row_i = choose_v_num(index_i, v1_un_index, v2_un_index, v1_de_index, v2_de_index, f1_un_index, f2_un_index, f1_de_index, f2_de_index);
                for (int index_j = 0; index_j < 8; index_j++) {
                    int col_j = choose_v_num(index_j, v1_un_index, v2_un_index, v1_de_index, v2_de_index, f1_un_index, f2_un_index, f1_de_index, f2_de_index);
                    for (int dim_i = 0; dim_i < 3; dim_i++) {
                        int hessisian_i = index_i * 3 + dim_i;
                        for (int dim_j = 0; dim_j < 3; dim_j++) {
                            int hessisian_j = index_j * 3 + dim_j;
                                tripletList[index_first + index_i*72 + index_j*9 + dim_i*3 + dim_j] = Eigen::Triplet<double>(row_i+dim_i, col_j+dim_j, H(hessisian_i, hessisian_j));
                        }
                    }
                }
            }
        }
    }

    int tri_index_first = 24 * 24 * meshes.uE.rows();

    #pragma omp parallel for
    for (int i = 0; i < F_undeformed.rows(); i++) {
        const int v1 = F_undeformed(i, 0);
        const int v2 = F_undeformed(i, 1);
        const int v3 = F_undeformed(i, 2);
        autodiff::ArrayXvar normal_un(13);
        normal_un(0) = V_undeformed(v1, 0);
        normal_un(1) = V_undeformed(v1, 1);
        normal_un(2) = V_undeformed(v1, 2);
        normal_un(3) = V_undeformed(v2, 0);
        normal_un(4) = V_undeformed(v2, 1);
        normal_un(5) = V_undeformed(v2, 2);
        normal_un(6) = V_undeformed(v3, 0);
        normal_un(7) = V_undeformed(v3, 1);
        normal_un(8) = V_undeformed(v3, 2);
        normal_un(9) = N_undeformed_opt(i, 0);
        normal_un(10) = N_undeformed_opt(i, 1);
        normal_un(11) = N_undeformed_opt(i, 2);
        normal_un(12) = meshes.weight_angle;
        autodiff::var energy = Compute_Normal_Energy(normal_un);
        using namespace autodiff;
        using namespace std;
        Eigen::VectorXd grad;
        Eigen::MatrixXd H = hessian(energy, normal_un, grad);
        // std::cout << "grad: " << grad.transpose() << std::endl;
        int index_first = tri_index_first + i * 12 * 12;
        int v1_un_index = v1 * 3 + V_un_start;
        int v2_un_index = v2 * 3 + V_un_start;
        int v3_un_index = v3 * 3 + V_un_start;
        int f1_un_index = i * 3 + N_un_start;

        for (int index_i = 0; index_i < 4; index_i++) {
            int row_i = choose_v_num(index_i, v1_un_index, v2_un_index, v3_un_index, f1_un_index, -1, -1, -1, -1);
            for (int index_j = 0; index_j < 4; index_j++) {
                int col_j = choose_v_num(index_j, v1_un_index, v2_un_index, v3_un_index, f1_un_index, -1, -1, -1, -1);
                for (int dim_i = 0; dim_i < 3; dim_i++) {
                    int hessisian_i = index_i * 3 + dim_i;
                    for (int dim_j = 0; dim_j < 3; dim_j++) {
                        int hessisian_j = index_j * 3 + dim_j;
                            tripletList[index_first + index_i*36 + index_j*9 + dim_i*3 + dim_j] = Eigen::Triplet<double>(row_i+dim_i, col_j+dim_j, H(hessisian_i, hessisian_j));
                    }
                }
            }
        }
    }
    tri_index_first += 12 * 12 * meshes.F_undeformed.rows();
    #pragma omp parallel for
    for (int i = 0; i < F_deformed.rows(); i++) {
        const int v1 = F_deformed(i, 0);
        const int v2 = F_deformed(i, 1);
        const int v3 = F_deformed(i, 2);
        autodiff::ArrayXvar normal_de(13);
        normal_de(0) = V_deformed(v1, 0);
        normal_de(1) = V_deformed(v1, 1);
        normal_de(2) = V_deformed(v1, 2);
        normal_de(3) = V_deformed(v2, 0);
        normal_de(4) = V_deformed(v2, 1);
        normal_de(5) = V_deformed(v2, 2);
        normal_de(6) = V_deformed(v3, 0);
        normal_de(7) = V_deformed(v3, 1);
        normal_de(8) = V_deformed(v3, 2);
        normal_de(9) = N_deformed_opt(i, 0);
        normal_de(10) = N_deformed_opt(i, 1);
        normal_de(11) = N_deformed_opt(i, 2);
        normal_de(12) = meshes.weight_angle;
        autodiff::var energy = Compute_Normal_Energy(normal_de);
        using namespace autodiff;
        using namespace std;
        Eigen::VectorXd grad;
        Eigen::MatrixXd H = hessian(energy, normal_de, grad);

        int index_first = tri_index_first + i * 12 * 12;
        int v1_de_index = v1 * 3 + V_de_start;
        int v2_de_index = v2 * 3 + V_de_start;
        int v3_de_index = v3 * 3 + V_de_start;
        int f1_de_index = i * 3 + N_de_start;
        for (int index_i = 0; index_i < 4; index_i++) {
            int row_i = choose_v_num(index_i, v1_de_index, v2_de_index, v3_de_index, f1_de_index, -1, -1, -1, -1);
            for (int index_j = 0; index_j < 4; index_j++) {
                int col_j = choose_v_num(index_j, v1_de_index, v2_de_index, v3_de_index, f1_de_index, -1, -1, -1, -1);
                for (int dim_i = 0; dim_i < 3; dim_i++) {
                    int hessisian_i = index_i * 3 + dim_i;
                    for (int dim_j = 0; dim_j < 3; dim_j++) {
                        int hessisian_j = index_j * 3 + dim_j;
                            tripletList[index_first + index_i*36 + index_j*9 + dim_i*3 + dim_j] = Eigen::Triplet<double>(row_i+dim_i, col_j+dim_j, H(hessisian_i, hessisian_j));
                    }
                }
            }
        }
    }
}