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
    meshes.Quad_Angle_sub_C = 0.0;
    Eigen::VectorXd &Quad_Angle_sub_Vector = meshes.Quad_Angle_sub_Vector;
    Eigen::VectorXd &Quad_Angle_opt = meshes.Quad_Angle_Vector;
    Quad_Angle_opt = Eigen::VectorXd::Zero(meshes.uE.rows());
    Quad_Angle_sub_Vector = Eigen::VectorXd::Zero(meshes.uE.rows());
    // igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed);
    // igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed);
    // igl::parallel_for(meshes.uE.rows(), [&](int i) {
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
            double energy = meshes.weight_angle_sub * std::pow(difference, 2)/(std::pow(difference, 2) + delta);
            #pragma omp atomic
            meshes.Quad_Angle_sub_C += energy;

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
    // },10000);
    // igl::parallel_for(F_undeformed.rows(), [&](int i) {
    // igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed_opt);
    // #pragma omp parallel for
    // for (int i = 0; i < F_undeformed.rows(); i++) {
    //     const int v1 = F_undeformed(i, 0);
    //     const int v2 = F_undeformed(i, 1);
    //     const int v3 = F_undeformed(i, 2);
    //     double energy = 0.0;
    //     energy += std::pow(N_undeformed_opt.row(i).dot(V_undeformed.row(v2) - V_undeformed.row(v1)), 2);
    //     energy += std::pow(N_undeformed_opt.row(i).dot(V_undeformed.row(v3) - V_undeformed.row(v2)), 2);
    //     energy += std::pow(N_undeformed_opt.row(i).dot(V_undeformed.row(v1) - V_undeformed.row(v3)), 2);
    //     energy += std::pow(N_undeformed_opt.row(i).squaredNorm() - 1, 2);

    //     energy += std::pow(N_deformed_opt.row(i).dot(V_deformed.row(v2) - V_deformed.row(v1)), 2);
    //     energy += std::pow(N_deformed_opt.row(i).dot(V_deformed.row(v3) - V_deformed.row(v2)), 2);
    //     energy += std::pow(N_deformed_opt.row(i).dot(V_deformed.row(v1) - V_deformed.row(v3)), 2);
    //     energy += std::pow(N_deformed_opt.row(i).squaredNorm() - 1, 2);
    //     #pragma omp atomic
    //     meshes.Quad_Angle_sub_C += energy;


        // #pragma omp atomic
        // meshes.Quad_Angle_sub_C += meshes.weight_angle * (N_undeformed_opt.row(i) - meshes.N_undeformed.row(i)).squaredNorm();
        // #pragma omp atomic
        // meshes.Quad_Angle_sub_C += meshes.weight_angle * (N_deformed_opt.row(i) - meshes.N_deformed.row(i)).squaredNorm();
    // }
    // },10000);
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
    meshes.Quad_Angle_C = 0.0;
    Eigen::MatrixXd &Quad_Angle_sub_grad = meshes.Quad_Angle_sub_grad;
    Quad_Angle_sub_grad = Eigen::MatrixXd::Zero(V_undeformed.rows()+V_deformed.rows(), 3);
    Eigen::MatrixXd &Quad_Angle_sub_N_grad = meshes.Quad_Angle_sub_N_grad;
    Quad_Angle_sub_N_grad = Eigen::MatrixXd::Zero(N_undeformed_opt.rows() + N_deformed_opt.rows(), 3);
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

            // double scale_un = meshes.weight_angle_sub * 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_un - angle_de);
            double scale_un = meshes.weight_angle_sub * 2 * delta * t / std::pow(std::pow(t, 2) + delta, 2);

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
                Quad_Angle_sub_grad(v1, j) += grad;
                grad = (scale_un * v1_v2 * v1_v2_de * 2 * (angle_un - angle_de) * (partial_atan_x * partial_x_v2 + partial_atan_y * partial_y_v2))(j) +
                    scale_un * difference * v1_v2_de * 2 * (v1_v2_vec)(j);
                #pragma omp atomic
                Quad_Angle_sub_grad(v2, j) += grad;
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

            // double scale_de = meshes.weight_angle_sub * 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_de - angle_un);

            // for (int j = 0; j < 3; j++) {
            //     double grad = (scale_de * (partial_atan_x * partial_x_n1 + partial_atan_y * partial_y_n1))(j);
            //     #pragma omp atomic
            //     Quad_Angle_sub_N_grad(f1 + N_undeformed_opt.rows(), j) += grad;
            //     grad = (scale_de * (partial_atan_x * partial_x_n2 + partial_atan_y * partial_y_n2))(j);
            //     #pragma omp atomic
            //     Quad_Angle_sub_N_grad(f2 + N_undeformed_opt.rows(), j) += grad;
            //     grad = (scale_de * (partial_atan_x * partial_x_v1 + partial_atan_y * partial_y_v1))(j);
            //     #pragma omp atomic
            //     Quad_Angle_sub_grad(v1 + V_undeformed.rows(), j) += grad;
            //     grad = (scale_de * (partial_atan_x * partial_x_v2 + partial_atan_y * partial_y_v2))(j);
            //     #pragma omp atomic
            //     Quad_Angle_sub_grad(v2 + V_undeformed.rows(), j) += grad;
            // }
            double scale_de = meshes.weight_angle_sub * 2 * delta * t / std::pow(std::pow(t, 2) + delta, 2);
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
                Quad_Angle_sub_grad(v1 + V_undeformed.rows(), j) += grad;
                grad = (scale_de * v1_v2 * v1_v2_de * 2 * (angle_de - angle_un) * (partial_atan_x * partial_x_v2 + partial_atan_y * partial_y_v2))(j) +
                    scale_de * difference * v1_v2 * 2 * (v1_v2_vec_de)(j);
                #pragma omp atomic
                Quad_Angle_sub_grad(v2 + V_undeformed.rows(), j) += grad;
            }
        }
    }
    // },10000);
    

    // igl::parallel_for(F_undeformed.rows(), [&](int i) {
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
            Quad_Angle_sub_grad(v1, j) += (meshes.weight_angle * 2 * (-N_v2_v1) * N_undeformed_opt.row(i) +
                meshes.weight_angle * 2 * (N_v1_v3) * N_undeformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v2, j) += (meshes.weight_angle * 2 * (N_v2_v1) * N_undeformed_opt.row(i) +
                meshes.weight_angle * 2 * (-N_v3_v2) * N_undeformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v3, j) += (meshes.weight_angle * 2 * (N_v3_v2) * N_undeformed_opt.row(i) +
                meshes.weight_angle * 2 * (-N_v1_v3) * N_undeformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_N_grad(i, j) += (meshes.weight_angle * 2 * (N_v2_v1) * (V_undeformed.row(v2) - V_undeformed.row(v1)) +
                meshes.weight_angle * 2 * (N_v3_v2) * (V_undeformed.row(v3) - V_undeformed.row(v2)) +
                meshes.weight_angle * 2 * (N_v1_v3) * (V_undeformed.row(v1) - V_undeformed.row(v3)) +
                meshes.weight_angle * 2 * (N_undeformed_opt.row(i).squaredNorm() - 1) * 2 * N_undeformed_opt.row(i))(j);
            // #pragma omp atomic
            // meshes.Quad_Angle_sub_grad(v1, j) += meshes.weight_angle * 2 
        }

        N_v2_v1 = N_deformed_opt.row(i).dot(V_deformed.row(v2) - V_deformed.row(v1));
        N_v3_v2 = N_deformed_opt.row(i).dot(V_deformed.row(v3) - V_deformed.row(v2));
        N_v1_v3 = N_deformed_opt.row(i).dot(V_deformed.row(v1) - V_deformed.row(v3));
        for (int j = 0; j < 3; j++){
            double grad = (2 * (-N_v2_v1) * N_deformed_opt.row(i) +
                2 * (N_v1_v3) * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v1 + V_undeformed.rows(), j) += grad;
            grad = (2 * (N_v2_v1) * N_deformed_opt.row(i) +
                2 * (-N_v3_v2) * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v2 + V_undeformed.rows(), j) += grad;
            grad = (2 * (N_v3_v2) * N_deformed_opt.row(i) +
                2 * (-N_v1_v3) * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v3 + V_undeformed.rows(), j) += grad;
            grad = (2 * (N_v2_v1) * (V_deformed.row(v2) - V_deformed.row(v1)) +
                2 * (N_v3_v2) * (V_deformed.row(v3) - V_deformed.row(v2)) +
                2 * (N_v1_v3) * (V_deformed.row(v1) - V_deformed.row(v3)) +
                2 * (N_deformed_opt.row(i).squaredNorm() - 1) * 2 * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_N_grad(i + N_undeformed_opt.rows(), j) += grad;
        }
    }
    // },10000);
    if (meshes.iteration%1000 == 0 && meshes.delta > 1e-6) {
        // meshes.delta = meshes.delta * 0.1;
        // meshes.weight_angle_sub = meshes.weight_angle_sub * 0.8;
    }
    std::cout << "delta: " << meshes.delta << std::endl;
}



void Compute_Quad_Angle_sub_sub(Meshes &meshes) {
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
    meshes.Quad_Angle_sub_C = 0.0;
    Eigen::VectorXd &Quad_Angle_sub_Vector = meshes.Quad_Angle_sub_Vector;
    Quad_Angle_sub_Vector = Eigen::VectorXd::Zero(meshes.uE.rows());
    // igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed);
    // igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed);
    // igl::parallel_for(meshes.uE.rows(), [&](int i) {
    meshes.selected_edges.clear();
    int count = 0;
     bool is_first;
    if (meshes.Quad_Angle_Vector.rows() == 0) {
        is_first = true;
        meshes.Quad_Angle_Vector = Eigen::VectorXd::Zero(meshes.uE.rows());
    }
    else {
        is_first = false;
    }
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
            #pragma omp atomic
            meshes.Quad_Angle_sub_C += meshes.weight_angle_sub * std::pow(difference, 2)/(std::pow(difference, 2) + delta);

            double real_angle_un, real_angle_de;
            Compute_Angle_sub(V_undeformed, F_undeformed, meshes.N_undeformed, f1, f2, v1, v2, real_angle_un);
            Compute_Angle_sub(V_deformed, F_deformed, meshes.N_deformed, f1, f2, v1, v2, real_angle_de);
             if (is_first) {
                if (meshes.Quad_Angle_sub_Vector(i) != 0) {
                    meshes.Quad_Angle_Vector[i] = angle_de;
                }
                else {
                    meshes.Quad_Angle_Vector[i] = angle_un;
                }
            }
        }
    }
    // },10000);
    // igl::parallel_for(F_undeformed.rows(), [&](int i) {
    // igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed_opt);
    // #pragma omp parallel for
    // for (int i = 0; i < F_undeformed.rows(); i++) {
    //     const int v1 = F_undeformed(i, 0);
    //     const int v2 = F_undeformed(i, 1);
    //     const int v3 = F_undeformed(i, 2);
    //     #pragma omp atomic
    //     meshes.Quad_Angle_sub_C += std::pow(N_undeformed_opt.row(i).dot(V_undeformed.row(v2) - V_undeformed.row(v1)), 2);
    //     #pragma omp atomic
    //     meshes.Quad_Angle_sub_C += std::pow(N_undeformed_opt.row(i).dot(V_undeformed.row(v3) - V_undeformed.row(v2)), 2);
    //     #pragma omp atomic
    //     meshes.Quad_Angle_sub_C += std::pow(N_undeformed_opt.row(i).dot(V_undeformed.row(v1) - V_undeformed.row(v3)), 2);
    //     #pragma omp atomic
    //     meshes.Quad_Angle_sub_C += std::pow(N_undeformed_opt.row(i).squaredNorm() - 1, 2);

    //     #pragma omp atomic
    //     meshes.Quad_Angle_sub_C += std::pow(N_deformed_opt.row(i).dot(V_deformed.row(v2) - V_deformed.row(v1)), 2);
    //     #pragma omp atomic
    //     meshes.Quad_Angle_sub_C += std::pow(N_deformed_opt.row(i).dot(V_deformed.row(v3) - V_deformed.row(v2)), 2);
    //     #pragma omp atomic
    //     meshes.Quad_Angle_sub_C += std::pow(N_deformed_opt.row(i).dot(V_deformed.row(v1) - V_deformed.row(v3)), 2);
    //     #pragma omp atomic
    //     meshes.Quad_Angle_sub_C += std::pow(N_deformed_opt.row(i).squaredNorm() - 1, 2);
    //     // #pragma omp atomic
    //     // meshes.Quad_Angle_sub_C += meshes.weight_angle * (N_undeformed_opt.row(i) - meshes.N_undeformed.row(i)).squaredNorm();
    //     // #pragma omp atomic
    //     // meshes.Quad_Angle_sub_C += meshes.weight_angle * (N_deformed_opt.row(i) - meshes.N_deformed.row(i)).squaredNorm();
    // }
    // },10000);
}

void Compute_Quad_derivatives_Angle_sub_sub(Meshes &meshes) {
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &N_undeformed_opt = meshes.N_undeformed_opt;
    Eigen::MatrixXd &N_deformed_opt = meshes.N_deformed_opt;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &EF = meshes.EF;
    const double delta = meshes.delta;
    meshes.Quad_Angle_C = 0.0;
    Eigen::MatrixXd &Quad_Angle_sub_grad = meshes.Quad_Angle_sub_grad;
    Quad_Angle_sub_grad = Eigen::MatrixXd::Zero(V_undeformed.rows()+V_deformed.rows(), 3);
    Eigen::MatrixXd &Quad_Angle_sub_N_grad = meshes.Quad_Angle_sub_N_grad;
    Quad_Angle_sub_N_grad = Eigen::MatrixXd::Zero(N_undeformed_opt.rows() + N_deformed_opt.rows(), 3);
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
            Compute_Angle_sub(V_undeformed, F_undeformed, N_undeformed_opt, f1, f2, v1, v2, angle_un);
            Compute_Angle_sub(V_deformed, F_deformed, N_deformed_opt, f1, f2, v1, v2, angle_de);
            angle_un = meshes.Quad_Angle_Vector[i];
            double v1_v2 = (V_undeformed.row(v2) - V_undeformed.row(v1)).norm();
            Eigen::RowVector3d v1_v2_normalized = (V_undeformed.row(v2) - V_undeformed.row(v1)) / v1_v2;
            double difference = std::pow(angle_un - angle_de, 2);
            double t = difference * v1_v2;
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

            // double scale_un = meshes.weight_angle_sub * 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_un - angle_de);
            double scale_un = meshes.weight_angle_sub * 2 * delta * t / std::pow(std::pow(t, 2) + delta, 2);


            for (int j = 0; j < 3; j++) {
                // #pragma omp atomic
                // Quad_Angle_sub_N_grad(f1, j) += (scale_un * (partial_atan_x * partial_x_n1 + partial_atan_y * partial_y_n1))(j);
                // #pragma omp atomic
                // Quad_Angle_sub_N_grad(f2, j) += (scale_un * (partial_atan_x * partial_x_n2 + partial_atan_y * partial_y_n2))(j);
                // #pragma omp atomic
                // Quad_Angle_sub_grad(v1, j) += (scale_un * (partial_atan_x * partial_x_v1 + partial_atan_y * partial_y_v1))(j);
                // #pragma omp atomic
                // Quad_Angle_sub_grad(v2, j) += (scale_un * (partial_atan_x * partial_x_v2 + partial_atan_y * partial_y_v2))(j);
                // double grad = (scale_un * (partial_atan_x * partial_x_n1 + partial_atan_y * partial_y_n1))(j);
                // #pragma omp atomic
                // Quad_Angle_sub_N_grad(f1, j) += grad;
                // grad = (scale_un * (partial_atan_x * partial_x_n2 + partial_atan_y * partial_y_n2))(j);
                // #pragma omp atomic
                // Quad_Angle_sub_N_grad(f2, j) += grad;
                // grad = (scale_un * (partial_atan_x * partial_x_v1 + partial_atan_y * partial_y_v1))(j);
                // #pragma omp atomic
                // Quad_Angle_sub_grad(v1, j) += grad;
                // grad = (scale_un * (partial_atan_x * partial_x_v2 + partial_atan_y * partial_y_v2))(j);
                // #pragma omp atomic
                // Quad_Angle_sub_grad(v2, j) += grad;
                double grad = (scale_un * v1_v2 * 2 * (angle_un - angle_de) * (partial_atan_x * partial_x_n1 + partial_atan_y * partial_y_n1))(j);
                #pragma omp atomic
                Quad_Angle_sub_N_grad(f1, j) += grad;
                grad = (scale_un * v1_v2 * 2 * (angle_un - angle_de) * (partial_atan_x * partial_x_n2 + partial_atan_y * partial_y_n2))(j);
                #pragma omp atomic
                Quad_Angle_sub_N_grad(f2, j) += grad;
                grad = (scale_un * v1_v2 * 2 * (angle_un - angle_de) * (partial_atan_x * partial_x_v1 + partial_atan_y * partial_y_v1))(j) +
                    scale_un * difference * (-v1_v2_normalized)(j);
                #pragma omp atomic
                Quad_Angle_sub_grad(v1, j) += grad;
                grad = (scale_un * v1_v2 * 2 * (angle_un - angle_de) * (partial_atan_x * partial_x_v1 + partial_atan_y * partial_y_v1))(j) +
                    scale_un * difference * (v1_v2_normalized)(j);
                #pragma omp atomic
                Quad_Angle_sub_grad(v2, j) += grad;
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

            // double scale_de = meshes.weight_angle_sub * 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_de - angle_un);
            double scale_de = meshes.weight_angle_sub * 2 * delta * t / std::pow(std::pow(t, 2) + delta, 2);
            for (int j = 0; j < 3; j++) {
                double grad = (scale_de * v1_v2 * 2 * (angle_de - angle_un) * (partial_atan_x * partial_x_n1 + partial_atan_y * partial_y_n1))(j);
                #pragma omp atomic
                Quad_Angle_sub_N_grad(f1 + N_undeformed_opt.rows(), j) += grad;
                grad = (scale_de * v1_v2 * 2 * (angle_de - angle_un) * (partial_atan_x * partial_x_n2 + partial_atan_y * partial_y_n2))(j);
                #pragma omp atomic
                Quad_Angle_sub_N_grad(f2 + N_undeformed_opt.rows(), j) += grad;
                grad = (scale_de * v1_v2 * 2 * (angle_de - angle_un) * (partial_atan_x * partial_x_v1 + partial_atan_y * partial_y_v1))(j);
                #pragma omp atomic
                Quad_Angle_sub_grad(v1 + V_undeformed.rows(), j) += grad;
                grad = (scale_de * v1_v2 * 2 * (angle_de - angle_un) * (partial_atan_x * partial_x_v2 + partial_atan_y * partial_y_v2))(j);
                #pragma omp atomic
                Quad_Angle_sub_grad(v2 + V_undeformed.rows(), j) += grad;
            }
        }
    }
    // },10000);
    

    // igl::parallel_for(F_undeformed.rows(), [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < F_undeformed.rows(); i++) {
        const int v1 = F_undeformed(i, 0);
        const int v2 = F_undeformed(i, 1);
        const int v3 = F_undeformed(i, 2);
        
        double N_v2_v1 = N_undeformed_opt.row(i).dot(V_undeformed.row(v2) - V_undeformed.row(v1));
        double N_v3_v2 = N_undeformed_opt.row(i).dot(V_undeformed.row(v3) - V_undeformed.row(v2));
        double N_v1_v3 = N_undeformed_opt.row(i).dot(V_undeformed.row(v1) - V_undeformed.row(v3));
        
        // for (int j = 0; j < 3; j++){
        //     #pragma omp atomic
        //     Quad_Angle_sub_grad(v1, j) += (meshes.weight_angle * 2 * (-N_v2_v1) * N_undeformed_opt.row(i) +
        //         meshes.weight_angle * 2 * (N_v1_v3) * N_undeformed_opt.row(i))(j);
        //     #pragma omp atomic
        //     Quad_Angle_sub_grad(v2, j) += (meshes.weight_angle * 2 * (N_v2_v1) * N_undeformed_opt.row(i) +
        //         meshes.weight_angle * 2 * (-N_v3_v2) * N_undeformed_opt.row(i))(j);
        //     #pragma omp atomic
        //     Quad_Angle_sub_grad(v3, j) += (meshes.weight_angle * 2 * (N_v3_v2) * N_undeformed_opt.row(i) +
        //         meshes.weight_angle * 2 * (-N_v1_v3) * N_undeformed_opt.row(i))(j);
        //     #pragma omp atomic
        //     Quad_Angle_sub_N_grad(i, j) += (meshes.weight_angle * 2 * (N_v2_v1) * (V_undeformed.row(v2) - V_undeformed.row(v1)) +
        //         meshes.weight_angle * 2 * (N_v3_v2) * (V_undeformed.row(v3) - V_undeformed.row(v2)) +
        //         meshes.weight_angle * 2 * (N_v1_v3) * (V_undeformed.row(v1) - V_undeformed.row(v3)) +
        //         meshes.weight_angle * 2 * (N_undeformed_opt.row(i).squaredNorm() - 1) * 2 * N_undeformed_opt.row(i))(j);
        //     // #pragma omp atomic
        //     // meshes.Quad_Angle_sub_grad(v1, j) += meshes.weight_angle * 2 
        // }
        for (int j = 0; j < 3; j++){
            double grad = (2 * (-N_v2_v1) * N_undeformed_opt.row(i) +
                2 * (N_v1_v3) * N_undeformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v1, j) += grad;
            grad = (2 * (N_v2_v1) * N_undeformed_opt.row(i) +
                2 * (-N_v3_v2) * N_undeformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v2, j) += grad;
            grad = (2 * (N_v3_v2) * N_undeformed_opt.row(i) +
                2 * (-N_v1_v3) * N_undeformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v3, j) += grad;
            grad = (2 * (N_v2_v1) * (V_undeformed.row(v2) - V_undeformed.row(v1)) +
                2 * (N_v3_v2) * (V_undeformed.row(v3) - V_undeformed.row(v2)) +
                2 * (N_v1_v3) * (V_undeformed.row(v1) - V_undeformed.row(v3)) +
                2 * (N_undeformed_opt.row(i).squaredNorm() - 1) * 2 * N_undeformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_N_grad(i, j) += grad;
        }

        N_v2_v1 = N_deformed_opt.row(i).dot(V_deformed.row(v2) - V_deformed.row(v1));
        N_v3_v2 = N_deformed_opt.row(i).dot(V_deformed.row(v3) - V_deformed.row(v2));
        N_v1_v3 = N_deformed_opt.row(i).dot(V_deformed.row(v1) - V_deformed.row(v3));
        for (int j = 0; j < 3; j++){
            double grad = (2 * (-N_v2_v1) * N_deformed_opt.row(i) +
                2 * (N_v1_v3) * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v1 + V_undeformed.rows(), j) += grad;
            grad = (2 * (N_v2_v1) * N_deformed_opt.row(i) +
                2 * (-N_v3_v2) * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v2 + V_undeformed.rows(), j) += grad;
            grad = (2 * (N_v3_v2) * N_deformed_opt.row(i) +
                2 * (-N_v1_v3) * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_grad(v3 + V_undeformed.rows(), j) += grad;
            grad = (2 * (N_v2_v1) * (V_deformed.row(v2) - V_deformed.row(v1)) +
                2 * (N_v3_v2) * (V_deformed.row(v3) - V_deformed.row(v2)) +
                2 * (N_v1_v3) * (V_deformed.row(v1) - V_deformed.row(v3)) +
                2 * (N_deformed_opt.row(i).squaredNorm() - 1) * 2 * N_deformed_opt.row(i))(j);
            #pragma omp atomic
            Quad_Angle_sub_N_grad(i + N_undeformed_opt.rows(), j) += grad;
        }
    }
    // },10000);
    if (meshes.iteration%1000 == 0 && meshes.delta > 1e-6) {
        // meshes.delta = meshes.delta * 0.1;
        // meshes.weight_angle_sub = meshes.weight_angle_sub * 0.8;
    }
    std::cout << "delta: " << meshes.delta << std::endl;
}


// void Compute_Newton_Angle(Meshes &meshes) {
//     Eigen::MatrixXi &F_deformed = meshes.F_deformed;
//     Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
//     Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
//     Eigen::MatrixXd &N_undeformed_opt = meshes.N_undeformed_opt;
//     Eigen::MatrixXd &N_deformed_opt = meshes.N_deformed_opt;
//     Eigen::MatrixXd &V_deformed = meshes.V_deformed;
//     Eigen::MatrixXi &EF = meshes.EF;
//     const double delta = meshes.delta;
//     meshes.Quad_Angle_C = 0.0;
//     Eigen::VectorXd &C_Angle = meshes.C_Angle;
//     C_Angle = Eigen::VectorXd::Zero((V_undeformed.rows()+V_deformed.rows()) * 3);
//     Eigen::VectorXd &C_Angle_N = meshes.C_Angle_N;
//     C_Angle_N = Eigen::VectorXd::Zero((N_undeformed_opt.rows() + N_deformed_opt.rows()) * 3);
//     // igl::parallel_for(meshes.uE.rows(), [&](int i) {
//     #pragma omp parallel for
//     for (int i = 0; i < meshes.uE.rows(); i++) {
//         if (!(EF(i, 0) == -1 || EF(i, 1) == -1)) {
//             std::vector<int> half_edges = meshes.uE2E[i];
//             const int num_faces = meshes.F_undeformed.rows();
//             const int f1 = half_edges[0] % num_faces;
//             const int f2 = half_edges[1] % num_faces;
//             const int c1 = half_edges[0] / num_faces;
//             const int c2 = half_edges[1] / num_faces;
//             const int v1 = meshes.F_undeformed(f1, (c1+1)%3);
//             const int v2 = meshes.F_undeformed(f1, (c1+2)%3);
//             const int v4 = meshes.F_undeformed(f1, c1);
//             const int v3 = meshes.F_undeformed(f2, c2);
//             double angle_un, angle_de;
//             Compute_Angle_sub(V_undeformed, F_undeformed, N_undeformed_opt, f1, f2, v1, v2, angle_un);
//             Compute_Angle_sub(V_deformed, F_deformed, N_deformed_opt, f1, f2, v1, v2, angle_de);
//             // angle_un = meshes.Quad_Angle_Vector[i];
//             double difference = std::pow(angle_un - angle_de, 2);
//             Eigen::RowVector3d n1 = N_undeformed_opt.row(f1);
//             Eigen::RowVector3d n2 = N_undeformed_opt.row(f2);

//             Eigen::RowVector3d V1 = V_undeformed.row(v1);
//             Eigen::RowVector3d V2 = V_undeformed.row(v2);

//             Eigen::RowVector3d n1_cross_n2 = n1.cross(n2);
//             double n1_dot_n2 = n1.dot(n2);
//             Eigen::RowVector3d v2_minus_v1 = V2 - V1;
//             double l_v2_minus_v1 = v2_minus_v1.norm();
//             double x = n1_dot_n2;
//             double y = n1_cross_n2.dot(v2_minus_v1.normalized());
//             double partial_atan_x = -y / (x * x + y * y);
//             double partial_atan_y = x / (x * x + y * y);
//             Eigen::RowVector3d partial_x_n1 = n2;
//             Eigen::RowVector3d partial_x_n2 = n1;
//             Eigen::RowVector3d partial_y_n1 = n2.cross(v2_minus_v1.normalized());
//             Eigen::RowVector3d partial_y_n2 = v2_minus_v1.normalized().cross(n1);
//             Eigen::RowVector3d partial_x_v1 = Eigen::RowVector3d::Zero();
//             Eigen::RowVector3d partial_x_v2 = Eigen::RowVector3d::Zero();
//             Eigen::RowVector3d partial_y_v1 = (-n1_cross_n2*l_v2_minus_v1 + n1_cross_n2.dot(v2_minus_v1)*v2_minus_v1/l_v2_minus_v1)/(l_v2_minus_v1*l_v2_minus_v1);
//             Eigen::RowVector3d partial_y_v2 = (n1_cross_n2*l_v2_minus_v1 - n1_cross_n2.dot(v2_minus_v1)*v2_minus_v1/l_v2_minus_v1)/(l_v2_minus_v1*l_v2_minus_v1);

//             double scale_un = meshes.weight_angle_sub * 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_un - angle_de);
            
//             for (int j = 0; j < 3; j++) {
//                 double grad = (scale_un * (partial_atan_x * partial_x_n1 + partial_atan_y * partial_y_n1))(j);
//                 #pragma omp atomic
//                 C_Angle_N(f1*3 + j) += grad;
//                 grad = (scale_un * (partial_atan_x * partial_x_n2 + partial_atan_y * partial_y_n2))(j);
//                 #pragma omp atomic
//                 C_Angle_N(f2*3 + j) += grad;
//                 grad = (scale_un * (partial_atan_x * partial_x_v1 + partial_atan_y * partial_y_v1))(j);
//                 #pragma omp atomic
//                 C_Angle(v1*3 + j) += grad;
//                 grad = (scale_un * (partial_atan_x * partial_x_v2 + partial_atan_y * partial_y_v2))(j);
//                 #pragma omp atomic
//                 C_Angle(v2*3 + j) += grad;
//             }

//             n1 = N_deformed_opt.row(f1);
//             n2 = N_deformed_opt.row(f2);

//             V1 = V_deformed.row(v1);
//             V2 = V_deformed.row(v2);

//             n1_cross_n2 = n1.cross(n2);
//             n1_dot_n2 = n1.dot(n2);
//             v2_minus_v1 = V2 - V1;
//             l_v2_minus_v1 = v2_minus_v1.norm();
//             x = n1_dot_n2;
//             y = n1_cross_n2.dot(v2_minus_v1.normalized());
//             partial_atan_x = -y / (x * x + y * y);
//             partial_atan_y = x / (x * x + y * y);
//             partial_x_n1 = n2;
//             partial_x_n2 = n1;
//             partial_y_n1 = n2.cross(v2_minus_v1.normalized());
//             partial_y_n2 = v2_minus_v1.normalized().cross(n1);
//             partial_x_v1 = Eigen::RowVector3d::Zero();
//             partial_x_v2 = Eigen::RowVector3d::Zero();
//             partial_y_v1 = (-n1_cross_n2*l_v2_minus_v1 + n1_cross_n2.dot(v2_minus_v1)*v2_minus_v1/l_v2_minus_v1)/(l_v2_minus_v1*l_v2_minus_v1);
//             partial_y_v2 = (n1_cross_n2*l_v2_minus_v1 - n1_cross_n2.dot(v2_minus_v1)*v2_minus_v1/l_v2_minus_v1)/(l_v2_minus_v1*l_v2_minus_v1);

//             double scale_de = meshes.weight_angle_sub * 2 * delta * difference / std::pow(std::pow(difference, 2) + delta, 2) * 2 * (angle_de - angle_un);
//             for (int j = 0; j < 3; j++) {
//                 double grad = (scale_de * (partial_atan_x * partial_x_n1 + partial_atan_y * partial_y_n1))(j);
//                 #pragma omp atomic
//                 C_Angle_N(f1*3 + j + N_undeformed_opt.rows()) += grad;
//                 grad = (scale_de * (partial_atan_x * partial_x_n2 + partial_atan_y * partial_y_n2))(j);
//                 #pragma omp atomic
//                 C_Angle_N(f2*3 + j + N_undeformed_opt.rows()) += grad;
//                 grad = (scale_de * (partial_atan_x * partial_x_v1 + partial_atan_y * partial_y_v1))(j);
//                 #pragma omp atomic
//                 C_Angle(v1*3 + j + V_undeformed.rows()*3) += grad;
//                 grad = (scale_de * (partial_atan_x * partial_x_v2 + partial_atan_y * partial_y_v2))(j);
//                 #pragma omp atomic
//                 C_Angle(v2*3 + j + V_undeformed.rows()*3) += grad;
//             }
//         }
//     }
    
//     #pragma omp parallel for
//     for (int i = 0; i < F_undeformed.rows(); i++) {
//         const int v1 = F_undeformed(i, 0);
//         const int v2 = F_undeformed(i, 1);
//         const int v3 = F_undeformed(i, 2);
        
//         double N_v2_v1 = N_undeformed_opt.row(i).dot(V_undeformed.row(v2) - V_undeformed.row(v1));
//         double N_v3_v2 = N_undeformed_opt.row(i).dot(V_undeformed.row(v3) - V_undeformed.row(v2));
//         double N_v1_v3 = N_undeformed_opt.row(i).dot(V_undeformed.row(v1) - V_undeformed.row(v3));

//         for (int j = 0; j < 3; j++){
//             double grad = (meshes.weight_angle * 2 * (-N_v2_v1) * N_undeformed_opt.row(i) +
//                 meshes.weight_angle * 2 * (N_v1_v3) * N_undeformed_opt.row(i))(j);
//             #pragma omp atomic
//             C_Angle(v1*3 + j) += grad;
//             grad = (meshes.weight_angle * 2 * (N_v2_v1) * N_undeformed_opt.row(i) +
//                 meshes.weight_angle * 2 * (-N_v3_v2) * N_undeformed_opt.row(i))(j);
//             #pragma omp atomic
//             C_Angle(v2*3 + j) += grad;
//             grad = (meshes.weight_angle * 2 * (N_v3_v2) * N_undeformed_opt.row(i) +
//                 meshes.weight_angle * 2 * (-N_v1_v3) * N_undeformed_opt.row(i))(j);
//             #pragma omp atomic
//             C_Angle(v3*3 + j) += grad;
//             grad = (meshes.weight_angle * 2 * (N_v2_v1) * (V_undeformed.row(v2) - V_undeformed.row(v1)) +
//             meshes.weight_angle * 2 * (N_v3_v2) * (V_undeformed.row(v3) - V_undeformed.row(v2)) +
//             meshes.weight_angle * 2 * (N_v1_v3) * (V_undeformed.row(v1) - V_undeformed.row(v3)) +
//             meshes.weight_angle * 2 * (N_undeformed_opt.row(i).squaredNorm() - 1) * 2 * N_undeformed_opt.row(i))(j);
//             #pragma omp atomic
//             C_Angle_N(i*3 + j) += grad;
//             // #pragma omp atomic
//             // meshes.Quad_Angle_sub_grad(v1, j) += meshes.weight_angle * 2 
//         }

//         N_v2_v1 = N_deformed_opt.row(i).dot(V_deformed.row(v2) - V_deformed.row(v1));
//         N_v3_v2 = N_deformed_opt.row(i).dot(V_deformed.row(v3) - V_deformed.row(v2));
//         N_v1_v3 = N_deformed_opt.row(i).dot(V_deformed.row(v1) - V_deformed.row(v3));
//         for (int j = 0; j < 3; j++){
//             double grad = (2 * (-N_v2_v1) * N_deformed_opt.row(i) +
//                 2 * (N_v1_v3) * N_deformed_opt.row(i))(j);
//             #pragma omp atomic
//             C_Angle(v1*3 + j + V_undeformed.rows()*3) += grad;
//             grad = (2 * (N_v2_v1) * N_deformed_opt.row(i) +
//                 2 * (-N_v3_v2) * N_deformed_opt.row(i))(j);
//             #pragma omp atomic
//             C_Angle(v2*3 + j + V_undeformed.rows()*3) += grad;
//             grad = (2 * (N_v3_v2) * N_deformed_opt.row(i) +
//                 2 * (-N_v1_v3) * N_deformed_opt.row(i))(j);
//             #pragma omp atomic
//             C_Angle(v3*3 + j + V_undeformed.rows()*3) += grad;
//             grad = (2 * (N_v2_v1) * (V_deformed.row(v2) - V_deformed.row(v1)) +
//                 2 * (N_v3_v2) * (V_deformed.row(v3) - V_deformed.row(v2)) +
//                 2 * (N_v1_v3) * (V_deformed.row(v1) - V_deformed.row(v3)) +
//                 2 * (N_deformed_opt.row(i).squaredNorm() - 1) * 2 * N_deformed_opt.row(i))(j);
//             #pragma omp atomic
//             C_Angle_N(i*3 + j + N_undeformed_opt.rows() * 3) += grad;
//         }
//     }
//     if (meshes.iteration%1000 == 0 && meshes.delta > 1e-6) {
//         // meshes.delta = meshes.delta * 0.1;
//         // meshes.weight_angle_sub = meshes.weight_angle_sub * 0.8;
//     }
//     std::cout << "delta: " << meshes.delta << std::endl;
// }

// // Helper function to add a 3x3 block matrix to the triplet list
// inline void AddBlockToTriplets(std::vector<Eigen::Triplet<double>>& triplets,
//     int start_row, int start_col,
//     const Eigen::Matrix3d& block)
// {
// for (int r = 0; r < 3; ++r) {
// for (int c = 0; c < 3; ++c) {
// if (block(r, c) != 0.0) { // Add only non-zero elements
// triplets.emplace_back(start_row + r, start_col + c, block(r, c));
// }
// }
// }
// }

// // Helper function to add a 3x1 vector * 1x3 vector (outer product) to triplets
// inline void AddOuterProductToTriplets(std::vector<Eigen::Triplet<double>>& triplets,
//            int start_row, int start_col,
//            const Eigen::RowVector3d& vec1,
//            const Eigen::RowVector3d& vec2,
//            double scale = 1.0)
// {
// if (scale == 0.0) return;
// for (int r = 0; r < 3; ++r) {
// for (int c = 0; c < 3; ++c) {
// double val = scale * vec1(r) * vec2(c);
// if (val != 0.0) {
// triplets.emplace_back(start_row + r, start_col + c, val);
// }
// }
// }
// }

// // Helper function for cross product matrix [v]x
// inline Eigen::Matrix3d CrossProductMatrix(const Eigen::RowVector3d& v) {
// Eigen::Matrix3d m;
// m <<  0, -v(2),  v(1),
// v(2),  0, -v(0),
// -v(1), v(0),  0;
// return m;
// }




// // --- Function to compute Dihedral Angle and its derivatives ---
// // Renamed v1->vA, v2->vB for clarity (edge vertices)
// void Compute_Angle_And_Derivatives(
//     const Eigen::RowVector3d& VA, const Eigen::RowVector3d& VB, // Edge vertices
//     const Eigen::RowVector3d& VC, const Eigen::RowVector3d& VD, // Opposite vertices (VC in face 1, VD in face 2)
//     const Eigen::RowVector3d& n1, const Eigen::RowVector3d& n2, // Face normals
//     DihedralDerivatives& derivs) // Output struct
// {
//     // --- Input variables ---
//     // Edge vector: e = VB - VA
//     // Normal vectors: n1, n2

//     // --- Intermediate values ---
//     Eigen::RowVector3d e = VB - VA;
//     double l_e = e.norm();
//     if (l_e < 1e-12) { // Degenerate edge
//         derivs = DihedralDerivatives(); // Reset to zeros
//         return;
//     }
//     Eigen::RowVector3d e_norm = e / l_e;

//     double n1_dot_n2 = n1.dot(n2);
//     Eigen::RowVector3d n1_cross_n2 = n1.cross(n2);

//     // Clamp n1_dot_n2 to avoid numerical issues with acos or atan2 near +/-1
//     double x = std::max(-1.0, std::min(1.0, n1_dot_n2));
//     double y = n1_cross_n2.dot(e_norm);

//     // --- Angle ---
//     derivs.angle = std::atan2(y, x); // Dihedral angle in [-pi, pi]

//     // --- First Derivatives (Gradients) ---
//     // Common term: den = x^2 + y^2
//     double den = x * x + y * y;
//     if (den < 1e-12) { // Normals are likely parallel or anti-parallel
//         // Derivatives are ill-defined or very large. Handle gracefully.
//         derivs = DihedralDerivatives(); // Reset to zero, but keep angle
//         derivs.angle = std::atan2(y, x);
//         return;
//     }

//     double inv_den = 1.0 / den;
//     double dAdx = -y * inv_den; // d(atan2)/dx
//     double dAdy = x * inv_den;  // d(atan2)/dy

//     // Derivatives of x and y w.r.t. inputs
//     Eigen::RowVector3d dx_dn1 = n2;
//     Eigen::RowVector3d dx_dn2 = n1;
//     Eigen::RowVector3d dx_dVA = Eigen::RowVector3d::Zero();
//     Eigen::RowVector3d dx_dVB = Eigen::RowVector3d::Zero();

//     Eigen::RowVector3d dy_dn1 = n2.cross(e_norm);
//     Eigen::RowVector3d dy_dn2 = e_norm.cross(n1); // = -n1.cross(e_norm)

//     Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
//     Eigen::Matrix3d e_norm_outer = e_norm.transpose() * e_norm; // 3x3 matrix e_norm^T * e_norm
//     Eigen::Matrix3d de_norm_dVA_mat = (-I + e_norm_outer) / l_e; // d(e_norm)/dVA
//     Eigen::Matrix3d de_norm_dVB_mat = ( I - e_norm_outer) / l_e; // d(e_norm)/dVB

//     // dy/dVA = dy/de_norm * de_norm/dVA = (n1 x n2) * de_norm_dVA_mat
//     // dy/dVB = dy/de_norm * de_norm/dVB = (n1 x n2) * de_norm_dVB_mat
//     Eigen::RowVector3d dy_dVA = n1_cross_n2 * de_norm_dVA_mat;
//     Eigen::RowVector3d dy_dVB = n1_cross_n2 * de_norm_dVB_mat;


//     // Chain rule for gradients dA/dp = (dA/dx)(dx/dp) + (dA/dy)(dy/dp)
//     derivs.grad_n1 = dAdx * dx_dn1 + dAdy * dy_dn1;
//     derivs.grad_n2 = dAdx * dx_dn2 + dAdy * dy_dn2;
//     derivs.grad_vA = dAdx * dx_dVA + dAdy * dy_dVA; // = dAdy * dy_dVA
//     derivs.grad_vB = dAdx * dx_dVB + dAdy * dy_dVB; // = dAdy * dy_dVB

//     // --- Second Derivatives (Hessians) ---
//     // Formula: H_ij = d/dpi (dA/dpj) = d/dpi [ (dA/dx)(dx/dpj) + (dA/dy)(dy/dpj) ]
//     // Chain rule expansion:
//     // H_ij = [ (d(dA/dx)/dpi) * (dx/dpj) + (dA/dx) * (d^2x/dpi dpj) ]
//     //      + [ (d(dA/dy)/dpi) * (dy/dpj) + (dA/dy) * (d^2y/dpi dpj) ]
//     // where:
//     // d(dA/dx)/dpi = (d^2A/dx^2)(dx/dpi) + (d^2A/dxdy)(dy/dpi)
//     // d(dA/dy)/dpi = (d^2A/dydx)(dx/dpi) + (d^2A/dy^2)(dy/dpi)
//     // H_ij = [ (dAxx*dx_dpi + dAxy*dy_dpi)*dx_dpj^T + dAdx * d2x_dpi_dpj ] -> Matrix form needs care
//     //      + [ (dAxy*dx_dpi + dAyy*dy_dpi)*dy_dpj^T + dAdy * d2y_dpi_dpj ]

//     // Derivatives of atan2(y,x)
//     double inv_den_sq = inv_den * inv_den;
//     double dAxx = 2.0 * x * y * inv_den_sq;        // d^2A/dx^2
//     double dAyy = -dAxx;                           // d^2A/dy^2
//     double dAxy = (y*y - x*x) * inv_den_sq;        // d^2A/dxdy = d^2A/dydx

//     // --- Second derivatives of x = n1.dot(n2) ---
//     Eigen::Matrix3d d2x_dn1_dn1 = Eigen::Matrix3d::Zero();
//     Eigen::Matrix3d d2x_dn1_dn2 = I; // d/dn1(dx/dn2) = d/dn1(n1) = I
//     Eigen::Matrix3d d2x_dn2_dn2 = Eigen::Matrix3d::Zero();
//     // All other d2x involving VA or VB are also zero matrices.

//     // --- Second derivatives of y = (n1 x n2) . e_norm ---
//     // These are the most complex parts. We need d/dpi(dy/dpj).

//     // d(dy/dn1)/dpi -> d/dpi( n2 x e_norm )
//     Eigen::Matrix3d d2y_dn1_dn1 = Eigen::Matrix3d::Zero(); // d/dn1(n2 x e_norm) = 0
//     Eigen::Matrix3d d2y_dn1_dn2 = CrossProductMatrix(e_norm); // d/dn2(n2 x e_norm) = I x e_norm = CrossProductMatrix(e_norm) (assuming n2 is the variable)
//     // d/dVA (n2 x e_norm) = n2 x (de_norm/dVA) = -CrossProductMatrix(n2) * de_norm_dVA_mat
//     Eigen::Matrix3d d2y_dn1_dVA = -CrossProductMatrix(n2) * de_norm_dVA_mat;
//     Eigen::Matrix3d d2y_dn1_dVB = -CrossProductMatrix(n2) * de_norm_dVB_mat;

//     // d(dy/dn2)/dpi -> d/dpi( e_norm x n1 )
//     Eigen::Matrix3d d2y_dn2_dn2 = Eigen::Matrix3d::Zero(); // d/dn2(e_norm x n1) = 0
//     // d/dVA (e_norm x n1) = (de_norm/dVA) x n1 = de_norm_dVA_mat * CrossProductMatrix(n1) ?? No, vector x vector -> Matrix * vector
//     // Let's use component notation or index notation carefully.
//     // Alternative: d/dVA(e_norm x n1)_k = d/dVA( eps_kji e_j n1_i ) = eps_kji (d(e_j)/dVA) n1_i
//     // This corresponds to de_norm_dVA_mat * n1 applied with cross product structure.
//     // Resulting matrix M should satisfy M*v = (de_norm_dVA_mat * v) x n1 for any vector v.
//     // d/dVA(e_norm x n1) -> acts on v -> (de_norm_dVA_mat * v) x n1 = -n1 x (de_norm_dVA_mat * v) = CrossProductMatrix(-n1) * de_norm_dVA_mat * v
//     Eigen::Matrix3d d2y_dn2_dVA = CrossProductMatrix(-n1) * de_norm_dVA_mat; // Transpose? Check index notation carefully. Let's assume this structure.
//     Eigen::Matrix3d d2y_dn2_dVB = CrossProductMatrix(-n1) * de_norm_dVB_mat;

//     // d(dy/dVA)/dpi -> d/dpi( (n1 x n2) * de_norm_dVA_mat )
//     // These require Hessians of e_norm (d^2 e_norm / dVA dVA, etc.) which are extremely complex.
//     // *** Approximating these Hessians as ZERO for now ***
//     // *** A full Hessian requires computing these terms ***
//     Eigen::Matrix3d d2y_dVA_dVA = Eigen::Matrix3d::Zero(); // Placeholder
//     Eigen::Matrix3d d2y_dVA_dVB = Eigen::Matrix3d::Zero(); // Placeholder
//     Eigen::Matrix3d d2y_dVB_dVB = Eigen::Matrix3d::Zero(); // Placeholder


//     // --- Assemble Hessian Blocks H_ij = d/dpi (dA/dpj) ---
//     // Use the formula derived above, being careful with matrix vs vector operations
//     // H_ij = (dAxx*dx_dpi + dAxy*dy_dpi)^T * dx_dpj + dAdx * d2x_dpi_dpj
//     //      + (dAxy*dx_dpi + dAyy*dy_dpi)^T * dy_dpj + dAdy * d2y_dpi_dpj
//     // Note: (vectorA^T * vectorB) computes outer product matrix A * B^T

//     // Components for H_n1_n1 = d/dn1(dA/dn1)
//     Eigen::RowVector3d dAdx_dn1 = dAxx * dx_dn1 + dAxy * dy_dn1;
//     Eigen::RowVector3d dAdy_dn1 = dAxy * dx_dn1 + dAyy * dy_dn1;
//     derivs.H_n1_n1 = dAdx_dn1.transpose() * dx_dn1 + dAdx * d2x_dn1_dn1 /*0*/
//                    + dAdy_dn1.transpose() * dy_dn1 + dAdy * d2y_dn1_dn1 /*0*/;

//     // Components for H_n1_n2 = d/dn1(dA/dn2)
//     // Need dx_dn2, dy_dn2
//     derivs.H_n1_n2 = dAdx_dn1.transpose() * dx_dn2 + dAdx * d2x_dn1_dn2 /*I*/
//                    + dAdy_dn1.transpose() * dy_dn2 + dAdy * d2y_dn1_dn2;

//     // Components for H_n1_vA = d/dn1(dA/dVA)
//     // Need dx_dVA=0, dy_dVA
//     derivs.H_vA_n1= (dAdx_dn1.transpose() * dx_dVA /*0*/ + dAdx * Eigen::Matrix3d::Zero() /*d2x_dn1_dVA*/
//                    + dAdy_dn1.transpose() * dy_dVA + dAdy * d2y_dn1_dVA).transpose(); // Transpose to match H_n1_vA

//     // Components for H_n1_vB = d/dn1(dA/dVB)
//     // Need dx_dVB=0, dy_dVB
//     derivs.H_vB_n1 = (dAdx_dn1.transpose() * dx_dVB /*0*/ + dAdx * Eigen::Matrix3d::Zero() /*d2x_dn1_dVB*/
//                    + dAdy_dn1.transpose() * dy_dVB + dAdy * d2y_dn1_dVB).transpose();


//     // Components for H_n2_n2 = d/dn2(dA/dn2)
//     Eigen::RowVector3d dAdx_dn2 = dAxx * dx_dn2 + dAxy * dy_dn2;
//     Eigen::RowVector3d dAdy_dn2 = dAxy * dx_dn2 + dAyy * dy_dn2;
//     derivs.H_n2_n2 = dAdx_dn2.transpose() * dx_dn2 + dAdx * d2x_dn2_dn2 /*0*/
//                    + dAdy_dn2.transpose() * dy_dn2 + dAdy * d2y_dn2_dn2 /*0*/;

//     // Components for H_n2_vA = d/dn2(dA/dVA)
//     derivs.H_vA_n2 = (dAdx_dn2.transpose() * dx_dVA /*0*/ + dAdx * Eigen::Matrix3d::Zero() /*d2x_dn2_dVA*/)
//                    + (dAdy_dn2.transpose() * dy_dVA + dAdy * d2y_dn2_dVA).transpose();

//     // Components for H_n2_vB = d/dn2(dA/dVB)
//     derivs.H_vB_n2 = (dAdx_dn2.transpose() * dx_dVB /*0*/ + dAdx * Eigen::Matrix3d::Zero() /*d2x_dn2_dVB*/)
//                    + (dAdy_dn2.transpose() * dy_dVB + dAdy * d2y_dn2_dVB).transpose();


//     // Components for H_vA_vA = d/dVA(dA/dVA)
//     Eigen::RowVector3d dAdx_dVA = dAxx * dx_dVA /*0*/ + dAxy * dy_dVA; // = dAxy * dy_dVA
//     Eigen::RowVector3d dAdy_dVA = dAxy * dx_dVA /*0*/ + dAyy * dy_dVA; // = dAyy * dy_dVA
//     derivs.H_vA_vA = dAdx_dVA.transpose() * dx_dVA /*0*/ + dAdx * Eigen::Matrix3d::Zero() /*d2x_dVA_dVA*/
//                    + dAdy_dVA.transpose() * dy_dVA + dAdy * d2y_dVA_dVA /* approx 0 */;

//     // Components for H_vA_vB = d/dVA(dA/dVB)
//     derivs.H_vA_vB = dAdx_dVA.transpose() * dx_dVB /*0*/ + dAdx * Eigen::Matrix3d::Zero() /*d2x_dVA_dVB*/
//                    + dAdy_dVA.transpose() * dy_dVB + dAdy * d2y_dVA_dVB /* approx 0 */;


//     // Components for H_vB_vB = d/dVB(dA/dVB)
//     Eigen::RowVector3d dAdx_dVB = dAxx * dx_dVB /*0*/ + dAxy * dy_dVB; // = dAxy * dy_dVB
//     Eigen::RowVector3d dAdy_dVB = dAxy * dx_dVB /*0*/ + dAyy * dy_dVB; // = dAyy * dy_dVB
//     derivs.H_vB_vB = dAdx_dVB.transpose() * dx_dVB /*0*/ + dAdx * Eigen::Matrix3d::Zero() /*d2x_dVB_dVB*/
//                    + dAdy_dVB.transpose() * dy_dVB + dAdy * d2y_dVB_dVB /* approx 0 */;

//     // Fill symmetric counterparts (if needed, or rely on adding both ij and ji terms later)
//     // Example: derivs.H_vB_vA = derivs.H_vA_vB.transpose(); // Do this if only upper/lower triangle is computed

//     // **************************************************************************
//     // *** END OF COMPLEX ANALYTICAL DERIVATIVES FOR HESSIAN OF ANGLE ******
//     // **************************************************************************
// }




// // Main Hessian Computation Function
// void Compute_Newton_derivatives_Angle(Meshes &meshes)
// {
// // Mesh data
// std::vector<Eigen::Triplet<double>> &triplets = meshes.tripletList_Angle;
// triplets.clear(); // Clear previous triplet list
//     // Mesh data
//     const Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
//     const Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
//     const Eigen::MatrixXd &N_undeformed_opt = meshes.N_undeformed_opt;
//     const Eigen::MatrixXd &V_deformed = meshes.V_deformed;
//     const Eigen::MatrixXi &F_deformed = meshes.F_deformed; // Should be same topology F_undeformed
//     const Eigen::MatrixXd &N_deformed_opt = meshes.N_deformed_opt;
//     const Eigen::MatrixXi &EF = meshes.EF; // Edge to face incidence
//     const Eigen::MatrixXi &uE = meshes.uE; // Unique edges (vertex pairs)
//     const std::vector<std::vector<int>>& uE2E = meshes.uE2E; // Map unique edge index to halfedges

//     const double delta = meshes.delta;
//     const double weight_angle_sub = meshes.weight_angle_sub; // Weight for dihedral term
//     const double weight_angle = meshes.weight_angle;       // Weight for normal term

//     // Dimensions
//     const int nV = V_undeformed.rows();
//     const int nF = F_undeformed.rows();
//     const int nE = uE.rows(); // Number of unique edges
//     const int total_vars = (nV + nV + nF + nF) * 3;

//     // Global index offsets
//     const int V_un_offset = 0;
//     const int V_de_offset = nV * 3;
//     const int N_un_offset = (nV + nV) * 3;
//     const int N_de_offset = (nV + nV + nF) * 3;

//     // Estimate triplet count (Overestimation is safer)
//     size_t estimated_triplets = nE * 576 + nF * 144 * 2 + 1000; // Rough estimate
//     // std::cout << "Estimated triplets: " << estimated_triplets << std::endl;

//     // Thread-local triplet storage
//     int max_threads = omp_get_max_threads();
//     if (max_threads <= 0) max_threads = 1; // Ensure at least one thread
//     std::vector<std::vector<Eigen::Triplet<double>>> T_thread(max_threads);
//     for(int i=0; i<max_threads; ++i) {
//          // Adjusted allocation per thread based on rough estimate
//         T_thread[i].reserve(estimated_triplets / max_threads + 500);
//     }


//     // --- Part 1: Hessian of Dihedral Angle Energy ---
//     #pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < nE; i++) {
//         int thread_id = omp_get_thread_num();
//         if (thread_id < 0 || thread_id >= max_threads) thread_id = 0; // Safety check

//         // Check for boundary edges
//         if (EF(i, 0) == -1 || EF(i, 1) == -1 || uE2E.empty() || i >= uE2E.size() || uE2E[i].size() < 2) {
//             continue; // Skip boundary or invalid edges
//         }

//         // Get faces and vertices for the edge
//         const std::vector<int>& half_edges = uE2E[i];
//         const int num_faces = nF; // F_undeformed.rows()
//         const int he1 = half_edges[0];
//         const int he2 = half_edges[1];

//         const int f1 = he1 % num_faces;
//         const int f2 = he2 % num_faces;
//         // Local indices c1, c2 are not strictly needed if we use uE directly

//         // Use unique edge vertices (vA, vB)
//         const int vA_idx = uE(i, 0);
//         const int vB_idx = uE(i, 1);
//         int vC_idx = -1, vD_idx = -1; // Opposite vertices

//         // Find vC (in f1, opposite edge AB) and vD (in f2, opposite edge AB)
//         for(int k=0; k<3; ++k) {
//             if(F_undeformed(f1, k) != vA_idx && F_undeformed(f1, k) != vB_idx) vC_idx = F_undeformed(f1, k);
//             if(F_undeformed(f2, k) != vA_idx && F_undeformed(f2, k) != vB_idx) vD_idx = F_undeformed(f2, k);
//         }
//          if (vC_idx == -1 || vD_idx == -1) {
//              // This might happen on non-manifold edges or if indexing is wrong
//              #pragma omp critical
//              {
//                 // std::cerr << "Warning: Could not find opposite vertices for edge " << i << " (" << vA_idx << "," << vB_idx << ") f1=" << f1 << " f2=" << f2 << std::endl;
//              }
//              continue;
//          }


//         // Get vertex positions and normals
//         Eigen::RowVector3d VA_un = V_undeformed.row(vA_idx);
//         Eigen::RowVector3d VB_un = V_undeformed.row(vB_idx);
//         Eigen::RowVector3d VC_un = V_undeformed.row(vC_idx); // Opposite vertex in f1
//         Eigen::RowVector3d VD_un = V_undeformed.row(vD_idx); // Opposite vertex in f2
//         Eigen::RowVector3d N1_un = N_undeformed_opt.row(f1);
//         Eigen::RowVector3d N2_un = N_undeformed_opt.row(f2);

//         Eigen::RowVector3d VA_de = V_deformed.row(vA_idx);
//         Eigen::RowVector3d VB_de = V_deformed.row(vB_idx);
//         Eigen::RowVector3d VC_de = V_deformed.row(vC_idx);
//         Eigen::RowVector3d VD_de = V_deformed.row(vD_idx);
//         Eigen::RowVector3d N1_de = N_deformed_opt.row(f1);
//         Eigen::RowVector3d N2_de = N_deformed_opt.row(f2);

//         // Compute angles and all derivatives (including Hessians)
//         DihedralDerivatives deriv_un, deriv_de;
//         Compute_Angle_And_Derivatives(VA_un, VB_un, VC_un, VD_un, N1_un, N2_un, deriv_un);
//         Compute_Angle_And_Derivatives(VA_de, VB_de, VC_de, VD_de, N1_de, N2_de, deriv_de);

//         // Compute energy function derivatives: f(t) = w * t^2 / (t^2 + delta), t = angle_un - angle_de
//         double t = deriv_un.angle - deriv_de.angle;
//         double t_sq = t * t;
//         double den_ft = t_sq + delta;
//         double den_ft_sq = den_ft * den_ft;
//         double den_ft_cub = den_ft_sq * den_ft;

//         if (std::abs(den_ft) < 1e-15) continue; // Avoid division by zero

//         // f'(t) = 2 * t * delta / (t^2 + delta)^2
//         double f_prime_t = 2.0 * t * delta / den_ft_sq;

//         // f''(t) = (2*delta^2 - 6*t^2*delta) / (t^2 + delta)^3
//         double f_double_prime_t = (2.0 * delta * delta - 6.0 * t_sq * delta) / den_ft_cub;

//         // Variables involved: V_un(A), V_un(B), N_un(f1), N_un(f2) -> Block Un
//         //                   V_de(A), V_de(B), N_de(f1), N_de(f2) -> Block De
//         // Total 8 vector variables (24 scalars)

//         // Global indices (start row/col for 3x3 blocks)
//         int vAun_r = V_un_offset + vA_idx * 3; int vBun_r = V_un_offset + vB_idx * 3;
//         int n1un_r = N_un_offset + f1 * 3;     int n2un_r = N_un_offset + f2 * 3;
//         int vAde_r = V_de_offset + vA_idx * 3; int vBde_r = V_de_offset + vB_idx * 3;
//         int n1de_r = N_de_offset + f1 * 3;     int n2de_r = N_de_offset + f2 * 3;

//         // --- Assemble Hessian Blocks using the formula ---
//         // H_ij = w * [ f''(t) * (dt/dXi)^T * (dt/dXj) + f'(t) * (d^2t / (dXi dXj)) ]
//         // dt/dX = dAngle_un/dX - dAngle_de/dX
//         // d2t/dXi dXj = d2Angle_un/dXi dXj - d2Angle_de/dXi dXj

//         // Macro to compute and add a block (handles un/de splitting)
//         #define ADD_DIHEDRAL_HESS_BLOCK(ROW_VAR_PREFIX, COL_VAR_PREFIX, \
//                                        ROW_VAR_IDX, COL_VAR_IDX,        \
//                                        GRAD_UN_ROW, GRAD_UN_COL, HESS_UN_ROW_COL, \
//                                        GRAD_DE_ROW, GRAD_DE_COL, HESS_DE_ROW_COL) \
//             {                                                                    \
//                 Eigen::RowVector3d dt_dX_row = Eigen::RowVector3d::Zero();       \
//                 Eigen::RowVector3d dt_dX_col = Eigen::RowVector3d::Zero();       \
//                 Eigen::Matrix3d d2t_dXrow_dXcol = Eigen::Matrix3d::Zero();       \
//                 int global_row_idx = 0;                                          \
//                 int global_col_idx = 0;                                          \
//                                                                                  \
//                 /* Determine dt/dX_row and global_row_idx */                     \
//                 if (ROW_VAR_PREFIX == vAun_r) { dt_dX_row = deriv_un.grad_vA; global_row_idx = vAun_r; } \
//                 else if (ROW_VAR_PREFIX == vBun_r) { dt_dX_row = deriv_un.grad_vB; global_row_idx = vBun_r; } \
//                 else if (ROW_VAR_PREFIX == n1un_r) { dt_dX_row = deriv_un.grad_n1; global_row_idx = n1un_r; } \
//                 else if (ROW_VAR_PREFIX == n2un_r) { dt_dX_row = deriv_un.grad_n2; global_row_idx = n2un_r; } \
//                 else if (ROW_VAR_PREFIX == vAde_r) { dt_dX_row = -deriv_de.grad_vA; global_row_idx = vAde_r; } \
//                 else if (ROW_VAR_PREFIX == vBde_r) { dt_dX_row = -deriv_de.grad_vB; global_row_idx = vBde_r; } \
//                 else if (ROW_VAR_PREFIX == n1de_r) { dt_dX_row = -deriv_de.grad_n1; global_row_idx = n1de_r; } \
//                 else if (ROW_VAR_PREFIX == n2de_r) { dt_dX_row = -deriv_de.grad_n2; global_row_idx = n2de_r; } \
//                                                                                  \
//                 /* Determine dt/dX_col and global_col_idx */                     \
//                  if (COL_VAR_PREFIX == vAun_r) { dt_dX_col = deriv_un.grad_vA; global_col_idx = vAun_r; } \
//                 else if (COL_VAR_PREFIX == vBun_r) { dt_dX_col = deriv_un.grad_vB; global_col_idx = vBun_r; } \
//                 else if (COL_VAR_PREFIX == n1un_r) { dt_dX_col = deriv_un.grad_n1; global_col_idx = n1un_r; } \
//                 else if (COL_VAR_PREFIX == n2un_r) { dt_dX_col = deriv_un.grad_n2; global_col_idx = n2un_r; } \
//                 else if (COL_VAR_PREFIX == vAde_r) { dt_dX_col = -deriv_de.grad_vA; global_col_idx = vAde_r; } \
//                 else if (COL_VAR_PREFIX == vBde_r) { dt_dX_col = -deriv_de.grad_vB; global_col_idx = vBde_r; } \
//                 else if (COL_VAR_PREFIX == n1de_r) { dt_dX_col = -deriv_de.grad_n1; global_col_idx = n1de_r; } \
//                 else if (COL_VAR_PREFIX == n2de_r) { dt_dX_col = -deriv_de.grad_n2; global_col_idx = n2de_r; } \
//                                                                                  \
//                 /* Determine d2t/dXi dXj based on block type */                   \
//                 bool row_is_un = (global_row_idx < N_un_offset);                 \
//                 bool col_is_un = (global_col_idx < N_un_offset);                 \
//                 if (row_is_un && col_is_un) { d2t_dXrow_dXcol = HESS_UN_ROW_COL; } \
//                 else if (!row_is_un && !col_is_un) { d2t_dXrow_dXcol = -HESS_DE_ROW_COL; } \
//                 /* else d2t is zero for mixed blocks */                          \
//                                                                                  \
//                 /* Compute Hessian block */                                      \
//                 Eigen::Matrix3d hess_block = weight_angle_sub * (                \
//                     f_double_prime_t * (dt_dX_row.transpose() * dt_dX_col) +     \
//                     f_prime_t * d2t_dXrow_dXcol                                  \
//                 );                                                               \
//                 /* Add to triplets */                                             \
//                 if (hess_block.squaredNorm() > 1e-20) { /* Avoid adding tiny blocks */ \
//                      AddBlockToTriplets(T_thread[thread_id], global_row_idx, global_col_idx, hess_block); \
//                 }                                                               \
//             }

//         // --- Add all 64 blocks (8x8 variables) ---
//         // This is verbose, but ensures all pairs are covered.
//         // We pass the relevant H_.. terms from deriv_un and deriv_de

//         // Row: vAun
//         ADD_DIHEDRAL_HESS_BLOCK(vAun_r, vAun_r, 0, 0, deriv_un.grad_vA, deriv_un.grad_vA, deriv_un.H_vA_vA, deriv_de.grad_vA, deriv_de.grad_vA, deriv_de.H_vA_vA);
//         ADD_DIHEDRAL_HESS_BLOCK(vAun_r, vBun_r, 0, 0, deriv_un.grad_vA, deriv_un.grad_vB, deriv_un.H_vA_vB, deriv_de.grad_vA, deriv_de.grad_vB, deriv_de.H_vA_vB);
//         ADD_DIHEDRAL_HESS_BLOCK(vAun_r, n1un_r, 0, 0, deriv_un.grad_vA, deriv_un.grad_n1, deriv_un.H_vA_n1, deriv_de.grad_vA, deriv_de.grad_n1, deriv_de.H_vA_n1);
//         ADD_DIHEDRAL_HESS_BLOCK(vAun_r, n2un_r, 0, 0, deriv_un.grad_vA, deriv_un.grad_n2, deriv_un.H_vA_n2, deriv_de.grad_vA, deriv_de.grad_n2, deriv_de.H_vA_n2);
//         ADD_DIHEDRAL_HESS_BLOCK(vAun_r, vAde_r, 0, 0, deriv_un.grad_vA, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_vA, deriv_de.grad_vA, deriv_de.H_vA_vA); // Mixed block example
//         ADD_DIHEDRAL_HESS_BLOCK(vAun_r, vBde_r, 0, 0, deriv_un.grad_vA, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_vA, deriv_de.grad_vB, deriv_de.H_vA_vB);
//         ADD_DIHEDRAL_HESS_BLOCK(vAun_r, n1de_r, 0, 0, deriv_un.grad_vA, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_vA, deriv_de.grad_n1, deriv_de.H_vA_n1);
//         ADD_DIHEDRAL_HESS_BLOCK(vAun_r, n2de_r, 0, 0, deriv_un.grad_vA, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_vA, deriv_de.grad_n2, deriv_de.H_vA_n2);

//         // Row: vBun
//         ADD_DIHEDRAL_HESS_BLOCK(vBun_r, vAun_r, 0, 0, deriv_un.grad_vB, deriv_un.grad_vA, deriv_un.H_vA_vB.transpose(), deriv_de.grad_vB, deriv_de.grad_vA, deriv_de.H_vA_vB.transpose()); // Use transpose H_vA_vB
//         ADD_DIHEDRAL_HESS_BLOCK(vBun_r, vBun_r, 0, 0, deriv_un.grad_vB, deriv_un.grad_vB, deriv_un.H_vB_vB, deriv_de.grad_vB, deriv_de.grad_vB, deriv_de.H_vB_vB);
//         ADD_DIHEDRAL_HESS_BLOCK(vBun_r, n1un_r, 0, 0, deriv_un.grad_vB, deriv_un.grad_n1, deriv_un.H_vB_n1, deriv_de.grad_vB, deriv_de.grad_n1, deriv_de.H_vB_n1);
//         ADD_DIHEDRAL_HESS_BLOCK(vBun_r, n2un_r, 0, 0, deriv_un.grad_vB, deriv_un.grad_n2, deriv_un.H_vB_n2, deriv_de.grad_vB, deriv_de.grad_n2, deriv_de.H_vB_n2);
//         ADD_DIHEDRAL_HESS_BLOCK(vBun_r, vAde_r, 0, 0, deriv_un.grad_vB, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_vB, deriv_de.grad_vA, deriv_de.H_vA_vB.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(vBun_r, vBde_r, 0, 0, deriv_un.grad_vB, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_vB, deriv_de.grad_vB, deriv_de.H_vB_vB);
//         ADD_DIHEDRAL_HESS_BLOCK(vBun_r, n1de_r, 0, 0, deriv_un.grad_vB, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_vB, deriv_de.grad_n1, deriv_de.H_vB_n1);
//         ADD_DIHEDRAL_HESS_BLOCK(vBun_r, n2de_r, 0, 0, deriv_un.grad_vB, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_vB, deriv_de.grad_n2, deriv_de.H_vB_n2);

//         // Row: n1un
//         ADD_DIHEDRAL_HESS_BLOCK(n1un_r, vAun_r, 0, 0, deriv_un.grad_n1, deriv_un.grad_vA, deriv_un.H_vA_n1.transpose(), deriv_de.grad_n1, deriv_de.grad_vA, deriv_de.H_vA_n1.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(n1un_r, vBun_r, 0, 0, deriv_un.grad_n1, deriv_un.grad_vB, deriv_un.H_vB_n1.transpose(), deriv_de.grad_n1, deriv_de.grad_vB, deriv_de.H_vB_n1.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(n1un_r, n1un_r, 0, 0, deriv_un.grad_n1, deriv_un.grad_n1, deriv_un.H_n1_n1, deriv_de.grad_n1, deriv_de.grad_n1, deriv_de.H_n1_n1);
//         ADD_DIHEDRAL_HESS_BLOCK(n1un_r, n2un_r, 0, 0, deriv_un.grad_n1, deriv_un.grad_n2, deriv_un.H_n1_n2, deriv_de.grad_n1, deriv_de.grad_n2, deriv_de.H_n1_n2);
//         ADD_DIHEDRAL_HESS_BLOCK(n1un_r, vAde_r, 0, 0, deriv_un.grad_n1, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_n1, deriv_de.grad_vA, deriv_de.H_vA_n1.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(n1un_r, vBde_r, 0, 0, deriv_un.grad_n1, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_n1, deriv_de.grad_vB, deriv_de.H_vB_n1.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(n1un_r, n1de_r, 0, 0, deriv_un.grad_n1, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_n1, deriv_de.grad_n1, deriv_de.H_n1_n1);
//         ADD_DIHEDRAL_HESS_BLOCK(n1un_r, n2de_r, 0, 0, deriv_un.grad_n1, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_n1, deriv_de.grad_n2, deriv_de.H_n1_n2);

//         // Row: n2un
//         ADD_DIHEDRAL_HESS_BLOCK(n2un_r, vAun_r, 0, 0, deriv_un.grad_n2, deriv_un.grad_vA, deriv_un.H_vA_n2.transpose(), deriv_de.grad_n2, deriv_de.grad_vA, deriv_de.H_vA_n2.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(n2un_r, vBun_r, 0, 0, deriv_un.grad_n2, deriv_un.grad_vB, deriv_un.H_vB_n2.transpose(), deriv_de.grad_n2, deriv_de.grad_vB, deriv_de.H_vB_n2.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(n2un_r, n1un_r, 0, 0, deriv_un.grad_n2, deriv_un.grad_n1, deriv_un.H_n1_n2.transpose(), deriv_de.grad_n2, deriv_de.grad_n1, deriv_de.H_n1_n2.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(n2un_r, n2un_r, 0, 0, deriv_un.grad_n2, deriv_un.grad_n2, deriv_un.H_n2_n2, deriv_de.grad_n2, deriv_de.grad_n2, deriv_de.H_n2_n2);
//         ADD_DIHEDRAL_HESS_BLOCK(n2un_r, vAde_r, 0, 0, deriv_un.grad_n2, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_n2, deriv_de.grad_vA, deriv_de.H_vA_n2.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(n2un_r, vBde_r, 0, 0, deriv_un.grad_n2, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_n2, deriv_de.grad_vB, deriv_de.H_vB_n2.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(n2un_r, n1de_r, 0, 0, deriv_un.grad_n2, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_n2, deriv_de.grad_n1, deriv_de.H_n1_n2.transpose());
//         ADD_DIHEDRAL_HESS_BLOCK(n2un_r, n2de_r, 0, 0, deriv_un.grad_n2, Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), deriv_de.grad_n2, deriv_de.grad_n2, deriv_de.H_n2_n2);

//         // Row: vAde
//         ADD_DIHEDRAL_HESS_BLOCK(vAde_r, vAun_r, 0, 0, deriv_un.grad_vA, deriv_un.grad_vA, deriv_un.H_vA_vA, Eigen::RowVector3d::Zero(), -deriv_de.grad_vA, deriv_de.H_vA_vA); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(vAde_r, vBun_r, 0, 0, deriv_un.grad_vB, deriv_un.grad_vA, deriv_un.H_vA_vB.transpose(), Eigen::RowVector3d::Zero(), -deriv_de.grad_vA, deriv_de.H_vA_vB.transpose()); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(vAde_r, n1un_r, 0, 0, deriv_un.grad_n1, deriv_un.grad_vA, deriv_un.H_vA_n1.transpose(), Eigen::RowVector3d::Zero(), -deriv_de.grad_vA, deriv_de.H_vA_n1.transpose()); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(vAde_r, n2un_r, 0, 0, deriv_un.grad_n2, deriv_un.grad_vA, deriv_un.H_vA_n2.transpose(), Eigen::RowVector3d::Zero(), -deriv_de.grad_vA, deriv_de.H_vA_n2.transpose()); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(vAde_r, vAde_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vA, -deriv_de.grad_vA, deriv_de.H_vA_vA); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(vAde_r, vBde_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vA, -deriv_de.grad_vB, deriv_de.H_vA_vB); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(vAde_r, n1de_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vA, -deriv_de.grad_n1, deriv_de.H_vA_n1); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(vAde_r, n2de_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vA, -deriv_de.grad_n2, deriv_de.H_vA_n2); // De-De

//         // Row: vBde
//         ADD_DIHEDRAL_HESS_BLOCK(vBde_r, vAun_r, 0, 0, deriv_un.grad_vA, deriv_un.grad_vB, deriv_un.H_vA_vB, Eigen::RowVector3d::Zero(), -deriv_de.grad_vB, deriv_de.H_vA_vB); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(vBde_r, vBun_r, 0, 0, deriv_un.grad_vB, deriv_un.grad_vB, deriv_un.H_vB_vB, Eigen::RowVector3d::Zero(), -deriv_de.grad_vB, deriv_de.H_vB_vB); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(vBde_r, n1un_r, 0, 0, deriv_un.grad_n1, deriv_un.grad_vB, deriv_un.H_vB_n1.transpose(), Eigen::RowVector3d::Zero(), -deriv_de.grad_vB, deriv_de.H_vB_n1.transpose()); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(vBde_r, n2un_r, 0, 0, deriv_un.grad_n2, deriv_un.grad_vB, deriv_un.H_vB_n2.transpose(), Eigen::RowVector3d::Zero(), -deriv_de.grad_vB, deriv_de.H_vB_n2.transpose()); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(vBde_r, vAde_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vA, -deriv_de.grad_vB, deriv_de.H_vA_vB.transpose()); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(vBde_r, vBde_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vB, -deriv_de.grad_vB, deriv_de.H_vB_vB); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(vBde_r, n1de_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vB, -deriv_de.grad_n1, deriv_de.H_vB_n1); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(vBde_r, n2de_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vB, -deriv_de.grad_n2, deriv_de.H_vB_n2); // De-De

//         // Row: n1de
//         ADD_DIHEDRAL_HESS_BLOCK(n1de_r, vAun_r, 0, 0, deriv_un.grad_vA, deriv_un.grad_n1, deriv_un.H_vA_n1, Eigen::RowVector3d::Zero(), -deriv_de.grad_n1, deriv_de.H_vA_n1); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(n1de_r, vBun_r, 0, 0, deriv_un.grad_vB, deriv_un.grad_n1, deriv_un.H_vB_n1, Eigen::RowVector3d::Zero(), -deriv_de.grad_n1, deriv_de.H_vB_n1); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(n1de_r, n1un_r, 0, 0, deriv_un.grad_n1, deriv_un.grad_n1, deriv_un.H_n1_n1, Eigen::RowVector3d::Zero(), -deriv_de.grad_n1, deriv_de.H_n1_n1); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(n1de_r, n2un_r, 0, 0, deriv_un.grad_n2, deriv_un.grad_n1, deriv_un.H_n1_n2.transpose(), Eigen::RowVector3d::Zero(), -deriv_de.grad_n1, deriv_de.H_n1_n2.transpose()); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(n1de_r, vAde_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vA, -deriv_de.grad_n1, deriv_de.H_vA_n1.transpose()); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(n1de_r, vBde_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vB, -deriv_de.grad_n1, deriv_de.H_vB_n1.transpose()); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(n1de_r, n1de_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_n1, -deriv_de.grad_n1, deriv_de.H_n1_n1); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(n1de_r, n2de_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_n1, -deriv_de.grad_n2, deriv_de.H_n1_n2); // De-De

//         // Row: n2de
//         ADD_DIHEDRAL_HESS_BLOCK(n2de_r, vAun_r, 0, 0, deriv_un.grad_vA, deriv_un.grad_n2, deriv_un.H_vA_n2, Eigen::RowVector3d::Zero(), -deriv_de.grad_n2, deriv_de.H_vA_n2); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(n2de_r, vBun_r, 0, 0, deriv_un.grad_vB, deriv_un.grad_n2, deriv_un.H_vB_n2, Eigen::RowVector3d::Zero(), -deriv_de.grad_n2, deriv_de.H_vB_n2); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(n2de_r, n1un_r, 0, 0, deriv_un.grad_n1, deriv_un.grad_n2, deriv_un.H_n1_n2, Eigen::RowVector3d::Zero(), -deriv_de.grad_n2, deriv_de.H_n1_n2); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(n2de_r, n2un_r, 0, 0, deriv_un.grad_n2, deriv_un.grad_n2, deriv_un.H_n2_n2, Eigen::RowVector3d::Zero(), -deriv_de.grad_n2, deriv_de.H_n2_n2); // Mixed
//         ADD_DIHEDRAL_HESS_BLOCK(n2de_r, vAde_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vA, -deriv_de.grad_n2, deriv_de.H_vA_n2.transpose()); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(n2de_r, vBde_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_vB, -deriv_de.grad_n2, deriv_de.H_vB_n2.transpose()); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(n2de_r, n1de_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_n1, -deriv_de.grad_n2, deriv_de.H_n1_n2.transpose()); // De-De
//         ADD_DIHEDRAL_HESS_BLOCK(n2de_r, n2de_r, 0, 0, Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero(), Eigen::Matrix3d::Zero(), -deriv_de.grad_n2, -deriv_de.grad_n2, deriv_de.H_n2_n2); // De-De


//         #undef ADD_DIHEDRAL_HESS_BLOCK

//     } // End parallel loop over edges

//     // --- Part 2: Hessian of Normal Regularization Energy ---
//     #pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < nF; i++) { // Loop over faces
//         int thread_id = omp_get_thread_num();
//          if (thread_id < 0 || thread_id >= max_threads) thread_id = 0; // Safety check


//         // Vertex indices for face i
//         const int v1_idx = F_undeformed(i, 0);
//         const int v2_idx = F_undeformed(i, 1);
//         const int v3_idx = F_undeformed(i, 2);
//         const int f_idx = i; // Normal index

//         // Global row/col starts for this face's variables
//         int v1un_r = V_un_offset + v1_idx * 3;
//         int v2un_r = V_un_offset + v2_idx * 3;
//         int v3un_r = V_un_offset + v3_idx * 3;
//         int n_un_r = N_un_offset + f_idx * 3;
//         int v1de_r = V_de_offset + v1_idx * 3;
//         int v2de_r = V_de_offset + v2_idx * 3;
//         int v3de_r = V_de_offset + v3_idx * 3;
//         int n_de_r = N_de_offset + f_idx * 3;

//         Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

//         // --- Undeformed Normal Energy ---
//         // Energy = w * [ (N ⋅ (v2 - v1))² + (N ⋅ (v3 - v2))² + (N ⋅ (v1 - v3))² + (N ⋅ N - 1)² ]
//         {
//             Eigen::RowVector3d N = N_undeformed_opt.row(f_idx);
//             Eigen::RowVector3d v1 = V_undeformed.row(v1_idx);
//             Eigen::RowVector3d v2 = V_undeformed.row(v2_idx);
//             Eigen::RowVector3d v3 = V_undeformed.row(v3_idx);
//             Eigen::RowVector3d e12 = v2 - v1;
//             Eigen::RowVector3d e23 = v3 - v2;
//             Eigen::RowVector3d e31 = v1 - v3;
//             double d1 = N.dot(e12);
//             double d2 = N.dot(e23);
//             double d3 = N.dot(e31);
//             double Nsq = N.squaredNorm();
//             double d4 = Nsq - 1.0;

//             Eigen::Matrix3d NNT = N.transpose() * N; // N * N^T outer product

//             // Term 1: (N ⋅ e12)² -> H = 2 * grad^T * grad + 2 * val * Hess(val)
//             // grad(N.e12) = [e12, -N, N, 0]^T
//             // Hess(N.e12) involves blocks like d/dN(e12), d/dv1(-N), d/dv2(N)
//             // d/dN(e12)=0, d/dv1(-N)=0, d/dv2(N)=0
//             // d/dN(-N)=-I, d/dN(N)=I
//             // Mixed Hessians: d/dN(e12)=0, d/dv1(e12)=-I, d/dv2(e12)=I
//             // d/dv1(-N) = 0, d/dv2(N) = 0
//             Eigen::Matrix3d H1_NN = 2.0 * e12.transpose() * e12; // 2*gradN^T*gradN + 2*d1*Hess_NN(0)
//             Eigen::Matrix3d H1_v1v1 = 2.0 * NNT; // 2*gradV1^T*gradV1 + 2*d1*Hess_v1v1(0)
//             Eigen::Matrix3d H1_v2v2 = 2.0 * NNT; // 2*gradV2^T*gradV2 + 2*d1*Hess_v2v2(0)
//             Eigen::Matrix3d H1_v1v2 = -2.0 * NNT; // 2*gradV1^T*gradV2 + 2*d1*Hess_v1v2(0)
//             // Mixed blocks H = 2*gradA^T*gradB + 2*d1*Hess_AB
//             Eigen::Matrix3d H1_Nv1 = 2.0 * e12.transpose() * (-N) + 2.0 * d1 * (-I); // 2*gradN^T*gradV1 + 2*d1*Hess_Nv1
//             Eigen::Matrix3d H1_Nv2 = 2.0 * e12.transpose() * (N) + 2.0 * d1 * (I); // 2*gradN^T*gradV2 + 2*d1*Hess_Nv2


//             // Term 2: (N ⋅ e23)²
//             Eigen::Matrix3d H2_NN = 2.0 * e23.transpose() * e23;
//             Eigen::Matrix3d H2_v2v2 = 2.0 * NNT;
//             Eigen::Matrix3d H2_v3v3 = 2.0 * NNT;
//             Eigen::Matrix3d H2_v2v3 = -2.0 * NNT;
//             Eigen::Matrix3d H2_Nv2 = 2.0 * e23.transpose() * (-N) + 2.0 * d2 * (-I);
//             Eigen::Matrix3d H2_Nv3 = 2.0 * e23.transpose() * (N) + 2.0 * d2 * (I);


//             // Term 3: (N ⋅ e31)²
//             Eigen::Matrix3d H3_NN = 2.0 * e31.transpose() * e31;
//             Eigen::Matrix3d H3_v3v3 = 2.0 * NNT;
//             Eigen::Matrix3d H3_v1v1 = 2.0 * NNT;
//             Eigen::Matrix3d H3_v3v1 = -2.0 * NNT;
//             Eigen::Matrix3d H3_Nv3 = 2.0 * e31.transpose() * (-N) + 2.0 * d3 * (-I);
//             Eigen::Matrix3d H3_Nv1 = 2.0 * e31.transpose() * (N) + 2.0 * d3 * (I);

//             // Term 4: (N⋅N - 1)²
//             // grad = 2*(N.N-1) * 2N = 4*d4*N
//             // Hess = 4*grad(d4)*N^T + 4*d4*grad(N) = 4*(2N)*N^T + 4*d4*I
//             Eigen::Matrix3d H4_NN = 8.0 * NNT + 4.0 * d4 * I; // Hessian w.r.t N
//             // Other Hessians involving V are zero for this term.

//             // Combine Hessians (multiplied by weight_angle)
//             // Blocks involving N_un
//             AddBlockToTriplets(T_thread[thread_id], n_un_r, n_un_r, weight_angle * (H1_NN + H2_NN + H3_NN + H4_NN));
//             AddBlockToTriplets(T_thread[thread_id], n_un_r, v1un_r, weight_angle * (H1_Nv1 + H3_Nv1));
//             AddBlockToTriplets(T_thread[thread_id], n_un_r, v2un_r, weight_angle * (H1_Nv2 + H2_Nv2));
//             AddBlockToTriplets(T_thread[thread_id], n_un_r, v3un_r, weight_angle * (H2_Nv3 + H3_Nv3));
//             // Symmetric blocks N-V are transposes
//             AddBlockToTriplets(T_thread[thread_id], v1un_r, n_un_r, weight_angle * (H1_Nv1.transpose() + H3_Nv1.transpose()));
//             AddBlockToTriplets(T_thread[thread_id], v2un_r, n_un_r, weight_angle * (H1_Nv2.transpose() + H2_Nv2.transpose()));
//             AddBlockToTriplets(T_thread[thread_id], v3un_r, n_un_r, weight_angle * (H2_Nv3.transpose() + H3_Nv3.transpose()));


//             // Blocks involving V_un (diagonal)
//             AddBlockToTriplets(T_thread[thread_id], v1un_r, v1un_r, weight_angle * (H1_v1v1 + H3_v1v1));
//             AddBlockToTriplets(T_thread[thread_id], v2un_r, v2un_r, weight_angle * (H1_v2v2 + H2_v2v2));
//             AddBlockToTriplets(T_thread[thread_id], v3un_r, v3un_r, weight_angle * (H2_v3v3 + H3_v3v3));

//             // Blocks involving V_un (off-diagonal)
//             AddBlockToTriplets(T_thread[thread_id], v1un_r, v2un_r, weight_angle * (H1_v1v2));
//             AddBlockToTriplets(T_thread[thread_id], v2un_r, v1un_r, weight_angle * (H1_v1v2.transpose())); // Symmetric part
//             AddBlockToTriplets(T_thread[thread_id], v2un_r, v3un_r, weight_angle * (H2_v2v3));
//             AddBlockToTriplets(T_thread[thread_id], v3un_r, v2un_r, weight_angle * (H2_v2v3.transpose())); // Symmetric part
//             AddBlockToTriplets(T_thread[thread_id], v3un_r, v1un_r, weight_angle * (H3_v3v1));
//             AddBlockToTriplets(T_thread[thread_id], v1un_r, v3un_r, weight_angle * (H3_v3v1.transpose())); // Symmetric part
//         }

//         // --- Deformed Normal Energy ---
//         // Same structure, just use deformed variables V_de, N_de
//          {
//             Eigen::RowVector3d N = N_deformed_opt.row(f_idx);
//             Eigen::RowVector3d v1 = V_deformed.row(v1_idx);
//             Eigen::RowVector3d v2 = V_deformed.row(v2_idx);
//             Eigen::RowVector3d v3 = V_deformed.row(v3_idx);
//             Eigen::RowVector3d e12 = v2 - v1;
//             Eigen::RowVector3d e23 = v3 - v2;
//             Eigen::RowVector3d e31 = v1 - v3;
//             double d1 = N.dot(e12);
//             double d2 = N.dot(e23);
//             double d3 = N.dot(e31);
//             double Nsq = N.squaredNorm();
//             double d4 = Nsq - 1.0;

//             Eigen::Matrix3d NNT = N.transpose() * N;

//             // Term 1: (N ⋅ e12)²
//             Eigen::Matrix3d H1_NN = 2.0 * e12.transpose() * e12;
//             Eigen::Matrix3d H1_v1v1 = 2.0 * NNT;
//             Eigen::Matrix3d H1_v2v2 = 2.0 * NNT;
//             Eigen::Matrix3d H1_v1v2 = -2.0 * NNT;
//             Eigen::Matrix3d H1_Nv1 = 2.0 * e12.transpose() * (-N) + 2.0 * d1 * (-I);
//             Eigen::Matrix3d H1_Nv2 = 2.0 * e12.transpose() * (N) + 2.0 * d1 * (I);

//             // Term 2: (N ⋅ e23)²
//             Eigen::Matrix3d H2_NN = 2.0 * e23.transpose() * e23;
//             Eigen::Matrix3d H2_v2v2 = 2.0 * NNT;
//             Eigen::Matrix3d H2_v3v3 = 2.0 * NNT;
//             Eigen::Matrix3d H2_v2v3 = -2.0 * NNT;
//             Eigen::Matrix3d H2_Nv2 = 2.0 * e23.transpose() * (-N) + 2.0 * d2 * (-I);
//             Eigen::Matrix3d H2_Nv3 = 2.0 * e23.transpose() * (N) + 2.0 * d2 * (I);

//             // Term 3: (N ⋅ e31)²
//             Eigen::Matrix3d H3_NN = 2.0 * e31.transpose() * e31;
//             Eigen::Matrix3d H3_v3v3 = 2.0 * NNT;
//             Eigen::Matrix3d H3_v1v1 = 2.0 * NNT;
//             Eigen::Matrix3d H3_v3v1 = -2.0 * NNT;
//             Eigen::Matrix3d H3_Nv3 = 2.0 * e31.transpose() * (-N) + 2.0 * d3 * (-I);
//             Eigen::Matrix3d H3_Nv1 = 2.0 * e31.transpose() * (N) + 2.0 * d3 * (I);

//             // Term 4: (N⋅N - 1)²
//             Eigen::Matrix3d H4_NN = 8.0 * NNT + 4.0 * d4 * I;

//             // Combine Hessians. Check if weight applies here too. Original gradient had no weight. Assume weight=1 for now.
//             const double current_weight = 1.0; // Or weight_angle if it should apply here

//             // Blocks involving N_de
//             AddBlockToTriplets(T_thread[thread_id], n_de_r, n_de_r, current_weight*(H1_NN + H2_NN + H3_NN + H4_NN));
//             AddBlockToTriplets(T_thread[thread_id], n_de_r, v1de_r, current_weight*(H1_Nv1 + H3_Nv1));
//             AddBlockToTriplets(T_thread[thread_id], n_de_r, v2de_r, current_weight*(H1_Nv2 + H2_Nv2));
//             AddBlockToTriplets(T_thread[thread_id], n_de_r, v3de_r, current_weight*(H2_Nv3 + H3_Nv3));
//             // Symmetric blocks N-V
//             AddBlockToTriplets(T_thread[thread_id], v1de_r, n_de_r, current_weight*(H1_Nv1.transpose() + H3_Nv1.transpose()));
//             AddBlockToTriplets(T_thread[thread_id], v2de_r, n_de_r, current_weight*(H1_Nv2.transpose() + H2_Nv2.transpose()));
//             AddBlockToTriplets(T_thread[thread_id], v3de_r, n_de_r, current_weight*(H2_Nv3.transpose() + H3_Nv3.transpose()));

//             // Blocks involving V_de (diagonal)
//             AddBlockToTriplets(T_thread[thread_id], v1de_r, v1de_r, current_weight*(H1_v1v1 + H3_v1v1));
//             AddBlockToTriplets(T_thread[thread_id], v2de_r, v2de_r, current_weight*(H1_v2v2 + H2_v2v2));
//             AddBlockToTriplets(T_thread[thread_id], v3de_r, v3de_r, current_weight*(H2_v3v3 + H3_v3v3));

//             // Blocks involving V_de (off-diagonal)
//             AddBlockToTriplets(T_thread[thread_id], v1de_r, v2de_r, current_weight*(H1_v1v2));
//             AddBlockToTriplets(T_thread[thread_id], v2de_r, v1de_r, current_weight*(H1_v1v2.transpose())); // Symmetric part
//             AddBlockToTriplets(T_thread[thread_id], v2de_r, v3de_r, current_weight*(H2_v2v3));
//             AddBlockToTriplets(T_thread[thread_id], v3de_r, v2de_r, current_weight*(H2_v2v3.transpose())); // Symmetric part
//             AddBlockToTriplets(T_thread[thread_id], v3de_r, v1de_r, current_weight*(H3_v3v1));
//             AddBlockToTriplets(T_thread[thread_id], v1de_r, v3de_r, current_weight*(H3_v3v1.transpose())); // Symmetric part
//          }

//     } // End parallel loop over faces


//     // --- Merge Triplets ---
//     triplets.clear();
//     size_t total_triplet_count = 0;
//     for(int i=0; i<max_threads; ++i) {
//         total_triplet_count += T_thread[i].size();
//     }
//     triplets.reserve(total_triplet_count);
//     // std::cout << "Actual triplets generated: " << total_triplet_count << std::endl;

//     for(int i=0; i<max_threads; ++i) {
//         // Move triplets if possible (C++11) to avoid copying large amounts of data
//         triplets.insert(triplets.end(), std::make_move_iterator(T_thread[i].begin()), std::make_move_iterator(T_thread[i].end()));
//         // Fallback for older compilers or if move is not desired:
//         // triplets.insert(triplets.end(), T_thread[i].begin(), T_thread[i].end());
//     }

//     // Optional: Update delta (as in original code) - Seems unrelated to Hessian computation itself.
//     // if (meshes.iteration%1000 == 0 && meshes.delta > 1e-6) {
//     //     meshes.delta *= 0.1;
//     //     meshes.weight_angle_sub *= 0.8;
//     // }
//     // std::cout << "delta (in Hessian func): " << meshes.delta << std::endl;
// }




// void Compute_Angle_sub(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &N, int f1, int f2,
//     const int v1, const int v2, double &angle) {
//     Eigen::Vector3d n1 = N.row(f1);
//     Eigen::Vector3d n2 = N.row(f2);
//     angle = atan2(n1.cross(n2).dot((V.row(v2) - V.row(v1)).normalized()), n1.dot(n2));
// }



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
            // double angle_un, angle_de;
            // Compute_Angle_sub(V_undeformed, F_undeformed, N_undeformed_opt, f1, f2, v1, v2, angle_un);
            // Compute_Angle_sub(V_deformed, F_deformed, N_deformed_opt, f1, f2, v1, v2, angle_de);
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
            // std::cout << "grad: " << grad.transpose() << std::endl;
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
        // std::cout << "grad: " << grad.transpose() << std::endl;
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
        // std::cout << "grad: " << grad.transpose() << std::endl;
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
    meshes.Quad_Angle_C = 0.0;
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
            // double angle_un, angle_de;
            // Compute_Angle_sub(V_undeformed, F_undeformed, N_undeformed_opt, f1, f2, v1, v2, angle_un);
            // Compute_Angle_sub(V_deformed, F_deformed, N_deformed_opt, f1, f2, v1, v2, angle_de);
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
            // std::cout << "grad: " << grad.transpose() << std::endl;
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