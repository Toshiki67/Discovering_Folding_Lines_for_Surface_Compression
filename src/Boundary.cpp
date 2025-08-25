//
// Created by 青木俊樹 on 4/11/24.
//

#include "Boundary.h"
#include <igl/boundary_loop.h>
#include <igl/parallel_for.h>
#include <omp.h>


void closest_point_line(const Eigen::Vector3d &p, const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                        Eigen::Vector3d &closest, double &distance) {
    Eigen::Vector3d v1 = p - p1;
    Eigen::Vector3d v2 = p2 - p1;
    double t = v1.dot(v2) / v2.dot(v2);
    t = std::max(0.0, std::min(1.0, t));
    closest = p1 + t * v2;
    distance = (closest - p).norm();
}

void initializeBoundary(Meshes &meshes) {
    igl::boundary_loop(meshes.F_refer, meshes.boundary_loop);
    std::cout << "Num of boundary loop: " << meshes.boundary_loop.size() << std::endl;
    std::cout << "F_undeformed.rows(): " << meshes.F_undeformed.rows() << std::endl;

    std::vector<std::vector<int>> boundary_loop_undeformed;
    igl::boundary_loop(meshes.F_undeformed, boundary_loop_undeformed);
    std::cout << "Num of boundary loop: " << meshes.boundary_loop.size() << std::endl;
    igl::parallel_for(boundary_loop_undeformed.size(), [&](int i) {
        std::cout << "i = " << i << std::endl;
        int index = boundary_loop_undeformed[i][0];
        // closest point on the boundary polyline
        Eigen::Vector3d p = meshes.V_undeformed.row(index);
        double closest_distance = 1e10;
        // closest point on the surface
        int minimum_j = -1;
        for (int j = 0; j < meshes.boundary_loop.size(); j++) {
            for (int k = 0; k < meshes.boundary_loop[j].size(); k++) {
                Eigen::Vector3d p1 = meshes.V_refer.row(meshes.boundary_loop[j][k]);
                Eigen::Vector3d p2 = meshes.V_refer.row(meshes.boundary_loop[j][(k + 1) % meshes.boundary_loop[j].size()]);
                // compute closest distance line p1 p2, point p
                Eigen::Vector3d closest;
                double distance;
                closest_point_line(p, p1, p2, closest, distance);
                if (distance < closest_distance) {
                    closest_distance = distance;
                    minimum_j = j;
                }
            }
        }
        for (int j = 0; j < boundary_loop_undeformed[i].size(); j++) {
            index = boundary_loop_undeformed[i][j];
            meshes.boundary_pair.push_back(std::make_pair(index, minimum_j));
        }
    },10000);
    std::cout << "Num of boundary: " << meshes.boundary_pair.size() << std::endl;
}

void Compute_Boundary_Constraints(Meshes &meshes) {
    meshes.boundary_closest_point.assign(meshes.boundary_pair.size(), std::make_pair(0, Eigen::Vector3d(0, 0, 0)));

    igl::parallel_for(meshes.boundary_pair.size(), [&](int i) {
        int index = meshes.boundary_pair[i].first;
        int index_surface = meshes.boundary_loop[meshes.boundary_pair[i].second][i];
        // closest point on the boundary polyline
        Eigen::Vector3d p = meshes.V_undeformed.row(index);
        double closest_distance = 1e10;
        // closest point on the surface
        Eigen::Vector3d closest_point;
        for (int j = 0; j < meshes.boundary_loop[meshes.boundary_pair[i].second].size(); j++) {
            Eigen::Vector3d p1 = meshes.V_refer.row(meshes.boundary_loop[meshes.boundary_pair[i].second][j]);
            Eigen::Vector3d p2 = meshes.V_refer.row(meshes.boundary_loop[meshes.boundary_pair[i].second][(j + 1) % meshes.boundary_loop[meshes.boundary_pair[i].second].size()]);
            // compute closest distance line p1 p2, point p
            Eigen::Vector3d closest;
            double distance;
            closest_point_line(p, p1, p2, closest, distance);
            if (distance < closest_distance) {
                closest_distance = distance;
                closest_point = closest;
            }
        }
        meshes.C(i + meshes.current_num) = (p-closest_point).norm();
        meshes.boundary_closest_point[i] = std::make_pair(index, closest_point);
    },10000);
    meshes.current_num += meshes.boundary_pair.size();
}



void Compute_derivatives_Boundary_Constraints(Meshes &meshes) {
    std::vector<Eigen::Triplet<double>> &tripletList = meshes.tripletList;
    igl::parallel_for(meshes.boundary_pair.size(), [&](int i) {
        int index = meshes.boundary_pair[i].first;
        Eigen::Vector3d p = meshes.V_undeformed.row(index);
        Eigen::Vector3d closest_point = meshes.boundary_closest_point[i].second;
        Eigen::Vector3d grad = (p - closest_point);
        double C_i = meshes.C(i + meshes.current_num);
        if (C_i < 1e-10) {
            tripletList[3 * i + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, index * 3, 0.0);
            tripletList[3 * i + 1 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, index * 3 + 1, 0.0);
            tripletList[3 * i + 2 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, index * 3 + 2, 0.0);
        } else{
            tripletList[3 * i + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, index * 3, grad(0)/C_i);
            tripletList[3 * i + 1 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, index * 3 + 1, grad(1)/C_i);
            tripletList[3 * i + 2 + meshes.current_triplet_num] = Eigen::Triplet<double>(i + meshes.current_num, index * 3 + 2, grad(2)/C_i);
        }
    },10000);
    meshes.current_triplet_num += meshes.boundary_pair.size() * 3;
    meshes.current_num += meshes.boundary_pair.size();
}


void Compute_Quad_Boundary_Constraints(Meshes &meshes) {
    meshes.boundary_closest_point.assign(meshes.boundary_pair.size(), std::make_pair(0, Eigen::Vector3d(0, 0, 0)));
    meshes.Quad_Boundary_C = 0.0;

    // igl::parallel_for(meshes.boundary_pair.size(), [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < meshes.boundary_pair.size(); i++) {
        int index = meshes.boundary_pair[i].first;
        int index_surface = meshes.boundary_loop[meshes.boundary_pair[i].second][i];
        // closest point on the boundary polyline
        Eigen::Vector3d p = meshes.V_undeformed.row(index);
        double closest_distance = 1e10;
        // closest point on the surface
        Eigen::Vector3d closest_point;
        for (int j = 0; j < meshes.boundary_loop[meshes.boundary_pair[i].second].size(); j++) {
            Eigen::Vector3d p1 = meshes.V_refer.row(meshes.boundary_loop[meshes.boundary_pair[i].second][j]);
            Eigen::Vector3d p2 = meshes.V_refer.row(meshes.boundary_loop[meshes.boundary_pair[i].second][(j + 1) % meshes.boundary_loop[meshes.boundary_pair[i].second].size()]);
            // compute closest distance line p1 p2, point p
            Eigen::Vector3d closest;
            double distance;
            closest_point_line(p, p1, p2, closest, distance);
            if (distance < closest_distance) {
                closest_distance = distance;
                closest_point = closest;
            }
        }
        #pragma omp atomic
        meshes.Quad_Boundary_C += (p-closest_point).norm();
        meshes.boundary_closest_point[i] = std::make_pair(index, closest_point);
    }
    // },10000);
}


void Compute_Quad_derivatives_Boundary_Constraints(Meshes &meshes) {
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXd &V_deform = meshes.V_deformed;
    Eigen::MatrixXd &Quad_Boundary_grad = meshes.Quad_Boundary_grad;
    Quad_Boundary_grad = Eigen::MatrixXd::Zero(V_undeformed.rows() + V_deform.rows(), 3);
    // igl::parallel_for(meshes.boundary_pair.size(), [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < meshes.boundary_pair.size(); i++) {
        int index = meshes.boundary_pair[i].first;
        Eigen::Vector3d p = meshes.V_undeformed.row(index);
        Eigen::Vector3d closest_point = meshes.boundary_closest_point[i].second;
        Eigen::Vector3d grad = (p - closest_point);
        double c = grad.norm();
        if (c < 1e-10)
            Quad_Boundary_grad.row(index) = Eigen::Vector3d(0, 0, 0);
        else
            Quad_Boundary_grad.row(index) = grad/c;
    }
    // },10000);
}

void Compute_Newton_Boundary_Constraints(Meshes &meshes) {
    meshes.boundary_closest_point.assign(meshes.boundary_pair.size(), std::make_pair(0, Eigen::Vector3d(0, 0, 0)));
    meshes.Quad_Boundary_C = 0.0;
    meshes.C_Boundary = Eigen::VectorXd::Zero(meshes.V_undeformed.rows() * 3 + meshes.V_deformed.rows() * 3);
    meshes.energy_Boundary = 0.0;
    // igl::parallel_for(meshes.boundary_pair.size(), [&](int i) {
    #pragma omp parallel for
    for (int i = 0; i < meshes.boundary_pair.size(); i++) {
        int index = meshes.boundary_pair[i].first;
        int index_surface = meshes.boundary_loop[meshes.boundary_pair[i].second][i];
        // closest point on the boundary polyline
        Eigen::Vector3d p = meshes.V_undeformed.row(index);
        double closest_distance = 1e10;
        // closest point on the surface
        Eigen::Vector3d closest_point;
        for (int j = 0; j < meshes.boundary_loop[meshes.boundary_pair[i].second].size(); j++) {
            Eigen::Vector3d p1 = meshes.V_refer.row(meshes.boundary_loop[meshes.boundary_pair[i].second][j]);
            Eigen::Vector3d p2 = meshes.V_refer.row(meshes.boundary_loop[meshes.boundary_pair[i].second][(j + 1) % meshes.boundary_loop[meshes.boundary_pair[i].second].size()]);
            // compute closest distance line p1 p2, point p
            Eigen::Vector3d closest;
            double distance;
            closest_point_line(p, p1, p2, closest, distance);
            if (distance < closest_distance) {
                closest_distance = distance;
                closest_point = closest;
            }
        }
        // #pragma omp atomic
        // meshes.Quad_Boundary_C += (p-closest_point).norm();
        double c = (p-closest_point).squaredNorm();
        #pragma omp atomic
        meshes.energy_Boundary += c * meshes.weight_boundary;
        for (int j = 0; j < 3; j++) {
            meshes.C_Boundary(index * 3 + j) = meshes.weight_boundary * 2 * (p(j) - closest_point(j));
        }
        meshes.boundary_closest_point[i] = std::make_pair(index, closest_point);
    }
    // },10000);
}


void Compute_Newton_derivatives_Boundary_Constraints(Meshes &meshes) {
    Eigen::SparseMatrix<double> &dC_dV = meshes.dC_Boundary_dV;
    std::vector<Eigen::Triplet<double>> tripletList;
    dC_dV = Eigen::SparseMatrix<double>(meshes.boundary_pair.size()*3, meshes.V_undeformed.rows() * 3 + meshes.V_deformed.rows() * 3);
    // initialize triplet list as 4*meshes.boundary_pair.size() * 3
    tripletList = std::vector<Eigen::Triplet<double>>(meshes.boundary_pair.size() * 3);
    #pragma omp parallel for
    for (int i = 0; i < meshes.boundary_pair.size(); i++) {
        int index = meshes.boundary_pair[i].first;
        Eigen::Vector3d p = meshes.V_undeformed.row(index);
        Eigen::Vector3d closest_point = meshes.boundary_closest_point[i].second;
        Eigen::Vector3d grad = (p - closest_point);
        double c = grad.squaredNorm();
        for (int j = 0; j < 3; j++) {
            tripletList[3 * i + j] = Eigen::Triplet<double>(index * 3 + j, index * 3 + j, meshes.weight_boundary * 2);
        }
    }
    // dC_dV.setFromTriplets(tripletList.begin(), tripletList.end());
    meshes.tripletList_Boundary = tripletList;
    meshes.current_triplet_num += meshes.boundary_pair.size() * 3;
}
