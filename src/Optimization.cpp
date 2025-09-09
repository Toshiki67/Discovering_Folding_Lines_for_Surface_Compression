#include "Optimization.h"
#include "Constraint.h"
#include "Adam.h"
#include <Eigen/QR>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SVD>
#include <igl/edges.h>
#include <igl/writeOBJ.h>

void matrix_to_vector(const Eigen::MatrixXd &V, Eigen::VectorXd &x) {
    Eigen::MatrixXd V_t = V.transpose();
    x = Eigen::VectorXd(Eigen::Map<Eigen::VectorXd> (V_t.data(), V_t.size()));
}

void two_matrix_to_vector(const Eigen::MatrixXd &V1, const Eigen::MatrixXd &V2, Eigen::VectorXd &x) {
    Eigen::VectorXd x1;
    matrix_to_vector(V1, x1);
    Eigen::VectorXd x2;
    matrix_to_vector(V2, x2);
    x.resize(x1.size() + x2.size());
    x << x1, x2;
}



void Minimize(Meshes &meshes, int num_iterations) {
    for (int i = 0; i < num_iterations; i++) {
        // std::cout << "///////////////////////////////////////////////" << std::endl;
        meshes.iteration = meshes.iteration + 1;
        Compute_Quad_Constraints(meshes);
        Compute_Quad_derivatives(meshes);
        Meshes meshes_new = meshes.create_meshes(meshes);
        Eigen::MatrixXd V_un_V_de = Eigen::MatrixXd::Zero(meshes.V_undeformed.rows() + meshes.V_deformed.rows(), 3);
        V_un_V_de.block(0, 0, meshes.V_undeformed.rows(), 3) = meshes.V_undeformed;
        V_un_V_de.block(meshes.V_undeformed.rows(), 0, meshes.V_deformed.rows(), 3) = meshes.V_deformed;
        Eigen::MatrixXd V_un_V_de_copy = V_un_V_de;
        meshes.adam.update(V_un_V_de, meshes.Quad_grad);
        Eigen::MatrixXd V_un_V_de_diff = V_un_V_de - V_un_V_de_copy;
        meshes.V_undeformed = V_un_V_de.block(0, 0, meshes.V_undeformed.rows(), 3);
        meshes.V_deformed = V_un_V_de.block(meshes.V_undeformed.rows(), 0, meshes.V_deformed.rows(), 3);

        Eigen::MatrixXd N_un_N_de = Eigen::MatrixXd::Zero(meshes.N_undeformed_opt.rows() + meshes.N_deformed_opt.rows(), 3);
        N_un_N_de.block(0, 0, meshes.N_undeformed_opt.rows(), 3) = meshes.N_undeformed_opt;
        N_un_N_de.block(meshes.N_undeformed_opt.rows(), 0, meshes.N_deformed_opt.rows(), 3) = meshes.N_deformed_opt;
        Eigen::MatrixXd N_un_N_de_copy = N_un_N_de;
        meshes.adam_n.update(N_un_N_de, meshes.Quad_N_grad);
        Eigen::MatrixXd N_un_N_de_diff = N_un_N_de - N_un_N_de_copy;
        meshes.N_undeformed_opt = N_un_N_de.block(0, 0, meshes.N_undeformed_opt.rows(), 3);
        meshes.N_deformed_opt = N_un_N_de.block(meshes.N_undeformed_opt.rows(), 0, meshes.N_deformed_opt.rows(), 3);

      
    }
}


void Newton(Meshes &meshes, int num_iterations) {
    for (int i = 0; i < num_iterations; i++) {
        std::cout << "///////////////////////////////////////////////" << std::endl;
        meshes.iteration = meshes.iteration + 1;
        Compute_Newton_Constraints(meshes);
        std::cout << "C.norm() = " << meshes.C_all.norm() << std::endl;
        std::cout << "meshes_optimized.energy_all = " << meshes.energy_all << std::endl;
        Compute_Newton_derivatives(meshes);
        std::cout << "dC_dV:" << meshes.dC_dV_all.rows() << " " << meshes.dC_dV_all.cols() << std::endl;
        // using PCG solver
        Eigen::setNbThreads(10);
        double lambda = 10.0;
        // max iterations = 1000
        for (int k = 0; k < 1; k++) {
            Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
            Eigen::SparseMatrix<double> dC_dV = meshes.dC_dV_all;
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_spd;
            solver_spd.compute(dC_dV);
            
            auto D = solver_spd.vectorD();
            if((D.array() > 0).all()) {
                std::cout << "SPD" << std::endl;
            } else {
                std::cout << "not SPD" << std::endl;
            }
            


            solver.compute(dC_dV);
            solver.setMaxIterations(1000);
            // Eigen::VectorXd dx = solver.solve(-meshes.C_all);
            Eigen::VectorXd dx = -meshes.C_all;
            // Eigen::VectorXd dx = -meshes.C_all;
            std::cout << "finish solve" << std::endl;
            // line search
            double lr = 1.0;
            double alpha = 0.25;
            for (int j = 0; j < 20; j++) {
                std::cout << "--- Newton step ---" << std::endl;
                std::cout << "lr = " << lr << std::endl;
                Meshes meshes_optimized = meshes;
                Eigen::MatrixXd V_update_un_matrix_sub = dx.block(0, 0, meshes.V_undeformed.rows()*3, 1);
                V_update_un_matrix_sub.resize(3, meshes.V_undeformed.rows());
                Eigen::MatrixXd V_update_un_matrix = V_update_un_matrix_sub.transpose();
                meshes_optimized.V_undeformed += V_update_un_matrix;
                Eigen::MatrixXd V_update_de_matrix_sub = dx.block(meshes.V_undeformed.rows()*3, 0, meshes.V_deformed.rows()*3, 1);
                V_update_de_matrix_sub.resize(3, meshes.V_deformed.rows());
                Eigen::MatrixXd V_update_de_matrix = V_update_de_matrix_sub.transpose();
                meshes_optimized.V_deformed += V_update_de_matrix;
                Compute_Newton_Constraints(meshes_optimized);
                std::cout << "meshes_optimized.C.norm() = " << meshes_optimized.C_all.norm() << std::endl;
                std::cout << "meshes_optimized.energy_all = " << meshes_optimized.energy_all << std::endl;
                if (meshes_optimized.energy_all < meshes.energy_all || j == 19) {
                    meshes.V_undeformed += V_update_un_matrix;
                    meshes.V_deformed += V_update_de_matrix;
                    // meshes.N_undeformed_opt += N_update_un_matrix;
                    // meshes.N_deformed_opt += N_update_de_matrix;
                    meshes.C_all = meshes_optimized.C_all;
                    break;
                }
                dx = dx * alpha;
                lr = lr * alpha;
            }
        }

    }
}