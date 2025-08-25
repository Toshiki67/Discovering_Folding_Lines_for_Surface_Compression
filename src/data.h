#ifndef FREEFORM_DATA_H
#define FREEFORM_DATA_H
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "Adam.h"

class Meshes {
    public:
        Meshes create_meshes(const Meshes &meshes) {
            Meshes new_meshes = meshes;
            return new_meshes;
        }

        std::vector<std::vector<int>> A;

        Eigen::MatrixXd V_deformed, V_undeformed, V_refer, V_deformed_target;
        Eigen::MatrixXi F_deformed, F_undeformed, F_refer;

        Eigen::MatrixXd Vel;
        Eigen::MatrixXd V_deformed_pre;

        std::vector<std::vector<int>> adjacency_list;

        Eigen::MatrixXd D1d, D2d;
        Eigen::VectorXd restShapeArea;

        std::vector<std::pair<int, int>> selected_edges;
        
        Eigen::SparseMatrix<double> L, L2;

        Eigen::MatrixXd Velocities_deformed, Velocities_undeformed;

        Eigen::VectorXi EMAP;
        Eigen::MatrixXi E, uE;
        std::vector<std::vector<int>> uE2E;
        Eigen::MatrixXi EF;
        Eigen::MatrixXi EI;

        Eigen::MatrixXd N_refer;
        Eigen::MatrixXd N_deformed;
        Eigen::MatrixXd N_undeformed;

        Eigen::MatrixXd N_deformed_opt;
        Eigen::MatrixXd N_undeformed_opt;

        Eigen::VectorXd sqrD;
        Eigen::VectorXi I;
        Eigen::MatrixXd Cp;

        Eigen::VectorXi E_line;


        std::vector<std::vector<int>> boundary_loop;
        std::vector<std::pair<int, int>> boundary_pair;
        std::vector<std::pair<int, Eigen::Vector3d>> boundary_closest_point;

        Eigen::VectorXd C;
        Eigen::SparseMatrix<double> dC_dV;
        std::vector<Eigen::Triplet<double>> tripletList;
        int num_constraints;
        int boundary_num;
        int current_num = 0;
        // int current_triplet_num = 0;
        double threshold = M_PI / 12;

        std::vector<std::vector<int>> self_collision_pairs;

        double p_depth = 1;//1e-1;

        double Quad_Closeness_C = 0.0;
        Eigen::VectorXd Quad_Closeness_Vector;
        Eigen::MatrixXd Quad_Closeness_grad;

        double Quad_Isometry_C = 0.0;
        Eigen::VectorXd Quad_Isometry_Vector;
        Eigen::MatrixXd Quad_Isometry_grad;

        double Quad_Boundary_C = 0.0;
        Eigen::VectorXd Quad_Boundary_Vector;
        Eigen::MatrixXd Quad_Boundary_grad;

        double Quad_Symmetric_C = 0.0;
        Eigen::VectorXd Quad_Symmetric_Vector;
        Eigen::MatrixXd Quad_Symmetric_grad;

        double Quad_Angle_C = 0.0;
        Eigen::VectorXd Quad_Angle_Vector;
        Eigen::MatrixXd Quad_Angle_grad;

        double Quad_Angle_sub_C = 0.0;
        Eigen::VectorXd Quad_Angle_sub_Vector;
        Eigen::MatrixXd Quad_Angle_sub_grad;
        Eigen::MatrixXd Quad_Angle_sub_N_grad;

        double Quad_Uni_C = 0.0;
        Eigen::VectorXd Quad_Uniform_Vector;
        Eigen::MatrixXd Quad_Uniform_grad;

        double Quad_ICP_C = 0.0;
        Eigen::VectorXd Quad_ICP_Vector;
        Eigen::MatrixXd Quad_ICP_grad;


        double Quad_C = 0.0;
        

        double weight_isometry = 800.0;
        // double weight_closeness = 1.0;
        double weight_closeness = 50;
        double weight_boundary = weight_isometry;
        double weight_symmetric = 0.1;//10;//0.1;
        double weight_angle = 40;//6400;//40;
        // double weight_angle_sub = 0.8;
        double weight_angle_sub = 0.8;
        // double weight_uniform = 1e-2 * 4.0;
        double weight_uniform = 1e-1 * 4.0;
        // double weight_angle = 0.0;
        // double weight_angle_sub = 0.0;
        // double weight_uniform = 0.0;
        // double weight_icp = 0.02;

        Eigen::MatrixXd Quad_grad;
        Eigen::MatrixXd Quad_N_grad;

        double delta = 1e-3;

        AdamOptimizer adam{1e-2 * 1, 0.5, 0.7, 1e-8};
        AdamOptimizer adam_n{1e-2 * 1, 0.5, 0.7, 1e-8};
        // AdamOptimizer adam{0.01, 0.9, 0.999, 1e-8};
        // AdamOptimizer adam_n{0.01, 0.9, 0.999, 1e-8};
        int iteration = 0;


        int current_triplet_num = 0;
        Eigen::VectorXd C_all;
        double energy_all = 0.0;
        Eigen::SparseMatrix<double> dC_dV_all;
        Eigen::VectorXd C_Isometry;
        Eigen::SparseMatrix<double> dC_Isometry_dV;
        std::vector<Eigen::Triplet<double>> tripletList_Isometry;
        double energy_Isometry = 0.0;
        Eigen::VectorXd C_Boundary;
        Eigen::SparseMatrix<double> dC_Boundary_dV;
        std::vector<Eigen::Triplet<double>> tripletList_Boundary;
        double energy_Boundary = 0.0;
        Eigen::VectorXd C_Closeness;
        Eigen::SparseMatrix<double> dC_Closeness_dV;
        std::vector<Eigen::Triplet<double>> tripletList_Closeness;
        double energy_Closeness = 0.0;
        Eigen::VectorXd C_Angle;
        Eigen::SparseMatrix<double> dC_Angle_dV;
        std::vector<Eigen::Triplet<double>> tripletList_Angle;
        Eigen::VectorXd C_Angle_N;
        double energy_Angle = 0.0;

        std::vector<Eigen::Triplet<double>> tripletList_Symmetric;
        Eigen::VectorXd C_Symmetric;
        double energy_Symmetric = 0.0;



};

#endif FREEFORM_DATA_H