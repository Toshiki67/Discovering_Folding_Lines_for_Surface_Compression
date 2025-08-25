#ifndef FREEFORM_CONNECTFOLD_H
#define FREEFORM_CONNECTFOLD_H
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <set>
#include "data.h"

std::pair<int, int> make_edge(int v1, int v2);
void expand_line(
    int current, 
    std::unordered_map<int, std::unordered_set<int>>& adjacency_list, 
    std::set<std::pair<int, int>>& visited_edges, 
    std::deque<int>& line, 
    bool forward);

void find_line(
    int start, 
    int end,
    std::unordered_map<int, std::unordered_set<int>>& adjacency_list, 
    std::set<std::pair<int, int>>& visited_edges, 
    std::vector<int>& line);

void Connect_Fold(Meshes &meshes, std::vector<std::vector<int>> &lines);


void Compute_Angle_MV(const Eigen::Vector3d n1, const Eigen::Vector3d n2,
    const Eigen::Vector3d v1, const Eigen::Vector3d v2, double &angle);

void normal(Eigen::Vector3d &n, const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const Eigen::Vector3d &v3);

void Verify_MV(Meshes &meshes, std::vector<std::vector<int>> &lines, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &V_un,  Eigen::MatrixXi &F_un, std::vector<int> &MV);
double calculate_zigzag_metric(const std::vector<int>& line_indices, const Eigen::MatrixXd& V, double sharp_angle_threshold_rad);

struct ConnectionCandidate {
    int target_line_idx = -1;
    int target_side = -1;       // 0 for head, 1 for tail of target line
    double dot_product = -2.0;  // Initialize to a value lower than any valid dot product
    int connection_node_idx = -1;

    bool isValid() const {
        return target_line_idx != -1;
    }
};

struct PotentialMergeOperation {
    int line_idx1 = -1;
    int side1 = -1; // 0 for head, 1 for tail of line_idx1
    int line_idx2 = -1;
    int side2 = -1; // 0 for head, 1 for tail of line_idx2
    // double quality_metric = -2.0; // Optional: e.g., dot_product, if needed for tie-breaking

    // For std::sort and std::unique (if using them with custom comparators is problematic, provide lambdas)
    bool operator<(const PotentialMergeOperation& other) const {
        if (line_idx1 != other.line_idx1) return line_idx1 < other.line_idx1;
        if (line_idx2 != other.line_idx2) return line_idx2 < other.line_idx2;
        if (side1 != other.side1) return side1 < other.side1;
        return side2 < other.side2;
    }

    bool operator==(const PotentialMergeOperation& other) const {
        return line_idx1 == other.line_idx1 &&
               line_idx2 == other.line_idx2 &&
               side1 == other.side1 &&
               side2 == other.side2;
    }
};




Eigen::Vector3d get_tangent_vector(
    const Eigen::MatrixXd& V,
    const std::vector<int>& line_nodes,
    int side // 0 for head, 1 for tail
);

void Merge_Mutually_Best_Collinear_Connections(
    const Eigen::MatrixXd& V,
    std::vector<std::vector<int>>& lines,
    std::vector<int>& MV
);

void Merge_Collinear_Connections(const Eigen::MatrixXd& V, std::vector<std::vector<int>>& lines,
    std::vector<int>& MV);

void Connect_Close_MV(Meshes &meshes, std::vector<std::vector<int>> &lines, std::vector<int> &MV);

#endif FREEFORM_CONNECTFOLD_H