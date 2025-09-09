#include "ConnectFold.h"
#include <igl/point_mesh_squared_distance.h>
#include <igl/per_face_normals.h>
#include <igl/parallel_for.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/per_face_normals.h>
#include <igl/barycenter.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/grad.h>
#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <stack>
#include <deque>
#include <set>

#include <vector>
#include <array>
#include <Eigen/Core> // Assuming Eigen is used as per V.row()
#include <cmath>      // For std::cos, M_PI
#include <algorithm>  // For std::sort, std::unique, std::reverse

#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/parula.h>
#include <igl/per_corner_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>

#include <omp.h>

std::pair<int, int> make_edge(int v1, int v2) {
    return {std::min(v1, v2), std::max(v1, v2)};
}

// Function to expand a line in one direction
void expand_line(
    int current, 
    std::unordered_map<int, std::unordered_set<int>>& adjacency_list, 
    std::set<std::pair<int, int>>& visited_edges, 
    std::deque<int>& line, 
    bool forward) {
    while (true) {
        auto& neighbors = adjacency_list[current];

        // Find unvisited neighbors
        std::vector<int> unvisited_neighbors;
        for (const auto& neighbor : neighbors) {
            auto edge = make_edge(current, neighbor);
            if (visited_edges.find(edge) == visited_edges.end()) {
                unvisited_neighbors.push_back(neighbor);
            }
        }

        // Stop if no unvisited neighbors or branching point/end
        if (unvisited_neighbors.empty() || neighbors.size() != 2) {
            break;
        }

        // Move to the next vertex
        int next = unvisited_neighbors[0];
        visited_edges.insert(make_edge(current, next));

        if (forward) {
            line.push_back(next); // Add to the end of the deque
        } else {
            line.push_front(next); // Add to the front of the deque
        }

        current = next;
    }
}

// Function to find a connected line
void find_line(
    int start, 
    int end,
    std::unordered_map<int, std::unordered_set<int>>& adjacency_list, 
    std::set<std::pair<int, int>>& visited_edges, 
    std::vector<int>& line) {
    std::deque<int> temp_line;
    temp_line.push_back(start);
    temp_line.push_back(end);

    // Expand in both directions
    expand_line(start, adjacency_list, visited_edges, temp_line, false);  // Forward direction
    expand_line(end, adjacency_list, visited_edges, temp_line, true); // Backward direction

    // Convert deque to vector
    line.assign(temp_line.begin(), temp_line.end());
}


void Connect_Fold(Meshes &meshes, std::vector<std::vector<int>> &lines) {
    Eigen::VectorXd &Quad_Angle_sub_Vector = meshes.Quad_Angle_sub_Vector;
    std::unordered_map<int, std::unordered_set<int>> adjacency_list;

    for (int i = 0; i < Quad_Angle_sub_Vector.size(); ++i) {
        if (Quad_Angle_sub_Vector[i] > 0) { // Folding line
            int v1 = meshes.uE(i, 0);
            int v2 = meshes.uE(i, 1);
            adjacency_list[v1].insert(v2);
            adjacency_list[v2].insert(v1);
        }
    }
    lines.clear();
        // Step 2: Find connected lines
    std::set<std::pair<int, int>> visited_edges;

    for (const auto& [vertex, neighbors] : adjacency_list) {
        for (const auto& neighbor : neighbors) {
            auto edge = make_edge(vertex, neighbor);
            if (visited_edges.find(edge) == visited_edges.end()) {
                std::vector<int> line;
                visited_edges.insert(edge);
                find_line(vertex, neighbor, adjacency_list, visited_edges, line);

                if (!line.empty()) {
                    lines.push_back(line);
                }
            }
        }
    }
    adjacency_list.clear();

    for (int i = 0; i < Quad_Angle_sub_Vector.size(); ++i) {
        if (Quad_Angle_sub_Vector[i] < 0) { // Folding line
            int v1 = meshes.uE(i, 0);
            int v2 = meshes.uE(i, 1);
            adjacency_list[v1].insert(v2);
            adjacency_list[v2].insert(v1);
        }
    }
        // Step 2: Find connected lines
    visited_edges.clear();

    for (const auto& [vertex, neighbors] : adjacency_list) {
        for (const auto& neighbor : neighbors) {
            auto edge = make_edge(vertex, neighbor);
            if (visited_edges.find(edge) == visited_edges.end()) {
                std::vector<int> line;
                visited_edges.insert(edge);
                find_line(vertex, neighbor, adjacency_list, visited_edges, line);

                if (!line.empty()) {
                    lines.push_back(line);
                }
            }
        }
    }
}


void Compute_Angle_MV(const Eigen::Vector3d n1, const Eigen::Vector3d n2,
    const Eigen::Vector3d v1, const Eigen::Vector3d v2, double &angle) {
    angle = atan2(n1.cross(n2).dot((v2 - v1).normalized()), n1.dot(n2));
}

void normal(Eigen::Vector3d &n, const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const Eigen::Vector3d &v3) {
    n = (v2 - v1).cross(v3 - v1);
    n.normalize();
}


void Verify_MV(Meshes &meshes, std::vector<std::vector<int>> &lines, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &V_un,  Eigen::MatrixXi &F_un, std::vector<int> &MV) {
    Eigen::MatrixXd V_refer = meshes.V_refer;
    Eigen::MatrixXi F_refer = meshes.F_refer;
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd N;
    // igl::per_face_normals(V, F, N);
    std::map<std::pair<int, int>, int> V2uE;
    // // Alternative discrete mean curvature
    // Eigen::MatrixXd HN;
    // Eigen::SparseMatrix<double> L,M,Minv;
    // igl::cotmatrix(V,F,L);
    // igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_VORONOI,M);
    // igl::invert_diag(M,Minv);
    // // Laplace-Beltrami of position
    // HN = -Minv*(L*V);
    // // Extract magnitude as mean curvature
    // Eigen::VectorXd H = HN.rowwise().norm();

    // // Compute curvature directions via quadric fitting
    // Eigen::MatrixXd PD1,PD2;
    // Eigen::VectorXd PV1,PV2;
    // igl::principal_curvature(V,F,PD1,PD2,PV1,PV2);
    // // mean curvature
    // H = 0.5*(PV1+PV2);

    Eigen::MatrixXd V_mean = (meshes.V_deformed + meshes.V_undeformed)/2;
    Eigen::MatrixXd N_mean;
    igl::per_face_normals(V_mean, F_deformed, N_mean);

    for (int i = 0; i < meshes.uE.rows(); i++) {
        int v1 = meshes.uE(i, 0);
        int v2 = meshes.uE(i, 1);
        V2uE[std::make_pair(std::min(v1, v2), std::max(v1, v2))] = i;
    }
    for (int i = 0; i < lines.size(); i++) {
        int MV_i = 0;
        std::cout << "i/lines.size(): " << i << "/" << lines.size() << std::endl;
        #pragma omp parallel for
        for (int j = 0; j < lines[i].size() - 1; j++) {
            // we only sample 10 edge for each line
            // if (j % (lines[i].size() / 10) != 0) {
            //     continue;
            // }
            int p1 = lines[i][j];
            int p2 = lines[i][j + 1];
            // if std::make_pair(std::min(p1, p2), std::max(p1, p2)) not in V2uE
            if (V2uE.find(std::make_pair(std::min(p1, p2), std::max(p1, p2))) == V2uE.end()) {
                continue;
            }
            int uEi = V2uE[std::make_pair(std::min(p1, p2), std::max(p1, p2))];
            std::vector<int> half_edges = meshes.uE2E[uEi];
            const int num_faces = meshes.F_deformed.rows();
            const int f1 = half_edges[0] % num_faces;
            const int f2 = half_edges[1] % num_faces;
            const int c1 = half_edges[0] / num_faces;
            const int c2 = half_edges[1] / num_faces;
            const int v1 = meshes.F_deformed(f1, (c1+1)%3);
            const int v2 = meshes.F_deformed(f1, (c1+2)%3);
            const int v4 = meshes.F_deformed(f1, c1);
            const int v3 = meshes.F_deformed(f2, c2);
            Eigen::MatrixXd P = Eigen::MatrixXd::Zero(4, 3);
            double angle_un, angle_de;
            Eigen::Vector3d n1 = meshes.N_deformed.row(f1);
            Eigen::Vector3d n2 = meshes.N_deformed.row(f2);
            Eigen::Vector3d V1 = V_deformed.row(v1);
            Eigen::Vector3d V2 = V_deformed.row(v2);
            // Compute_Angle_MV(n1, n2, V1, V2, angle_de);

            // if (std::abs(angle_de) > 0) {
                V1 = V_mean.row(v1);
                V2 = V_mean.row(v2);
                n1 = N_mean.row(f1);
                n2 = N_mean.row(f2);

                Compute_Angle_MV(n1, n2, V1, V2, angle_de);
                
            n1 = meshes.N_undeformed.row(f1);
            n2 = meshes.N_undeformed.row(f2);
            V1 = meshes.V_undeformed.row(v1);
            V2 = meshes.V_undeformed.row(v2);
            Compute_Angle_MV(n1, n2, V1, V2, angle_un);
            // if (std::abs(angle_de) > M_PI - 0.1) {
            //     continue;
            // }
            if (angle_de > angle_un) {
                #pragma omp atomic
                MV_i++;
            }
            else {
                #pragma omp atomic
                MV_i--;
            }
        }
        // std::cout << "MV_i: " << MV_i << std::endl;
        if (MV_i > 0) {
            MV.push_back(1);
        }
        else {
            MV.push_back(-1);
        }
    }
}


// Function to calculate tangent vector at a line's endpoint
Eigen::Vector3d get_tangent_vector(
    const Eigen::MatrixXd& V,
    const std::vector<int>& line_nodes,
    int side // 0 for head, 1 for tail
) {
    if (line_nodes.size() < 2) {
        return Eigen::Vector3d::Zero(); // Should not happen for valid lines
    }
    if (side == 0) { // Head
        return (V.row(line_nodes[1]) - V.row(line_nodes[0])).normalized();
    } else { // Tail
        return (V.row(line_nodes.back()) - V.row(line_nodes[line_nodes.size() - 2])).normalized();
    }
}


void Merge_Mutually_Best_Collinear_Connections(
    const Eigen::MatrixXd& V,
    std::vector<std::vector<int>>& lines,
    std::vector<int>& MV
) {
    double cos_collinear_threshold = std::cos(M_PI / 6); // 30 degrees
    bool merged_in_iteration = true;

    while (merged_in_iteration) {
        merged_in_iteration = false;
        if (lines.size() < 2) break;

        // Each line has two potential endpoints for connection.
        // best_candidates_for_endpoints[line_idx][0] is for head of line_idx
        // best_candidates_for_endpoints[line_idx][1] is for tail of line_idx
        std::vector<std::array<ConnectionCandidate, 2>> best_candidates_for_endpoints(lines.size());

        // --- Part 1: For each endpoint of each line, find its best connection candidate ---
        for (int i = 0; i < lines.size(); ++i) {
            if (lines[i].size() < 2) continue;

            for (int self_side_i = 0; self_side_i < 2; ++self_side_i) { // 0 for head of line i, 1 for tail of line i
                ConnectionCandidate current_best_for_this_endpoint;
                // Initialize just below threshold, so only candidates >= threshold are considered "better"
                current_best_for_this_endpoint.dot_product = cos_collinear_threshold - 1e-9;


                int v_i_conn_node_idx = (self_side_i == 0) ? lines[i].front() : lines[i].back();
                Eigen::Vector3d tangent_i;
                if (self_side_i == 0) { // head of i, tangent i[0]->i[1]
                    tangent_i = (V.row(lines[i][1]) - V.row(v_i_conn_node_idx)).normalized();
                } else { // tail of i, tangent i[end-1]->i[end-2] (tangent points away from connection node)
                    tangent_i = (V.row(lines[i][lines[i].size()-2]) - V.row(v_i_conn_node_idx)).normalized();
                }

                for (int j = 0; j < lines.size(); ++j) {
                    if (i == j || lines[j].size() < 2) continue;
                    if (MV[i] != MV[j]) continue; // Ensure same material/group

                    for (int target_side_j = 0; target_side_j < 2; ++target_side_j) { // 0 for head of line j, 1 for tail of line j
                        int v_j_conn_node_idx = (target_side_j == 0) ? lines[j].front() : lines[j].back();

                        if (v_i_conn_node_idx == v_j_conn_node_idx) { // Connection point matches
                            Eigen::Vector3d tangent_j_for_merge;
                            double current_dot_product = -2.0; // Default invalid dot product

                                if (lines[j].size() < 2) continue; // Should be caught earlier, but good for safety
          
                                if (self_side_i == 1 && target_side_j == 0) { // tail_i connects to head_j (i -> j)
                                    // tangent_i is i[end-2]->i[end] (points "out" of i's tail)
                                    // For j to continue, j's tangent should be j[0]->j[1] (points "out" of j's head)
                                    tangent_j_for_merge = (V.row(lines[j][1]) - V.row(v_j_conn_node_idx)).normalized();
                                    current_dot_product = -tangent_i.dot(tangent_j_for_merge); // Should be close to 1 for collinear
                                } else if (self_side_i == 0 && target_side_j == 1) { // head_i connects to tail_j (j -> i)
                                    // tangent_i is i[1]->i[0] (points "out" of i's head)
                                    // For j to precede, j's tangent should be j[end-1]->j[end-2] (points "out" of j's tail)
                                    tangent_j_for_merge = (V.row(lines[j][lines[j].size()-2]) - V.row(v_j_conn_node_idx)).normalized();
                                    current_dot_product = -tangent_i.dot(tangent_j_for_merge); // Should be close to 1
                                } else if (self_side_i == 0 && target_side_j == 0) { // head_i connects to head_j (rev_j -> i or rev_i -> j)
                                    // tangent_i is i[1]->i[0] (out of i's head)
                                    // tangent_j for rev_j should be j[0]->j[1] (out of j's head, to be reversed)
                                    // For rev_j -> i: tangent of rev_j (at its new head, orig j's head) is j[1]->j[0]
                                    // For rev_i -> j: tangent of rev_i (at its new head, orig i's head) is i[1]->i[0]
                                    // The prompt's logic:
                                    // tangent_j_for_merge = (V.row(v_j_conn_node_idx) - V.row(lines[j][1])).normalized(); // This is j[0] <- j[1] (INWARD for j)
                                    // Let's use the same definition as other cases: tangent_j "outward" from connection point
                                    tangent_j_for_merge = (V.row(lines[j][1]) - V.row(v_j_conn_node_idx)).normalized(); // j[0]->j[1]
                                    current_dot_product = -tangent_i.dot(tangent_j_for_merge); // Expect near 1 if rev_i then j, or rev_j then i
                                } else if (self_side_i == 1 && target_side_j == 1) { // tail_i connects to tail_j (i -> rev_j or j -> rev_i)
                                    // tangent_i is i[end-2]->i[end] (out of i's tail)
                                    // tangent_j for rev_j should be j[end-1]->j[end-2] (out of j's tail, to be reversed)
                                    // The prompt's logic:
                                    // tangent_j_for_merge = (V.row(v_j_conn_node_idx) - V.row(lines[j][lines[j].size()-2])).normalized(); // This is j[end-1] <- j[end-2] (INWARD for j)
                                    tangent_j_for_merge = (V.row(lines[j][lines[j].size()-2]) - V.row(v_j_conn_node_idx)).normalized(); // j[end-1]->j[end-2]
                                    current_dot_product = -tangent_i.dot(tangent_j_for_merge); // Expect near 1
                                }// end v_i_conn_node_idx == v_j_conn_node_idx

                            if (current_dot_product > current_best_for_this_endpoint.dot_product) {
                                current_best_for_this_endpoint.target_line_idx = j;
                                current_best_for_this_endpoint.target_side = target_side_j;
                                current_best_for_this_endpoint.dot_product = current_dot_product;
                                current_best_for_this_endpoint.connection_node_idx = v_i_conn_node_idx;
                            }
                        } // end if connection point matches
                    } // end loop target_side_j
                } // end loop j
                best_candidates_for_endpoints[i][self_side_i] = current_best_for_this_endpoint;
            } // end loop self_side_i
        } // end loop i (finding best candidates for each endpoint)


        // --- Part 2: Find mutually best pairs above threshold and prepare merge operations ---
        std::vector<PotentialMergeOperation> merge_ops_this_iteration;
        for (int i = 0; i < lines.size(); ++i) {
            if (lines[i].size() < 2) continue;

            for (int self_side_i = 0; self_side_i < 2; ++self_side_i) {
                const auto& cand_i = best_candidates_for_endpoints[i][self_side_i];

                // Check 1: Is cand_i valid and its dot product above the threshold?
                if (!cand_i.isValid() || cand_i.dot_product < cos_collinear_threshold) {
                    continue;
                }

                int j = cand_i.target_line_idx;
                int side_on_j_targeted_by_i = cand_i.target_side;

                // Basic sanity checks for j
                if (j < 0 || j >= lines.size() || lines[j].size() < 2 || i == j) {
                    continue;
                }

                // Check 2: Does the corresponding endpoint on line j also choose line i's current endpoint,
                // and is its dot product also above the threshold?
                const auto& cand_j = best_candidates_for_endpoints[j][side_on_j_targeted_by_i];
                std::cout << "i: " << i << std::endl;
                std::cout << "j: " << j << std::endl;
                if (cand_j.isValid()){
                    std::cout << "i, j: " << i << ", " << j << std::endl;
                    std::cout << "target_line_idx: " << cand_j.target_line_idx << std::endl;
                    std::cout << "target_side: " << cand_j.target_side << std::endl;
                    std::cout << "dot_product: " << cand_j.dot_product << std::endl;
                    std::cout << "self_side_i: " << self_side_i << std::endl;
                }

                if (cand_j.isValid() && cand_j.dot_product >= cos_collinear_threshold &&
                    cand_j.target_line_idx == i &&
                    cand_j.target_side == self_side_i &&
                    cand_i.connection_node_idx == cand_j.connection_node_idx) { // Mutual best connection at the same vertex

                    PotentialMergeOperation op;
                    // Canonical form: line_idx1 < line_idx2
                    if (i < j) {
                        op.line_idx1 = i;
                        op.side1 = self_side_i;
                        op.line_idx2 = j;
                        op.side2 = side_on_j_targeted_by_i;
                    } else {
                        op.line_idx1 = j;
                        op.side1 = side_on_j_targeted_by_i;
                        op.line_idx2 = i;
                        op.side2 = self_side_i;
                    }
                    // op.quality_metric = cand_i.dot_product; // Store if needed for sorting/selection
                    merge_ops_this_iteration.push_back(op);
                }
            }
        }
        std::cout << "merge_ops_this_iteration.size(): " << merge_ops_this_iteration.size() << std::endl;

        // Sort and unique the operations to avoid redundant processing or conflicts if any
        if (!merge_ops_this_iteration.empty()) {
            std::sort(merge_ops_this_iteration.begin(), merge_ops_this_iteration.end(),
                [](const PotentialMergeOperation& a, const PotentialMergeOperation& b) {
                    if (a.line_idx1 != b.line_idx1) return a.line_idx1 < b.line_idx1;
                    if (a.line_idx2 != b.line_idx2) return a.line_idx2 < b.line_idx2;
                    if (a.side1 != b.side1) return a.side1 < b.side1;
                    return a.side2 < b.side2;
                });
            merge_ops_this_iteration.erase(
                std::unique(merge_ops_this_iteration.begin(), merge_ops_this_iteration.end(),
                    [](const PotentialMergeOperation& a, const PotentialMergeOperation& b) {
                        return a.line_idx1 == b.line_idx1 &&
                               a.line_idx2 == b.line_idx2 &&
                               a.side1 == b.side1 &&
                               a.side2 == b.side2;
                    }),
                merge_ops_this_iteration.end());
        }

        

        // --- Part 3: Process the merge operations ---
        std::vector<bool> line_has_been_merged_this_pass(lines.size(), false);

        for (const auto& op : merge_ops_this_iteration) {
            if (line_has_been_merged_this_pass[op.line_idx1] || line_has_been_merged_this_pass[op.line_idx2]) {
                continue; // One of the lines already participated in a merge this pass
            }

            // Ensure lines are still valid (e.g., not emptied by a previous unrelated operation if logic changes)
            if (lines[op.line_idx1].size() < 1 || lines[op.line_idx2].size() < 1) { // < 1 because they might be single point after a merge
                continue;
            }


            std::vector<int> line1_nodes_original = lines[op.line_idx1]; // Copy original
            std::vector<int> line2_nodes_original = lines[op.line_idx2]; // Copy original
            std::vector<int> new_merged_line_nodes;
            bool current_merge_performed = false;

            // Determine which line is "line1" and "line2" based on op's canonical form
            int actual_line_idx1 = op.line_idx1;
            int actual_side1 = op.side1;
            std::vector<int>& line1_nodes_ref = lines[actual_line_idx1]; // Modifiable ref
            const auto& line1_nodes_const = line1_nodes_original;


            int actual_line_idx2 = op.line_idx2;
            int actual_side2 = op.side2;
            std::vector<int>& line2_nodes_ref = lines[actual_line_idx2]; // Modifiable ref
            const auto& line2_nodes_const = line2_nodes_original;


            // The connection node is common and should appear only once.
            // line1_nodes_ref will absorb line2_nodes_ref.

            if (actual_side1 == 1 && actual_side2 == 0) { // Tail of line1 connects to Head of line2 (L1 -> L2)
                new_merged_line_nodes = line1_nodes_const;
                new_merged_line_nodes.insert(new_merged_line_nodes.end(), line2_nodes_const.begin() + 1, line2_nodes_const.end());
                current_merge_performed = true;
            } else if (actual_side1 == 0 && actual_side2 == 1) { // Head of line1 connects to Tail of line2 (L2 -> L1)
                new_merged_line_nodes = line2_nodes_const;
                new_merged_line_nodes.insert(new_merged_line_nodes.end(), line1_nodes_const.begin() + 1, line1_nodes_const.end());
                current_merge_performed = true;
            } else if (actual_side1 == 0 && actual_side2 == 0) { // Head of line1 connects to Head of line2 (rev(L1) -> L2 or rev(L2) -> L1)
                                                                // Based on op canonical form, we'll decide how to merge.
                                                                // The original logic in your prompt implies a specific ordering.
                                                                // If op.line_idx1 is 'i' and op.line_idx2 is 'j', and self_side_i=0, target_side_j=0
                                                                // This was (rev_j -> i) or (rev_i -> j).
                                                                // Let's assume we merge into op.line_idx1
                                                                // If op.side1 (line1's connecting side) is head, and op.side2 (line2's side) is head:
                                                                // reverse line1, then append line2 (skipping common node)
                std::vector<int> temp_line1 = line1_nodes_const;
                std::reverse(temp_line1.begin(), temp_line1.end());
                new_merged_line_nodes = temp_line1;
                new_merged_line_nodes.insert(new_merged_line_nodes.end(), line2_nodes_const.begin() + 1, line2_nodes_const.end());
                current_merge_performed = true;
            } else if (actual_side1 == 1 && actual_side2 == 1) { // Tail of line1 connects to Tail of line2 (L1 -> rev(L2) or L2 -> rev(L1))
                                                                // If op.side1 is tail, and op.side2 is tail:
                                                                // line1, then append reversed line2 (skipping common node)
                std::vector<int> temp_line2 = line2_nodes_const;
                std::reverse(temp_line2.begin(), temp_line2.end());
                new_merged_line_nodes = line1_nodes_const;
                new_merged_line_nodes.insert(new_merged_line_nodes.end(), temp_line2.begin() + 1, temp_line2.end());
                current_merge_performed = true;
            }

            if (current_merge_performed) {
                line1_nodes_ref = new_merged_line_nodes; // Store merged line in the first line's slot
                line2_nodes_ref.clear();                 // Mark the second line as absorbed

                line_has_been_merged_this_pass[op.line_idx1] = true;
                line_has_been_merged_this_pass[op.line_idx2] = true;
                merged_in_iteration = true;
            }
        }

        // --- Part 4: Compact the lines and MV vectors ---
        if (merged_in_iteration) { // Only compact if something actually changed
            std::vector<std::vector<int>> next_lines;
            std::vector<int> next_MV;
            next_lines.reserve(lines.size());
            next_MV.reserve(MV.size());

            for (size_t k = 0; k < lines.size(); ++k) {
                if (!lines[k].empty()) { // Only keep non-absorbed (non-empty) lines
                    next_lines.push_back(lines[k]);
                    if (k < MV.size()) { // Safety check
                        next_MV.push_back(MV[k]);
                    } else {
                        // This should not happen if MV is always correctly sized with lines.
                        // Handle error or assign a default MV value if necessary.
                        // std::cerr << "Warning: MV vector size mismatch during compaction." << std::endl;
                    }
                }
            }
            lines = std::move(next_lines);
            MV = std::move(next_MV);
        }
    } // end while(merged_in_iteration)
}


double calculate_zigzag_metric(const std::vector<int>& line_indices, const Eigen::MatrixXd& V, double sharp_angle_threshold_rad) {
    if (line_indices.size() < 3) return 0.0;
    int sharp_turns = 0;
    double sum_angle_diff = 0.0;
    for (size_t k = 0; k < line_indices.size() - 2; ++k) {
        Eigen::Vector3d p1 = V.row(line_indices[k]);
        Eigen::Vector3d p2 = V.row(line_indices[k+1]);
        Eigen::Vector3d p3 = V.row(line_indices[k+2]);
        Eigen::Vector3d v1 = (p1 - p2).normalized(); // B->A
        Eigen::Vector3d v2 = (p3 - p2).normalized(); // B->C
        double angle = std::acos(std::max(-1.0, std::min(1.0, v1.dot(v2)))); // Angle at p2
        if (angle < sharp_angle_threshold_rad) { // 鋭角な曲がり
            sharp_turns++;
            sum_angle_diff += (sharp_angle_threshold_rad - angle);
        } // 角度の差を加算
    }
    // return (double)sharp_turns / (line_indices.size() - 2); // 鋭角な曲がりの割合
    return sum_angle_diff/ (line_indices.size() - 2); // 鋭角な曲がりの割合
}

void Connect_Close_MV(Meshes &meshes, std::vector<std::vector<int>> &lines, std::vector<int> &MV) {
    Eigen::MatrixXd V = meshes.V_undeformed;
    std::vector<int> delete_list;

    for (int i = 0; i < lines.size(); i++) {
        double total_length = 0;
        double sum_angle = 0;
        double sum_real = 0;
        for (int j = 0; j < lines[i].size() - 1; j++) {
            int p1 = lines[i][j];
            int p2 = lines[i][j + 1];
            double length = (V.row(p1) - V.row(p2)).norm();
            total_length += length;
        }
        if (lines[i][0] == lines[i].back() && total_length < 8.0) {
            delete_list.push_back(i);
        }
    }
    std::cout << "delete_list.size(): " << delete_list.size() << std::endl;
    //sort delete_list in descending order
    std::sort(delete_list.begin(), delete_list.end(), std::greater<int>());
    for (int i = 0; i < delete_list.size(); i++) {
        lines.erase(lines.begin() + delete_list[i]);
        MV.erase(MV.begin() + delete_list[i]);
    }

    Merge_Mutually_Best_Collinear_Connections(V, lines, MV);
    // double threshold = 2.0;
    double threshold = 0.0;
    std::vector<int> delete_list_angle;
    std::map<std::pair<int, int>, int> V2uE;
    for (int i = 0; i < meshes.uE.rows(); i++) {
        int v1 = meshes.uE(i, 0);
        int v2 = meshes.uE(i, 1);
        V2uE[std::make_pair(std::min(v1, v2), std::max(v1, v2))] = i;
    }
    for (int i = 0; i < lines.size(); i++) {
        int MV_i = 0;
        double sum_angle = 0;
        double sum_real = 0;
        double total_length = 0;
        for (int j = 0; j < lines[i].size() - 1; j++) {
            int p1 = lines[i][j];
            int p2 = lines[i][j + 1];
            int uEi = V2uE[std::make_pair(std::min(p1, p2), std::max(p1, p2))];
            double angle_diff_opt = meshes.Quad_Angle_Vector[uEi];
            double angle_diff_real = std::pow(meshes.Quad_Angle_sub_Vector[uEi], 2);
            sum_real += angle_diff_real;
            sum_angle += angle_diff_opt;
            double length = (V.row(p1) - V.row(p2)).norm();
            total_length += length;
        }
        double zigzag_metric = calculate_zigzag_metric(lines[i], V, M_PI *5/6);
        if (total_length < 4.0){
            // delete_list_angle.push_back(i);
            int max_count = 0;
            int j_count_head = 0;
            int j_count_tail = 0;
            for (int j = 0; j< lines.size(); j++){
                if (i == j){
                    continue;
                }
                for (int k = 0; k < lines[j].size(); k++){
                    if (lines[i][0] == lines[j][k]){
                        j_count_head = 1;
                    }
                    if (lines[i].back() == lines[j][k]){
                        j_count_tail = 1;
                    }
                }
            }
            max_count = j_count_head + j_count_tail;
            if (max_count < 2){
                delete_list_angle.push_back(i);
            }
        }
        if (zigzag_metric > 0.1 && total_length < 6.0 && std::find(delete_list_angle.begin(), delete_list_angle.end(), i) == delete_list_angle.end()) {
            delete_list_angle.push_back(i);
        }
        if (total_length < 3.0 && std::find(delete_list_angle.begin(), delete_list_angle.end(), i) == delete_list_angle.end()) {
            delete_list_angle.push_back(i);
        }
        if (sum_angle < 0.01 && std::find(delete_list_angle.begin(), delete_list_angle.end(), i) == delete_list_angle.end()) {
            delete_list_angle.push_back(i);
        }
    }
    // delete the lines with small sum angle
    // sort delete_list in descending order
    std::sort(delete_list_angle.begin(), delete_list_angle.end(), std::greater<int>());
    for (int i = 0; i < delete_list_angle.size(); i++) {
        lines.erase(lines.begin() + delete_list_angle[i]);
        MV.erase(MV.begin() + delete_list_angle[i]);
    }
    // std::cout << "lines.size(): " << lines.size() << std::endl;
    // std::vector<int> connection;
    // for (int i = 0; i < lines.size(); i++) {
    //     bool connected = false;
    //     std::cout << lines[i].size() << std::endl;
    //     std::pair<int, double> closest_head = {-1, 1000000};
    //     std::pair<int, double> closest_tail = {-1, 1000000};
    //     int connect_head = 0;
    //     int connect_tail = 0;

    //     for (int j = 0; j < lines.size(); j++) {
    //         if (i == j) {
    //             continue;
    //         }
    //         // if (MV[i] != MV[j]) {
    //         //     continue;
    //         // })
    //         double head = 1.0;
    //         double tail = 1.0;

    //         int head_i = lines[i].front();
    //         int tail_i = lines[i].back();
    //         int head_j = lines[j].front();
    //         int tail_j = lines[j].back();
    //         Eigen::Vector3d head_i_v = V.row(head_i);
    //         Eigen::Vector3d tail_i_v = V.row(tail_i);
    //         Eigen::Vector3d head_i_vec = (V.row(head_i) - V.row(lines[i][1])).normalized();
    //         Eigen::Vector3d tail_i_vec = (V.row(tail_i) - V.row(lines[i][lines[i].size() - 2])).normalized();
    //         Eigen::Vector3d head_j_v = V.row(head_j);
    //         Eigen::Vector3d tail_j_v = V.row(tail_j);
    //         Eigen::Vector3d head_j_vec = (V.row(head_j) - V.row(lines[j][1])).normalized();
    //         Eigen::Vector3d tail_j_vec = (V.row(tail_j) - V.row(lines[j][lines[j].size() - 2])).normalized();
    //         if (head_i == head_j && head_i_vec.dot(head_j_vec) < 0.0 && MV[i] == MV[j]){
    //             connect_head = 1;
    //         }
    //         if (head_i == tail_j && head_i_vec.dot(tail_j_vec) < 0.0 && MV[i] == MV[j]){
    //             connect_head = 1;
    //         }
    //         if (tail_i == head_j && tail_i_vec.dot(head_j_vec) < 0.0 && MV[i] == MV[j]){
    //             connect_tail = 1;
    //         }
    //         if (tail_i == tail_j && tail_i_vec.dot(tail_j_vec) < 0.0 && MV[i] == MV[j]){
    //             connect_tail = 1;
    //         }
    //     }
    //     int con = connect_head + connect_tail;
    //     connection.push_back(con);

    // }
    // std::vector<int> delete_list;
    // for (int i = 0; i < lines.size(); i++) {
    //     double total_length = 0;
    //     double sum_angle = 0;
    //     double sum_real = 0;
    //     for (int j = 0; j < lines[i].size() - 1; j++) {
    //         int p1 = lines[i][j];
    //         int p2 = lines[i][j + 1];
    //         int uEi = V2uE[std::make_pair(std::min(p1, p2), std::max(p1, p2))];
    //         double angle_diff_opt = meshes.Quad_Angle_Vector[uEi];
    //         double angle_diff_real = meshes.Quad_Angle_sub_Vector[uEi];
    //         sum_real += angle_diff_real;
    //         sum_angle += angle_diff_opt;
    //         double length = (V.row(p1) - V.row(p2)).norm();
    //         total_length += length;
    //     }
    //     if (total_length < 3.0 && connection[i] < 2) {
    //         delete_list.push_back(i);
    //     } else if (total_length < 2.0 && connection[i] == 2) {
    //         delete_list.push_back(i);
    //     } else if (sum_angle < 0.01 && std::abs(sum_real) < 2.0) {
    //         delete_list.push_back(i);
    //     } 
    // }
    // std::cout << "delete_list.size(): " << delete_list.size() << std::endl;
    // //sort delete_list in descending order
    // std::sort(delete_list.begin(), delete_list.end(), std::greater<int>());
    // for (int i = 0; i < delete_list.size(); i++) {
    //     lines.erase(lines.begin() + delete_list[i]);
    //     MV.erase(MV.begin() + delete_list[i]);
    // }
    // std::cout << "lines.size(): " << lines.size() << std::endl;

    // std::vector<int> connection_whichever;
    // for (int i = 0; i < lines.size(); i++) {
    //     bool connected = false;
    //     std::cout << lines[i].size() << std::endl;
    //     std::pair<int, double> closest_head = {-1, 1000000};
    //     std::pair<int, double> closest_tail = {-1, 1000000};
    //     int connect_whichever_head = 0;
    //     int connect_whichever_tail = 0;

    //     for (int j = 0; j < lines.size(); j++) {
    //         if (i == j) {
    //             continue;
    //         }
    //         // if (MV[i] != MV[j]) {
    //         //     continue;
    //         // })
    //         double head = 1.0;
    //         double tail = 1.0;

    //         int head_i = lines[i].front();
    //         int tail_i = lines[i].back();
    //         int head_j = lines[j].front();
    //         int tail_j = lines[j].back();
    //         Eigen::Vector3d head_i_v = V.row(head_i);
    //         Eigen::Vector3d tail_i_v = V.row(tail_i);
    //         Eigen::Vector3d head_i_vec = (V.row(head_i) - V.row(lines[i][1])).normalized();
    //         Eigen::Vector3d tail_i_vec = (V.row(tail_i) - V.row(lines[i][lines[i].size() - 2])).normalized();
    //         Eigen::Vector3d head_j_v = V.row(head_j);
    //         Eigen::Vector3d tail_j_v = V.row(tail_j);
    //         Eigen::Vector3d head_j_vec = (V.row(head_j) - V.row(lines[j][1])).normalized();
    //         Eigen::Vector3d tail_j_vec = (V.row(tail_j) - V.row(lines[j][lines[j].size() - 2])).normalized();
    //         if (head_i == head_j){
    //             connect_whichever_head = 1;
    //         }
    //         if (head_i == tail_j){
    //             connect_whichever_head = 1;
    //         }
    //     }

    //     int con_whichever = connect_whichever_head + connect_whichever_tail;
    //     connection_whichever.push_back(con_whichever);
    // }
    // delete_list.clear();

    
    // std::cout << "lines.size(): " << lines.size() << std::endl;


    for (int i = 0; i < lines.size(); i++) {
        bool connected = false;
        std::cout << lines[i].size() << std::endl;
        std::pair<int, double> closest_head = {-1, 1000000};
        std::pair<int, double> closest_tail = {-1, 1000000};
        for (int j = 0; j < lines.size(); j++) {
            if (i == j) {
                continue;
            }
            // if (MV[i] != MV[j]) {
            //     continue;
            // })
            double head = 1.0;
            double tail = 1.0;

            int head_i = lines[i].front();
            int tail_i = lines[i].back();
            int head_j = lines[j].front();
            int tail_j = lines[j].back();
            Eigen::Vector3d head_i_v = V.row(head_i);
            Eigen::Vector3d tail_i_v = V.row(tail_i);
            Eigen::Vector3d head_i_vec = (V.row(head_i) - V.row(lines[i][1])).normalized();
            Eigen::Vector3d tail_i_vec = (V.row(tail_i) - V.row(lines[i][lines[i].size() - 2])).normalized();
            Eigen::Vector3d head_j_v = V.row(head_j);
            Eigen::Vector3d tail_j_v = V.row(tail_j);
            Eigen::Vector3d head_j_vec = (V.row(head_j) - V.row(lines[j][1])).normalized();
            Eigen::Vector3d tail_j_vec = (V.row(tail_j) - V.row(lines[j][lines[j].size() - 2])).normalized();

            Eigen::Vector3d connect_head_i_tail_j = (head_i_v - tail_j_v).normalized();
            Eigen::Vector3d connect_tail_i_head_j = (tail_i_v - head_j_v).normalized();
            Eigen::Vector3d connect_head_i_head_j = (head_i_v - head_j_v).normalized();
            Eigen::Vector3d connect_tail_i_tail_j = (tail_i_v - tail_j_v).normalized();
            if ((head_i_v - head_j_v).norm() < closest_head.second && head_i != head_j && head_i_vec.dot(head_j_vec) < 0.0 && head_i_vec.dot(connect_head_i_head_j) < 0.0 && MV[i] == MV[j]) {
                // add head_j to the head of line i
                closest_head = {head_j, (head_i_v - head_j_v).norm()};
            }
            if ((head_i_v - tail_j_v).norm() < closest_head.second && head_i != tail_j && head_i_vec.dot(tail_j_vec) < 0.0 && head_i_vec.dot(connect_head_i_tail_j) < 0.0 &&  MV[i] == MV[j]) {
                // add tail_j to the head of line i
                closest_head = {tail_j, (head_i_v - tail_j_v).norm()};
            }
            if ((tail_i_v - head_j_v).norm() < closest_tail.second && tail_i != head_j && tail_i_vec.dot(head_j_vec) < 0.0 && head_i_vec.dot(connect_tail_i_head_j) < 0.0 && MV[i] == MV[j]) {
                // add head_j to the tail of line i
                closest_tail = {head_j, (tail_i_v - head_j_v).norm()};
            }
            if ((tail_i_v - tail_j_v).norm() < closest_tail.second && tail_i != tail_j && tail_i_vec.dot(tail_j_vec) < 0.0 && head_i_vec.dot(connect_tail_i_tail_j) < 0.0 && MV[i] == MV[j]) {
                // add tail_j to the tail of line i
                closest_tail = {tail_j, (tail_i_v - tail_j_v).norm()};
            }
        }
        if (closest_head.second < 3.0) {
            lines[i].insert(lines[i].begin(), closest_head.first);
        }
        if (closest_tail.second < 3.0) {
            lines[i].push_back(closest_tail.first);
        }

    }
    // Merge_Single_Connection(lines, MV);
    Verify_MV(meshes, lines, V, meshes.F_deformed, meshes.V_undeformed, meshes.F_undeformed, MV);
}