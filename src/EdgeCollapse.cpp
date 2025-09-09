#include "EdgeCollapse.h"
#include <cmath>
#include <iostream>
#include <igl/edge_topology.h>
#include <igl/edge_lengths.h>
#include <igl/unique_edge_map.h>
#include <igl/squared_edge_lengths.h>
#include <igl/internal_angles.h>
#include <igl/flip_edge.h>
#include <igl/unique_edge_map.h>
#include <igl/edge_flaps.h>

#include <igl/intersection_blocking_collapse_edge_callbacks.h>
#include <igl/qslim_optimal_collapse_edge_callbacks.h>
#include <igl/per_vertex_point_to_plane_quadrics.h>
#include <igl/STR.h>
#include <igl/connect_boundary_to_infinity.h>
#include <igl/decimate.h>
#include <igl/max_faces_stopping_condition.h>
#include <igl/point_simplex_squared_distance.h>
#include <igl/edge_flaps.h>
#include <igl/decimate_callback_types.h>
#include <igl/find.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/edge_topology.h>
#include <igl/unique_edge_map.h>
#include <igl/edge_flaps.h>
#include <igl/is_edge_manifold.h>
#include <igl/infinite_cost_stopping_condition.h>
#include <igl/predicates/find_self_intersections.h>
#include <igl/remove_unreferenced.h>
#include <igl/doublearea.h>


#include "Boundary.h"

void Compute_EdgeCollapse(Meshes &meshes) {
    Eigen::MatrixXd VO;
    Eigen::MatrixXd VO_undeformed;
    Eigen::MatrixXi FO;
    igl::connect_boundary_to_infinity(meshes.V_deformed,meshes.V_undeformed,meshes.F_undeformed,VO,VO_undeformed,FO);
    Eigen::VectorXi EMAP;
    Eigen::MatrixXi E,EF,EI;
    igl::edge_flaps(FO,E,EMAP,EF,EI);

    igl::decimate_cost_and_placement_callback cost_and_placement;
    igl::decimate_pre_collapse_callback  pre_collapse;
    igl::decimate_post_collapse_callback post_collapse;

    // Quadrics per vertex
    typedef std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double> Quadric;
    std::vector<Quadric> quadrics;
      igl::per_vertex_point_to_plane_quadrics(VO,FO,EMAP,EF,EI,quadrics);
    // State variables keeping track of edge we just collapsed
    int v1 = -1;
    int v2 = -1;
    // Callbacks for computing and updating metric
    igl::qslim_optimal_collapse_edge_callbacks(
      E,quadrics,v1,v2, cost_and_placement, pre_collapse,post_collapse);

    igl::AABB<Eigen::MatrixXd, 3> * tree = new igl::AABB<Eigen::MatrixXd, 3>();
    tree->init(meshes.V_deformed,meshes.F_deformed);
    tree->validate();


    igl::intersection_blocking_collapse_edge_callbacks(
      pre_collapse, post_collapse, // These will get copied as needed
      tree,
      pre_collapse, post_collapse);
    

    Eigen::MatrixXd U;
    Eigen::MatrixXd U_undeformed;
    Eigen::MatrixXi G;
    Eigen::VectorXi J,I;
    int m = meshes.F_undeformed.rows();
    const int orig_m = m;

    // to make N_deformed_opt and N_undeformed_opt size same as FO, add 0,0,0 rows to the end
    int additional_rows = FO.rows() - meshes.N_deformed_opt.rows();
    meshes.N_deformed_opt.conservativeResize(FO.rows(), 3);
    meshes.N_undeformed_opt.conservativeResize(FO.rows(), 3);
    meshes.N_deformed_opt.bottomLeftCorner(additional_rows, meshes.N_deformed_opt.cols()) = Eigen::MatrixXd::Zero(additional_rows, meshes.N_deformed_opt.cols());
    meshes.N_undeformed_opt.bottomLeftCorner(additional_rows, meshes.N_undeformed_opt.cols()) = Eigen::MatrixXd::Zero(additional_rows, meshes.N_undeformed_opt.cols());
    const bool ret = igl::decimate(
      VO, VO_undeformed, FO,
      cost_and_placement,
      igl::infinite_cost_stopping_condition(cost_and_placement),
      pre_collapse,
      post_collapse,
      E, EMAP, EF, EI, meshes.N_deformed_opt, meshes.N_undeformed_opt,
      U, U_undeformed, G, J, I);
    G = G(igl::find((J.array()<orig_m).eval()), igl::placeholders::all).eval();
    meshes.N_deformed_opt = meshes.N_deformed_opt(igl::find((J.array()<orig_m).eval()), igl::placeholders::all).eval();
    meshes.N_undeformed_opt = meshes.N_undeformed_opt(igl::find((J.array()<orig_m).eval()), igl::placeholders::all).eval();
    {
      Eigen::VectorXi I_sub;
      Eigen::VectorXi _;
      igl::remove_unreferenced(Eigen::MatrixXd(U),Eigen::MatrixXi(G),U,G, _, I_sub);
      Eigen::MatrixXd U_ = U_undeformed;
      U_undeformed = U_(I_sub.derived(),igl::placeholders::all);
    }
    meshes.V_deformed = U;
    meshes.V_undeformed = U_undeformed;
    meshes.F_undeformed = G;
    meshes.F_deformed = G;
    meshes.uE2E.clear();
    meshes.N_deformed_opt.conservativeResize(meshes.F_deformed.rows(), 3);
    meshes.N_undeformed_opt.conservativeResize(meshes.F_undeformed.rows(), 3);
    Eigen::VectorXi EI_;
    Eigen::MatrixXd EV;
    Eigen::MatrixXi IF,EE;
    Eigen::Array<bool,Eigen::Dynamic,1> CP;
    igl::predicates::find_self_intersections(meshes.V_deformed,meshes.F_deformed,IF,CP,EV,EE,EI_);
}


void Compute_subdivide_mesh(Meshes &meshes) {
    Eigen::VectorXd area;
    igl::doublearea(meshes.V_undeformed, meshes.F_undeformed, area);

    double threshold = 4.0;
    Eigen::MatrixXd V_sub_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi F_sub_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd V_sub_deformed = meshes.V_deformed;

    Eigen::MatrixXd N_sub_undeformed = meshes.N_undeformed_opt;
    Eigen::MatrixXd N_sub_deformed = meshes.N_deformed_opt;

    Eigen::VectorXi F_over_1 = (area.array() > threshold).cast<int>();

    // get the index of the faces that are over the threshold
    std::vector<int> index;
    for (int i = 0; i < F_over_1.size(); i++) {
        if (F_over_1(i) == 1) {
            index.push_back(i);
        }
    }
    // flip index
    std::reverse(index.begin(), index.end());

    V_sub_undeformed.conservativeResize(V_sub_undeformed.rows() + index.size(), 3);
    F_sub_undeformed.conservativeResize(F_sub_undeformed.rows() + index.size() * 3, 3);
    N_sub_undeformed.conservativeResize(N_sub_undeformed.rows() + index.size() * 3, 3);
    N_sub_deformed.conservativeResize(N_sub_deformed.rows() + index.size() * 3, 3);
    V_sub_deformed.conservativeResize(V_sub_deformed.rows() + index.size(), 3);
    #pragma omp parallel for
    for (int i = 0; i < index.size(); i++) {
        int face_index = index[i];
        Eigen::RowVector3d v0 = V_sub_undeformed.row(F_sub_undeformed(face_index, 0));
        Eigen::RowVector3d v1 = V_sub_undeformed.row(F_sub_undeformed(face_index, 1));
        Eigen::RowVector3d v2 = V_sub_undeformed.row(F_sub_undeformed(face_index, 2));
        Eigen::RowVector3d v_center = (v0 + v1 + v2) / 3.0;
        V_sub_undeformed.row(V_sub_undeformed.rows() - index.size() + i) = v_center;

        Eigen::RowVector3d v0_deformed = V_sub_deformed.row(F_sub_undeformed(face_index, 0));
        Eigen::RowVector3d v1_deformed = V_sub_deformed.row(F_sub_undeformed(face_index, 1));
        Eigen::RowVector3d v2_deformed = V_sub_deformed.row(F_sub_undeformed(face_index, 2));
        Eigen::RowVector3d v_center_deformed = (v0_deformed + v1_deformed + v2_deformed) / 3.0;
        V_sub_deformed.row(V_sub_deformed.rows() - index.size() + i) = v_center_deformed;

        int v0_index = F_sub_undeformed(face_index, 0);
        int v1_index = F_sub_undeformed(face_index, 1);
        int v2_index = F_sub_undeformed(face_index, 2);
        int v_new_index = V_sub_undeformed.rows() - index.size() + i;
        F_sub_undeformed.row(F_sub_undeformed.rows() - index.size()*3 + i * 3) = Eigen::RowVector3i(v0_index, v1_index, v_new_index);
        F_sub_undeformed.row(F_sub_undeformed.rows() - index.size()*3 + i * 3 + 1) = Eigen::RowVector3i(v1_index, v2_index, v_new_index);
        F_sub_undeformed.row(F_sub_undeformed.rows() - index.size()*3 + i * 3 + 2) = Eigen::RowVector3i(v2_index, v0_index, v_new_index);

        Eigen::RowVector3d n0 = ((v1 - v0).cross(v_center - v1)).normalized();
        Eigen::RowVector3d n1 = ((v2 - v1).cross(v_center - v2)).normalized();
        Eigen::RowVector3d n2 = ((v0 - v2).cross(v_center - v0)).normalized();
        N_sub_undeformed.row(F_sub_undeformed.rows() - index.size()*3 + i * 3) = n0;
        N_sub_undeformed.row(F_sub_undeformed.rows() - index.size()*3 + i * 3 + 1) = n1;
        N_sub_undeformed.row(F_sub_undeformed.rows() - index.size()*3 + i * 3 + 2) = n2;

        Eigen::RowVector3d n0_deformed = ((v1_deformed - v0_deformed).cross(v_center_deformed - v1_deformed)).normalized();
        Eigen::RowVector3d n1_deformed = ((v2_deformed - v1_deformed).cross(v_center_deformed - v2_deformed)).normalized();
        Eigen::RowVector3d n2_deformed = ((v0_deformed - v2_deformed).cross(v_center_deformed - v0_deformed)).normalized();
        N_sub_deformed.row(F_sub_undeformed.rows() - index.size()*3 + i * 3) = n0_deformed;
        N_sub_deformed.row(F_sub_undeformed.rows() - index.size()*3 + i * 3 + 1) = n1_deformed;
        N_sub_deformed.row(F_sub_undeformed.rows() - index.size()*3 + i * 3 + 2) = n2_deformed;
    }
    //delete the old faces
    for (int i = 0; i < index.size(); i++) {
      int face_index = index[i];
      F_sub_undeformed.row(face_index) = F_sub_undeformed.row(F_sub_undeformed.rows() - 1 - i);
      N_sub_undeformed.row(face_index) = N_sub_undeformed.row(N_sub_undeformed.rows() - 1 - i);
      N_sub_deformed.row(face_index) = N_sub_deformed.row(N_sub_deformed.rows() - 1 - i);
    }
    F_sub_undeformed.conservativeResize(F_sub_undeformed.rows() - index.size(), 3);
    N_sub_undeformed.conservativeResize(N_sub_undeformed.rows() - index.size(), 3);
    N_sub_deformed.conservativeResize(N_sub_deformed.rows() - index.size(), 3);

    meshes.V_undeformed = V_sub_undeformed;
    meshes.F_undeformed = F_sub_undeformed;
    meshes.F_deformed = F_sub_undeformed;
    meshes.V_deformed = V_sub_deformed;
    meshes.N_undeformed_opt = N_sub_undeformed;
    meshes.N_deformed_opt = N_sub_deformed;


}