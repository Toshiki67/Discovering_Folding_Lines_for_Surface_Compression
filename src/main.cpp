#include <iostream>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Dense>

#include <Eigen/Core>

#include <cassert>
#include <filesystem>
#include <fstream>

#include <glob.h>

#include "Constraint.h"
#include "Optimization.h"
#include "Boundary.h"
#include "EdgeFlip.h"
#include "EdgeCollapse.h"
#include "data.h"
#include "SymmetricDirichlet.h"
#include "Uniform.h"
#include "Angle_sub.h"
#include "SelfCollision.h"
#include <igl/project.h>
#include <igl/edge_topology.h>
#include <igl/per_face_normals.h>
#include <igl/centroid.h>
#include <igl/unique_edge_map.h>
#include <igl/edge_flaps.h>
#include <igl/copyleft/cgal/is_self_intersecting.h>
#include <igl/predicates/find_self_intersections.h>
#include <igl/remove_unreferenced.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/adjacency_list.h>
#include <chrono>

#include <omp.h>


Meshes meshes;
int x_width = -100;
std::vector<std::string> files;
int save_iteration = 0;


char *getCmdOption(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

void get_adjacent_faces(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &V_refer, Eigen::MatrixXd &V_deformed) {
    int target_faces = 15000;

    // スタートするフェイス（最初はフェイス0とする）
    // int start_face = 3000;
    int start_face = 25000;

    Eigen::SparseMatrix<int> A;
    igl::facet_adjacency_matrix(F, A);

    // 選択されたフェイスのインデックスを格納
    std::vector<int> selected_faces;
    selected_faces.push_back(start_face);

    // 探索用データ構造
    std::queue<int> face_queue;
    std::set<int> visited_faces;

    face_queue.push(start_face);
    visited_faces.insert(start_face);

    // 隣接フェイスを探索
    while (!face_queue.empty() && selected_faces.size() < target_faces) {
        int current_face = face_queue.front();
        face_queue.pop();

        // 隣接フェイスを探索
        for (Eigen::SparseMatrix<int>::InnerIterator it(A, current_face); it; ++it) {
            if (it.value() == 0) {
                continue;
            }
            int neighbor_face = it.row(); // 隣接フェイス

            // 未訪問なら追加
            if (visited_faces.count(neighbor_face) == 0) {
                face_queue.push(neighbor_face);
                visited_faces.insert(neighbor_face);
                selected_faces.push_back(neighbor_face);

                // 必要数に達したら終了
                if (selected_faces.size() >= target_faces) {
                    break;
                }
            }
        }
    }

    // 新しいメッシュを構築
    Eigen::MatrixXi F_new(selected_faces.size(), 3);
    for (size_t i = 0; i < selected_faces.size(); ++i) {
        F_new.row(i) = F.row(selected_faces[i]);
    }

    Eigen::MatrixXi F_new_new;
    Eigen::MatrixXd V_new;
    Eigen::VectorXi I;
    // igl::remove_unreferenced(V, F_new, V_new, F_new_new, I);

    Eigen::MatrixXd V_refer_new;
    Eigen::MatrixXd V_deformed_new;
    // igl::remove_unreferenced(V_refer, F, V_refer_new, F_new_new, I);
    // igl::remove_unreferenced(V_deformed, F, V_deformed_new, F_new_new, I);
    // V_refer = V_refer_new;
    // V_deformed = V_deformed_new;
    
    F = F_new;
    // V = V_new;
    // if adjacent faces are only 1, then delete that face
    std::vector<int> delete_faces;
    igl::facet_adjacency_matrix(F, A);
    for (int i = 0; i < F.rows(); i++) {
        if (A.outerIndexPtr()[i + 1] - A.outerIndexPtr()[i] == 1) {
            delete_faces.push_back(i);
        }
    }
    Eigen::MatrixXi F_new_new_new(F.rows() - delete_faces.size(), 3);
    int count = 0;
    for (int i = 0; i < F.rows(); i++) {
        if (std::find(delete_faces.begin(), delete_faces.end(), i) == delete_faces.end()) {
            F_new_new_new.row(count) = F.row(i);
            count++;
        }
    }
    F = F_new_new_new;
    igl::remove_unreferenced(V, F, V_new, F_new_new, I);

    // Eigen::MatrixXd V_refer_new;
    // Eigen::MatrixXd V_deformed_new;
    igl::remove_unreferenced(V_refer, F, V_refer_new, F_new_new, I);
    igl::remove_unreferenced(V_deformed, F, V_deformed_new, F_new_new, I);
    V_refer = V_refer_new;
    V_deformed = V_deformed_new;
    F = F_new_new;
    V = V_new;
}

void visualizeEdgeSubdivision(
    const Eigen::MatrixXi &uE, // エッジ情報
    const Eigen::MatrixXd &V, // 頂点位置
    const Eigen::VectorXd &Quad_Angle_sub_Vector, // 可視化したいエッジを指定
    Eigen::MatrixXd &P,
    Eigen::MatrixXd &C
) {
    // 可視化用のエッジ分割点を格納
    std::vector<Eigen::RowVector3d> subdivision_points;
    std::vector<Eigen::RowVector3d> subdivision_colors;

    for (int i = 0; i < uE.rows(); ++i) {
        if (Quad_Angle_sub_Vector(i) == 0.0) {
            continue; // "1" の要素に対応するエッジのみを処理
        }

        // エッジの始点と終点
        Eigen::RowVector3d start = V.row(uE(i, 0));
        Eigen::RowVector3d end = V.row(uE(i, 1));
        for (int j = 0; j < 3; ++j) {
            subdivision_points.push_back(start + (j + 1) * (end - start) / 4.0);
            if (Quad_Angle_sub_Vector(i) == 1.0) {
                subdivision_colors.push_back(Eigen::RowVector3d(1, 0, 0));
            } else {
                subdivision_colors.push_back(Eigen::RowVector3d(0, 0, 1));
            }
        }
    }
    P = Eigen::MatrixXd(subdivision_points.size(), 3);
    for (int i = 0; i < subdivision_points.size(); ++i) {
        P.row(i) = subdivision_points[i];
    }
    C = Eigen::MatrixXd(subdivision_points.size(), 3);
    for (int i = 0; i < subdivision_colors.size(); ++i) {
        C.row(i) = subdivision_colors[i];
    }
}

void Initialization(Meshes &mesh_sub) {
    igl::unique_edge_map(mesh_sub.F_undeformed, mesh_sub.E, mesh_sub.uE, mesh_sub.EMAP, mesh_sub.uE2E);
    igl::edge_flaps(mesh_sub.F_undeformed, mesh_sub.uE, mesh_sub.EMAP, mesh_sub.EF, mesh_sub.EI);
    initializeBoundary(mesh_sub);
    SymmetricDirichlet_initailize(mesh_sub);
    Uniform_initialize(mesh_sub);
    igl::adjacency_list(mesh_sub.F_undeformed, mesh_sub.A);
}


void EdgeFLip(Meshes &mesh_sub){
    Fold_EdgeFlip(mesh_sub);
    // Compute_EdgeFlip(mesh_sub);
    Meshes new_meshes;
    new_meshes.V_deformed = mesh_sub.V_deformed;
    new_meshes.F_deformed = mesh_sub.F_deformed;
    new_meshes.V_undeformed = mesh_sub.V_undeformed;
    new_meshes.F_undeformed = mesh_sub.F_deformed;
    new_meshes.V_refer = mesh_sub.V_refer;
    new_meshes.F_refer = mesh_sub.F_refer;
    new_meshes.N_deformed_opt = mesh_sub.N_deformed_opt;
    new_meshes.N_undeformed_opt = mesh_sub.N_undeformed_opt;
    new_meshes.N_refer = mesh_sub.N_refer;
    Initialization(new_meshes);
    new_meshes.delta = mesh_sub.delta;
    new_meshes.weight_angle = mesh_sub.weight_angle;
    new_meshes.weight_angle_sub = mesh_sub.weight_angle_sub;
    mesh_sub = new_meshes;
}

void EdgeCollapse(Meshes &mesh_sub){
    Compute_Quad_Angle_sub(mesh_sub);
    Compute_EdgeCollapse(mesh_sub);
    Meshes new_meshes;
    new_meshes.V_deformed = mesh_sub.V_deformed;
    new_meshes.F_deformed = mesh_sub.F_deformed;
    new_meshes.V_undeformed = mesh_sub.V_undeformed;
    new_meshes.F_undeformed = mesh_sub.F_deformed;
    new_meshes.V_refer = mesh_sub.V_refer;
    new_meshes.F_refer = mesh_sub.F_refer;
    new_meshes.N_deformed_opt = mesh_sub.N_deformed_opt;
    new_meshes.N_undeformed_opt = mesh_sub.N_undeformed_opt;
    new_meshes.N_refer = mesh_sub.N_refer;
    Compute_subdivide_mesh(new_meshes);

    Initialization(new_meshes);
    new_meshes.delta = mesh_sub.delta;
    new_meshes.weight_angle = mesh_sub.weight_angle;
    new_meshes.weight_angle_sub = mesh_sub.weight_angle_sub;
    mesh_sub = new_meshes;
    // mesh_sub.delta = 1e-4;
}


int main(int argc, char *argv[]) {
    std::cout << "omp_get_max_threads: " << omp_get_max_threads() << std::endl;
    
    igl::default_num_threads(12);
    char *folder_char = getCmdOption(argv, argv + argc, "-f");
    char *start_delta_char = getCmdOption(argv, argv + argc, "-d");
    std::string start_delta_string = std::string(start_delta_char);
    double start_delta = std::stod(start_delta_string);
    meshes.delta = std::pow(0.1, start_delta);
    std::string folder = std::string(folder_char);
    // std::string undeformed_file = folder + "/undeformed.obj";
    std::string search_path = folder + "/particle_*.obj";
    std::cout << "search_path: " << search_path << std::endl;
    glob_t glob_result;
    glob(search_path.c_str(), GLOB_TILDE, NULL, &glob_result);
    for(unsigned int i=0; i<glob_result.gl_pathc; ++i){
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);


    std::string undeformed_file = folder + "/0.obj";
    // std::string deformed_file = folder + "/230.obj";
    std::string deformed_file = folder + "/" + std::to_string((files.size() - 1)*10) + ".obj";
    // std::string deformed_file = folder + "/deformed_24900.obj";
    // std::string deformed_file = folder + "/30.obj";
    std::string reference_file = folder + "/0.obj";

    char *remesh = getCmdOption(argv, argv + argc, "-r");
    std::string remesh_string = std::string(remesh);
    if (remesh_string == "1") {
        undeformed_file = folder + "/V_undeformed.obj";
        deformed_file = folder + "/V_deformed.obj";
        // reference_file = folder + "V_undeformed.obj";
    } 


    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &V_refer = meshes.V_refer;
    Eigen::MatrixXi &F_refer = meshes.F_refer;

    igl::readOBJ(deformed_file, V_deformed, F_deformed);
    igl::readOBJ(undeformed_file, V_undeformed, F_undeformed);
    igl::readOBJ(reference_file, V_refer, F_refer);

    // flip F column

    // F_deformed.col(0).swap(F_deformed.col(1));
    // F_undeformed.col(0).swap(F_undeformed.col(1));
    // F_refer.col(0).swap(F_refer.col(1));
    

    V_deformed *= 100;
    V_undeformed *= 100;
    V_refer *= 100;

    // get_adjacent_faces(F_deformed, V_deformed, V_refer, V_undeformed);
    // F_refer = F_deformed;
    // F_undeformed = F_deformed;

    meshes.V_deformed_pre = V_deformed;
    meshes.Vel = meshes.V_deformed - meshes.V_deformed_pre;



    // bool is_intersecting_df = igl::copyleft::cgal::is_self_intersecting(V_deformed, F_deformed);
    // bool is_intersecting_uf = igl::copyleft::cgal::is_self_intersecting(V_undeformed, F_undeformed);
    // std::cout << "is_intersecting_df: " << is_intersecting_df << std::endl;
    // std::cout << "is_intersecting_uf: " << is_intersecting_uf << std::endl;


    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V_refer, F_refer);

    std::cout << "V_deformed: " << V_deformed.rows() << " " << V_deformed.cols() << std::endl;
    std::cout << "V_undeformed: " << V_undeformed.rows() << " " << V_undeformed.cols() << std::endl;
    std::cout << "V_refer: " << V_refer.rows() << " " << V_refer.cols() << std::endl;

    bool is_optimizing = false;

    igl::unique_edge_map(F_undeformed, meshes.E, meshes.uE, meshes.EMAP, meshes.uE2E);
    igl::edge_flaps(meshes.F_undeformed, meshes.uE, meshes.EMAP, meshes.EF, meshes.EI);

    igl::per_face_normals(V_refer, F_refer, meshes.N_refer);
    initializeBoundary(meshes);
    SymmetricDirichlet_initailize(meshes);
    Uniform_initialize(meshes);
    igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed_opt);
    igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed_opt);
    igl::adjacency_list(meshes.F_undeformed, meshes.A);




    // initializeBoundary(V_refer_surface, F_refer_surface, V_refer, F_refer);

    // create directory with day for saving results
    time_t now = time(0);
    tm *ltm = localtime(&now);
    int iteration = 0;
    igl::writeOBJ(folder + "/deformed_" + std::to_string(iteration) + ".obj", V_deformed, F_deformed);
    igl::writeOBJ(folder + "/undeformed_" + std::to_string(iteration) + ".obj", V_refer, F_refer);
    igl::writeOBJ(folder + "/refer_" + std::to_string(iteration) + ".obj", V_refer, F_refer);
    iteration++;

    // Set the callback function for the viewer
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) {
        if (is_optimizing) {
            Newton(meshes);
            std::cout << "Save results..." << std::endl;
            igl::writeOBJ(folder + "/deformed_" + std::to_string(iteration) + ".obj", V_deformed, F_deformed);
            igl::writeOBJ(folder + "/refer_" + std::to_string(iteration) + ".obj", V_refer, F_refer);
            iteration++;
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            // Update the mesh
            viewer.data(0).set_mesh(V_deformed, F_deformed);
            // add point V_deformed 0 vertex
            viewer.append_mesh();
            //V_refer move y direction and make new mesh
            Eigen::MatrixXd V_undeformed_new = V_undeformed.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(1).set_mesh(V_undeformed_new, F_undeformed);
            viewer.append_mesh();
            Eigen::MatrixXd V_refer_new = V_refer.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(2).set_mesh(V_refer_new, F_refer);
            viewer.append_mesh();
            Eigen::MatrixXd V_undeformed_second = V_undeformed.rowwise() + Eigen::RowVector3d(x_width, 0, 0);
            viewer.data(3).set_mesh(V_undeformed_second, F_undeformed);
            is_optimizing = false;
        }
        

        // int i = 0;
        // Eigen::Vector2i face = meshes.EF.row(i);
        // int f1 = face(0);
        // int f2 = face(1);
        // std::cout << "f1: " << f1 << " f2: " << f2 << std::endl;
        // // get the angles
        // Eigen::MatrixXd points(4, 3);
        // points.row(0) = V_undeformed.row(F_undeformed(f1, (meshes.EI(i, 0) + 1) % 3)) + Eigen::RowVector3d(x_width*2, 0, 0);
        // points.row(1) = V_undeformed.row(F_undeformed(f1, (meshes.EI(i, 0) + 2) % 3)) + Eigen::RowVector3d(x_width*2, 0, 0);
        // points.row(2) = V_undeformed.row(F_undeformed(f2, (meshes.EI(i, 1) + 1) % 3))+ Eigen::RowVector3d(x_width*2 + 0.01, 0, 0);
        // points.row(3) = V_undeformed.row(F_undeformed(f2, (meshes.EI(i, 1) + 2) % 3)) + Eigen::RowVector3d(x_width*2 + 0.01, 0, 0);
        // Eigen::MatrixXd Colors(4, 3);
        // Colors << 1, 0, 0,
        //         0, 1, 0,
        //         0, 0, 1,
        //         1, 1, 0;
        // viewer.data(0).add_points(points, Colors);

        return false;
    };

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int modifiers) {
        if (key == ' ') {
            is_optimizing = !is_optimizing;
        }
        if (key == 'A') {
            std::cout << "EdgeFlip" << std::endl;
            EdgeFLip(meshes);
            // bool is_intersecting_df = igl::copyleft::cgal::is_self_intersecting(V_deformed, F_deformed);
            // bool is_intersecting_uf = igl::copyleft::cgal::is_self_intersecting(V_undeformed, F_undeformed);
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            // Update the mesh
            viewer.data(0).set_mesh(V_deformed, F_deformed);
            // add point V_deformed 0 vertex
            viewer.append_mesh();
            //V_refer move y direction and make new mesh
            Eigen::MatrixXd V_undeformed_new = V_undeformed.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(1).set_mesh(V_undeformed_new, F_undeformed);
            viewer.append_mesh();
            Eigen::MatrixXd V_refer_new = V_refer.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(2).set_mesh(V_refer_new, F_refer);
            viewer.append_mesh();
            Eigen::MatrixXd V_undeformed_second = V_undeformed.rowwise() + Eigen::RowVector3d(x_width, 0, 0);
            viewer.data(3).set_mesh(V_undeformed_second, F_undeformed);
            is_optimizing = false;
        }
        if (key == 'B') {
            std::cout << "EdgeCollapse" << std::endl;
            EdgeCollapse(meshes);

            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            // Update the mesh
            viewer.data(0).set_mesh(V_deformed, F_deformed);
            // add point V_deformed 0 vertex
            viewer.append_mesh();
            //V_refer move y direction and make new mesh
            Eigen::MatrixXd V_undeformed_new = V_undeformed.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(1).set_mesh(V_undeformed_new, F_undeformed);
            viewer.append_mesh();
            Eigen::MatrixXd V_refer_new = V_refer.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(2).set_mesh(V_refer_new, F_refer);
            viewer.append_mesh();
            Eigen::MatrixXd V_undeformed_second = V_undeformed.rowwise() + Eigen::RowVector3d(x_width, 0, 0);
            viewer.data(3).set_mesh(V_undeformed_second, F_undeformed);
            is_optimizing = false;
        }
        // if (key == 'C') {
        //     Eigen::VectorXi EI;
        //     Eigen::MatrixXd EV;
        //     Eigen::MatrixXi IF,EE;
        //     Eigen::Array<bool,Eigen::Dynamic,1> CP;
        //     igl::predicates::find_self_intersections(meshes.V_deformed,meshes.F_deformed,IF,CP,EV,EE,EI);
        //     std::cout << "Found " << IF.rows() << " self intersecting pairs" << std::endl;
        //     viewer.data().set_edges(EV,EE, Eigen::RowVector3d(1,1,1));
        //     viewer.data().set_points(EV, Eigen::RowVector3d(1,1,1));
        //     Eigen::VectorXi I;
        //     igl::unique(IF,I);
        //     Eigen::VectorXd D = Eigen::MatrixXd::Zero(meshes.F_deformed.rows(),1);
        //     D(I).setConstant(1.0);
        //     viewer.data(0).set_data(D,0,1,igl::COLOR_MAP_TYPE_MAGMA);
        // }
        if (key == 'D') {
            Minimize(meshes, 100);
            viewer.data().clear();
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            // Update the mesh
            viewer.data(0).set_mesh(V_deformed, F_deformed);
            // add point V_deformed 0 vertex
            viewer.append_mesh();
            Eigen::MatrixXd P_deformed, C_deformed;
            visualizeEdgeSubdivision(meshes.uE, V_deformed, meshes.Quad_Angle_sub_Vector, P_deformed, C_deformed);
            viewer.data(0).add_points(P_deformed, C_deformed);
            viewer.data(0).point_size = 8;
            //V_refer move y direction and make new mesh
            Eigen::MatrixXd V_undeformed_new = V_undeformed.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(1).set_mesh(V_undeformed_new, F_undeformed);
            viewer.append_mesh();
            Eigen::MatrixXd V_refer_new = V_refer.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(2).set_mesh(V_refer_new, F_refer);
            viewer.append_mesh();
            Eigen::MatrixXd V_undeformed_second = V_undeformed.rowwise() + Eigen::RowVector3d(x_width, 0, 0);
            viewer.data(3).set_mesh(V_undeformed_second, F_undeformed);
            // set point size
            viewer.data(3).point_size = 8;
            Eigen::MatrixXd P, C;
            visualizeEdgeSubdivision(meshes.uE, V_undeformed_second, meshes.Quad_Angle_sub_Vector, P, C);
            viewer.data(3).add_points(P, C);
            return true;
        }
        if (key == 'V') {
            Compute_Quad_Angle_sub(meshes);
            viewer.data().clear();
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            // Update the mesh
            viewer.data(0).set_mesh(V_deformed, F_deformed);
            // add point V_deformed 0 vertex
            viewer.append_mesh();
            Eigen::MatrixXd P_deformed, C_deformed;
            visualizeEdgeSubdivision(meshes.uE, V_deformed, meshes.Quad_Angle_sub_Vector, P_deformed, C_deformed);
            viewer.data(0).add_points(P_deformed, C_deformed);
            viewer.data(0).point_size = 8;
            //V_refer move y direction and make new mesh
            Eigen::MatrixXd V_undeformed_new = V_undeformed.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(1).set_mesh(V_undeformed_new, F_undeformed);
            viewer.append_mesh();
            Eigen::MatrixXd V_refer_new = V_refer.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(2).set_mesh(V_refer_new, F_refer);
            viewer.append_mesh();
            Eigen::MatrixXd V_undeformed_second = V_undeformed.rowwise() + Eigen::RowVector3d(x_width, 0, 0);
            viewer.data(3).set_mesh(V_undeformed_second, F_undeformed);
            // set point size
            viewer.data(3).point_size = 8;
            Eigen::MatrixXd P, C;
            visualizeEdgeSubdivision(meshes.uE, V_undeformed_second, meshes.Quad_Angle_sub_Vector, P, C);
            viewer.data(3).add_points(P, C);
            return true;
        }
        if (key == 'R') {
            AdamOptimizer new_adam{1e-2, 0.9, 0.999, 1e-8};
            AdamOptimizer new_adam_n{1e-2, 0.9, 0.999, 1e-8};
            meshes.adam = new_adam;
            meshes.adam_n = new_adam_n;
        }
        if (key == 'E') {
            meshes.delta = meshes.delta * 0.1;
        }
        if (key == 'T') {
            meshes.delta = meshes.delta * 10;
        }
        if (key == 'Y') {
            igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed_opt);
            igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed_opt);
        }
        if (key == 'U') {
            meshes.weight_angle = meshes.weight_angle * 2.0;
            std::cout << "weight_angle_sub: " << meshes.weight_angle_sub << std::endl;
            std::cout << "weight_angle: " << meshes.weight_angle << std::endl;
        }
        if (key == 'I') {
            meshes.weight_angle = meshes.weight_angle * 0.5;
            std::cout << "weight_angle_sub: " << meshes.weight_angle_sub << std::endl;
            std::cout << "weight_angle: " << meshes.weight_angle << std::endl;
        }
        if (key == 'O') {
            meshes.weight_angle_sub = meshes.weight_angle_sub * 2.0;
            std::cout << "weight_angle_sub: " << meshes.weight_angle_sub << std::endl;
            std::cout << "weight_angle: " << meshes.weight_angle << std::endl;
        }
        if (key == 'P') {
            meshes.weight_angle_sub = meshes.weight_angle_sub * 0.5;
            std::cout << "weight_angle_sub: " << meshes.weight_angle_sub << std::endl;
            std::cout << "weight_angle: " << meshes.weight_angle << std::endl;
        }
        if (key == 'S')
        {
            while (meshes.delta < 1e-4*9.0){
                Minimize(meshes, 100);
                // EdgeFLip(meshes);
                // EdgeCollapse(meshes);
                igl::writeOBJ(folder + "/deformed_e_" + std::to_string(int(-log10(meshes.delta))) + ".obj", meshes.V_deformed, meshes.F_deformed);
                igl::writeOBJ(folder + "/undeformed_" + std::to_string(int(-log10(meshes.delta))) + ".obj", meshes.V_undeformed, meshes.F_undeformed);
                meshes.delta = meshes.delta * 10;
            }
            igl::writeOBJ(folder + "/deformed_" + std::to_string(1) + ".obj", meshes.V_deformed, meshes.F_deformed);
            igl::writeOBJ(folder + "/undeformed_" + std::to_string(1) + ".obj", meshes.V_undeformed, meshes.F_undeformed);
            std::ofstream file_undeformed(folder + "/undeformed_" + std::to_string(1) + ".txt");
            if (file_undeformed.is_open()) {
                file_undeformed << meshes.N_undeformed_opt << std::endl;
            } else {
                std::cerr << "Unable to open file for writing." << std::endl;
            }
            std::ofstream file_deformed(folder + "/deformed_" + std::to_string(1) + ".txt");
            if (file_deformed.is_open()) {
                file_deformed << meshes.N_deformed_opt << std::endl;
            } else {
                std::cerr << "Unable to open file for writing." << std::endl;
            }

            int save_iteration = 0;
            bool decrease = false;
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 20; j++)
                {
                    Minimize(meshes, 100);
                    save_iteration = save_iteration + 100;
                    if ((j+1) % 5 == 0) {
                        EdgeFLip(meshes);
                        EdgeCollapse(meshes);
                    }
                    igl::writeOBJ(folder + "/deformed_" + std::to_string(save_iteration) + ".obj", meshes.V_deformed, meshes.F_deformed);
                    igl::writeOBJ(folder + "/undeformed_" + std::to_string(save_iteration) + ".obj", meshes.V_undeformed, meshes.F_undeformed);
                    std::ofstream file_undeformed(folder + "/undeformed_" + std::to_string(save_iteration) + ".txt");
                    if (file_undeformed.is_open()) {
                        file_undeformed << meshes.N_undeformed_opt << std::endl;
                    } else {
                        std::cerr << "Unable to open file for writing." << std::endl;
                    }
                    std::ofstream file_deformed(folder + "/deformed_" + std::to_string(save_iteration) + ".txt");
                    if (file_deformed.is_open()) {
                        file_deformed << meshes.N_deformed_opt << std::endl;
                    } else {
                        std::cerr << "Unable to open file for writing." << std::endl;
                    }
                    if (meshes.delta > 1e-4 * 9.0&& j >= 10)
                        break;
                }
                // igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed_opt);
                // igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed_opt);
                // if (decrease)
                //     meshes.delta = meshes.delta * 0.1;
                // else
                //     meshes.delta = meshes.delta * 10;
                // if (meshes.delta < 1e-6 * 9.0 && decrease)
                //     decrease = false;
                // if (meshes.delta > 1e-3 * 0.9 && !decrease)
                //     decrease = true;
                    
                if (meshes.delta > 1e-7 * 9.0){
                    meshes.delta = meshes.delta * 0.1;
                    igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed_opt);
                    igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed_opt);
                }
                
            } 
        }
        if (key == 'Z')
        {
            std::chrono::system_clock::time_point  start, end;
            start = std::chrono::system_clock::now();
            for (int num = 0; num < 10; num++){
                bool decrease = false;
                for (int i = 0; i < 100; i++)
                {
                    for (int j = 0; j < 20; j++)
                    {
                        Minimize(meshes, 100);
                        save_iteration = save_iteration + 100;
                        // if (save_iteration % 40000 == 0) {
                        //     if (meshes.delta > 1e-5 * 9.0) {
                        //         meshes.delta = meshes.delta * 0.1;
                        //         // igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed_opt);
                        //         // igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed_opt);
                        //     }
                        // }
                        if ((j+1) % 1 == 0) {
                            meshes.Vel = meshes.V_deformed - meshes.V_deformed_pre;
                            // igl::writeOBJ(folder + "/pre_deformed_" + std::to_string(save_iteration) + ".obj", meshes.V_deformed, meshes.F_deformed);
                            Compute_SelfCollision(meshes);
                            GetIntermidiateVelocityContactResolved(meshes);
                            Compute_SelfCollision(meshes);
                            if (save_iteration > 50000){
                                EdgeFLip(meshes);
                            }
                            EdgeCollapse(meshes);
                            meshes.V_deformed_pre = meshes.V_deformed;
                            if (save_iteration % 1000 == 0) {
                                igl::writeOBJ(folder + "/deformed_" + std::to_string(save_iteration) + ".obj", meshes.V_deformed, meshes.F_deformed);
                                igl::writeOBJ(folder + "/undeformed_" + std::to_string(save_iteration) + ".obj", meshes.V_undeformed, meshes.F_undeformed);
                                end = std::chrono::system_clock::now();
                                double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
                                std::ofstream file_time(folder + "/FLE_time_" + std::to_string(save_iteration) + ".txt");
                                if (file_time.is_open()) {
                                    file_time << elapsed << std::endl;
                                } else {
                                    std::cerr << "Unable to open file for writing." << std::endl;
                                }
                                // close the file
                                file_time.close();
                                std::ofstream file_undeformed(folder + "/undeformed_" + std::to_string(save_iteration) + ".txt");
                                if (file_undeformed.is_open()) {
                                    file_undeformed << meshes.N_undeformed_opt << std::endl;
                                } else {
                                    std::cerr << "Unable to open file for writing." << std::endl;
                                }
                                // close the file
                                file_undeformed.close();
                                std::ofstream file_deformed(folder + "/deformed_" + std::to_string(save_iteration) + ".txt");
                                if (file_deformed.is_open()) {
                                    file_deformed << meshes.N_deformed_opt << std::endl;
                                } else {
                                    std::cerr << "Unable to open file for writing." << std::endl;
                                }
                                // close the file
                                file_deformed.close();
                            }
                        }
                        if (meshes.delta > 1e-4 * 9.0&& j >= 10)
                            break;
                    } 
                } 
            }
        }
        if (key == 'X')
        {
            bool decrease = false;
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 20; j++)
                {
                    Minimize(meshes, 100);
                    save_iteration = save_iteration + 100;
                    if ((j+1) % 5 == 0) {
                        int simplify = 3;
                        for (int k = 0; k < simplify; k++)
                        {
                            meshes.delta = meshes.delta * 10;
                            Minimize(meshes, 10);
                        }
                        for (int k = 0; k < simplify; k++)
                        {
                            meshes.delta = meshes.delta * 0.1;
                        }
                        EdgeFLip(meshes);
                        EdgeCollapse(meshes);
                    }
                    igl::writeOBJ(folder + "/deformed_" + std::to_string(save_iteration) + ".obj", meshes.V_deformed, meshes.F_deformed);
                    igl::writeOBJ(folder + "/undeformed_" + std::to_string(save_iteration) + ".obj", meshes.V_undeformed, meshes.F_undeformed);
                    std::ofstream file_undeformed(folder + "/undeformed_" + std::to_string(save_iteration) + ".txt");
                    if (file_undeformed.is_open()) {
                        file_undeformed << meshes.N_undeformed_opt << std::endl;
                    } else {
                        std::cerr << "Unable to open file for writing." << std::endl;
                    }
                    std::ofstream file_deformed(folder + "/deformed_" + std::to_string(save_iteration) + ".txt");
                    if (file_deformed.is_open()) {
                        file_deformed << meshes.N_deformed_opt << std::endl;
                    } else {
                        std::cerr << "Unable to open file for writing." << std::endl;
                    }
                    if (meshes.delta > 1e-4 * 9.0&& j >= 10)
                        break;
                } 
            } 
        }

        if (key == 'N')
        {
            std::chrono::system_clock::time_point  start, end;
            start = std::chrono::system_clock::now();
            for (int num = 0; num < 1; num++){
                bool decrease = false;
                for (int i = 0; i < 1; i++)
                {
                    for (int j = 0; j < 1; j++)
                    {
                        Newton(meshes, 10);
                        // save_iteration = save_iteration + 100;
                        // if ((j+1) % 1 == 0) {
                        //     meshes.Vel = meshes.V_deformed - meshes.V_deformed_pre;
                        //     // igl::writeOBJ(folder + "/pre_deformed_" + std::to_string(save_iteration) + ".obj", meshes.V_deformed, meshes.F_deformed);
                        //     Compute_SelfCollision(meshes);
                        //     GetIntermidiateVelocityContactResolved(meshes);
                        //     Compute_SelfCollision(meshes);
                        //     if (save_iteration > 50000){
                        //         EdgeFLip(meshes);
                        //     }
                        //     EdgeCollapse(meshes);
                        //     meshes.V_deformed_pre = meshes.V_deformed;
                        //     if (save_iteration % 1000 == 0) {
                        //         // igl::writeOBJ(folder + "/deformed_" + std::to_string(save_iteration) + ".obj", meshes.V_deformed, meshes.F_deformed);
                        //         // igl::writeOBJ(folder + "/undeformed_" + std::to_string(save_iteration) + ".obj", meshes.V_undeformed, meshes.F_undeformed);
                        //         end = std::chrono::system_clock::now();
                        //         double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
                        //         std::ofstream file_time(folder + "/FLE_time_" + std::to_string(save_iteration) + ".txt");
                        //         if (file_time.is_open()) {
                        //             file_time << elapsed << std::endl;
                        //         } else {
                        //             std::cerr << "Unable to open file for writing." << std::endl;
                        //         }
                        //         std::ofstream file_undeformed(folder + "/undeformed_" + std::to_string(save_iteration) + ".txt");
                        //         if (file_undeformed.is_open()) {
                        //             file_undeformed << meshes.N_undeformed_opt << std::endl;
                        //         } else {
                        //             std::cerr << "Unable to open file for writing." << std::endl;
                        //         }
                        //         std::ofstream file_deformed(folder + "/deformed_" + std::to_string(save_iteration) + ".txt");
                        //         if (file_deformed.is_open()) {
                        //             file_deformed << meshes.N_deformed_opt << std::endl;
                        //         } else {
                        //             std::cerr << "Unable to open file for writing." << std::endl;
                        //         }
                        //     }
                        // }
                        // if (meshes.delta > 1e-4 * 9.0&& j >= 10)
                        //     break;
                    } 
                } 
            }
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            // Update the mesh
            viewer.data(0).set_mesh(V_deformed, F_deformed);
            // add point V_deformed 0 vertex
            viewer.append_mesh();
            //V_refer move y direction and make new mesh
            Eigen::MatrixXd V_undeformed_new = V_undeformed.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(1).set_mesh(V_undeformed_new, F_undeformed);
            viewer.append_mesh();
            Eigen::MatrixXd V_refer_new = V_refer.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(2).set_mesh(V_refer_new, F_refer);
            viewer.append_mesh();
            Eigen::MatrixXd V_undeformed_second = V_undeformed.rowwise() + Eigen::RowVector3d(x_width, 0, 0);
            viewer.data(3).set_mesh(V_undeformed_second, F_undeformed);
        }
        return false;
    };


    viewer.data(0).clear();
    viewer.data(1).clear();
    viewer.data(2).clear();
    viewer.data(3).clear();
    // Update the mesh
    viewer.data(0).set_mesh(V_deformed, F_deformed);
    // add point V_deformed 0 vertex
    viewer.append_mesh();
    //V_refer move y direction and make new mesh
    Eigen::MatrixXd V_undeformed_new = V_undeformed.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
    viewer.data(1).set_mesh(V_undeformed_new, F_undeformed);
    viewer.append_mesh();
    Eigen::MatrixXd V_refer_new = V_refer.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
    viewer.data(2).set_mesh(V_refer_new, F_refer);
    viewer.append_mesh();
    Eigen::MatrixXd V_undeformed_second = V_undeformed.rowwise() + Eigen::RowVector3d(x_width, 0, 0);
    viewer.data(3).set_mesh(V_undeformed_second, F_undeformed);




    viewer.launch();


    return 0;
}
