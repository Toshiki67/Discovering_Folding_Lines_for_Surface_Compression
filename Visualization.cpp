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
#include "ConnectFold.h"
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
#include <random>
#include <omp.h>


#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/parula.h>
#include <igl/per_corner_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/doublearea.h>

Meshes meshes;
int x_width = -100;
std::vector<std::string> files;
std::vector<std::vector<int>> lines;
std::vector<int> MV;


char *getCmdOption(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

Eigen::RowVector3d random_color() {
    static std::random_device rd; // Seed generator
    static std::mt19937 rng(rd()); // Random number engine
    static std::uniform_real_distribution<> dist(0.0, 1.0); // Uniform distribution

    return Eigen::RowVector3d(dist(rng), dist(rng), dist(rng));
}

void visualize(const Eigen::MatrixXd& V,
               const Eigen::MatrixXi& uE,
               const std::vector<std::vector<int>>& lines,
               const std::vector<int>& MV,
               Eigen::MatrixXd& P, Eigen::MatrixXd& C) {
    std::vector<Eigen::RowVector3d> subdivision_points;
    std::vector<Eigen::RowVector3d> subdivision_colors;

    for (int line_i = 0; line_i < lines.size(); ++line_i) {
        const auto& line = lines[line_i];
        Eigen::RowVector3d color;
        if (MV[line_i] == 1) {
            color = Eigen::RowVector3d(1, 0, 0);
        } else {
            color = Eigen::RowVector3d(0, 0, 1);
        }
         // Assign a unique color to each line

        for (size_t i = 0; i < line.size() - 1; ++i) {
            int start_idx = line[i];
            int end_idx = line[i + 1];

            Eigen::RowVector3d start = V.row(start_idx);
            Eigen::RowVector3d end = V.row(end_idx);

            // Subdivide the edge into points
            for (int j = 0; j < 3; ++j) {
                subdivision_points.push_back(start + (j + 1) * (end - start) / 4.0);
                subdivision_colors.push_back(color);
            }
        }
    }

    // Convert points and colors to Eigen::MatrixXd for visualization
    P = Eigen::MatrixXd(subdivision_points.size(), 3);
    C = Eigen::MatrixXd(subdivision_colors.size(), 3);

    for (size_t i = 0; i < subdivision_points.size(); ++i) {
        P.row(i) = subdivision_points[i];
        C.row(i) = subdivision_colors[i];
    }
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
            if (Quad_Angle_sub_Vector(i) > 0.0) {
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
    Initialization(new_meshes);
    new_meshes.delta = mesh_sub.delta;
    mesh_sub = new_meshes;
    // mesh_sub.delta = 1e-4;
}


void read_txt(std::string filename, Eigen::MatrixXd &matrix) {
    std::vector<std::vector<double>> data; // Temporary storage for rows of the matrix
    std::string line;
    std::ifstream file(filename);


    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::vector<double> row;
        double value;

        while (stream >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    // Convert vector of vectors to Eigen::MatrixXd
    int rows = data.size();
    int cols = data[0].size();
    matrix = Eigen::MatrixXd(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = data[i][j];
        }
    }
}



int main(int argc, char *argv[]) {
    std::cout << "omp_get_max_threads: " << omp_get_max_threads() << std::endl;
    
    igl::default_num_threads(12);
    char *folder_char = getCmdOption(argv, argv + argc, "-f");
    char *start_delta_char = getCmdOption(argv, argv + argc, "-d");
    std::string start_delta_string = std::string(start_delta_char);
    double start_delta = std::stod(start_delta_string);
    // meshes.delta = 1e-6;
    std::string folder = std::string(folder_char);
    // std::string undeformed_file = folder + "/undeformed.obj";
    std::string search_path = folder + "/deformed_*.obj";
    std::cout << "search_path: " << search_path << std::endl;
    glob_t glob_result;
    glob(search_path.c_str(), GLOB_TILDE, NULL, &glob_result);
    for(unsigned int i=0; i<glob_result.gl_pathc; ++i){
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    int biggest = 0;
    for (int i = 0; i < files.size(); i++) {
        std::string filename = files[i];
        // split by _ and get the last element
        std::vector<std::string> split_string;
        std::string delimiter = "_";
        size_t pos = 0;
        std::string token;
        while ((pos = filename.find(delimiter)) != std::string::npos) {
            token = filename.substr(0, pos);
            split_string.push_back(token);
            filename.erase(0, pos + delimiter.length());
        }
        split_string.push_back(filename);
        std::string number = split_string[split_string.size() - 1];
        // delete .obj
        number = number.substr(0, number.size() - 4);
        int num = std::stoi(number);
        if (num > biggest) {
            biggest = num;
        }
    }
    char *biggest_char = getCmdOption(argv, argv + argc, "-b");
    if (biggest_char != nullptr) {
        biggest = std::stoi(std::string(biggest_char));
    }
    // biggest = 400000;

    std::string search_path_deformed = folder + "/particle_*.obj";
    std::cout << "search_path: " << search_path_deformed << std::endl;
    glob_t glob_result_deformed;
    glob(search_path_deformed.c_str(), GLOB_TILDE, NULL, &glob_result_deformed);
    std::vector<std::string> files_deformed;
    for(unsigned int i=0; i<glob_result_deformed.gl_pathc; ++i){
        files_deformed.push_back(std::string(glob_result_deformed.gl_pathv[i]));
    }
    globfree(&glob_result_deformed);
    int biggest_deformed = 0;
    for (int i = 0; i < files_deformed.size(); i++) {
        std::string filename = files_deformed[i];
        // split by _ and get the last element
        std::vector<std::string> split_string;
        std::string delimiter = "_";
        size_t pos = 0;
        std::string token;
        while ((pos = filename.find(delimiter)) != std::string::npos) {
            token = filename.substr(0, pos);
            split_string.push_back(token);
            filename.erase(0, pos + delimiter.length());
        }
        split_string.push_back(filename);
        std::string number = split_string[split_string.size() - 1];
        // delete .obj
        number = number.substr(0, number.size() - 4);
        int num = std::stoi(number);
        if (num > biggest_deformed) {
            biggest_deformed = num;
        }
    }

    // biggest = 400000;

    
    std::string undeformed_file = folder + "/undeformed_" + std::to_string(biggest) + ".obj";
    // std::string deformed_file = folder + "/result_2260.obj";
    // std::string undeformed_file = folder + "/result_0.obj";
    std::string deformed_file = folder + "/deformed_" + std::to_string(biggest) + ".obj";
    // std::string deformed_file = folder + "/30.obj";
    std::string reference_file = folder + "/0.obj";

    std::string initial_deformed_file = folder + "/" + std::to_string(biggest_deformed) + ".obj";

    Eigen::MatrixXd V_initial_deformed;
    Eigen::MatrixXi F_initial_deformed;
    igl::readOBJ(initial_deformed_file, V_initial_deformed, F_initial_deformed);
    V_initial_deformed = V_initial_deformed * 100;

    std::cout << "undeformed_file: " << undeformed_file << std::endl;
    std::cout << "deformed_file: " << deformed_file << std::endl;


    

    // std::string initial_deformed_file = folder + "/deformed_e_" + start_delta_string + ".obj";
    // std::string initial_deformed_file = folder + "/smoothed_mesh.obj";

    Eigen::MatrixXd V_initial;
    Eigen::MatrixXi F_initial;
    // igl::readOBJ(initial_deformed_file, V_initial, F_initial);

    // initial_deformed_file = folder + "/undeformed_1.obj";

    Eigen::MatrixXd V_initial_un;
    Eigen::MatrixXi F_initial_un;
    // igl::readOBJ(initial_deformed_file, V_initial_un, F_initial_un);


    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &V_refer = meshes.V_refer;
    Eigen::MatrixXi &F_refer = meshes.F_refer;

    igl::readOBJ(deformed_file, V_deformed, F_deformed);
    igl::readOBJ(undeformed_file, V_undeformed, F_undeformed);
    igl::readOBJ(reference_file, V_refer, F_refer);

    V_refer = V_refer * 100;

    read_txt(folder + "/deformed_" + std::to_string(biggest) + ".txt", meshes.N_deformed_opt);
    read_txt(folder + "/undeformed_" + std::to_string(biggest) + ".txt", meshes.N_undeformed_opt);

    // flip F column

    // F_deformed.col(0).swap(F_deformed.col(1));
    // F_undeformed.col(0).swap(F_undeformed.col(1));
    // F_refer.col(0).swap(F_refer.col(1));


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
    // igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed_opt);
    // igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed_opt);
    igl::adjacency_list(meshes.F_undeformed, meshes.A);
    // igl::per_face_normals(V_deformed, F_deformed, meshes.N_deformed_opt);
    // igl::per_face_normals(V_undeformed, F_undeformed, meshes.N_undeformed_opt);




    // initializeBoundary(V_refer_surface, F_refer_surface, V_refer, F_refer);

    // create directory with day for saving results
    time_t now = time(0);
    tm *ltm = localtime(&now);
    int iteration = 0;
    // igl::writeOBJ(folder + "/deformed_" + std::to_string(iteration) + ".obj", V_deformed, F_deformed);
    // igl::writeOBJ(folder + "/undeformed_" + std::to_string(iteration) + ".obj", V_refer, F_refer);
    iteration++;

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int modifiers) {
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
            meshes.delta = 1e-4;
        }
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
        if(key == 'E') {
            // compute area and visualize
            Eigen::VectorXd area_undeformed;
            igl::doublearea(V_undeformed, F_undeformed, area_undeformed);
            Eigen::VectorXd area_deformed;
            igl::doublearea(V_deformed, F_deformed, area_deformed);
            // compute the ratio
            Eigen::VectorXd ratio = area_deformed.array() / area_undeformed.array();
            std::cout << "ratio: " << ratio.transpose() << std::endl;
            //visualize the ratio if it is 1.5x or more red, if it is 0.5x or less blue, else colormap

            Eigen::MatrixXd C;
            // igl::parula(ratio, C);
            // set color to the mesh

        }


        if (key == 'V') {
            Compute_Quad_Angle_sub(meshes);
            viewer.data().clear();
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            viewer.data(4).clear();
            // Update the mesh
            viewer.data(0).set_mesh(V_deformed, F_deformed);
            // add point V_deformed 0 vertex
            viewer.append_mesh();
            Eigen::MatrixXd P_deformed, C_deformed;
            visualizeEdgeSubdivision(meshes.uE, V_deformed, meshes.Quad_Angle_sub_Vector, P_deformed, C_deformed);
            viewer.data(0).add_points(P_deformed, C_deformed);
            viewer.data(0).point_size = 8;
            //V_refer move y direction and make new mesh
            // Eigen::MatrixXd V_undeformed_new = V_undeformed.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            // viewer.data(1).set_mesh(V_undeformed_new, F_undeformed);
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
            viewer.append_mesh();
            Eigen::MatrixXd V_initial_deformed_new = V_initial_deformed.rowwise() - Eigen::RowVector3d(x_width, 0, 0);
            viewer.data(4).set_mesh(V_initial_deformed_new, F_initial_deformed);
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

        if (key == ' ') {
            

            // V_undeformed = V_deformed * 0.03 + V_undeformed * 0.97;

            igl::writeOBJ(folder + "/extracted.obj", V_undeformed, F_undeformed);

            std::string mountain;
            std::string valley;
            for (int i = 0; i < lines.size(); i++) {
                std::cout << "lines: " << lines[i].size() << std::endl;
                if (MV[i] == 1) {
                    for (int j = 0; j < lines[i].size(); j++) {
                        Eigen::RowVector3d p = V_undeformed.row(lines[i][j]);
                        mountain += std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]) + ",";
                    }
                    mountain += "\n";
                } else {
                    for (int j = 0; j < lines[i].size(); j++) {
                        Eigen::RowVector3d p = V_undeformed.row(lines[i][j]);
                        valley += std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]) + ",";
                    }
                    valley += "\n";
                }
            }
            std::ofstream mountain_file(folder + "/mountain.txt");
            mountain_file << mountain;
            mountain_file.close();
            std::ofstream valley_file(folder + "/valley.txt");
            valley_file << valley;
            valley_file.close();


            std::string mountain_deformed;
            std::string valley_deformed;
            for (int i = 0; i < lines.size(); i++) {
                std::cout << "lines: " << lines[i].size() << std::endl;
                if (MV[i] == 1) {
                    for (int j = 0; j < lines[i].size(); j++) {
                        Eigen::RowVector3d p = V_deformed.row(lines[i][j]);
                        mountain_deformed += std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]) + ",";
                    }
                    mountain_deformed += "\n";
                } else {
                    for (int j = 0; j < lines[i].size(); j++) {
                        Eigen::RowVector3d p = V_deformed.row(lines[i][j]);
                        valley_deformed += std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]) + ",";
                    }
                    valley_deformed += "\n";
                }
            }
            std::ofstream mountain_file_deformed(folder + "/mountain_deformed.txt");
            mountain_file_deformed << mountain_deformed;
            mountain_file_deformed.close();
            std::ofstream valley_file_deformed(folder + "/valley_deformed.txt");
            valley_file_deformed << valley_deformed;
            valley_file_deformed.close();


            // Eigen::MatrixXd V_undeformed_N;
            // igl::per_vertex_normals(V_undeformed, F_undeformed, V_undeformed_N);
            // Eigen::MatrixXd V_sum = Eigen::MatrixXd::Zero(V_undeformed.rows(), 3);
            // for (int i = 0; i < lines.size(); i++) {
            //     for (int j = 0; j < lines[i].size(); j++) {
            //         if (MV[i] == 1)
            //             V_sum.row(lines[i][j]) = V_sum.row(lines[i][j]) + V_undeformed_N.row(lines[i][j]) * 0.2;
            //         else
            //             V_sum.row(lines[i][j]) = V_sum.row(lines[i][j]) - V_undeformed_N.row(lines[i][j]) * 0.2;
            //     }
            // }

            // for (int i = 0; i < V_undeformed.rows(); i++) {
            //     if (V_sum.row(i).norm() > 1e-5) {
            //         V_undeformed.row(i) = V_undeformed.row(i) + V_sum.row(i).normalized() * 0.3;
            //     }
            // }
            V_undeformed = V_deformed * 0.2 + V_undeformed * 0.8;
            igl::writeOBJ(folder + "/extracted_fold.obj", V_undeformed, F_undeformed);
            mountain = "";
            valley = "";
            for (int i = 0; i < lines.size(); i++) {
                std::cout << "lines: " << lines[i].size() << std::endl;
                if (MV[i] == 1) {
                    for (int j = 0; j < lines[i].size(); j++) {
                        Eigen::RowVector3d p = V_undeformed.row(lines[i][j]);
                        mountain += std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]) + ",";
                    }
                    mountain += "\n";
                } else {
                    for (int j = 0; j < lines[i].size(); j++) {
                        Eigen::RowVector3d p = V_undeformed.row(lines[i][j]);
                        valley += std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]) + ",";
                    }
                    valley += "\n";
                }
            }
            std::ofstream mountain_file2(folder + "/mountain_fold.txt");
            mountain_file2 << mountain;
            mountain_file2.close();
            std::ofstream valley_file2(folder + "/valley_fold.txt");
            valley_file2 << valley;
            valley_file2.close();



            viewer.data().clear();
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            viewer.data(4).clear();
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
            visualize(V_undeformed_second, meshes.uE, lines, MV, P, C);
            viewer.data(3).add_points(P, C);
            viewer.append_mesh();
            Eigen::MatrixXd V_initial_deformed_new = V_initial_deformed.rowwise() - Eigen::RowVector3d(x_width, 0, 0);
            viewer.data(4).set_mesh(V_initial_deformed_new, F_initial_deformed);
        }

        if (key == 'C') {
            if (lines.empty()){
                Connect_Fold(meshes, lines);
                Verify_MV(meshes, lines, V_initial, F_initial, V_initial_un, F_initial_un, MV);
                Connect_Close_MV(meshes, lines, MV);
            }
            std::cout << "lines: " << lines.size() << std::endl;
            viewer.data().clear();
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            viewer.data(4).clear();
            // Update the mesh
            viewer.data(0).set_mesh(V_deformed, F_deformed);
            // add point V_deformed 0 vertex
            viewer.append_mesh();
            Eigen::MatrixXd P_deformed, C_deformed;
            visualizeEdgeSubdivision(meshes.uE, V_deformed, meshes.Quad_Angle_sub_Vector, P_deformed, C_deformed);
            viewer.data(0).add_points(P_deformed, C_deformed);
            viewer.data(0).point_size = 8;
            //V_refer move y direction and make new mesh
            // Eigen::MatrixXd V_undeformed_new = V_undeformed.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            // viewer.data(1).set_mesh(V_undeformed_new, F_undeformed);
            viewer.append_mesh();
            Eigen::MatrixXd V_refer_new = V_refer.rowwise() + Eigen::RowVector3d(x_width*2, 0, 0);
            viewer.data(2).set_mesh(V_refer_new, F_refer);
            viewer.append_mesh();
            Eigen::MatrixXd V_undeformed_second = V_undeformed.rowwise() + Eigen::RowVector3d(x_width, 0, 0);
            viewer.data(3).set_mesh(V_undeformed_second, F_undeformed);
            // set point size
            viewer.data(3).point_size = 8;
            Eigen::MatrixXd P, C;
            visualize(V_undeformed_second, meshes.uE, lines, MV, P, C);
            viewer.data(3).add_points(P, C);
             viewer.append_mesh();
            Eigen::MatrixXd V_initial_deformed_new = V_initial_deformed.rowwise() - Eigen::RowVector3d(x_width, 0, 0);
            viewer.data(4).set_mesh(V_initial_deformed_new, F_initial_deformed);
            return true;
        }

        if (key == 'F') {
            Eigen::MatrixXd V_mean = (meshes.V_deformed*0.2 + meshes.V_undeformed*0.8);
            meshes.V_deformed = V_mean;
            igl::unique_edge_map(F_undeformed, meshes.E, meshes.uE, meshes.EMAP, meshes.uE2E);
            igl::edge_flaps(meshes.F_undeformed, meshes.uE, meshes.EMAP, meshes.EF, meshes.EI);

            igl::per_face_normals(V_refer, F_refer, meshes.N_refer);
            SymmetricDirichlet_initailize(meshes);
            Uniform_initialize(meshes);
            Minimize_sub(meshes, 100);
            viewer.data().clear();
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            viewer.data(4).clear();
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
        }
        if (key == 'O') {
            Eigen::MatrixXd HN;
            Eigen::SparseMatrix<double> L,M,Minv;
            igl::cotmatrix(V_initial,F_initial,L);
            igl::massmatrix(V_initial,F_initial,igl::MASSMATRIX_TYPE_VORONOI,M);
            igl::invert_diag(M,Minv);
            // Laplace-Beltrami of position
            HN = -Minv*(L*V_initial);
            // Extract magnitude as mean curvature
            Eigen::VectorXd H = HN.rowwise().norm();

            // Compute curvature directions via quadric fitting
            Eigen::MatrixXd PD1,PD2;
            Eigen::VectorXd PV1,PV2;
            igl::principal_curvature(V_initial,F_initial,PD1,PD2,PV1,PV2);
            // mean curvature
            H = 0.5*(PV1+PV2);
            viewer.data().clear();
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            viewer.data(4).clear();
            viewer.data().set_mesh(V_initial, F_initial);

            viewer.data().set_data(H);
            const double avg = igl::avg_edge_length(V_initial,F_initial);

            // Draw a red segment parallel to the maximal curvature direction
            const Eigen::RowVector3d red(0.8,0.2,0.2),blue(0.2,0.2,0.8);
            viewer.data().add_edges(V_initial + PD1*avg, V_initial - PD1*avg, red);

            // Draw a blue segment parallel to the minimal curvature direction
            viewer.data().add_edges(V_initial + PD2*avg, V_initial - PD2*avg, blue);
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
// press "vc ""