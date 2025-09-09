#include <iostream>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
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
#include <igl/unique_edge_map.h>
#include <igl/edge_flaps.h>
#include <igl/copyleft/cgal/is_self_intersecting.h>
#include <igl/predicates/find_self_intersections.h>
#include <igl/remove_unreferenced.h>
#include <igl/adjacency_list.h>
#include <random>


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
    static std::random_device rd;
    static std::mt19937 rng(rd());
    static std::uniform_real_distribution<> dist(0.0, 1.0);

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

        for (size_t i = 0; i < line.size() - 1; ++i) {
            int start_idx = line[i];
            int end_idx = line[i + 1];

            Eigen::RowVector3d start = V.row(start_idx);
            Eigen::RowVector3d end = V.row(end_idx);

            for (int j = 0; j < 3; ++j) {
                subdivision_points.push_back(start + (j + 1) * (end - start) / 4.0);
                subdivision_colors.push_back(color);
            }
        }
    }

    P = Eigen::MatrixXd(subdivision_points.size(), 3);
    C = Eigen::MatrixXd(subdivision_colors.size(), 3);

    for (size_t i = 0; i < subdivision_points.size(); ++i) {
        P.row(i) = subdivision_points[i];
        C.row(i) = subdivision_colors[i];
    }
}

void visualizeEdgeSubdivision(
    const Eigen::MatrixXi &uE,
    const Eigen::MatrixXd &V,
    const Eigen::VectorXd &Quad_Angle_sub_Vector,
    Eigen::MatrixXd &P,
    Eigen::MatrixXd &C
) {
    std::vector<Eigen::RowVector3d> subdivision_points;
    std::vector<Eigen::RowVector3d> subdivision_colors;

    for (int i = 0; i < uE.rows(); ++i) {
        if (Quad_Angle_sub_Vector(i) == 0.0) {
            continue;
        }

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




void read_txt(std::string filename, Eigen::MatrixXd &matrix) {
    std::vector<std::vector<double>> data;
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
    char *reference = getCmdOption(argv, argv + argc, "-r");
    char *biggest_char = getCmdOption(argv, argv + argc, "-b");
    std::string reference_file = std::string(reference);
    char *start_delta_char = getCmdOption(argv, argv + argc, "-s");
    std::string start_delta_string = std::string(start_delta_char);
    double start_delta = std::stod(start_delta_string);
    std::string folder = std::string(folder_char);

    std::string search_path = folder + "/deformed_*.obj";
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
    if (biggest_char != nullptr) {
        biggest = std::stoi(std::string(biggest_char));
    }

    
    std::string undeformed_file = folder + "/undeformed_" + std::to_string(biggest) + ".obj";
    std::string deformed_file = folder + "/deformed_" + std::to_string(biggest) + ".obj";



    Eigen::MatrixXd V_initial;
    Eigen::MatrixXi F_initial;

    Eigen::MatrixXd V_initial_un;
    Eigen::MatrixXi F_initial_un;


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

    igl::adjacency_list(meshes.F_undeformed, meshes.A);


    time_t now = time(0);
    tm *ltm = localtime(&now);
    int iteration = 0;
    iteration++;

    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);
    

     menu.callback_draw_viewer_window = [](){};

    menu.callback_draw_custom_window = [&]() {
    ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        if (ImGui::Button("Visualize the folding line", ImVec2(-1,0))) {
            Compute_Quad_Angle_sub(meshes);
            viewer.data().clear();
            viewer.data(0).clear();
            viewer.data(1).clear();
            viewer.data(2).clear();
            viewer.data(3).clear();
            viewer.data(4).clear();
            viewer.data(0).set_mesh(V_deformed, F_deformed);
            viewer.append_mesh();
            Eigen::MatrixXd P_deformed, C_deformed;
            visualizeEdgeSubdivision(meshes.uE, V_deformed, meshes.Quad_Angle_sub_Vector, P_deformed, C_deformed);
            viewer.data(0).add_points(P_deformed, C_deformed);
            viewer.data(0).point_size = 8;
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

        if (ImGui::Button("Connect the folding line", ImVec2(-1,0))) {
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
            viewer.data(0).set_mesh(V_deformed, F_deformed);
            viewer.append_mesh();
            Eigen::MatrixXd P_deformed, C_deformed;
            visualizeEdgeSubdivision(meshes.uE, V_deformed, meshes.Quad_Angle_sub_Vector, P_deformed, C_deformed);
            viewer.data(0).add_points(P_deformed, C_deformed);
            viewer.data(0).point_size = 8;
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
            return true;
        }

        if (ImGui::Button("Save the folding line", ImVec2(-1,0))) {
            Compute_Quad_Angle_sub(meshes);
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