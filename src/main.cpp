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
// #include <igl/project.h>
#include <igl/edge_topology.h>
#include <igl/per_face_normals.h>
// #include <igl/centroid.h>
#include <igl/unique_edge_map.h>
#include <igl/edge_flaps.h>
#include <igl/copyleft/cgal/is_self_intersecting.h>
#include <igl/predicates/find_self_intersections.h>
// #include <igl/remove_unreferenced.h>
// #include <igl/facet_adjacency_matrix.h>
#include <igl/adjacency_list.h>


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
}


int main(int argc, char *argv[]) {
    std::cout << "omp_get_max_threads: " << omp_get_max_threads() << std::endl;
    
    igl::default_num_threads(12);
    char *reference = getCmdOption(argv, argv + argc, "-r");
    char *undeformed = getCmdOption(argv, argv + argc, "-u");
    char *deformed = getCmdOption(argv, argv + argc, "-d");
    char *start_delta_char = getCmdOption(argv, argv + argc, "-s");
    std::string start_delta_string = std::string(start_delta_char);
    double start_delta = std::stod(start_delta_string);
    meshes.delta = std::pow(0.1, start_delta);
    std::string reference_file = std::string(reference);
    std::string undeformed_file = std::string(undeformed);
    std::string deformed_file = std::string(deformed);
    // get folder path from undeformed_file
    std::string folder = undeformed_file.substr(0, undeformed_file.find_last_of("/"));


    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &V_refer = meshes.V_refer;
    Eigen::MatrixXi &F_refer = meshes.F_refer;

    igl::readOBJ(deformed_file, V_deformed, F_deformed);
    igl::readOBJ(undeformed_file, V_undeformed, F_undeformed);
    igl::readOBJ(reference_file, V_refer, F_refer);
    

    V_deformed *= 100;
    V_undeformed *= 100;
    V_refer *= 100;

    meshes.V_deformed_pre = V_deformed;
    meshes.Vel = meshes.V_deformed - meshes.V_deformed_pre;



    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V_refer, F_refer);

    std::cout << "V_deformed: " << V_deformed.rows() << " " << V_deformed.cols() << std::endl;
    std::cout << "V_undeformed: " << V_undeformed.rows() << " " << V_undeformed.cols() << std::endl;
    std::cout << "V_refer: " << V_refer.rows() << " " << V_refer.cols() << std::endl;


    igl::unique_edge_map(F_undeformed, meshes.E, meshes.uE, meshes.EMAP, meshes.uE2E);
    igl::edge_flaps(meshes.F_undeformed, meshes.uE, meshes.EMAP, meshes.EF, meshes.EI);

    igl::per_face_normals(V_refer, F_refer, meshes.N_refer);
    initializeBoundary(meshes);
    SymmetricDirichlet_initailize(meshes);
    Uniform_initialize(meshes);
    igl::per_face_normals(meshes.V_undeformed, meshes.F_undeformed, meshes.N_undeformed_opt);
    igl::per_face_normals(meshes.V_deformed, meshes.F_deformed, meshes.N_deformed_opt);
    igl::adjacency_list(meshes.F_undeformed, meshes.A);

    int iteration = 0;
    igl::writeOBJ(folder + "/deformed_" + std::to_string(iteration) + ".obj", V_deformed, F_deformed);
    igl::writeOBJ(folder + "/undeformed_" + std::to_string(iteration) + ".obj", V_refer, F_refer);
    igl::writeOBJ(folder + "/refer_" + std::to_string(iteration) + ".obj", V_refer, F_refer);
    iteration++;

    int max_iteration = 100;

    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);
    

     menu.callback_draw_viewer_window = [](){};
    menu.callback_draw_custom_window = [&]()
  {
     // menu size
    // ImGui::SetNextWindowSize(ImVec2(6000, 2000), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    // Add new group
    if (ImGui::CollapsingHeader("Adam", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::InputInt("000 iterations", &max_iteration))
        {
            if (max_iteration < 1)
                max_iteration = 1;
        }
      // Add a button
      if (ImGui::Button("Minimize", ImVec2(-1,0)))
      {
        for (int num = 0; num < max_iteration*10; num++){    
                Minimize(meshes, 100);
                save_iteration = save_iteration + 100;
                meshes.Vel = meshes.V_deformed - meshes.V_deformed_pre;
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
                std::cout << "iteration: " << save_iteration << std::endl;
            }
      }
    }
    ImGui::End();
  };

    

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int modifiers) {

        if (key == 'N')
        {
            for (int num = 0; num < 1; num++){
                for (int i = 0; i < 1; i++)
                {
                    for (int j = 0; j < 1; j++)
                    {
                        Newton(meshes, 10);
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
    viewer.data(0).set_mesh(V_deformed, F_deformed);
    viewer.append_mesh();
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
