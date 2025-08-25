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


#include "SelfCollision.h"
#include "data.h"

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

int main(int argc, char *argv[]) {
    char *file_char = getCmdOption(argv, argv + argc, "-f");
    std::string file_name = file_char;
    igl::readOBJ(file_name, meshes.V_refer, meshes.F_refer);
    igl::readOBJ(file_name, meshes.V_undeformed, meshes.F_undeformed);
    igl::readOBJ(file_name, meshes.V_deformed, meshes.F_deformed);
    Eigen::MatrixXd &V_deformed = meshes.V_deformed;
    Eigen::MatrixXi &F_deformed = meshes.F_deformed;
    Eigen::MatrixXd &V_undeformed = meshes.V_undeformed;
    Eigen::MatrixXi &F_undeformed = meshes.F_undeformed;
    Eigen::MatrixXd &V_refer = meshes.V_refer;
    Eigen::MatrixXi &F_refer = meshes.F_refer;

    // get file name not including directory name
    std::string file_name_no_dir = file_name.substr(file_name.find_last_of("/\\") + 1);
    // no .obj
    file_name_no_dir = file_name_no_dir.substr(0, file_name_no_dir.find(".obj"));
    file_name_no_dir = file_name_no_dir.substr(file_name_no_dir.find_last_of("_\\") + 1);
    std::cout << "file_name_no_dir: " << file_name_no_dir << std::endl;
    int file_num = std::stoi(file_name_no_dir);
    file_num -= 100;
    std::string folder = file_name.substr(0, file_name.find_last_of("/\\"));
    std::string file_num_name = folder + "/deformed_" + std::to_string(file_num) + ".obj";
    Eigen::MatrixXi F_;
    igl::readOBJ(file_num_name, meshes.V_deformed_pre, F_);
    meshes.Vel = meshes.V_deformed - meshes.V_deformed_pre;




    Compute_SelfCollision(meshes);
    // visualize
    igl::opengl::glfw::Viewer viewer;

    viewer.data().set_mesh(V_deformed, F_deformed);
    // viewer.data().set_mesh(meshes.V_deformed_pre, F_deformed);
    viewer.data().set_face_based(true);
    Eigen::VectorXi EI;
    Eigen::MatrixXd EV;
    Eigen::MatrixXi IF,EE;
    Eigen::Array<bool,Eigen::Dynamic,1> CP;
    igl::predicates::find_self_intersections(V_deformed,F_deformed,IF,CP,EV,EE,EI);
    for (int i = 0; i < IF.rows(); i++) {
        int face_id_0 = IF(i, 0);
        int face_id_1 = IF(i, 1);
        int v1 = F_deformed(face_id_0, 0);
        int v2 = F_deformed(face_id_0, 1);
        int v3 = F_deformed(face_id_0, 2);
        int v4 = F_deformed(face_id_1, 0);
        int v5 = F_deformed(face_id_1, 1);
        int v6 = F_deformed(face_id_1, 2);
        Eigen::RowVector3d p1 = V_deformed.row(v1);
        Eigen::RowVector3d p2 = V_deformed.row(v2);
        Eigen::RowVector3d p3 = V_deformed.row(v3);
        Eigen::RowVector3d p4 = V_deformed.row(v4);
        Eigen::RowVector3d p5 = V_deformed.row(v5);
        Eigen::RowVector3d p6 = V_deformed.row(v6);
        viewer.data().add_points(p1, Eigen::RowVector3d(1, 0, 0));
        viewer.data().add_points(p2, Eigen::RowVector3d(1, 0, 0));
        viewer.data().add_points(p3, Eigen::RowVector3d(1, 0, 0));
        viewer.data().add_points(p4, Eigen::RowVector3d(0, 0, 1));
        viewer.data().add_points(p5, Eigen::RowVector3d(0, 0, 1));
        viewer.data().add_points(p6, Eigen::RowVector3d(0, 0, 1));

    }
    // visulize self_collision_pairs
    // for (int i = 0; i < meshes.self_collision_pairs.size(); i++) {
    //     std::cout << "self_collision_pairs: " << meshes.self_collision_pairs[i][0] << std::endl;
    //     int v1 = meshes.self_collision_pairs[i][0];
    //     int v2 = meshes.self_collision_pairs[i][1];
    //     int v3 = meshes.self_collision_pairs[i][2];
    //     int v4 = meshes.self_collision_pairs[i][3];
    //     int type = meshes.self_collision_pairs[i][4];
    //     std::cout << "v1" << v1 << "v2" << v2 << "v3" << v3 << "v4" << v4 << "type" << type << std::endl;
    //     if (type == 0) { // edge-edge
    //         //visaulize points pair
    //         viewer.data().add_points(V_deformed.row(v1), Eigen::RowVector3d(1, 0, 0));
    //         viewer.data().add_points(V_deformed.row(v2), Eigen::RowVector3d(1, 0, 0));
    //         viewer.data().add_points(V_deformed.row(v3), Eigen::RowVector3d(0, 0, 1));
    //         viewer.data().add_points(V_deformed.row(v4), Eigen::RowVector3d(0, 0, 1));
    //     }
    //     else if (type == 1) { // face-vertex
    //         viewer.data().add_points(V_deformed.row(v1), Eigen::RowVector3d(1, 0, 0));
    //         viewer.data().add_points(V_deformed.row(v2), Eigen::RowVector3d(1, 0, 0));
    //         viewer.data().add_points(V_deformed.row(v3), Eigen::RowVector3d(1, 0, 0));
    //         viewer.data().add_points(V_deformed.row(v4), Eigen::RowVector3d(1, 0, 0));
    //     }
    // }
    // change point size
    viewer.data().point_size = 20;



    // if user push space GetIntermidiateVelocityContactResolved(Meshes &meshes)
    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int modifiers) {
        if (key == ' ') {
            // GetIntermidiateVelocityContactResolved(meshes);
            std::vector< std::set<int> > aRIZ;
            Eigen::MatrixXd Vel0 = meshes.Vel;
            igl::adjacency_list(meshes.F_deformed, meshes.adjacency_list);
            for (int itr = 0; itr<1;itr++){
                Compute_SelfCollision(meshes);
                int nnode_riz = 0;
                for(int iriz=0;iriz<aRIZ.size();iriz++){
                nnode_riz += aRIZ[iriz].size();
                }
                std::cout << "  RIZ iter: " << itr << "    Contact Elem Size: " << meshes.self_collision_pairs.size() << "   NNode In RIZ: " << nnode_riz << std::endl;
                if( meshes.self_collision_pairs.size() == 0 ){
                    std::cout << "Resolved All Collisions : " << std::endl;
                    break;
                }
                MakeRigidImpactZone(aRIZ, meshes.self_collision_pairs, meshes.adjacency_list);
                for(int iriz=0;iriz<aRIZ.size();iriz++){
                    nnode_riz += aRIZ[iriz].size();
                    for (auto key : aRIZ[iriz]) {
                        std::cout << key << std::endl;
                    }
                }
                std::cout << "  RIZ iter: " << aRIZ.size() << "    Contact Elem Size: " << meshes.self_collision_pairs.size() << "   NNode In RIZ: " << nnode_riz << std::endl;
                ApplyRigidImpactZone
                (meshes.Vel, aRIZ,
                meshes.V_deformed_pre,
                Vel0);
                meshes.V_deformed = meshes.V_deformed_pre + meshes.Vel;
            }
            viewer.data().clear();
            // Update the mesh
            viewer.data().set_mesh(V_deformed, F_deformed);
            for (int i = 0; i < aRIZ.size(); i++) {
                for (auto it = aRIZ[i].begin(); it != aRIZ[i].end(); it++) {
                    viewer.data().add_points(V_deformed.row(*it), Eigen::RowVector3d(1, 0, 0));
                }
            }
            
            // for (int i = 0; i < meshes.self_collision_pairs.size(); i++) {
            //     int v1 = meshes.self_collision_pairs[i][0];
            //     int v2 = meshes.self_collision_pairs[i][1];
            //     int v3 = meshes.self_collision_pairs[i][2];
            //     int v4 = meshes.self_collision_pairs[i][3];
            //     int type = meshes.self_collision_pairs[i][4];
            //     if (type == 0) { // edge-edge
            //         //visaulize points pair
            //         viewer.data().add_points(V_deformed.row(v1), Eigen::RowVector3d(1, 0, 0));
            //         viewer.data().add_points(V_deformed.row(v2), Eigen::RowVector3d(1, 0, 0));
            //         viewer.data().add_points(V_deformed.row(v3), Eigen::RowVector3d(0, 0, 1));
            //         viewer.data().add_points(V_deformed.row(v4), Eigen::RowVector3d(0, 0, 1));
            //     }
            //     else if (type == 1) { // face-vertex
            //         viewer.data().add_points(V_deformed.row(v1), Eigen::RowVector3d(1, 0, 0));
            //         viewer.data().add_points(V_deformed.row(v2), Eigen::RowVector3d(1, 0, 0));
            //         viewer.data().add_points(V_deformed.row(v3), Eigen::RowVector3d(1, 0, 0));
            //         viewer.data().add_points(V_deformed.row(v4), Eigen::RowVector3d(1, 0, 0));
            //     }
            // }
        }
        return false;
    };

    viewer.launch();
            

            
}


