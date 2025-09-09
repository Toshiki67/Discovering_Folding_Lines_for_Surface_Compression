#include "EdgeFlip.h"
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
#include <igl/is_edge_manifold.h>
#include <igl/predicates/find_self_intersections.h>

#include "Boundary.h"


void Compute_EdgeFlip(Meshes &meshes){
    Eigen::MatrixXd angles;
    igl::internal_angles(meshes.V_undeformed, meshes.F_undeformed, angles);
    auto corner = [](
        const Eigen::RowVector3d & x, 
        const Eigen::RowVector3d & y, 
        const Eigen::RowVector3d & z)
        {
        Eigen::RowVector3d v1 = (x-y).normalized();
        Eigen::RowVector3d v2 = (z-y).normalized();
        double s = v1.cross(v2).norm();
        double c = v1.dot(v2);
        return atan2(s, c);
    };
    

    for (int i = 0; i < meshes.uE.rows(); i++) {
        Eigen::Vector2i face = meshes.EF.row(i);
        int face1 = face(0);
        int face2 = face(1);
        if (face2 == -1 || face1 == -1) {
            continue;
        }

        double angle1_f1 = angles(face1, (meshes.EI(i, 0) + 1) % 3);
        double angle2_f1 = angles(face1, (meshes.EI(i, 0) + 2) % 3);
        double angle1_f2 = angles(face2, (meshes.EI(i, 1) + 1) % 3);
        double angle2_f2 = angles(face2, (meshes.EI(i, 1) + 2) % 3);
        if ((angle1_f1 < meshes.threshold && angle2_f1 < meshes.threshold) || (angle1_f2 < meshes.threshold && angle2_f2 < meshes.threshold)) {
            if (angle1_f1 + angle2_f2 > M_PI || angle2_f1 + angle1_f2 > M_PI) {
                continue;
            }
            std::vector<int> half_edges = meshes.uE2E[i];
            const size_t num_faces = meshes.F_undeformed.rows();
            const size_t f1 = half_edges[0] % num_faces;
            const size_t f2 = half_edges[1] % num_faces;
            const size_t c1 = half_edges[0] / num_faces;
            const size_t c2 = half_edges[1] / num_faces;
            const size_t v1 = meshes.F_undeformed(f1, (c1+1)%3);
            const size_t v2 = meshes.F_undeformed(f1, (c1+2)%3);
            const size_t v4 = meshes.F_undeformed(f1, c1);
            const size_t v3 = meshes.F_undeformed(f2, c2);
            double angle413 = corner(meshes.V_undeformed.row(v4), meshes.V_undeformed.row(v1), meshes.V_undeformed.row(v3));
            double angle134 = corner(meshes.V_undeformed.row(v1), meshes.V_undeformed.row(v3), meshes.V_undeformed.row(v4));
            double angle341 = corner(meshes.V_undeformed.row(v3), meshes.V_undeformed.row(v4), meshes.V_undeformed.row(v1));
            double angle324 = corner(meshes.V_undeformed.row(v3), meshes.V_undeformed.row(v2), meshes.V_undeformed.row(v4));
            double angle243 = corner(meshes.V_undeformed.row(v2), meshes.V_undeformed.row(v4), meshes.V_undeformed.row(v3));
            double angle432 = corner(meshes.V_undeformed.row(v4), meshes.V_undeformed.row(v3), meshes.V_undeformed.row(v2));
            angles(f1, 0) = angle413;
            angles(f1, 1) = angle134;
            angles(f1, 2) = angle341;
            angles(f2, 0) = angle324;
            angles(f2, 1) = angle243;
            angles(f2, 2) = angle432;
            int e41 = meshes.EMAP(f1 + ((c1 + 2) % 3) * num_faces); //41
            int e13 = meshes.EMAP(f2 + ((c2 + 1) % 3) * num_faces); //13
            int e32 = meshes.EMAP(f2 + ((c2 + 2) % 3) * num_faces); //32
            int e24 = meshes.EMAP(f1 + ((c1 + 1) % 3) * num_faces); //24
            if (meshes.EF(e13, 0) == f2)
                meshes.EF(e13, 0) = f1;
            else
                meshes.EF(e13, 1) = f1;
            if (meshes.EF(e24, 0) == f2)
                meshes.EF(e24, 0) = f1;
            else
                meshes.EF(e24, 1) = f1;

            
            if (meshes.EF(e41, 0) == f1)
                meshes.EI(e41, 0) = 1;
            else
                meshes.EF(e41, 1) = 1;
            if (meshes.EF(e13, 0) == f1)
                meshes.EI(e13, 0) = 2;
            else
                meshes.EI(e13, 1) = 2;

            if (meshes.EF(e32, 0) == f2)
                meshes.EI(e32, 0) = 1;
            else
                meshes.EI(e32, 1) = 1;
            if (meshes.EF(e24, 0) == f2)
                meshes.EI(e24, 0) = 2;
            else
                meshes.EI(e24, 1) = 2;
            std::vector<std::vector<int>> uE2E = meshes.uE2E;
            Eigen::MatrixXi F = meshes.F_undeformed;
            Eigen::MatrixXi E = meshes.E;
            Eigen::MatrixXi uE = meshes.uE;
            Eigen::VectorXi EMAP = meshes.EMAP;
            igl::flip_edge(F, E,uE,EMAP,uE2E,i);
            if (igl::is_edge_manifold(F)){
                meshes.F_undeformed = F;
                meshes.E = E;
                meshes.uE = uE;
                meshes.EMAP = EMAP;
                meshes.uE2E = uE2E;
            }
        }
    }
    meshes.F_deformed = meshes.F_undeformed;
    meshes.uE2E.clear();
    igl::unique_edge_map(meshes.F_undeformed, meshes.E, meshes.uE, meshes.EMAP, meshes.uE2E);
    igl::edge_flaps(meshes.F_undeformed, meshes.uE, meshes.EMAP, meshes.EF, meshes.EI);
    initializeBoundary(meshes);
}


Eigen::Vector3d normal(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1, const Eigen::Vector3d &p2) {
    const Eigen::Vector3d n0 = (p1 - p0).cross(p2 - p0);
    const Eigen::Vector3d n1 = (p2 - p1).cross(p0 - p1);
    const Eigen::Vector3d n2 = (p0 - p2).cross(p1 - p2);
    Eigen::Vector3d N;

    for(int d = 0;d<3;d++)
    {

      const std::function<double(double,double,double)> sum3 =
        [&sum3](double a, double b, double c)->double
      {
        if(fabs(c)>fabs(a))
        {
          return sum3(c,b,a);
        }
        // c < a
        if(fabs(c)>fabs(b))
        {
          return sum3(a,c,b);
        }
        // c < a, c < b
        if(fabs(b)>fabs(a))
        {
          return sum3(b,a,c);
        }
        return (a+b)+c;
      };

      N(d) = sum3(n0(d),n1(d),n2(d));
    }
    return N.normalized();
}

void Fold_EdgeFlip_no_intersections(Meshes &meshes){
    Meshes mesh_sub = meshes;
    for (int i = 0; i < meshes.uE.rows(); i++) {
        if (meshes.Quad_Angle_sub_Vector(i) == 0)
            continue;
        Eigen::Vector2i face = meshes.EF.row(i);
        int face1 = face(0);
        int face2 = face(1);
        if (face2 == -1 || face1 == -1) {
            continue;
        }
        std::vector<int> half_edges = meshes.uE2E[i];
        const size_t num_faces = meshes.F_undeformed.rows();
        const size_t f1 = half_edges[0] % num_faces;
        const size_t f2 = half_edges[1] % num_faces;
        const size_t c1 = half_edges[0] / num_faces;
        const size_t c2 = half_edges[1] / num_faces;
        const size_t v1 = meshes.F_undeformed(f1, (c1+1)%3);
        const size_t v2 = meshes.F_undeformed(f1, (c1+2)%3);
        const size_t v4 = meshes.F_undeformed(f1, c1);
        const size_t v3 = meshes.F_undeformed(f2, c2);
        int e41 = meshes.EMAP(f1 + ((c1 + 2) % 3) * num_faces); //41
        int e13 = meshes.EMAP(f2 + ((c2 + 1) % 3) * num_faces); //13
        int e32 = meshes.EMAP(f2 + ((c2 + 2) % 3) * num_faces); //32
        int e24 = meshes.EMAP(f1 + ((c1 + 1) % 3) * num_faces); //24
        Eigen::Vector3d vec_12 = (meshes.V_undeformed.row(v2) - meshes.V_undeformed.row(v1)).normalized();
        Eigen::Vector3d vec_32 = (meshes.V_undeformed.row(v2) - meshes.V_undeformed.row(v3)).normalized();
        Eigen::Vector3d vec_14 = (meshes.V_undeformed.row(v4) - meshes.V_undeformed.row(v1)).normalized();
        Eigen::Vector3d vec_13 = (meshes.V_undeformed.row(v3) - meshes.V_undeformed.row(v1)).normalized();
        Eigen::Vector3d vec_42 = (meshes.V_undeformed.row(v2) - meshes.V_undeformed.row(v4)).normalized();
        double threshold = 0.01;
        double threshold_sub = 0.7;
        bool is_fold = (meshes.Quad_Angle_sub_Vector(e41) != 0 && meshes.Quad_Angle_sub_Vector(e32) != 0 && vec_12.dot(vec_32) > threshold && vec_12.dot(vec_14) > threshold && vec_14.dot(vec_32) > threshold_sub);
        if (is_fold){
            meshes.Quad_Angle_sub_Vector(e32) = 0;
            meshes.Quad_Angle_sub_Vector(e41) = 0;
        } else {
            is_fold = (meshes.Quad_Angle_sub_Vector(e13) != 0 && meshes.Quad_Angle_sub_Vector(e24) != 0 && vec_12.dot(vec_13) > threshold && vec_12.dot(vec_42) > threshold && vec_42.dot(vec_13) > threshold_sub);
            if (is_fold){
                meshes.Quad_Angle_sub_Vector(e13) = 0;
                meshes.Quad_Angle_sub_Vector(e24) = 0;
            }
        }
        if (is_fold) {
            std::vector<int> half_edges = meshes.uE2E[i];
            const size_t num_faces = meshes.F_undeformed.rows();
            const size_t f1 = half_edges[0] % num_faces;
            const size_t f2 = half_edges[1] % num_faces;
            const size_t c1 = half_edges[0] / num_faces;
            const size_t c2 = half_edges[1] / num_faces;
            const size_t v1 = meshes.F_undeformed(f1, (c1+1)%3);
            const size_t v2 = meshes.F_undeformed(f1, (c1+2)%3);
            const size_t v4 = meshes.F_undeformed(f1, c1);
            const size_t v3 = meshes.F_undeformed(f2, c2);
            int e41 = meshes.EMAP(f1 + ((c1 + 2) % 3) * num_faces); //41
            int e13 = meshes.EMAP(f2 + ((c2 + 1) % 3) * num_faces); //13
            int e32 = meshes.EMAP(f2 + ((c2 + 2) % 3) * num_faces); //32
            int e24 = meshes.EMAP(f1 + ((c1 + 1) % 3) * num_faces); //24
            if (meshes.EF(e13, 0) == f2)
                meshes.EF(e13, 0) = f1;
            else
                meshes.EF(e13, 1) = f1;
            if (meshes.EF(e24, 0) == f2)
                meshes.EF(e24, 0) = f1;
            else
                meshes.EF(e24, 1) = f1;
            if (meshes.EF(e41, 0) == f1)
                meshes.EI(e41, 0) = 1;
            else
                meshes.EF(e41, 1) = 1;
            if (meshes.EF(e13, 0) == f1)
                meshes.EI(e13, 0) = 2;
            else
                meshes.EI(e13, 1) = 2;

            if (meshes.EF(e32, 0) == f2)
                meshes.EI(e32, 0) = 1;
            else
                meshes.EI(e32, 1) = 1;
            if (meshes.EF(e24, 0) == f2)
                meshes.EI(e24, 0) = 2;
            else
                meshes.EI(e24, 1) = 2;
            std::vector<std::vector<int>> uE2E = meshes.uE2E;
            Eigen::MatrixXi F = meshes.F_undeformed;
            Eigen::MatrixXi E = meshes.E;
            Eigen::MatrixXi uE = meshes.uE;
            Eigen::VectorXi EMAP = meshes.EMAP;
            igl::flip_edge(F, E,uE,EMAP,uE2E,i);
            meshes.N_undeformed_opt.row(f1) = normal(meshes.V_undeformed.row(v1), meshes.V_undeformed.row(v3), meshes.V_undeformed.row(v4));
            meshes.N_undeformed_opt.row(f2) = normal(meshes.V_undeformed.row(v2), meshes.V_undeformed.row(v4), meshes.V_undeformed.row(v3));

            meshes.N_deformed_opt.row(f1) = normal(meshes.V_deformed.row(v1), meshes.V_deformed.row(v3), meshes.V_deformed.row(v4));
            meshes.N_deformed_opt.row(f2) = normal(meshes.V_deformed.row(v2), meshes.V_deformed.row(v4), meshes.V_deformed.row(v3));

            if (igl::is_edge_manifold(F)){
                meshes.F_undeformed = F;
                meshes.E = E;
                meshes.uE = uE;
                meshes.EMAP = EMAP;
                meshes.uE2E = uE2E;
            }
            Eigen::VectorXi EI;
            Eigen::MatrixXd EV;
            Eigen::MatrixXi IF,EE;
            Eigen::Array<bool,Eigen::Dynamic,1> CP;
            igl::predicates::find_self_intersections(meshes.V_deformed,meshes.F_undeformed,IF,CP,EV,EE,EI);
            if (IF.rows() != 0)
                meshes = mesh_sub;
            else
                mesh_sub = meshes;
        }

    }
    meshes.F_deformed = meshes.F_undeformed;
}


void Fold_EdgeFlip(Meshes &meshes){
    Meshes mesh_sub = meshes;
    Eigen::VectorXi EI;
    Eigen::MatrixXd EV;
    Eigen::MatrixXi IF,EE;
    Eigen::Array<bool,Eigen::Dynamic,1> CP;
    igl::predicates::find_self_intersections(meshes.V_deformed,meshes.F_deformed,IF,CP,EV,EE,EI);
    for (int i = 0; i < meshes.uE.rows(); i++) {
        if (meshes.Quad_Angle_sub_Vector(i) == 0)
            continue;
        Eigen::Vector2i face = meshes.EF.row(i);
        int face1 = face(0);
        int face2 = face(1);
        if (face2 == -1 || face1 == -1) {
            continue;
        }
        std::vector<int> half_edges = meshes.uE2E[i];
        const size_t num_faces = meshes.F_undeformed.rows();
        const size_t f1 = half_edges[0] % num_faces;
        const size_t f2 = half_edges[1] % num_faces;
        const size_t c1 = half_edges[0] / num_faces;
        const size_t c2 = half_edges[1] / num_faces;
        const size_t v1 = meshes.F_undeformed(f1, (c1+1)%3);
        const size_t v2 = meshes.F_undeformed(f1, (c1+2)%3);
        const size_t v4 = meshes.F_undeformed(f1, c1);
        const size_t v3 = meshes.F_undeformed(f2, c2);
        int e41 = meshes.EMAP(f1 + ((c1 + 2) % 3) * num_faces); //41
        int e13 = meshes.EMAP(f2 + ((c2 + 1) % 3) * num_faces); //13
        int e32 = meshes.EMAP(f2 + ((c2 + 2) % 3) * num_faces); //32
        int e24 = meshes.EMAP(f1 + ((c1 + 1) % 3) * num_faces); //24
        Eigen::Vector3d vec_12 = (meshes.V_undeformed.row(v2) - meshes.V_undeformed.row(v1)).normalized();
        Eigen::Vector3d vec_32 = (meshes.V_undeformed.row(v2) - meshes.V_undeformed.row(v3)).normalized();
        Eigen::Vector3d vec_14 = (meshes.V_undeformed.row(v4) - meshes.V_undeformed.row(v1)).normalized();
        Eigen::Vector3d vec_13 = (meshes.V_undeformed.row(v3) - meshes.V_undeformed.row(v1)).normalized();
        Eigen::Vector3d vec_42 = (meshes.V_undeformed.row(v2) - meshes.V_undeformed.row(v4)).normalized();
        double threshold = 0.01;
        double threshold_sub = 0.7;
        bool is_fold = (meshes.Quad_Angle_sub_Vector(e41) != 0 && meshes.Quad_Angle_sub_Vector(e32) != 0 && vec_12.dot(vec_32) > threshold && vec_12.dot(vec_14) > threshold && vec_14.dot(vec_32) > threshold_sub);
        if (is_fold){
            meshes.Quad_Angle_sub_Vector(e32) = 0;
            meshes.Quad_Angle_sub_Vector(e41) = 0;
        } else {
            is_fold = (meshes.Quad_Angle_sub_Vector(e13) != 0 && meshes.Quad_Angle_sub_Vector(e24) != 0 && vec_12.dot(vec_13) > threshold && vec_12.dot(vec_42) > threshold && vec_42.dot(vec_13) > threshold_sub);
            if (is_fold){
                meshes.Quad_Angle_sub_Vector(e13) = 0;
                meshes.Quad_Angle_sub_Vector(e24) = 0;
            }
        }
        if (is_fold) {
            std::vector<int> half_edges = meshes.uE2E[i];
            const size_t num_faces = meshes.F_undeformed.rows();
            const size_t f1 = half_edges[0] % num_faces;
            const size_t f2 = half_edges[1] % num_faces;
            const size_t c1 = half_edges[0] / num_faces;
            const size_t c2 = half_edges[1] / num_faces;
            const size_t v1 = meshes.F_undeformed(f1, (c1+1)%3);
            const size_t v2 = meshes.F_undeformed(f1, (c1+2)%3);
            const size_t v4 = meshes.F_undeformed(f1, c1);
            const size_t v3 = meshes.F_undeformed(f2, c2);
            int e41 = meshes.EMAP(f1 + ((c1 + 2) % 3) * num_faces); //41
            int e13 = meshes.EMAP(f2 + ((c2 + 1) % 3) * num_faces); //13
            int e32 = meshes.EMAP(f2 + ((c2 + 2) % 3) * num_faces); //32
            int e24 = meshes.EMAP(f1 + ((c1 + 1) % 3) * num_faces); //24
            if (meshes.EF(e13, 0) == f2)
                meshes.EF(e13, 0) = f1;
            else
                meshes.EF(e13, 1) = f1;
            if (meshes.EF(e24, 0) == f2)
                meshes.EF(e24, 0) = f1;
            else
                meshes.EF(e24, 1) = f1;
            if (meshes.EF(e41, 0) == f1)
                meshes.EI(e41, 0) = 1;
            else
                meshes.EF(e41, 1) = 1;
            if (meshes.EF(e13, 0) == f1)
                meshes.EI(e13, 0) = 2;
            else
                meshes.EI(e13, 1) = 2;

            if (meshes.EF(e32, 0) == f2)
                meshes.EI(e32, 0) = 1;
            else
                meshes.EI(e32, 1) = 1;
            if (meshes.EF(e24, 0) == f2)
                meshes.EI(e24, 0) = 2;
            else
                meshes.EI(e24, 1) = 2;
            std::vector<std::vector<int>> uE2E = meshes.uE2E;
            Eigen::MatrixXi F = meshes.F_undeformed;
            Eigen::MatrixXi E = meshes.E;
            Eigen::MatrixXi uE = meshes.uE;
            Eigen::VectorXi EMAP = meshes.EMAP;
            igl::flip_edge(F, E,uE,EMAP,uE2E,i);
            meshes.N_undeformed_opt.row(f1) = normal(meshes.V_undeformed.row(v1), meshes.V_undeformed.row(v3), meshes.V_undeformed.row(v4));
            meshes.N_undeformed_opt.row(f2) = normal(meshes.V_undeformed.row(v2), meshes.V_undeformed.row(v4), meshes.V_undeformed.row(v3));

            meshes.N_deformed_opt.row(f1) = normal(meshes.V_deformed.row(v1), meshes.V_deformed.row(v3), meshes.V_deformed.row(v4));
            meshes.N_deformed_opt.row(f2) = normal(meshes.V_deformed.row(v2), meshes.V_deformed.row(v4), meshes.V_deformed.row(v3));

            if (igl::is_edge_manifold(F)){
                meshes.F_undeformed = F;
                meshes.E = E;
                meshes.uE = uE;
                meshes.EMAP = EMAP;
                meshes.uE2E = uE2E;
            }
        }

    }
    meshes.F_deformed = meshes.F_undeformed;
    igl::predicates::find_self_intersections(meshes.V_deformed,meshes.F_deformed,IF,CP,EV,EE,EI);
    if (IF.rows() == 0){
        return;
    }
    Fold_EdgeFlip_no_intersections(mesh_sub);
    meshes = mesh_sub;
    Eigen::VectorXi EI_;
    Eigen::MatrixXd EV_;
    Eigen::MatrixXi IF_,EE_;
    Eigen::Array<bool,Eigen::Dynamic,1> CP_;
    igl::predicates::find_self_intersections(meshes.V_deformed,meshes.F_deformed,IF_,CP_,EV_,EE_,EI_);
}

