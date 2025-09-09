#ifndef FREEFORM_BOUNDARY_H
#define FREEFORM_BOUNDARY_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

void closest_point_line(const Eigen::Vector3d &p, const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                        Eigen::Vector3d &closest, double &distance);

void initializeBoundary(Meshes &meshes);

void Compute_Quad_Boundary_Constraints(Meshes &meshes);
void Compute_Quad_derivatives_Boundary_Constraints(Meshes &meshes);

void Compute_Newton_Boundary_Constraints(Meshes &meshes);
void Compute_Newton_derivatives_Boundary_Constraints(Meshes &meshes);

#endif FREEFORM_BOUNDARY_H
