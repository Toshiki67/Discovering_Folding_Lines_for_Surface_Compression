#ifndef FREEFORM_ANGLE_H
#define FREEFORM_ANGLE_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

void Compute_perpendicular(const Eigen::Vector3d &p, const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                           Eigen::Vector3d &perpendicular);

void Compute_Angle(Meshes &meshes);

void Compute_Quad_Angle(Meshes &meshes);

void Compute_Quad_derivatives_Angle(Meshes &meshes);

int choose(int num, int v1_un, int v2_un, int v3_un, int v4_un, int v1_de, int v2_de, int v3_de, int v4_de);

autodiff::dual2nd Compute_Direct_Angle_energy(
    const autodiff::ArrayXdual2nd& x, const autodiff::dual2nd &weight, const autodiff::dual2nd &delta);

void Compute_Newton_Direct_Angle(Meshes &meshes);

void Compute_Newton_derivatives_Direct_Angle(Meshes &meshes);
#endif FREEFORM_ANGLE_H