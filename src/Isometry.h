#ifndef FREEFORM_ISOMETRY_H
#define FREEFORM_ISOMETRY_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

void Compute_Quad_Isometry(Meshes &meshes);

void Compute_Quad_derivatives_Isometry(Meshes &meshes);

void Compute_Quad_Isometry_sub(Meshes &meshes);

void Compute_Quad_derivatives_Isometry_sub(Meshes &meshes);

int choose(int num, int v1, int v2, int v3, int v4);

autodiff::dual2nd ComputeIsometry_energy(
    const autodiff::ArrayXdual2nd& x, const autodiff::dual2nd &weight);

void Compute_Newton_Isometry(Meshes &meshes);

void Compute_Newton_derivatives_Isometry(Meshes &meshes);

#endif FREEFORM_ISOMETRY_H
