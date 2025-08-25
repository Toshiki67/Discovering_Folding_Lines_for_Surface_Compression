//
// Created by 青木俊樹 on 4/8/24.
//

#ifndef FREEFORM_CONSTRAINT_H
#define FREEFORM_CONSTRAINT_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

void Compute_Constraints(Meshes &meshes);

void Compute_derivatives(Meshes &meshes);

void Compute_Quad_Constraints(Meshes &meshes);

void Compute_Quad_derivatives(Meshes &meshes);

void Compute_Quad_Constraints_sub(Meshes &meshes);

void Compute_Quad_derivatives_sub(Meshes &meshes);

void Compute_Newton_Constraints(Meshes &meshes);

void Compute_Newton_derivatives(Meshes &meshes);

#endif FREEFORM_CONSTRAINT_H
