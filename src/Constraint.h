#ifndef FREEFORM_CONSTRAINT_H
#define FREEFORM_CONSTRAINT_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

void Compute_Quad_Constraints(Meshes &meshes);

void Compute_Quad_derivatives(Meshes &meshes);

void Compute_Newton_Constraints(Meshes &meshes);

void Compute_Newton_derivatives(Meshes &meshes);

#endif FREEFORM_CONSTRAINT_H
