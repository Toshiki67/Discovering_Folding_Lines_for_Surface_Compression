#ifndef FREEFORM_UNIFORM_H
#define FREEFORM_UNIFORM_H
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

void Uniform_initialize(Meshes &meshes);
void Compute_Quad_Uniform(Meshes &meshes);
void Compute_Quad_derivatives_Uniform(Meshes &meshes);

void Compute_Quad_Uniform_sub(Meshes &meshes);
void Compute_Quad_derivatives_Uniform_sub(Meshes &meshes);


#endif FREEFORM_UNIFORM_H