#ifndef FREEFORM_CLOSENESS_H
#define FREEFORM_CLOSENESS_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

void Compute_Quad_Closeness(Meshes &meshes);
void Compute_Quad_derivatives_Closeness(Meshes &meshes);

void Compute_Newton_Closeness(Meshes &meshes);
void Compute_Newton_derivatives_Closeness(Meshes &meshes);



#endif FREEFORM_CLOSENESS_H
