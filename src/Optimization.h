#ifndef FREEFORM_OPTIMIZATION_H
#define FREEFORM_OPTIMIZATION_H
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

void Minimize(Meshes &meshes, int num_iterations);


void Newton(Meshes &meshes, int num_iterations);

#endif FREEFORM_OPTIMIZATION_H
