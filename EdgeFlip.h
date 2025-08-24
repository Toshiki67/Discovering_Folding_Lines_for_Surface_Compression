#ifndef FREEFORM_EDGEFLIP_H
#define FREEFORM_EDGEFLIP_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

void Compute_EdgeFlip(Meshes &meshes);

Eigen::Vector3d normal(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);

void Fold_EdgeFlip_no_intersections(Meshes &meshes);

void Fold_EdgeFlip(Meshes &meshes);


#endif FREEFORM_EDGEFLIP_H