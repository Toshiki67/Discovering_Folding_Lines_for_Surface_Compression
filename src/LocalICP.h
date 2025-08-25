//
// Created by 青木俊樹 on 4/10/24.
//

#ifndef FREEFORM_LOCALICP_H
#define FREEFORM_LOCALICP_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

void ICP(const Eigen::MatrixXd &V, const Eigen::MatrixXd &V_target, Eigen::MatrixXd &R, Eigen::VectorXd &t, Eigen::VectorXd &V_result);

void Compute_Quad_LocalICP(Meshes &meshes);

void Compute_Quad_derivatives_LocalICP(Meshes &meshes);

#endif FREEFORM_LOCALICP_H
