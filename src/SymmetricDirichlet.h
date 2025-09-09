#ifndef FREEFORM_SYMMETRIC_H
#define FREEFORM_SYMMETRIC_H
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

void SymmetricDirichlet_initailize(Meshes &meshes);
void computeSurfaceGradientPerFace(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::MatrixXd &D1, Eigen::MatrixXd &D2);

void Compute_Quad_SymmetricDirichlet(Meshes &meshes);
void Compute_Quad_derivatives_SymmetricDirichlet(Meshes &meshes);


autodiff::dual2nd Compute_SymmetricDirichlet(const autodiff::ArrayXdual2nd& x, const autodiff::ArrayXdual2nd& d1d, const autodiff::ArrayXdual2nd& d2d, const autodiff::dual2nd &area, const autodiff::dual2nd &weight);
int choose(int num, int v1, int v2, int v3);
void Compute_Newton_SymmetricDirichlet(Meshes &meshes);
void Compute_Newton_derivatives_SymmetricDirichlet(Meshes &meshes);


#endif FREEFORM_SYMMETRIC_H