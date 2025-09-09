#ifndef FREEFORM_ANGLESUB_H
#define FREEFORM_ANGLESUB_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

void Compute_Angle_sub(Meshes &meshes);

void Compute_Quad_Angle_sub(Meshes &meshes);

void Compute_Quad_derivatives_Angle_sub(Meshes &meshes);

void Set_selected_edges(Meshes &meshes);


autodiff::dual2nd Compute_Angle(const autodiff::VectorXdual2nd& x, const autodiff::dual & weight, const autodiff::dual& delta);
autodiff::var Compute_Normal_Energy(const autodiff::ArrayXvar& x);

void Compute_Newton_Angle(Meshes &meshes);
int choose_v_num(int num, int v1, int v2, int v3, int v4, int n1, int n2, int n3, int n4);
void Compute_Newton_derivatives_Angle(Meshes &meshes);

#endif FREEFORM_ANGLESUB_H