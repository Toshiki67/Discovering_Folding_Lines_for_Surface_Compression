#ifndef FREEFORM_SELFCOLLISION_H
#define FREEFORM_SELFCOLLISION_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "data.h"
#include <iostream>
#include <vector>
#include <set>



double TripleProduct(const Eigen::RowVector3d &a, const Eigen::RowVector3d &b, const Eigen::RowVector3d &c);

double EvaluateCubic(double r2, double k0, double k1, double k2, double k3);

void BisectRangeCubicRoot(int &icnt, double &r0, double &r1, double &v0, double &v1, double &k0, double &k1, double &k2, double &k3);

double FindRootCubic(double r0, double r1, double v0, double v1, double k0, double k1, double k2, double k3);

double FindCoplanerInterp(const Eigen::RowVector3d& s0, const Eigen::RowVector3d& s1, const Eigen::RowVector3d& s2, const Eigen::RowVector3d& s3,
    const Eigen::RowVector3d& e0, const Eigen::RowVector3d& e1, const Eigen::RowVector3d& e2, const Eigen::RowVector3d& e3);

double Distance_FV(const Eigen::RowVector3d &Va, const Eigen::RowVector3d &Vb, 
    const Eigen::RowVector3d &Vc, const Eigen::RowVector3d &Vd,
    double &w0, double &w1);

double Distance_EE(const Eigen::RowVector3d &Va, const Eigen::RowVector3d &Vb, 
    const Eigen::RowVector3d &Vc, const Eigen::RowVector3d &Vd,
    double &ratio_ab, double &ratio_cd);

bool Is_FV(const int a, const int b, const int c, const int d,
    const Eigen::RowVector3d &Va, const Eigen::RowVector3d &Vb, 
    const Eigen::RowVector3d &Vc, const Eigen::RowVector3d &Vd,
    const double threshold);

bool Is_EE(const int a, const int b, const int c, const int d,
    const Eigen::RowVector3d &Va, const Eigen::RowVector3d &Vb, 
    const Eigen::RowVector3d &Vc, const Eigen::RowVector3d &Vd,
    const double threshold);

void Compute_SelfCollision(Meshes &meshes);

void Compute_SelfCollisionImpulse(std::vector<std::vector<int>> &self_collision_pairs,
    double delta,
    Meshes &meshes);

void Compute_SelfCollisionImpulse_CCD(std::vector<std::vector<int>> &self_collision_pairs,
    double delta,
    Meshes &meshes);

void MakeRigidImpactZone
    (std::vector< std::set<int> >& aRIZ, // (in,ou)RIZに属する節点のインデックスの集合の配列
     const std::vector<std::vector<int>>& aContactElem, // 自己交差する接触要素の配列
     const std::vector<std::vector<int>>& aEdge);
void CalcInvMat3(double ainv[], const double a[]);
void ApplyRigidImpactZone
(Eigen::MatrixXd& aUVWm, // (in,out)RIZで更新された中間速度
 ////
 const std::vector< std::set<int> >& aRIZ,  // (in)各RIZに属する節点の集合(set)の配列
 const Eigen::MatrixXd& aXYZ, // (in) 前ステップの節点の位置の配列
 const Eigen::MatrixXd& aUVWm0);
 void GetIntermidiateVelocityContactResolved(Meshes &meshes);
#endif FREEFORM_SELFCOLLISION_H