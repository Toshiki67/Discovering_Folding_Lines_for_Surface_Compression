#include <igl/predicates/find_self_intersections.h>
#include <igl/adjacency_list.h>
#include <cmath>
#include <iostream>
#include "SelfCollision.h"


double TripleProduct(const Eigen::RowVector3d &a, const Eigen::RowVector3d &b, const Eigen::RowVector3d &c) {
    return a.dot(b.cross(c));
}

double EvaluateCubic(double r2, double k0, double k1, double k2, double k3){
    return k0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2;
}

void BisectRangeCubicRoot(int &icnt, double &r0, double &r1, double &v0, double &v1, double &k0, double &k1, double &k2, double &k3){
    icnt--;
    if (icnt <= 0)
        return;
    double r2 = (r0 + r1) / 2;
    double v2 = EvaluateCubic(r2, k0, k1, k2, k3);
    if( v0*v2 < 0 ){ r1 = r2; } // r0とr2の間で符号が変化する
    else{            r0 = r2; } // r1とr2の間で符号が変化する
    BisectRangeCubicRoot(icnt,r0,r1,v0,v2,k0,k1,k2,k3);
}

double FindRootCubic(double r0, double r1, double v0, double v1, double k0, double k1, double k2, double k3) {
    int icnt = 15;
    BisectRangeCubicRoot(icnt, r0,r1, v0,v1, k0,k1,k2,k3);
    return 0.5 * (r0 + r1);
}

double FindCoplanerInterp(const Eigen::RowVector3d& s0, const Eigen::RowVector3d& s1, const Eigen::RowVector3d& s2, const Eigen::RowVector3d& s3,
    const Eigen::RowVector3d& e0, const Eigen::RowVector3d& e1, const Eigen::RowVector3d& e2, const Eigen::RowVector3d& e3)
{
    const Eigen::RowVector3d x1 = s1-s0;
    const Eigen::RowVector3d x2 = s2-s0;
    const Eigen::RowVector3d x3 = s3-s0;
    const Eigen::RowVector3d v1 = e1-e0-x1;
    const Eigen::RowVector3d v2 = e2-e0-x2;
    const Eigen::RowVector3d v3 = e3-e0-x3;
    // 三次関数の係数の計算
    const double k0 = TripleProduct(x3,x1,x2);
    const double k1 = TripleProduct(v3,x1,x2)+TripleProduct(x3,v1,x2)+TripleProduct(x3,x1,v2);
    const double k2 = TripleProduct(v3,v1,x2)+TripleProduct(v3,x1,v2)+TripleProduct(x3,v1,v2);
    const double k3 = TripleProduct(v3,v1,v2);
    double r0=-0.0;
    double r1=+1.0;
    const double f0 = EvaluateCubic(r0,k0,k1,k2,k3);
    const double f1 = EvaluateCubic(r1,k0,k1,k2,k3);
    double det = k2*k2-3*k1*k3;
    if( std::abs(k3) < 1.0e-10 && std::abs(k2) > 1.0e-10 ){ // quadric function、二次関数
      double r2 = -k1/(2*k2); // 極値をとるr
      const double f2 = EvaluateCubic(r2, k0,k1,k2,k3);
      if( r2 > 0 && r2 < 1 ){
        if(      f0*f2 < 0 ){
          return FindRootCubic(r0,r2, f0,f2, k0,k1,k2,k3);
  
        }
        else if( f2*f1 < 0 ){
          return FindRootCubic(r2,r1, f2,f1, k0,k1,k2,k3);
        }
      }
    }
    if( det > 0 && std::abs(k3) > 1.0e-10 ) // cubic function with two extream value、三次関数で極値がある場合
    {
      double r3 = (-k2-std::sqrt(det))/(3*k3); // 極値をとる小さい方のr
      const double f3 = EvaluateCubic(r3, k0,k1,k2,k3);
      if( r3 > 0 && r3 < 1 ){
        if(      f0*f3 < 0 ){
          return FindRootCubic(r0,r3, f0,f3, k0,k1,k2,k3);
        }
        else if( f3*f1 < 0 ){
          return FindRootCubic(r3,r1, f3,f1, k0,k1,k2,k3);
        }
      }
      double r4 = (-k2+std::sqrt(det))/(3*k3); // 極値をとる大きい方のr
      const double f4 = EvaluateCubic(r4, k0,k1,k2,k3);
      if( r3 > 0 && r3 < 1 && r4 > 0 && r4 < 1 ){
        if( f3*f4 < 0 ){
          return FindRootCubic(r3,r4, f3,f4, k0,k1,k2,k3);
        }
      }
      if( r4 > 0 && r4 < 1 ){
        if(      f0*f4 < 0 ){
          return FindRootCubic(r0,r4, f0,f4, k0,k1,k2,k3);
        }
        else if( f4*f1 < 0 ){
          return FindRootCubic(r4,r1, f4,f1, k0,k1,k2,k3);
        }
      }
    }
    // monotonus function、0と１の間で短調増加関数
    if( f0*f1 > 0 ){ return -1; } // 根がない場合
    return FindRootCubic(r0,r1, f0,f1, k0,k1,k2,k3);
  }

double Distance_FV(const Eigen::RowVector3d &Va, const Eigen::RowVector3d &Vb, 
    const Eigen::RowVector3d &Vc, const Eigen::RowVector3d &Vd,
    double &w0, double &w1) {
    Eigen::RowVector3d Vac = Va - Vc;
    Eigen::RowVector3d Vbc = Vb - Vc;
    Eigen::RowVector3d Vdc = Vd - Vc;
    double t0 = Vac.dot(Vac);
    double t1 = Vbc.dot(Vbc);
    double t2 = Vac.dot(Vbc);
    double t3 = Vac.dot(Vdc);
    double t4 = Vbc.dot(Vdc);
    double det = t0 * t1 - t2 * t2;
    double invdet = 1.0 / det;
    w0 = (t1 * t3 - t2 * t4) * invdet;
    w1 = (t0 * t4 - t2 * t3) * invdet;
    const double w2 = 1.0 - w0 - w1;
    Eigen::RowVector3d P = w0 * Va + w1 * Vb + w2 * Vc;
    double height = (P - Vd).norm();
    return height;
}

double Distance_EE(const Eigen::RowVector3d &Va, const Eigen::RowVector3d &Vb, 
    const Eigen::RowVector3d &Vc, const Eigen::RowVector3d &Vd,
    double &ratio_ab, double &ratio_cd) {

        const Eigen::RowVector3d &ba = Vb - Va;
        const Eigen::RowVector3d &dc = Vd - Vc;
        if (ba.cross(dc).norm() < 1e-10) {
            Eigen::RowVector3d P = Va - Vc;
            Eigen::RowVector3d dc_copy = dc;
            dc_copy.normalize();
            Eigen::RowVector3d vert = P - P.dot(dc_copy) * dc_copy;
            double dist = vert.norm();
            double la = Va.dot(dc_copy);
            double lb = Vb.dot(dc_copy);
            double lc = Vc.dot(dc_copy);
            double ld = Vd.dot(dc_copy);
            double min_ab = (la < lb) ? la : lb;
            double max_ab = (la > lb) ? la : lb;
            double min_cd = (lc < ld) ? lc : ld;
            double max_cd = (lc > ld) ? lc : ld;
            double lm;
            if (max_ab < min_cd)
                lm = (max_ab + min_cd) / 2;
            else if (max_cd < min_ab)
                lm = (max_cd + min_ab) / 2;
            else if (max_ab < max_cd)
                lm = (max_ab + min_cd) / 2;
            else
                lm = (max_cd + min_ab) / 2;
            ratio_ab = (lm - la) / (lb - la);
            ratio_cd = (lm - lc) / (ld - lc);
            return dist;
        }
        double t0 = ba.dot(ba);
        double t1 = dc.dot(dc);
        double t2 = ba.dot(dc);
        double t3 = ba.dot(Vc-Va);
        double t4 = dc.dot(Vc-Va);
        double det = t0 * t1 - t2 * t2;
        double invdet = 1.0 / det;
        ratio_ab = (t1 * t3 - t2 * t4) * invdet;
        ratio_cd = (t2 * t3 - t0 * t4) * invdet;
        Eigen::RowVector3d P = Va + ratio_ab * ba;
        Eigen::RowVector3d Q = Vc + ratio_cd * dc;
        return (P - Q).norm();
    }

double Height_FV(const Eigen::RowVector3d &Va, const Eigen::RowVector3d &Vb, 
    const Eigen::RowVector3d &Vc, const Eigen::RowVector3d &Vd){
        // get normal vector
  double dtmp_x = (Vb(1)-Va(1))*(Vc(2)-Va(2))-(Vb(2)-Va(2))*(Vc(1)-Va(1));
  double dtmp_y = (Vb(2)-Va(2))*(Vc(0)-Va(0))-(Vb(0)-Va(0))*(Vc(2)-Va(2));
  double dtmp_z = (Vb(0)-Va(0))*(Vc(1)-Va(1))-(Vb(1)-Va(1))*(Vc(0)-Va(0));
    
  // normalize normal vector
  const double dtmp1 = 1.0 / std::sqrt( dtmp_x*dtmp_x + dtmp_y*dtmp_y + dtmp_z*dtmp_z );
  dtmp_x *= dtmp1;
  dtmp_y *= dtmp1;
  dtmp_z *= dtmp1;
    
  return (Vd(0)-Va(0))*dtmp_x+(Vd(1)-Va(1))*dtmp_y+(Vd(2)-Va(2))*dtmp_z;
    }

bool Is_FV(const int a, const int b, const int c, const int d,
    const Eigen::RowVector3d &Va, const Eigen::RowVector3d &Vb, 
    const Eigen::RowVector3d &Vc, const Eigen::RowVector3d &Vd,
    const double threshold) {
    if (d == a || d == b || d == c) 
        return false;
    double w0, w1;
    double height = Height_FV(Va, Vb, Vc, Vd);
    double dist = Distance_FV(Va, Vb, Vc, Vd, w0, w1);
    const double w2 = 1.0 - w0 - w1;
    if (height > threshold) 
        return false;
    if (dist > threshold)
        return false;
    if (w0 < 0 || w0 > 1) 
        return false;
    if (w1 < 0 || w1 > 1)
        return false;
    if (w2 < 0 || w2 > 1)
        return false;
    return true;
}


bool Is_EE(const int a, const int b, const int c, const int d,
    const Eigen::RowVector3d &Va, const Eigen::RowVector3d &Vb, 
    const Eigen::RowVector3d &Vc, const Eigen::RowVector3d &Vd,
    const double threshold) {
    if (a == c || b == c || a == d || b == d) 
        return false;
    if (Vc(0)+threshold < Va(0) && Vc(0)+threshold < Vb(0) && Vd(0)+threshold < Va(0) && Vd(0)+threshold < Vb(0))
        return false;
    if (Vc(0)-threshold > Va(0) && Vc(0)-threshold > Vb(0) && Vd(0)-threshold > Va(0) && Vd(0)-threshold > Vb(0))
        return false;
    if (Vc(1)+threshold < Va(1) && Vc(1)+threshold < Vb(1) && Vd(1)+threshold < Va(1) && Vd(1)+threshold < Vb(1))
        return false;
    if (Vc(1)-threshold > Va(1) && Vc(1)-threshold > Vb(1) && Vd(1)-threshold > Va(1) && Vd(1)-threshold > Vb(1))
        return false;
    if (Vc(2)+threshold < Va(2) && Vc(2)+threshold < Vb(2) && Vd(2)+threshold < Va(2) && Vd(2)+threshold < Vb(2))
        return false;
    if (Vc(2)-threshold > Va(2) && Vc(2)-threshold > Vb(2) && Vd(2)-threshold > Va(2) && Vd(2)-threshold > Vb(2))
        return false;
    double ratio_ab, ratio_cd;
    double dist = Distance_EE(Va, Vb, Vc, Vd, ratio_ab, ratio_cd);
    if (dist > threshold) 
        return false;
    if (ratio_ab < 0 || ratio_ab > 1)
        return false;
    if (ratio_cd < 0 || ratio_cd > 1)
        return false;
    Eigen::RowVector3d P = Va + ratio_ab * (Vb - Va);
    Eigen::RowVector3d Q = Vc + ratio_cd * (Vd - Vc);
    if ((P - Q).norm() > threshold)
        return false;
    return true;
    }


void Compute_SelfCollision(Meshes &meshes) {
    Eigen::MatrixXi &F = meshes.F_deformed;
    Eigen::MatrixXd &V = meshes.V_deformed;
    Eigen::VectorXi EI;
    Eigen::MatrixXd EV;
    Eigen::MatrixXi IF,EE;
    Eigen::Array<bool,Eigen::Dynamic,1> CP;
    igl::predicates::find_self_intersections(V,F,IF,CP,EV,EE,EI);
    std::cout << "Found " << IF.rows() << " self intersecting pairs" << std::endl;
    // verify the self-intersecting pairs is edge-edge or face-vertex
    std::vector<std::vector<int>> &self_collision_pairs = meshes.self_collision_pairs;
    self_collision_pairs.clear();
    for (int i = 0; i < IF.rows(); i++) {
        int f1 = IF(i, 0);
        int f2 = IF(i, 1);
        int v1 = F(f1, 0);
        int v2 = F(f1, 1);
        int v3 = F(f1, 2);
        int v4 = F(f2, 0);
        int v5 = F(f2, 1);
        int v6 = F(f2, 2);
        for (int j = 0; j < 3; j++) {
            if (Is_FV(v1, v2, v3, v4, V.row(v1), V.row(v2), V.row(v3), V.row(v4), meshes.p_depth)) 
                self_collision_pairs.push_back({v1, v2, v3, v4, 1});
            if (Is_FV(v1, v2, v3, v5, V.row(v1), V.row(v2), V.row(v3), V.row(v5), meshes.p_depth))
                self_collision_pairs.push_back({v1, v2, v3, v5, 1});
            if (Is_FV(v1, v2, v3, v6, V.row(v1), V.row(v2), V.row(v3), V.row(v6), meshes.p_depth))
                self_collision_pairs.push_back({v1, v2, v3, v6, 1});
            if (Is_FV(v4, v5, v6, v1, V.row(v4), V.row(v5), V.row(v6), V.row(v1), meshes.p_depth))
                self_collision_pairs.push_back({v4, v5, v6, v1, 1});
            if (Is_FV(v4, v5, v6, v2, V.row(v4), V.row(v5), V.row(v6), V.row(v2), meshes.p_depth))
                self_collision_pairs.push_back({v4, v5, v6, v2, 1});
            if (Is_FV(v4, v5, v6, v3, V.row(v4), V.row(v5), V.row(v6), V.row(v3), meshes.p_depth))
                self_collision_pairs.push_back({v4, v5, v6, v3, 1});
            
            if (Is_EE(v1, v2, v4, v5, V.row(v1), V.row(v2), V.row(v4), V.row(v5), meshes.p_depth))
                self_collision_pairs.push_back({v1, v2, v4, v5, 0});
            if (Is_EE(v1, v2, v5, v6, V.row(v1), V.row(v2), V.row(v5), V.row(v6), meshes.p_depth))
                self_collision_pairs.push_back({v1, v2, v5, v6, 0});
            if (Is_EE(v1, v2, v6, v4, V.row(v1), V.row(v2), V.row(v6), V.row(v4), meshes.p_depth))
                self_collision_pairs.push_back({v1, v2, v6, v4, 0});
            if (Is_EE(v2, v3, v4, v5, V.row(v2), V.row(v3), V.row(v4), V.row(v5), meshes.p_depth))
                self_collision_pairs.push_back({v2, v3, v4, v5, 0});
            if (Is_EE(v2, v3, v5, v6, V.row(v2), V.row(v3), V.row(v5), V.row(v6), meshes.p_depth))
                self_collision_pairs.push_back({v2, v3, v5, v6, 0});
            if (Is_EE(v2, v3, v6, v4, V.row(v2), V.row(v3), V.row(v6), V.row(v4), meshes.p_depth))
                self_collision_pairs.push_back({v2, v3, v6, v4, 0});
            if (Is_EE(v3, v1, v4, v5, V.row(v3), V.row(v1), V.row(v4), V.row(v5), meshes.p_depth))
                self_collision_pairs.push_back({v3, v1, v4, v5, 0});
            if (Is_EE(v3, v1, v5, v6, V.row(v3), V.row(v1), V.row(v5), V.row(v6), meshes.p_depth))
                self_collision_pairs.push_back({v3, v1, v5, v6, 0});
            if (Is_EE(v3, v1, v6, v4, V.row(v3), V.row(v1), V.row(v6), V.row(v4), meshes.p_depth))
                self_collision_pairs.push_back({v3, v1, v6, v4, 0});
        }
    }
}

void Compute_SelfCollisionImpulse(std::vector<std::vector<int>> &self_collision_pairs,
    double delta,
    Meshes &meshes) {
        double stiffness = 300;
        for (int i = 0; i < self_collision_pairs.size(); i++) {
        std::vector<int> &pair = self_collision_pairs[i];
            const int p0 = pair[0];
            const int p1 = pair[1];
            const int p2 = pair[2];
            const int p3 = pair[3];
            const int type = pair[4];
            Eigen::RowVector3d P0 = meshes.V_deformed.row(p0);
            Eigen::RowVector3d P1 = meshes.V_deformed.row(p1);
            Eigen::RowVector3d P2 = meshes.V_deformed.row(p2);
            Eigen::RowVector3d P3 = meshes.V_deformed.row(p3);
            Eigen::RowVector3d Vel0 = meshes.Vel.row(p0);
            Eigen::RowVector3d Vel1 = meshes.Vel.row(p1);
            Eigen::RowVector3d Vel2 = meshes.Vel.row(p2);
            Eigen::RowVector3d Vel3 = meshes.Vel.row(p3);
            if (type == 1) { // face vertex
                double w0, w1;
                double dist = Distance_FV(P0, P1, P2, P3, w0, w1);
                if (w0 < 0 || w0 > 1) 
                    continue;
                if (w1 < 0 || w1 > 1)
                    continue;
                if (dist > delta) 
                    continue;
                double w2 = 1.0 - w0 - w1;
                Eigen::RowVector3d P = w0 * P0 + w1 * P1 + w2 * P2;
                Eigen::RowVector3d norm = P3 - P;
                norm = norm / norm.norm();
                double p_depth = delta - (P3-P).dot(norm);
                double rel_v = (Vel3 - w0*Vel0 - w1*Vel1).dot(norm);
                if (rel_v > 0.1*p_depth)
                    continue;
                double imp_el = stiffness * p_depth;
                double imp_ie = 0.1*p_depth - rel_v;
                double imp_min = std::min(imp_el, imp_ie);
                double imp_mod = 2*imp_min/(1+w0*w0+w1*w1+w2*w2);
                imp_mod *= 0.25;
                meshes.Vel.row(p0) += -norm * imp_mod * w0;
                meshes.Vel.row(p1) += -norm * imp_mod * w1;
                meshes.Vel.row(p2) += -norm * imp_mod * w2;
                meshes.Vel.row(p3) += norm * imp_mod;
            } else { //edge edge
                double w01, w23;
                double dist = Distance_EE(P0, P1, P2, P3, w01, w23);
                if (w01 < 0 || w01 > 1) 
                    continue;
                    if (w23 < 0 || w23 > 1)
                    continue;
                if (dist > delta) 
                    continue;
                Eigen::RowVector3d c01 = (1-w01) * P0 + w01 * P1;
                Eigen::RowVector3d c23 = (1-w23) * P2 + w23 * P3;
                Eigen::RowVector3d norm = c23 - c01;
                norm = norm / norm.norm();
                double p_depth = delta - (c23-c01).norm();
                double rel_v = ((1-w23)*Vel2 + w23*Vel3 - (1-w01)*Vel0 - w01*Vel1).dot(norm);
                if (rel_v > 0.1*p_depth)
                    continue;
                double imp_el = stiffness * p_depth;
                double imp_ie = 0.1*p_depth - rel_v;
                double imp_min = std::min(imp_el, imp_ie);
                double imp_mod = 2*imp_min/(w01*w01 + (1-w01)*(1-w01) + w23*w23 + (1-w23)*(1-w23));
                imp_mod *= 0.25;
                meshes.Vel.row(p0) += -norm * imp_mod * (1-w01);
                meshes.Vel.row(p1) += -norm * imp_mod * w01;
                meshes.Vel.row(p2) += norm * imp_mod * (1-w23);
                meshes.Vel.row(p3) += norm * imp_mod * w23;
            }
        }
    }

void Compute_SelfCollisionImpulse_CCD(std::vector<std::vector<int>> &self_collision_pairs,
    double delta,
    Meshes &meshes) {
        double stiffness = 300;
        for (int i = 0; i < self_collision_pairs.size(); i++) {
        std::vector<int> &pair = self_collision_pairs[i];
            const int p0 = pair[0];
            const int p1 = pair[1];
            const int p2 = pair[2];
            const int p3 = pair[3];
            const int type = pair[4];
            Eigen::RowVector3d P0 = meshes.V_deformed.row(p0);
            Eigen::RowVector3d P1 = meshes.V_deformed.row(p1);
            Eigen::RowVector3d P2 = meshes.V_deformed.row(p2);
            Eigen::RowVector3d P3 = meshes.V_deformed.row(p3);
            Eigen::RowVector3d Vel0 = meshes.Vel.row(p0);
            Eigen::RowVector3d Vel1 = meshes.Vel.row(p1);
            Eigen::RowVector3d Vel2 = meshes.Vel.row(p2);
            Eigen::RowVector3d Vel3 = meshes.Vel.row(p3);
            double t = FindCoplanerInterp(P0, P1, P2, P3,P0+Vel0, P1+Vel1, P2+Vel2, P3+Vel3);
            if( t < 0 || t > 1 ) continue;
            if(type == 1){ // face-vtx
                double w0,w1;
                {        
                    Eigen::RowVector3d p0m = P0 + t*Vel0;
                    Eigen::RowVector3d p1m = P1 + t*Vel1;
                    Eigen::RowVector3d p2m = P2 + t*Vel2;
                    Eigen::RowVector3d p3m = P3 + t*Vel3;
                    double dist = Distance_FV(p0m, p1m, p2m, p3m, w0,w1);
                    if( w0 < 0 || w0 > 1 ) continue;
                    if( w1 < 0 || w1 > 1 ) continue;
                    if( dist > delta ) continue;
                }
                double w2 = 1.0 - w0 - w1;
                Eigen::RowVector3d Pc = w0*P0 + w1*P1 + w2*P2;
                Eigen::RowVector3d norm = P3 - Pc; 
                norm = norm / norm.norm();
                double rel_v = (Vel3-w0*Vel0-w1*Vel1-w2*Vel2).dot(norm); // relative velocity (positive if separating)
                if( rel_v > 0.1*delta ) continue; // separating
                double imp = (0.1*delta-rel_v);
                double imp_mod = 2*imp/(1.0+w0*w0+w1*w1+w2*w2);
                imp_mod *= 0.1;
                meshes.Vel.row(p0) += -norm*imp_mod*w0;
                meshes.Vel.row(p1) += -norm*imp_mod*w1;
                meshes.Vel.row(p2) += -norm*imp_mod*w2;
                meshes.Vel.row(p3) += norm*imp_mod;
              }
              else{ // edge-edge
                double w01,w23;
                {
                    Eigen::RowVector3d p0m = P0 + t*Vel0;
                    Eigen::RowVector3d p1m = P1 + t*Vel1;
                    Eigen::RowVector3d p2m = P2 + t*Vel2;
                    Eigen::RowVector3d p3m = P3 + t*Vel3;
                    double dist = Distance_EE(p0m, p1m, p2m, p3m, w01,w23);
                    if( w01 < 0 || w01 > 1 ) continue;
                    if( w23 < 0 || w23 > 1 ) continue;
                    if( dist > delta ) continue;
                }      
                Eigen::RowVector3d c01 = (1-w01)*P0 + w01*P1;
                Eigen::RowVector3d c23 = (1-w23)*P2 + w23*P3;
                Eigen::RowVector3d norm = (c23-c01);
                norm = norm / norm.norm();
                double rel_v = ((1-w23)*Vel2+w23*Vel3-(1-w01)*Vel0-w01*Vel1).dot(norm);
                if( rel_v > 0.1*delta ) continue; // separating
                double imp = (0.1*delta-rel_v); // reasonable
                double imp_mod = 2*imp/( w01*w01+(1-w01)*(1-w01) + w23*w23+(1-w23)*(1-w23) );
                imp_mod *= 0.1;
                meshes.Vel.row(p0) += -norm*imp_mod*(1-w01);
                meshes.Vel.row(p1) += -norm*imp_mod*w01;
                meshes.Vel.row(p2) += norm*imp_mod*(1-w23);
                meshes.Vel.row(p3) += norm*imp_mod*w23;
        }
    }
}


void MakeRigidImpactZone
(std::vector< std::set<int> >& aRIZ, // (in,ou)RIZに属する節点のインデックスの集合の配列
 const std::vector<std::vector<int>>& aContactElem, // 自己交差する接触要素の配列
 const std::vector<std::vector<int>>& aEdge) // 三角形メッシュの辺の配列
{
  for(int ice=0;ice<aContactElem.size();ice++){
    const std::vector<int>& ce = aContactElem[ice];
    const int n[4] = {ce[0], ce[1], ce[2], ce[3]};
    std::set<int> ind_inc; // 接触要素が接するRIZの集合
    for(int i=0;i<4;i++){
      const int ino = n[i];
      for(int iriz=0;iriz<aRIZ.size();iriz++){
        if( aRIZ[iriz].find(ino) != aRIZ[iriz].end() ){
          ind_inc.insert(iriz);
        }
        else{
          for(int jno: aEdge[ino]){
            if( aRIZ[iriz].find(jno) != aRIZ[iriz].end() ){
              ind_inc.insert(iriz);  break;
            }
          }
        }
      }
    }
    if( ind_inc.size() == 0 ){ // 接触要素はどのRIZにも属していない
      int ind0 = (int)aRIZ.size();
      aRIZ.resize(ind0+1);
      aRIZ[ind0].insert(n[0]); aRIZ[ind0].insert(n[1]); aRIZ[ind0].insert(n[2]); aRIZ[ind0].insert(n[3]);
    }
    else if( ind_inc.size() == 1 ){ // 接触要素は一つのRIZに接する
      int ind0 = *(ind_inc.begin());
      aRIZ[ind0].insert(n[0]); aRIZ[ind0].insert(n[1]); aRIZ[ind0].insert(n[2]); aRIZ[ind0].insert(n[3]);
    }
    else{ // overlapping two reagion，接触要素が複数のRIZに接するー＞接する複数のRIZをマージする
      std::vector< std::set<int> > aRIZ1; // マージされた後のRIZの配列
      for(int iriz=0;iriz<aRIZ.size();iriz++){ // 接さないRIZをコピー
        if( ind_inc.find(iriz) != ind_inc.end() ) continue;
        aRIZ1.push_back( aRIZ[iriz] );
      }
      // マージしたRIZを，aRIZ1の最後に追加
      int ind0 = (int)aRIZ1.size();
      aRIZ1.resize(ind0+1);
      for(std::set<int>::iterator itr=ind_inc.begin();itr!=ind_inc.end();itr++){
        int ind1 = *itr;
        assert( ind1 < aRIZ.size() );
        for(std::set<int>::iterator jtr=aRIZ[ind1].begin();jtr!=aRIZ[ind1].end();jtr++){
          aRIZ1[ind0].insert(*jtr);
        }
      }
      aRIZ1[ind0].insert(n[0]); aRIZ1[ind0].insert(n[1]); aRIZ1[ind0].insert(n[2]); aRIZ1[ind0].insert(n[3]);
      aRIZ = aRIZ1;
    }
  }
}

void CalcInvMat3(double ainv[], const double a[])
{
	const double det =
  + a[0]*a[4]*a[8] + a[3]*a[7]*a[2] + a[6]*a[1]*a[5]
  - a[0]*a[7]*a[5] - a[6]*a[4]*a[2] - a[3]*a[1]*a[8];
	const double inv_det = 1.0/det;
  
	ainv[0] = inv_det*(a[4]*a[8]-a[5]*a[7]);
	ainv[1] = inv_det*(a[2]*a[7]-a[1]*a[8]);
	ainv[2] = inv_det*(a[1]*a[5]-a[2]*a[4]);
  
	ainv[3] = inv_det*(a[5]*a[6]-a[3]*a[8]);
	ainv[4] = inv_det*(a[0]*a[8]-a[2]*a[6]);
	ainv[5] = inv_det*(a[2]*a[3]-a[0]*a[5]);
  
	ainv[6] = inv_det*(a[3]*a[7]-a[4]*a[6]);
	ainv[7] = inv_det*(a[1]*a[6]-a[0]*a[7]);
	ainv[8] = inv_det*(a[0]*a[4]-a[1]*a[3]);
}

void ApplyRigidImpactZone
(Eigen::MatrixXd& aUVWm, // (in,out)RIZで更新された中間速度
 ////
 const std::vector< std::set<int> >& aRIZ,  // (in)各RIZに属する節点の集合(set)の配列
 const Eigen::MatrixXd& aXYZ, // (in) 前ステップの節点の位置の配列
 const Eigen::MatrixXd& aUVWm0) // (in) RIZを使う前の中間速度
{
  for(int iriz=0;iriz<aRIZ.size();iriz++){
    std::vector<int> aInd; // index of points belong to this RIZ
    for(std::set<int>::iterator jtr=aRIZ[iriz].begin();jtr!=aRIZ[iriz].end();jtr++){
      aInd.push_back(*jtr);
    }
    Eigen::RowVector3d gc(0,0,0); // 重心位置
    Eigen::RowVector3d av(0,0,0); // 平均速度
    for(int iv=0;iv<aInd.size();iv++){
      int ino = aInd[iv];
      gc += aXYZ.row(ino);
      av += aUVWm0.row(ino);
    }
    gc /= (double)aInd.size();
    av /= (double)aInd.size();
    Eigen::RowVector3d L(0,0,0); // 角運動量
    double I[9] = {0,0,0, 0,0,0, 0,0,0}; // 慣性テンソル
    // Eigen::MatrixXd I = Eigen::MatrixXd::Zero(3,3);
    for(int iv=0;iv<aInd.size();iv++){
      int ino = aInd[iv];
      Eigen::RowVector3d p = aXYZ.row(ino);
      Eigen::RowVector3d v = aUVWm0.row(ino);
    //   L += Cross(p-gc,v-av);
        L += (p-gc).cross(v-av);
    Eigen::RowVector3d q = p-gc;
    //   I[0] += v.dot(v) - q(0)*q(0);  I[1] +=     - q(0)*q(1);  I[2] +=     - q(0)*q(2);
    //   I[3] +=     - q(1)*q(0);  I[4] += v.dot(v) - q(1)*q(1);  I[5] +=     - q(1)*q(2);
    //   I[6] +=     - q(2)*q(0);  I[7] +=     - q(2)*q(1);  I[8] += v.dot(v) - q(2)*q(2);
    I[0] += q[1]*q[1] + q[2]*q[2];  I[1] += -q[0]*q[1];           I[2] += -q[0]*q[2];
    I[3] += -q[1]*q[0];             I[4] += q[0]*q[0] + q[2]*q[2];I[5] += -q[1]*q[2];
    I[6] += -q[2]*q[0];             I[7] += -q[2]*q[1];           I[8] += q[0]*q[0] + q[1]*q[1];
    }
    // 角速度を求める
    double Iinv[9];
    CalcInvMat3(Iinv,I);
    Eigen::RowVector3d omg;
    omg(0) = Iinv[0]*L(0) + Iinv[1]*L(1) + Iinv[2]*L(2);
    omg(1) = Iinv[3]*L(0) + Iinv[4]*L(1) + Iinv[5]*L(2);
    omg(2) = Iinv[6]*L(0) + Iinv[7]*L(1) + Iinv[8]*L(2);
    // 中間速度の更新
    for(int iv=0;iv<aInd.size();iv++){
      int ino = aInd[iv];
      Eigen::RowVector3d p = aXYZ.row(ino);
      Eigen::RowVector3d rot = - (p-gc).cross(omg);
      aUVWm(ino, 0) = av(0) + rot(0);
      aUVWm(ino, 1) = av(1) + rot(1);
      aUVWm(ino, 2) = av(2) + rot(2);
    }
  }
}

void GetIntermidiateVelocityContactResolved(Meshes &meshes) {
    Compute_SelfCollision(meshes);
    if (meshes.self_collision_pairs.size() == 0)
        return;
    Compute_SelfCollisionImpulse(meshes.self_collision_pairs, meshes.p_depth, meshes);
    meshes.V_deformed = meshes.V_deformed_pre + meshes.Vel;
    for(int itr=0;itr<10;itr++){
        Compute_SelfCollision(meshes);
        if (meshes.self_collision_pairs.size() == 0)
            return;
        Compute_SelfCollisionImpulse_CCD(meshes.self_collision_pairs, meshes.p_depth, meshes);
        meshes.V_deformed = meshes.V_deformed_pre + meshes.Vel;
    }
    std::vector< std::set<int> > aRIZ;
    Eigen::MatrixXd Vel0 = meshes.Vel;
    igl::adjacency_list(meshes.F_deformed, meshes.adjacency_list);
    for (int itr = 0; itr<100;itr++){
        Compute_SelfCollision(meshes);
        if (meshes.self_collision_pairs.size() == 0)
            return;
        int nnode_riz = 0;
        for(int iriz=0;iriz<aRIZ.size();iriz++){
        nnode_riz += aRIZ[iriz].size();
        }
        std::cout << "  RIZ iter: " << itr << "    Contact Elem Size: " << meshes.self_collision_pairs.size() << "   NNode In RIZ: " << nnode_riz << std::endl;
        if( meshes.self_collision_pairs.size() == 0 ){
            std::cout << "Resolved All Collisions : " << std::endl;
            break;
        }
        MakeRigidImpactZone(aRIZ, meshes.self_collision_pairs, meshes.adjacency_list);
        ApplyRigidImpactZone
        (meshes.Vel, aRIZ,
         meshes.V_deformed_pre,
         Vel0);
         meshes.V_deformed = meshes.V_deformed_pre + meshes.Vel;
    }

}