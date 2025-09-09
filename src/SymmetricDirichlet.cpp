#include "SymmetricDirichlet.h"
#include <iostream>
#include <igl/local_basis.h>
#include <igl/doublearea.h>
#include <igl/per_face_normals.h>
#include <igl/parallel_for.h>
#include <omp.h>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>


void computeSurfaceGradientPerFace(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::MatrixXd &D1, Eigen::MatrixXd &D2)
{
    Eigen::MatrixXd F1, F2, F3;
    igl::local_basis(V, F, F1, F2, F3);
    const int Fn = F.rows();  const int vn = V.rows();

    Eigen::MatrixXd Dx(Fn,3), Dy(Fn, 3), Dz(Fn, 3);
    Eigen::MatrixXd fN; igl::per_face_normals(V, F, fN);
    Eigen::VectorXd Ar; igl::doublearea(V, F, Ar);
    Eigen::PermutationMatrix<3> perm;

    Eigen::Vector3i Pi;
    Pi << 1, 2, 0;
    Eigen::PermutationMatrix<3> P = Eigen::PermutationMatrix<3>(Pi);

    for (int i = 0; i < Fn; i++) {
        int i1 = F(i, 0);
        int i2 = F(i, 1);
        int i3 = F(i, 2);

        Eigen::Matrix3d e;
        e.col(0) = V.row(i2) - V.row(i1);
        e.col(1) = V.row(i3) - V.row(i2);
        e.col(2) = V.row(i1) - V.row(i3);;

        Eigen::Vector3d Fni = fN.row(i);
        double Ari = Ar(i);

        Eigen::Matrix3d n_M;
        n_M << 0, -Fni(2), Fni(1), Fni(2), 0, -Fni(0), -Fni(1), Fni(0), 0;
        Eigen::VectorXi R(3); R << 0, 1, 2;
        Eigen::VectorXi C(3); C << 3 * i + 2, 3 * i, 3 * i + 1;
        Eigen::Matrix3d res = ((1. / Ari)*(n_M*e))*P;

        Dx.row(i) = res.row(0);
        Dy.row(i) = res.row(1);
        Dz.row(i) = res.row(2);
    }
    D1 = F1.col(0).asDiagonal()*Dx + F1.col(1).asDiagonal()*Dy + F1.col(2).asDiagonal()*Dz;
    D2 = F2.col(0).asDiagonal()*Dx + F2.col(1).asDiagonal()*Dy + F2.col(2).asDiagonal()*Dz;
}	


void SymmetricDirichlet_initailize(Meshes &meshes) 
{
	Eigen::MatrixXd D1cols_un, D2cols_un;
	computeSurfaceGradientPerFace(meshes.V_undeformed, meshes.F_undeformed, D1cols_un, D2cols_un);
    Eigen::VectorXd restShapeArea_un;
	igl::doublearea(meshes.V_undeformed, meshes.F_undeformed, restShapeArea_un);
	restShapeArea_un /= 2;

    Eigen::MatrixXd D1cols_de, D2cols_de;
    computeSurfaceGradientPerFace(meshes.V_deformed, meshes.F_deformed, D1cols_de, D2cols_de);
    Eigen::VectorXd restShapeArea_de;
    igl::doublearea(meshes.V_deformed, meshes.F_deformed, restShapeArea_de);
    restShapeArea_de /= 2;

    meshes.D1d = Eigen::MatrixXd::Zero(D1cols_un.rows() + D1cols_de.rows(), 3);
    meshes.D2d = Eigen::MatrixXd::Zero(D2cols_un.rows() + D2cols_de.rows(), 3);

    meshes.D1d.block(0, 0, D1cols_un.rows(), 3) = D1cols_un;
    meshes.D1d.block(D1cols_un.rows(), 0, D1cols_de.rows(), 3) = D1cols_de;
    meshes.D2d.block(0, 0, D2cols_un.rows(), 3) = D2cols_un;
    meshes.D2d.block(D2cols_un.rows(), 0, D2cols_de.rows(), 3) = D2cols_de;

    meshes.restShapeArea = Eigen::VectorXd::Zero(restShapeArea_un.rows() + restShapeArea_de.rows());
    meshes.restShapeArea.segment(0, restShapeArea_un.rows()) = restShapeArea_un;
    meshes.restShapeArea.segment(restShapeArea_un.rows(), restShapeArea_de.rows()) = restShapeArea_de;
}


void Compute_Quad_SymmetricDirichlet(Meshes &meshes) 
{
    meshes.Quad_Symmetric_C = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < meshes.F_undeformed.rows(); i++) {
        int v0_index = meshes.F_undeformed(i, 0);
		int v1_index = meshes.F_undeformed(i, 1);
		int v2_index = meshes.F_undeformed(i, 2);
		Eigen::RowVector3d V0 = meshes.V_undeformed.row(v0_index);
		Eigen::RowVector3d V1 = meshes.V_undeformed.row(v1_index);
		Eigen::RowVector3d V2 = meshes.V_undeformed.row(v2_index);
		Eigen::RowVector3d e10 = V1 - V0;
		Eigen::RowVector3d e20 = V2 - V0;
		Eigen::RowVector3d B1 = e10 / e10.norm();
        Eigen::RowVector3d B2 = B1.cross(e20).cross(B1).normalized();
		Eigen::RowVector3d Xi;
        Xi << V0.dot(B1), V1.dot(B1), V2.dot(B1);
		Eigen::RowVector3d Yi;
        Yi << V0.dot(B2), V1.dot(B2), V2.dot(B2);
		//prepare jacobian		
		const double a = meshes.D1d.row(i).dot(Xi);
		const double b = meshes.D1d.row(i).dot(Yi);
		const double c = meshes.D2d.row(i).dot(Xi);
		const double d = meshes.D2d.row(i).dot(Yi);
		const double detJ = a * d - b * c;
		const double detJ2 = detJ * detJ;
		const double a2 = a * a;
		const double b2 = b * b;
		const double c2 = c * c;
		const double d2 = d * d;
		double energy = 0.5 * (1 + 1 / detJ2) * (a2 + b2 + c2 + d2);
        #pragma omp atomic
		meshes.Quad_Symmetric_C += meshes.restShapeArea(i) * energy;
    }


    #pragma omp parallel for
    for (int i = 0; i < meshes.F_deformed.rows(); i++) {
        int v0_index = meshes.F_deformed(i, 0);
		int v1_index = meshes.F_deformed(i, 1);
		int v2_index = meshes.F_deformed(i, 2);
		Eigen::RowVector3d V0 = meshes.V_deformed.row(v0_index);
		Eigen::RowVector3d V1 = meshes.V_deformed.row(v1_index);
		Eigen::RowVector3d V2 = meshes.V_deformed.row(v2_index);
		Eigen::RowVector3d e10 = V1 - V0;
		Eigen::RowVector3d e20 = V2 - V0;
		Eigen::RowVector3d B1 = e10 / e10.norm();
        Eigen::RowVector3d B2 = B1.cross(e20).cross(B1).normalized();
		Eigen::RowVector3d Xi;
        Xi << V0.dot(B1), V1.dot(B1), V2.dot(B1);
		Eigen::RowVector3d Yi;
        Yi << V0.dot(B2), V1.dot(B2), V2.dot(B2);
		//prepare jacobian		
		const double a = meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(Xi);
		const double b = meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(Yi);
		const double c = meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(Xi);
		const double d = meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(Yi);
		const double detJ = a * d - b * c;
		const double detJ2 = detJ * detJ;
		const double a2 = a * a;
		const double b2 = b * b;
		const double c2 = c * c;
		const double d2 = d * d;
		double energy = 0.5 * (1 + 1 / detJ2) * (a2 + b2 + c2 + d2);
        #pragma omp atomic
		meshes.Quad_Symmetric_C += meshes.restShapeArea(i + meshes.F_undeformed.rows()) * energy;
    }
}


void Compute_Quad_derivatives_SymmetricDirichlet(Meshes &meshes)
{
	meshes.Quad_Symmetric_grad = Eigen::MatrixXd::Zero(meshes.V_undeformed.rows() + meshes.V_deformed.rows(), 3);

    #pragma omp parallel for
    for (int i = 0; i < meshes.F_undeformed.rows(); i++) {
		int v0_index = meshes.F_undeformed(i, 0);
		int v1_index = meshes.F_undeformed(i, 1);
		int v2_index = meshes.F_undeformed(i, 2);
		Eigen::RowVector3d V0 = meshes.V_undeformed.row(v0_index);
		Eigen::RowVector3d V1 = meshes.V_undeformed.row(v1_index);
		Eigen::RowVector3d V2 = meshes.V_undeformed.row(v2_index);
		Eigen::RowVector3d e10 = V1 - V0;
		Eigen::RowVector3d e20 = V2 - V0;
		Eigen::RowVector3d B1 = e10 / e10.norm();
        Eigen::RowVector3d B2 = B1.cross(e20).cross(B1).normalized();
		Eigen::RowVector3d Xi;
        Xi << V0.dot(B1), V1.dot(B1), V2.dot(B1);
		Eigen::RowVector3d Yi;
        Yi << V0.dot(B2), V1.dot(B2), V2.dot(B2);	
		const double a = meshes.D1d.row(i).dot(Xi);
		const double b = meshes.D1d.row(i).dot(Yi);
		const double c = meshes.D2d.row(i).dot(Xi);
		const double d = meshes.D2d.row(i).dot(Yi);
		const double detJ = a * d - b * c;
		const double det2 = detJ * detJ;
		const double a2 = a * a;
		const double b2 = b * b;
		const double c2 = c * c;
		const double d2 = d * d;
		const double det3 = std::pow(detJ, 3);
		const double Fnorm = a2 + b2 + c2 + d2;

		Eigen::VectorXd de_dJ(4);
        de_dJ <<
			meshes.restShapeArea(i) * (a + a / det2 - d * Fnorm / det3),
			meshes.restShapeArea(i) * (b + b / det2 + c * Fnorm / det3),
			meshes.restShapeArea(i) * (c + c / det2 + b * Fnorm / det3),
			meshes.restShapeArea(i) * (d + d / det2 - a * Fnorm / det3);
		double Norm_e10_3 = std::pow(e10.norm(), 3);
		Eigen::RowVector3d B2_b2 = e10.cross(e20).cross(e10);
		double Norm_B2 = B2_b2.norm();
		double Norm_B2_2 = pow(Norm_B2, 2);
		Eigen::RowVector3d B2_dxyz0, B2_dxyz1;
		double B2_dnorm0, B2_dnorm1;
		Eigen::RowVector3d db1_dX, db2_dX, XX, YY;
		
        B2_dxyz0 << -e10(1) * e20(1) - e10(2) * e20(2), 2 * e10(0) * e20(1) - e10(1) * e20(0), -e10(2) * e20(0) + 2 * e10(0) * e20(2);
        B2_dxyz1 << std::pow(e10(1), 2) + std::pow(e10(2), 2), -e10(0) * e10(1), -e10(0) * e10(2);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        B2_dnorm1 = B2_b2.dot(B2_dxyz1) / Norm_B2;
        db2_dX << -(B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2 - (B2_dxyz1(0) * Norm_B2 - B2_b2(0) * B2_dnorm1) / Norm_B2_2,
            -(B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2 - (B2_dxyz1(1) * Norm_B2 - B2_b2(1) * B2_dnorm1) / Norm_B2_2,
            -(B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2 - (B2_dxyz1(2) * Norm_B2 - B2_b2(2) * B2_dnorm1) / Norm_B2_2;
        db1_dX << -(std::pow(e10(1), 2) + std::pow(e10(2), 2)) / Norm_e10_3, (e10(1) * e10(0)) / Norm_e10_3, (e10(2) * e10(0)) / Norm_e10_3;
        XX << V0.dot(db1_dX) + B1(0), V1.dot(db1_dX), V2.dot(db1_dX);
        YY << V0.dot(db2_dX) + B2(0), V1.dot(db2_dX), V2.dot(db2_dX);
        Eigen::VectorXd dj_dx(4);
        dj_dx << meshes.D1d.row(i).dot(XX), meshes.D1d.row(i).dot(YY), meshes.D2d.row(i).dot(XX), meshes.D2d.row(i).dot(YY);
		
        #pragma omp atomic
		meshes.Quad_Symmetric_grad(v0_index, 0) += de_dJ.dot(dj_dx);

        B2_dxyz0 << -e10(1) * e20(1) - e10(2) * e20(2), 2 * e10(0) * e20(1) - e10(1) * e20(0), -e10(2) * e20(0) + 2 * e10(0) * e20(2);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;

		db2_dX << 
			(B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2;
        db1_dX << -(-(std::pow(e10(1), 2) + std::pow(e10(2), 2)) / Norm_e10_3), -((e10(1) * e10(0)) / Norm_e10_3), -((e10(2) * e10(0)) / Norm_e10_3);
        XX << V0.dot(db1_dX), V1.dot(db1_dX) + B1(0), V2.dot(db1_dX);
        YY << V0.dot(db2_dX), V1.dot(db2_dX) + B2(0), V2.dot(db2_dX);
		dj_dx << meshes.D1d.row(i).dot(XX), meshes.D1d.row(i).dot(YY), meshes.D2d.row(i).dot(XX), meshes.D2d.row(i).dot(YY);
        #pragma omp atomic
		meshes.Quad_Symmetric_grad(v1_index, 0) += de_dJ.dot(dj_dx);

        B2_dxyz0 << std::pow(e10(1), 2) + std::pow(e10(2), 2), -e10(0) * e10(1), -e10(0) * e10(2);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        db2_dX << B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0 / Norm_B2_2,
            B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0 / Norm_B2_2,
            B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0 / Norm_B2_2;
        XX << 0, 0, B1(0);
        YY << V0.dot(db2_dX), V1.dot(db2_dX), V2.dot(db2_dX) + B2(0);
        dj_dx << meshes.D1d.row(i).dot(XX), meshes.D1d.row(i).dot(YY), meshes.D2d.row(i).dot(XX), meshes.D2d.row(i).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v2_index, 0) += de_dJ.dot(dj_dx);


        B2_dxyz0 << -e10(1) * e20(1) + 2 * e10(1) * e20(0), -e10(2) * e20(2) - e20(0) * e10(0), 2 * e10(1) * e20(2) - e10(2) * e20(1);
        B2_dxyz1 << -e10(1) * e10(0), std::pow(e10(2), 2) + std::pow(e10(0), 2), -e10(2) * e10(1);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        B2_dnorm1 = B2_b2.dot(B2_dxyz1) / Norm_B2;
        db2_dX << -((B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(0) * Norm_B2 - B2_b2(0) * B2_dnorm1) / Norm_B2_2),
            -((B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(1) * Norm_B2 - B2_b2(1) * B2_dnorm1) / Norm_B2_2),
            -((B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(2) * Norm_B2 - B2_b2(2) * B2_dnorm1) / Norm_B2_2);
        db1_dX << ((e10(1) * e10(0)) / Norm_e10_3), (-(std::pow(e10(0), 2) + std::pow(e10(2), 2)) / Norm_e10_3), ((e10(2) * e10(1)) / Norm_e10_3);
        XX << V0.dot(db1_dX) + B1(1), V1.dot(db1_dX), V2.dot(db1_dX);
        YY << V0.dot(db2_dX) + B2(1), V1.dot(db2_dX), V2.dot(db2_dX);
		dj_dx << meshes.D1d.row(i).dot(XX), meshes.D1d.row(i).dot(YY), meshes.D2d.row(i).dot(XX), meshes.D2d.row(i).dot(YY);
        #pragma omp atomic
		meshes.Quad_Symmetric_grad(v0_index, 1) += de_dJ.dot(dj_dx);


        B2_dxyz0 << -e10(0) * e20(1) + 2 * e10(1) * e20(0), -e10(2) * e20(2) - e20(0) * e10(0), 2 * e10(1) * e20(2) - e10(2) * e20(1);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        db2_dX << ((B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2),
            ((B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2),
            ((B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2);
        db1_dX << -((e10(1) * e10(0)) / Norm_e10_3), -(-(pow(e10(0), 2) + pow(e10(2), 2)) / Norm_e10_3), -((e10(2) * e10(1)) / Norm_e10_3);
        XX << V0.dot(db1_dX), V1.dot(db1_dX) + B1(1), V2.dot(db1_dX);
        YY << V0.dot(db2_dX), V1.dot(db2_dX) + B2(1), V2.dot(db2_dX);
        dj_dx << meshes.D1d.row(i).dot(XX), meshes.D1d.row(i).dot(YY), meshes.D2d.row(i).dot(XX), meshes.D2d.row(i).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v1_index, 1) += de_dJ.dot(dj_dx);


        B2_dxyz0 << -e10(1) * e10(0), std::pow(e10(2), 2) + std::pow(e10(0), 2), -e10(2) * e10(1);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        db2_dX << (B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2,
            (B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2,
            (B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2;
        XX << 0, 0, B1(1);
        YY << V0.dot(db2_dX), V1.dot(db2_dX), V2.dot(db2_dX) + B2(1);
        dj_dx << meshes.D1d.row(i).dot(XX), meshes.D1d.row(i).dot(YY), meshes.D2d.row(i).dot(XX), meshes.D2d.row(i).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v2_index, 1) += de_dJ.dot(dj_dx);


        B2_dxyz0 << 2 * e10(2) * e20(0) - e10(0) * e20(2), -e10(1) * e20(2) + 2 * e10(2) * e20(1), -e10(0) * e20(0) - e10(1) * e20(1);
        B2_dxyz1 << -e10(0) * e10(2), -e10(2) * e10(1), std::pow(e10(0), 2) + std::pow(e10(1), 2);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        B2_dnorm1 = B2_b2.dot(B2_dxyz1) / Norm_B2;
        db2_dX << -((B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(0) * Norm_B2 - B2_b2(0) * B2_dnorm1) / Norm_B2_2),
			-((B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(1) * Norm_B2 - B2_b2(1) * B2_dnorm1) / Norm_B2_2),
			-((B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(2) * Norm_B2 - B2_b2(2) * B2_dnorm1) / Norm_B2_2);
        db1_dX << ((e10(2) * e10(0)) / Norm_e10_3), ((e10(2) * e10(1)) / Norm_e10_3), (-(pow(e10(0), 2) + pow(e10(1), 2)) / Norm_e10_3);
        XX << V0.dot(db1_dX) + B1(2), V1.dot(db1_dX), V2.dot(db1_dX);
        YY << V0.dot(db2_dX) + B2(2), V1.dot(db2_dX), V2.dot(db2_dX);
        dj_dx << meshes.D1d.row(i).dot(XX), meshes.D1d.row(i).dot(YY), meshes.D2d.row(i).dot(XX), meshes.D2d.row(i).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v0_index, 2) += de_dJ.dot(dj_dx);


        B2_dxyz0 << 2 * e10(2) * e20(0) - e10(0) * e20(2), -e10(1) * e20(2) + 2 * e10(2) * e20(1), -e10(0) * e20(0) - e10(1) * e20(1);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        db2_dX << ((B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2),
            ((B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2),
            ((B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2);
        db1_dX << -((e10(2) * e10(0)) / Norm_e10_3), -((e10(2) * e10(1)) / Norm_e10_3), -(-(std::pow(e10(0), 2) + std::pow(e10(1), 2)) / Norm_e10_3);
        XX << V0.dot(db1_dX), V1.dot(db1_dX) + B1(2), V2.dot(db1_dX);
        YY << V0.dot(db2_dX), V1.dot(db2_dX) + B2(2), V2.dot(db2_dX);
        dj_dx << meshes.D1d.row(i).dot(XX), meshes.D1d.row(i).dot(YY), meshes.D2d.row(i).dot(XX), meshes.D2d.row(i).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v1_index, 2) += de_dJ.dot(dj_dx);

        B2_dxyz0 << -e10(0) * e10(2), -e10(2) * e10(1), std::pow(e10(0), 2) + std::pow(e10(1), 2);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        db2_dX << (B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2,
            (B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2,
            (B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2;
        XX << 0, 0, B1(2);
        YY << V0.dot(db2_dX), V1.dot(db2_dX), V2.dot(db2_dX) + B2(2);
        dj_dx << meshes.D1d.row(i).dot(XX), meshes.D1d.row(i).dot(YY), meshes.D2d.row(i).dot(XX), meshes.D2d.row(i).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v2_index, 2) += de_dJ.dot(dj_dx);
    }


    #pragma omp parallel for
    for (int i = 0; i < meshes.F_deformed.rows(); i++) {
		int v0_index = meshes.F_deformed(i, 0);
		int v1_index = meshes.F_deformed(i, 1);
		int v2_index = meshes.F_deformed(i, 2);
		Eigen::RowVector3d V0 = meshes.V_deformed.row(v0_index);
		Eigen::RowVector3d V1 = meshes.V_deformed.row(v1_index);
		Eigen::RowVector3d V2 = meshes.V_deformed.row(v2_index);
		Eigen::RowVector3d e10 = V1 - V0;
		Eigen::RowVector3d e20 = V2 - V0;
		Eigen::RowVector3d B1 = e10 / e10.norm();
        Eigen::RowVector3d B2 = B1.cross(e20).cross(B1).normalized();
		Eigen::RowVector3d Xi;
        Xi << V0.dot(B1), V1.dot(B1), V2.dot(B1);
		Eigen::RowVector3d Yi;
        Yi << V0.dot(B2), V1.dot(B2), V2.dot(B2);
		
		const double a = meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(Xi);
		const double b = meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(Yi);
		const double c = meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(Xi);
		const double d = meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(Yi);
		const double detJ = a * d - b * c;
		const double det2 = detJ * detJ;
		const double a2 = a * a;
		const double b2 = b * b;
		const double c2 = c * c;
		const double d2 = d * d;
		const double det3 = std::pow(detJ, 3);
		const double Fnorm = a2 + b2 + c2 + d2;

		Eigen::VectorXd de_dJ(4);
        de_dJ <<
			meshes.restShapeArea(i + meshes.F_undeformed.rows()) * (a + a / det2 - d * Fnorm / det3),
			meshes.restShapeArea(i + meshes.F_undeformed.rows()) * (b + b / det2 + c * Fnorm / det3),
			meshes.restShapeArea(i + meshes.F_undeformed.rows()) * (c + c / det2 + b * Fnorm / det3),
			meshes.restShapeArea(i + meshes.F_undeformed.rows()) * (d + d / det2 - a * Fnorm / det3);
		double Norm_e10_3 = std::pow(e10.norm(), 3);
		Eigen::RowVector3d B2_b2 = e10.cross(e20).cross(e10);
		double Norm_B2 = B2_b2.norm();
		double Norm_B2_2 = pow(Norm_B2, 2);
		Eigen::RowVector3d B2_dxyz0, B2_dxyz1;
		double B2_dnorm0, B2_dnorm1;
		Eigen::RowVector3d db1_dX, db2_dX, XX, YY;
		
        B2_dxyz0 << -e10(1) * e20(1) - e10(2) * e20(2), 2 * e10(0) * e20(1) - e10(1) * e20(0), -e10(2) * e20(0) + 2 * e10(0) * e20(2);
        B2_dxyz1 << std::pow(e10(1), 2) + std::pow(e10(2), 2), -e10(0) * e10(1), -e10(0) * e10(2);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        B2_dnorm1 = B2_b2.dot(B2_dxyz1) / Norm_B2;
        db2_dX << -(B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2 - (B2_dxyz1(0) * Norm_B2 - B2_b2(0) * B2_dnorm1) / Norm_B2_2,
            -(B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2 - (B2_dxyz1(1) * Norm_B2 - B2_b2(1) * B2_dnorm1) / Norm_B2_2,
            -(B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2 - (B2_dxyz1(2) * Norm_B2 - B2_b2(2) * B2_dnorm1) / Norm_B2_2;
        db1_dX << -(std::pow(e10(1), 2) + std::pow(e10(2), 2)) / Norm_e10_3, (e10(1) * e10(0)) / Norm_e10_3, (e10(2) * e10(0)) / Norm_e10_3;
        XX << V0.dot(db1_dX) + B1(0), V1.dot(db1_dX), V2.dot(db1_dX);
        YY << V0.dot(db2_dX) + B2(0), V1.dot(db2_dX), V2.dot(db2_dX);
        Eigen::VectorXd dj_dx(4);
        dj_dx << meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(YY), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(YY);
		#pragma omp atomic
		meshes.Quad_Symmetric_grad(v0_index + meshes.V_undeformed.rows(), 0) += de_dJ.dot(dj_dx);

        B2_dxyz0 << -e10(1) * e20(1) - e10(2) * e20(2), 2 * e10(0) * e20(1) - e10(1) * e20(0), -e10(2) * e20(0) + 2 * e10(0) * e20(2);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;

		db2_dX << 
			(B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2;
        db1_dX << -(-(std::pow(e10(1), 2) + std::pow(e10(2), 2)) / Norm_e10_3), -((e10(1) * e10(0)) / Norm_e10_3), -((e10(2) * e10(0)) / Norm_e10_3);
        XX << V0.dot(db1_dX), V1.dot(db1_dX) + B1(0), V2.dot(db1_dX);
        YY << V0.dot(db2_dX), V1.dot(db2_dX) + B2(0), V2.dot(db2_dX);
		dj_dx << meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(YY), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(YY);
        #pragma omp atomic
		meshes.Quad_Symmetric_grad(v1_index + meshes.V_undeformed.rows(), 0) += de_dJ.dot(dj_dx);

        B2_dxyz0 << std::pow(e10(1), 2) + std::pow(e10(2), 2), -e10(0) * e10(1), -e10(0) * e10(2);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        db2_dX << B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0 / Norm_B2_2,
            B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0 / Norm_B2_2,
            B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0 / Norm_B2_2;
        XX << 0, 0, B1(0);
        YY << V0.dot(db2_dX), V1.dot(db2_dX), V2.dot(db2_dX) + B2(0);
        dj_dx << meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(YY), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v2_index + meshes.V_undeformed.rows(), 0) += de_dJ.dot(dj_dx);


        B2_dxyz0 << -e10(1) * e20(1) + 2 * e10(1) * e20(0), -e10(2) * e20(2) - e20(0) * e10(0), 2 * e10(1) * e20(2) - e10(2) * e20(1);
        B2_dxyz1 << -e10(1) * e10(0), std::pow(e10(2), 2) + std::pow(e10(0), 2), -e10(2) * e10(1);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        B2_dnorm1 = B2_b2.dot(B2_dxyz1) / Norm_B2;
        db2_dX << -((B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(0) * Norm_B2 - B2_b2(0) * B2_dnorm1) / Norm_B2_2),
            -((B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(1) * Norm_B2 - B2_b2(1) * B2_dnorm1) / Norm_B2_2),
            -((B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(2) * Norm_B2 - B2_b2(2) * B2_dnorm1) / Norm_B2_2);
        db1_dX << ((e10(1) * e10(0)) / Norm_e10_3), (-(std::pow(e10(0), 2) + std::pow(e10(2), 2)) / Norm_e10_3), ((e10(2) * e10(1)) / Norm_e10_3);
        XX << V0.dot(db1_dX) + B1(1), V1.dot(db1_dX), V2.dot(db1_dX);
        YY << V0.dot(db2_dX) + B2(1), V1.dot(db2_dX), V2.dot(db2_dX);
		dj_dx << meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(YY), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(YY);
        #pragma omp atomic
		meshes.Quad_Symmetric_grad(v0_index + meshes.V_undeformed.rows(), 1) += de_dJ.dot(dj_dx);


        B2_dxyz0 << -e10(0) * e20(1) + 2 * e10(1) * e20(0), -e10(2) * e20(2) - e20(0) * e10(0), 2 * e10(1) * e20(2) - e10(2) * e20(1);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        db2_dX << ((B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2),
            ((B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2),
            ((B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2);
        db1_dX << -((e10(1) * e10(0)) / Norm_e10_3), -(-(pow(e10(0), 2) + pow(e10(2), 2)) / Norm_e10_3), -((e10(2) * e10(1)) / Norm_e10_3);
        XX << V0.dot(db1_dX), V1.dot(db1_dX) + B1(1), V2.dot(db1_dX);
        YY << V0.dot(db2_dX), V1.dot(db2_dX) + B2(1), V2.dot(db2_dX);
        dj_dx << meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(YY), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v1_index + meshes.V_undeformed.rows(), 1) += de_dJ.dot(dj_dx);


        B2_dxyz0 << -e10(1) * e10(0), std::pow(e10(2), 2) + std::pow(e10(0), 2), -e10(2) * e10(1);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        db2_dX << (B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2,
            (B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2,
            (B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2;
        XX << 0, 0, B1(1);
        YY << V0.dot(db2_dX), V1.dot(db2_dX), V2.dot(db2_dX) + B2(1);
        dj_dx << meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(YY), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v2_index + meshes.V_undeformed.rows(), 1) += de_dJ.dot(dj_dx);


        B2_dxyz0 << 2 * e10(2) * e20(0) - e10(0) * e20(2), -e10(1) * e20(2) + 2 * e10(2) * e20(1), -e10(0) * e20(0) - e10(1) * e20(1);
        B2_dxyz1 << -e10(0) * e10(2), -e10(2) * e10(1), std::pow(e10(0), 2) + std::pow(e10(1), 2);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        B2_dnorm1 = B2_b2.dot(B2_dxyz1) / Norm_B2;
        db2_dX << -((B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(0) * Norm_B2 - B2_b2(0) * B2_dnorm1) / Norm_B2_2),
			-((B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(1) * Norm_B2 - B2_b2(1) * B2_dnorm1) / Norm_B2_2),
			-((B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1(2) * Norm_B2 - B2_b2(2) * B2_dnorm1) / Norm_B2_2);
        db1_dX << ((e10(2) * e10(0)) / Norm_e10_3), ((e10(2) * e10(1)) / Norm_e10_3), (-(pow(e10(0), 2) + pow(e10(1), 2)) / Norm_e10_3);
        XX << V0.dot(db1_dX) + B1(2), V1.dot(db1_dX), V2.dot(db1_dX);
        YY << V0.dot(db2_dX) + B2(2), V1.dot(db2_dX), V2.dot(db2_dX);
        dj_dx << meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(YY), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v0_index + meshes.V_undeformed.rows(), 2) += de_dJ.dot(dj_dx);


        B2_dxyz0 << 2 * e10(2) * e20(0) - e10(0) * e20(2), -e10(1) * e20(2) + 2 * e10(2) * e20(1), -e10(0) * e20(0) - e10(1) * e20(1);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        db2_dX << ((B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2),
            ((B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2),
            ((B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2);
        db1_dX << -((e10(2) * e10(0)) / Norm_e10_3), -((e10(2) * e10(1)) / Norm_e10_3), -(-(std::pow(e10(0), 2) + std::pow(e10(1), 2)) / Norm_e10_3);
        XX << V0.dot(db1_dX), V1.dot(db1_dX) + B1(2), V2.dot(db1_dX);
        YY << V0.dot(db2_dX), V1.dot(db2_dX) + B2(2), V2.dot(db2_dX);
        dj_dx << meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(YY), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v1_index + meshes.V_undeformed.rows(), 2) += de_dJ.dot(dj_dx);

        B2_dxyz0 << -e10(0) * e10(2), -e10(2) * e10(1), std::pow(e10(0), 2) + std::pow(e10(1), 2);
        B2_dnorm0 = B2_b2.dot(B2_dxyz0) / Norm_B2;
        db2_dX << (B2_dxyz0(0) * Norm_B2 - B2_b2(0) * B2_dnorm0) / Norm_B2_2,
            (B2_dxyz0(1) * Norm_B2 - B2_b2(1) * B2_dnorm0) / Norm_B2_2,
            (B2_dxyz0(2) * Norm_B2 - B2_b2(2) * B2_dnorm0) / Norm_B2_2;
        XX << 0, 0, B1(2);
        YY << V0.dot(db2_dX), V1.dot(db2_dX), V2.dot(db2_dX) + B2(2);
        dj_dx << meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D1d.row(i + meshes.F_undeformed.rows()).dot(YY), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(XX), meshes.D2d.row(i + meshes.F_undeformed.rows()).dot(YY);
        #pragma omp atomic
        meshes.Quad_Symmetric_grad(v2_index + meshes.V_undeformed.rows(), 2) += de_dJ.dot(dj_dx);
    }
}





autodiff::dual2nd Compute_SymmetricDirichlet(const autodiff::ArrayXdual2nd& x, const autodiff::ArrayXdual2nd& d1d, const autodiff::ArrayXdual2nd& d2d, const autodiff::dual2nd &area, const autodiff::dual2nd &weight) 
{
    using namespace autodiff;
    const autodiff::dual2nd v0_x = x(0);
    const autodiff::dual2nd v0_y = x(1);
    const autodiff::dual2nd v0_z = x(2);
    const autodiff::dual2nd v1_x = x(3);
    const autodiff::dual2nd v1_y = x(4);
    const autodiff::dual2nd v1_z = x(5);
    const autodiff::dual2nd v2_x = x(6);
    const autodiff::dual2nd v2_y = x(7);
    const autodiff::dual2nd v2_z = x(8);
    const autodiff::dual2nd e10_x = v1_x - v0_x;
    const autodiff::dual2nd e10_y = v1_y - v0_y;
    const autodiff::dual2nd e10_z = v1_z - v0_z;
    const autodiff::dual2nd e20_x = v2_x - v0_x;
    const autodiff::dual2nd e20_y = v2_y - v0_y;
    const autodiff::dual2nd e20_z = v2_z - v0_z;
    const autodiff::dual2nd B1_x = e10_x / pow(e10_x * e10_x + e10_y * e10_y + e10_z * e10_z, 0.5);
    const autodiff::dual2nd B1_y = e10_y / pow(e10_x * e10_x + e10_y * e10_y + e10_z * e10_z, 0.5);
    const autodiff::dual2nd B1_z = e10_z / pow(e10_x * e10_x + e10_y * e10_y + e10_z * e10_z, 0.5);


    const autodiff::dual2nd b1_cross_e20_x = B1_y * e20_z - B1_z * e20_y;
    const autodiff::dual2nd b1_cross_e20_y = B1_z * e20_x - B1_x * e20_z;
    const autodiff::dual2nd b1_cross_e20_z = B1_x * e20_y - B1_y * e20_x;
    const autodiff::dual2nd b1_cross_e20_b1_x = b1_cross_e20_y * B1_z - b1_cross_e20_z * B1_y;
    const autodiff::dual2nd b1_cross_e20_b1_y = b1_cross_e20_z * B1_x - b1_cross_e20_x * B1_z;
    const autodiff::dual2nd b1_cross_e20_b1_z = b1_cross_e20_x * B1_y - b1_cross_e20_y * B1_x;
    const autodiff::dual2nd B2_x = b1_cross_e20_b1_x / pow(b1_cross_e20_b1_x * b1_cross_e20_b1_x + b1_cross_e20_b1_y * b1_cross_e20_b1_y + b1_cross_e20_b1_z * b1_cross_e20_b1_z, 0.5);
    const autodiff::dual2nd B2_y = b1_cross_e20_b1_y / pow(b1_cross_e20_b1_x * b1_cross_e20_b1_x + b1_cross_e20_b1_y * b1_cross_e20_b1_y + b1_cross_e20_b1_z * b1_cross_e20_b1_z, 0.5);
    const autodiff::dual2nd B2_z = b1_cross_e20_b1_z / pow(b1_cross_e20_b1_x * b1_cross_e20_b1_x + b1_cross_e20_b1_y * b1_cross_e20_b1_y + b1_cross_e20_b1_z * b1_cross_e20_b1_z, 0.5);


    const autodiff::dual2nd Xi_x = v0_x * B1_x + v0_y * B1_y + v0_z * B1_z;
    const autodiff::dual2nd Xi_y = v1_x * B1_x + v1_y * B1_y + v1_z * B1_z;
    const autodiff::dual2nd Xi_z = v2_x * B1_x + v2_y * B1_y + v2_z * B1_z;
    const autodiff::dual2nd Yi_x = v0_x * B2_x + v0_y * B2_y + v0_z * B2_z;
    const autodiff::dual2nd Yi_y = v1_x * B2_x + v1_y * B2_y + v1_z * B2_z;
    const autodiff::dual2nd Yi_z = v2_x * B2_x + v2_y * B2_y + v2_z * B2_z;
    
    const autodiff::dual2nd a = d1d(0) * Xi_x + d1d(1) * Xi_y + d1d(2) * Xi_z;
    const autodiff::dual2nd b = d1d(0) * Yi_x + d1d(1) * Yi_y + d1d(2) * Yi_z;
    const autodiff::dual2nd c = d2d(0) * Xi_x + d2d(1) * Xi_y + d2d(2) * Xi_z;
    const autodiff::dual2nd d = d2d(0) * Yi_x + d2d(1) * Yi_y + d2d(2) * Yi_z;

    const autodiff::dual2nd detJ = a * d - b * c;
    const autodiff::dual2nd detJ2 = detJ * detJ;
    const autodiff::dual2nd a2 = a * a;
    const autodiff::dual2nd b2 = b * b;
    const autodiff::dual2nd c2 = c * c;
    const autodiff::dual2nd d2 = d * d;
    const autodiff::dual2nd energy = 0.5 * (1 + 1 / detJ2) * (a2 + b2 + c2 + d2);
    return area * energy * weight;
}

void Compute_Newton_SymmetricDirichlet(Meshes &meshes) 
{
    meshes.energy_Symmetric = 0;
    meshes.C_Symmetric = Eigen::VectorXd::Zero(meshes.V_undeformed.rows()*3 + meshes.V_deformed.rows() * 3);

    #pragma omp parallel for
    for (int i = 0; i < meshes.F_undeformed.rows(); i++) {
        autodiff::ArrayXdual2nd x(9);
        x(0) = meshes.V_undeformed(meshes.F_undeformed(i, 0), 0);
        x(1) = meshes.V_undeformed(meshes.F_undeformed(i, 0), 1);
        x(2) = meshes.V_undeformed(meshes.F_undeformed(i, 0), 2);
        x(3) = meshes.V_undeformed(meshes.F_undeformed(i, 1), 0);
        x(4) = meshes.V_undeformed(meshes.F_undeformed(i, 1), 1);
        x(5) = meshes.V_undeformed(meshes.F_undeformed(i, 1), 2);
        x(6) = meshes.V_undeformed(meshes.F_undeformed(i, 2), 0);
        x(7) = meshes.V_undeformed(meshes.F_undeformed(i, 2), 1);
        x(8) = meshes.V_undeformed(meshes.F_undeformed(i, 2), 2);
        autodiff::ArrayXdual2nd d1d(3);
        d1d(0) = meshes.D1d(i, 0);
        d1d(1) = meshes.D1d(i, 1);
        d1d(2) = meshes.D1d(i, 2);
        autodiff::ArrayXdual2nd d2d(3);
        d2d(0) = meshes.D2d(i, 0);
        d2d(1) = meshes.D2d(i, 1);
        d2d(2) = meshes.D2d(i, 2);
        autodiff::dual2nd area = meshes.restShapeArea(i);
        autodiff::dual2nd weight = meshes.weight_symmetric;
        using namespace autodiff;
        autodiff::dual2nd energy;
        Eigen::VectorXd grad = gradient(Compute_SymmetricDirichlet, wrt(x), at(x, d1d, d2d, area, weight), energy);
        double energy_double = static_cast<double>(energy);
        #pragma omp atomic
        meshes.energy_Symmetric += energy_double;
        int v0_index = meshes.F_undeformed(i, 0);
        int v1_index = meshes.F_undeformed(i, 1);
        int v2_index = meshes.F_undeformed(i, 2);
        for (int dim = 0; dim < 3; dim++) {
            #pragma omp atomic
            meshes.C_Symmetric(v0_index*3 + dim) += grad(dim);
            #pragma omp atomic
            meshes.C_Symmetric(v1_index*3 + dim) += grad(dim + 3);
            #pragma omp atomic
            meshes.C_Symmetric(v2_index*3 + dim) += grad(dim + 6);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < meshes.F_deformed.rows(); i++) {
        autodiff::ArrayXdual2nd x(9);
        x(0) = meshes.V_deformed(meshes.F_deformed(i, 0), 0);
        x(1) = meshes.V_deformed(meshes.F_deformed(i, 0), 1);
        x(2) = meshes.V_deformed(meshes.F_deformed(i, 0), 2);
        x(3) = meshes.V_deformed(meshes.F_deformed(i, 1), 0);
        x(4) = meshes.V_deformed(meshes.F_deformed(i, 1), 1);
        x(5) = meshes.V_deformed(meshes.F_deformed(i, 1), 2);
        x(6) = meshes.V_deformed(meshes.F_deformed(i, 2), 0);
        x(7) = meshes.V_deformed(meshes.F_deformed(i, 2), 1);
        x(8) = meshes.V_deformed(meshes.F_deformed(i, 2), 2);
        autodiff::ArrayXdual2nd d1d(3);
        d1d(0) = meshes.D1d(i + meshes.F_undeformed.rows(), 0);
        d1d(1) = meshes.D1d(i + meshes.F_undeformed.rows(), 1);
        d1d(2) = meshes.D1d(i + meshes.F_undeformed.rows(), 2);
        autodiff::ArrayXdual2nd d2d(3);
        d2d(0) = meshes.D2d(i + meshes.F_undeformed.rows(), 0);
        d2d(1) = meshes.D2d(i + meshes.F_undeformed.rows(), 1);
        d2d(2) = meshes.D2d(i + meshes.F_undeformed.rows(), 2);
        autodiff::dual2nd area = meshes.restShapeArea(i + meshes.F_undeformed.rows());
        autodiff::dual2nd weight = meshes.weight_symmetric;
        using namespace autodiff;
        autodiff::dual2nd energy;
        Eigen::VectorXd grad = gradient(Compute_SymmetricDirichlet, wrt(x), at(x, d1d, d2d, area, weight), energy);
        double energy_double = static_cast<double>(energy);
        #pragma omp atomic
        meshes.energy_Symmetric += energy_double;
        int v0_index = meshes.F_deformed(i, 0);
        int v1_index = meshes.F_deformed(i, 1);
        int v2_index = meshes.F_deformed(i, 2);
        for (int dim = 0; dim < 3; dim++) {
            #pragma omp atomic
            meshes.C_Symmetric(v0_index*3 + dim + meshes.V_undeformed.rows() * 3) += grad(dim);
            #pragma omp atomic
            meshes.C_Symmetric(v1_index*3 + dim + meshes.V_undeformed.rows() * 3) += grad(dim + 3);
            #pragma omp atomic
            meshes.C_Symmetric(v2_index*3 + dim + meshes.V_undeformed.rows() * 3) += grad(dim + 6);
        }
    }
}

int choose(int num, int v1, int v2, int v3) {
    if (num == 0) {
        return v1;
    }
    else if (num == 1) {
        return v2;
    }
    else {
        return v3;
    }
}


void Compute_Newton_derivatives_SymmetricDirichlet(Meshes &meshes) 
{
    std::vector<Eigen::Triplet<double>> &tripletList = meshes.tripletList_Symmetric;
    tripletList = std::vector<Eigen::Triplet<double>>(9 * 9 * meshes.F_undeformed.rows() + 9 * 9 * meshes.F_deformed.rows());
    #pragma omp parallel for
    for (int i = 0; i < meshes.F_undeformed.rows(); i++) {
        autodiff::ArrayXdual2nd x(9);
        x(0) = meshes.V_undeformed(meshes.F_undeformed(i, 0), 0);
        x(1) = meshes.V_undeformed(meshes.F_undeformed(i, 0), 1);
        x(2) = meshes.V_undeformed(meshes.F_undeformed(i, 0), 2);
        x(3) = meshes.V_undeformed(meshes.F_undeformed(i, 1), 0);
        x(4) = meshes.V_undeformed(meshes.F_undeformed(i, 1), 1);
        x(5) = meshes.V_undeformed(meshes.F_undeformed(i, 1), 2);
        x(6) = meshes.V_undeformed(meshes.F_undeformed(i, 2), 0);
        x(7) = meshes.V_undeformed(meshes.F_undeformed(i, 2), 1);
        x(8) = meshes.V_undeformed(meshes.F_undeformed(i, 2), 2);
        autodiff::ArrayXdual2nd d1d(3);
        d1d(0) = meshes.D1d(i, 0);
        d1d(1) = meshes.D1d(i, 1);
        d1d(2) = meshes.D1d(i, 2);
        autodiff::ArrayXdual2nd d2d(3);
        d2d(0) = meshes.D2d(i, 0);
        d2d(1) = meshes.D2d(i, 1);
        d2d(2) = meshes.D2d(i, 2);
        autodiff::dual2nd area = meshes.restShapeArea(i);
        autodiff::dual2nd weight = meshes.weight_symmetric;
        using namespace autodiff;
        autodiff::dual2nd energy;
        autodiff::VectorXdual grad;
        // Eigen::MatrixXd grad = gradient(Compute_SymmetricDirichlet, wrt(x), at(x, d1d, d2d, area, weight), energy);
        Eigen::MatrixXd H = hessian(Compute_SymmetricDirichlet, wrt(x), at(x, d1d, d2d, area, weight), energy, grad);

        int v0_index = meshes.F_undeformed(i, 0);
        int v1_index = meshes.F_undeformed(i, 1);
        int v2_index = meshes.F_undeformed(i, 2);
        
        for (int idx_i = 0; idx_i < 3; idx_i++) {
            int v_n_index_1 = choose(idx_i, v0_index, v1_index, v2_index);
            for (int idx_j = 0; idx_j < 3; idx_j++) {
                int v_n_index_2 = choose(idx_j, v0_index, v1_index, v2_index);
                for (int dim_i = 0; dim_i < 3; dim_i++) {
                    int hessian_index_i = idx_i * 3 + dim_i;
                    for (int dim_j = 0; dim_j < 3; dim_j++) {
                        int hessian_index_j = idx_j * 3 + dim_j;
                        tripletList[idx_i * 27 + idx_j * 9 + dim_i * 3 + dim_j] = Eigen::Triplet<double>(v_n_index_1 * 3 + dim_i, v_n_index_2 * 3 + dim_j, H(hessian_index_i, hessian_index_j));
                    }
                }
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < meshes.F_deformed.rows(); i++) {
        autodiff::ArrayXdual2nd x(9);
        x(0) = meshes.V_deformed(meshes.F_deformed(i, 0), 0);
        x(1) = meshes.V_deformed(meshes.F_deformed(i, 0), 1);
        x(2) = meshes.V_deformed(meshes.F_deformed(i, 0), 2);
        x(3) = meshes.V_deformed(meshes.F_deformed(i, 1), 0);
        x(4) = meshes.V_deformed(meshes.F_deformed(i, 1), 1);
        x(5) = meshes.V_deformed(meshes.F_deformed(i, 1), 2);
        x(6) = meshes.V_deformed(meshes.F_deformed(i, 2), 0);
        x(7) = meshes.V_deformed(meshes.F_deformed(i, 2), 1);
        x(8) = meshes.V_deformed(meshes.F_deformed(i, 2), 2);
        autodiff::ArrayXdual2nd d1d(3);
        d1d(0) = meshes.D1d(i + meshes.F_undeformed.rows(), 0);
        d1d(1) = meshes.D1d(i + meshes.F_undeformed.rows(), 1);
        d1d(2) = meshes.D1d(i + meshes.F_undeformed.rows(), 2);
        autodiff::ArrayXdual2nd d2d(3);
        d2d(0) = meshes.D2d(i + meshes.F_undeformed.rows(), 0);
        d2d(1) = meshes.D2d(i + meshes.F_undeformed.rows(), 1);
        d2d(2) = meshes.D2d(i + meshes.F_undeformed.rows(), 2);
        autodiff::dual2nd area = meshes.restShapeArea(i + meshes.F_undeformed.rows());
        autodiff::dual2nd weight = meshes.weight_symmetric;
        using namespace autodiff;
        autodiff::dual2nd energy;
        autodiff::VectorXdual grad;
        // Eigen::MatrixXd grad = gradient(Compute_SymmetricDirichlet, wrt(x), at(x, d1d, d2d, area, weight), energy);
        Eigen::MatrixXd H = hessian(Compute_SymmetricDirichlet, wrt(x), at(x, d1d, d2d, area, weight), energy, grad);

        int v0_index = meshes.F_deformed(i, 0) + meshes.V_undeformed.rows();
        int v1_index = meshes.F_deformed(i, 1) + meshes.V_undeformed.rows();
        int v2_index = meshes.F_deformed(i, 2) + meshes.V_undeformed.rows();

        for (int idx_i = 0; idx_i < 3; idx_i++) {
            int v_n_index_1 = choose(idx_i, v0_index, v1_index, v2_index);
            for (int idx_j = 0; idx_j < 3; idx_j++) {
                int v_n_index_2 = choose(idx_j, v0_index, v1_index, v2_index);
                for (int dim_i = 0; dim_i < 3; dim_i++) {
                    int hessian_index_i = idx_i * 3 + dim_i;
                    for (int dim_j = 0; dim_j < 3; dim_j++) {
                        int hessian_index_j = idx_j * 3 + dim_j;
                        tripletList[idx_i *
                            27 + idx_j * 9 + dim_i * 3 + dim_j + meshes.F_undeformed.rows() * 81] = Eigen::Triplet<double>(v_n_index_1 * 3 + dim_i + meshes.V_undeformed.rows() * 3, v_n_index_2 * 3 + dim_j + meshes.V_undeformed.rows() * 3, H(hessian_index_i, hessian_index_j));
                    }
                }
            }
        }
    }
}

