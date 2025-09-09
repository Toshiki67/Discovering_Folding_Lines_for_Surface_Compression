import numpy as np
from random import random
import taichi as ti
import math
import os
import scipy.linalg as la


E = 200 # Young's modulus

nu = 0.3  # Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
kx = lambda_0/2 * 5
ky = kx

E_box = E*10
nu_box = 0.1
mu_box, lambda_box = E_box / (2 * (1 + nu_box)), E_box * nu_box / ((1 + nu_box) * (1 - 2 * nu_box))


r = 0.001

yield_stress = 1e12


@ti.func
def normal_stress(f_dg_p):
    U, sig, V = ti.svd(f_dg_p)
    J = 1.0
    for d in ti.static(range(3)):
        J *= sig[d, d]
    stress = 2 * mu_box * 1 * (f_dg_p - U @ V.transpose()) @ f_dg_p.transpose() + ti.Matrix.identity(
        float, 3) * lambda_box * 1 * J * (J - 1)
    return stress


@ti.func
def decompose(f, rho):
    e1 = ti.Vector([1, 0, 0])
    e2 = ti.Vector([0, 1, 0])
    b1 = f @ rho.transpose() @ e1
    b2 = f @ rho.transpose() @ e2
    b = ti.Matrix.cols([b1, b2, ti.Vector([0.0, 0.0, 0.0])])
    u_d, sig_d, v_d = ti.svd(b)
    return u_d, sig_d, v_d

@ti.func
def von_mises(f_dg_p, f_e, f_p, u_1, u_2, u_3, f_dm_p, sig_p):
    f_e_tr = f_dg_p @ f_p.inverse()
    d1_p = f_p @ ti.Vector([u_1[0, 0], u_1[1, 0], u_1[2, 0]])
    d2_p = f_p @ ti.Vector([u_2[0, 0], u_2[1, 0], u_2[2, 0]])
    d3_p = d1_p.cross(d2_p)
    d2_p = d3_p.cross(d1_p)
    d1_p = ti.math.normalize(d1_p)
    d2_p = ti.math.normalize(d2_p)
    d3_p = ti.math.normalize(d3_p)
    rho_t = ti.Matrix.zero(float, 3, 3)
    for i in ti.static(range(3)):
        rho_t[i, 0] = d1_p[i]
        rho_t[i, 1] = d2_p[i]
        rho_t[i, 2] = d3_p[i]
    rho_p = rho_t.transpose()
    u_e_p, sig_e_p, v_e_p = decompose(f_e_tr, rho_p)
    # rho_p = ti.Matrix.identity(float, 3)
    # u_e_p, sig_e_p, v_e_p = ti.svd(f_e_tr)
    new_fe = u_e_p @ sig_e_p @ v_e_p.transpose() @ rho_p
    sig_e_p[2, 2] = 1.0
    epsilon = ti.Matrix.zero(float, 2, 2)
    average_epsilon = 0.0
    for i in ti.static(range(2)):
        if sig_e_p[i, i] > 0.01:
            epsilon[i, i] = ti.math.log(sig_e_p[i, i])
        else:
            epsilon[i, i] = ti.math.log(0.01)
        average_epsilon += epsilon[i, i]
    average_epsilon /= 2.0
    tau = 2.0*mu_0*epsilon + lambda_0*average_epsilon*2.0*ti.Matrix.identity(float, 2)
    sum_tau = tau.trace()
    cond = tau - sum_tau/2.0*ti.Matrix.identity(float, 2)
    f_e_final = ti.Matrix.zero(float, 3, 3)
    if cond.norm() > yield_stress:
        epsilon_hat = epsilon - average_epsilon*ti.Matrix.identity(float, 2)
        print(epsilon_hat.norm())
        epsilon_hat_norm = epsilon_hat.norm() + 1e-6
        delta_gamma = epsilon_hat_norm - yield_stress/(2.0*mu_0)
        epsilon = epsilon - (delta_gamma/epsilon_hat_norm)*epsilon_hat
        sig_e_final = ti.Matrix([[ti.exp(epsilon[0, 0]), 0.0, 0.0], [0.0, ti.exp(epsilon[1, 1]), 0.0], [0.0, 0.0, 1.0]])
        # sig_e_final = ti.Matrix([[ti.exp(epsilon[0, 0]), 0.0, 0.0], [0.0, ti.exp(epsilon[1, 1]), 0.0], [0.0, 0.0, 1.0]])
        f_e_final = u_e_p @ sig_e_final @ v_e_p.transpose() @ rho_p
        print("sig_e_p", sig_e_p, "sig_e_final", sig_e_final)
    else:
        f_e_final = u_e_p @ sig_e_p @ v_e_p.transpose() @ rho_p
    return f_e_final



@ti.func
def plasticity_von_mises(f_dg_p, f_e, f_p, u_1, u_2, u_3, f_dm_p, sig_p):
    f_e = f_dg_p @ f_p.inverse()
    G = ti.Matrix.zero(float, 3, 3)
    d1 = ti.Vector([u_1[0, 0], u_1[1, 0], u_1[2, 0]])
    d2 = ti.Vector([u_2[0, 0], u_2[1, 0], u_2[2, 0]])
    d3 = ti.Vector([u_3[0, 0], u_3[1, 0], u_3[2, 0]])
    d1 = f_p@d1
    d2 = f_p@d2
    d3 = d1.cross(d2)
    d2 = d3.cross(d1)

    d1 = ti.math.normalize(d1)
    d2 = ti.math.normalize(d2)
    d3 = ti.math.normalize(d3)
    G = ti.Matrix.zero(float, 3, 3)
    for i in ti.static(range(3)):
        G[i, 0] = d1[i]
        G[i, 1] = d2[i]
        G[i, 2] = d3[i]

    U, S, V = ti.svd(G.transpose() @ f_e @ G)

    if S[0, 0] < 0:
        for i in ti.static(range(3)):
            U[0, i] = -U[0, i]
        S[0, 0] = -S[0, 0]
    if S[1, 1] < 0:
        for i in ti.static(range(3)):
            U[1, i] = -U[1, i]
        S[1, 1] = -S[1, 1]

    log_S = ti.Matrix([[ti.math.log(S[0, 0]), 0, 0], [0, ti.math.log(S[1, 1]), 0], [0, 0, 0]])

    tau = 2 * mu_0 * log_S + lambda_0 * log_S.trace() * ti.Matrix.identity(float, 3)
    tau_sig = tau
    tau_1 = tau_sig[0, 0]
    tau_2 = tau_sig[1, 1]

    x = yield_stress
    constraint_0 = tau_2 < tau_1 + x
    constraint_1 = tau_2 > tau_1 - x

    constraint_2 = tau_2 < -tau_1 + x
    constraint_3 = tau_2 > -tau_1 - x

    if (not constraint_0) and (not constraint_2):
        tau_1 = 0
        tau_2 = x
    elif (not constraint_1) and (not constraint_3):
        tau_1 = 0
        tau_2 = -x
    elif (not constraint_1) and (not constraint_2):
        tau_1 = x
        tau_2 = 0
    elif (not constraint_0) and (not constraint_3):
        tau_1 = -x
        tau_2 = 0
    elif constraint_0 and constraint_1 and (not constraint_2):
        z = tau_2 - tau_1
        tau_2 = x/2 + z/2.0
        tau_1 = x - tau_2
    elif constraint_0 and constraint_1 and (not constraint_3):
        z = tau_2 - tau_1
        tau_2 = - x/2 + z/2.0
        tau_1 = - x - tau_2
    elif constraint_2 and constraint_3 and (not constraint_0):
        z = tau_2 + tau_1
        tau_2 = x/2 + z/2.0
        tau_1 = tau_2 - x
    elif constraint_2 and constraint_3 and (not constraint_1):
        z = tau_2 + tau_1
        tau_2 = -x/2 + z/2.0
        tau_1 = tau_2 + x


    tau_vec = ti.Vector([tau_1, tau_2])
    coeff = ti.Matrix.zero(float, 2, 2)
    coeff = ti.Matrix([[2 * mu_0 + lambda_0, lambda_0], [lambda_0, 2 * mu_0 + lambda_0]])
    coeff_inv = coeff.inverse()
    log_cauchy = coeff_inv @ tau_vec
    sig = ti.Matrix([[ti.math.exp(log_cauchy[0]), 0, 0], [0, ti.math.exp(log_cauchy[1]), 0], [0, 0, S[2, 2]]])
    new_F = G@U@sig@V.transpose()@G.transpose()
    return new_F


@ti.func
def stress_ori_neo_hookean(f_dg_p, f_e, f_p, u1, u2, u3, u_p, v_r_p, rho_p, sig_p, f_dm_p):
    g_d = (((1 - f_dm_p) ** 2) * (1 - r)) + r
    # g_d = 1


    p_u_1 = f_p @ ti.Vector([u1[0, 0], u1[1, 0], u1[2, 0]])
    p_u_2 = f_p @ ti.Vector([u2[0, 0], u2[1, 0], u2[2, 0]])
    p_u_3 = p_u_1.cross(p_u_2)
    p_u_2 = p_u_3.cross(p_u_1)
    p_u_1 = ti.math.normalize(p_u_1)
    p_u_2 = ti.math.normalize(p_u_2)
    p_u_3 = ti.math.normalize(p_u_3)
    f_e_rho = ti.Matrix.zero(float, 3, 3)
    for i in range(3):
        f_e_rho[i, 0] = p_u_1[i]
        f_e_rho[i, 1] = p_u_2[i]
        f_e_rho[i, 2] = p_u_3[i]
    f_e_rho = f_e_rho.transpose()
    u_e_p, sig_e_p, v_e_p = decompose(f_e, f_e_rho)
    sig_e_p[2, 2] = 1.0
    J = sig_e_p[0, 0] * sig_e_p[1, 1] * sig_e_p[2, 2]
    new_fe = u_e_p @ sig_e_p @ v_e_p.transpose() @ f_e_rho
    new_fe_inv = new_fe.inverse()

    d_sig_mu = ti.Matrix.zero(float, 3, 3)
    d_sig_mu[0,0] = mu_0*(sig_e_p[0, 0] - sig_e_p[1, 1])
    d_sig_mu[1,1] = mu_0*(sig_e_p[1, 1] - sig_e_p[0, 0])

    d_sig_lambda = ti.Matrix.zero(float, 3, 3)
    d_sig_lambda[0, 0] = lambda_0*(sig_e_p[0, 0]*sig_e_p[1, 1] - 1)*sig_e_p[1, 1]
    d_sig_lambda[1, 1] = lambda_0*(sig_e_p[0, 0]*sig_e_p[1, 1] - 1)*sig_e_p[0, 0]

    neo_mu_fe = u_e_p @ d_sig_mu @ v_e_p.transpose() @ f_e_rho
    neo_lam_fe = u_e_p @ d_sig_lambda @ v_e_p.transpose() @ f_e_rho

    f_p_inv = f_p.inverse()
    zeros_00 = ti.Matrix.zero(float, 3, 3)
    zeros_00[0, 0] = zeros_00[0, 0] + 1.0
    fe_f_00 = zeros_00@f_p_inv
    zeros_01 = ti.Matrix.zero(float, 3, 3)
    zeros_01[0, 1] = zeros_01[0, 1] + 1.0
    fe_f_01 = zeros_01@f_p_inv
    zeros_02 = ti.Matrix.zero(float, 3, 3)
    zeros_02[0, 2] = zeros_02[0, 2] + 1.0
    fe_f_02 = zeros_02@f_p_inv
    zeros_10 = ti.Matrix.zero(float, 3, 3)
    zeros_10[1, 0] = zeros_10[1, 0] + 1.0
    fe_f_10 = zeros_10@f_p_inv
    zeros_11 = ti.Matrix.zero(float, 3, 3)
    zeros_11[1, 1] = zeros_11[1, 1] + 1.0
    fe_f_11 = zeros_11@f_p_inv
    zeros_12 = ti.Matrix.zero(float, 3, 3)
    zeros_12[1, 2] = zeros_12[1, 2] + 1.0
    fe_f_12 = zeros_12@f_p_inv
    zeros_20 = ti.Matrix.zero(float, 3, 3)
    zeros_20[2, 0] = zeros_20[2, 0] + 1.0
    fe_f_20 = zeros_20@f_p_inv
    zeros_21 = ti.Matrix.zero(float, 3, 3)
    zeros_21[2, 1] = zeros_21[2, 1] + 1.0
    fe_f_21 = zeros_21@f_p_inv
    zeros_22 = ti.Matrix.zero(float, 3, 3)
    zeros_22[2, 2] = zeros_22[2, 2] + 1.0
    fe_f_22 = zeros_22@f_p_inv

    neo_mu_f = ti.Matrix.zero(float, 3, 3)

    neo_mu_f[0, 0] = neo_mu_fe[0,0] * fe_f_00[0,0] + neo_mu_fe[0, 1] * fe_f_00[0, 1] + neo_mu_fe[0, 2] * fe_f_00[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_00[1,0] + neo_mu_fe[1, 1] * fe_f_00[1, 1] + neo_mu_fe[1, 2] * fe_f_00[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_00[2,0] + neo_mu_fe[2, 1] * fe_f_00[2, 1] + neo_mu_fe[2, 2] * fe_f_00[2, 2]
    neo_mu_f[0, 1] = neo_mu_fe[0,0] * fe_f_01[0,0] + neo_mu_fe[0, 1] * fe_f_01[0, 1] + neo_mu_fe[0, 2] * fe_f_01[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_01[1,0] + neo_mu_fe[1, 1] * fe_f_01[1, 1] + neo_mu_fe[1, 2] * fe_f_01[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_01[2,0] + neo_mu_fe[2, 1] * fe_f_01[2, 1] + neo_mu_fe[2, 2] * fe_f_01[2, 2]
    neo_mu_f[0, 2] = neo_mu_fe[0,0] * fe_f_02[0,0] + neo_mu_fe[0, 1] * fe_f_02[0, 1] + neo_mu_fe[0, 2] * fe_f_02[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_02[1,0] + neo_mu_fe[1, 1] * fe_f_02[1, 1] + neo_mu_fe[1, 2] * fe_f_02[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_02[2,0] + neo_mu_fe[2, 1] * fe_f_02[2, 1] + neo_mu_fe[2, 2] * fe_f_02[2, 2]
    neo_mu_f[1, 0] = neo_mu_fe[0,0] * fe_f_10[0,0] + neo_mu_fe[0, 1] * fe_f_10[0, 1] + neo_mu_fe[0, 2] * fe_f_10[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_10[1,0] + neo_mu_fe[1, 1] * fe_f_10[1, 1] + neo_mu_fe[1, 2] * fe_f_10[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_10[2,0] + neo_mu_fe[2, 1] * fe_f_10[2, 1] + neo_mu_fe[2, 2] * fe_f_10[2, 2]
    neo_mu_f[1, 1] = neo_mu_fe[0,0] * fe_f_11[0,0] + neo_mu_fe[0, 1] * fe_f_11[0, 1] + neo_mu_fe[0, 2] * fe_f_11[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_11[1,0] + neo_mu_fe[1, 1] * fe_f_11[1, 1] + neo_mu_fe[1, 2] * fe_f_11[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_11[2,0] + neo_mu_fe[2, 1] * fe_f_11[2, 1] + neo_mu_fe[2, 2] * fe_f_11[2, 2]
    neo_mu_f[1, 2] = neo_mu_fe[0,0] * fe_f_12[0,0] + neo_mu_fe[0, 1] * fe_f_12[0, 1] + neo_mu_fe[0, 2] * fe_f_12[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_12[1,0] + neo_mu_fe[1, 1] * fe_f_12[1, 1] + neo_mu_fe[1, 2] * fe_f_12[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_12[2,0] + neo_mu_fe[2, 1] * fe_f_12[2, 1] + neo_mu_fe[2, 2] * fe_f_12[2, 2]
    neo_mu_f[2, 0] = neo_mu_fe[0,0] * fe_f_20[0,0] + neo_mu_fe[0, 1] * fe_f_20[0, 1] + neo_mu_fe[0, 2] * fe_f_20[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_20[1,0] + neo_mu_fe[1, 1] * fe_f_20[1, 1] + neo_mu_fe[1, 2] * fe_f_20[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_20[2,0] + neo_mu_fe[2, 1] * fe_f_20[2, 1] + neo_mu_fe[2, 2] * fe_f_20[2, 2]
    neo_mu_f[2, 1] = neo_mu_fe[0,0] * fe_f_21[0,0] + neo_mu_fe[0, 1] * fe_f_21[0, 1] + neo_mu_fe[0, 2] * fe_f_21[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_21[1,0] + neo_mu_fe[1, 1] * fe_f_21[1, 1] + neo_mu_fe[1, 2] * fe_f_21[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_21[2,0] + neo_mu_fe[2, 1] * fe_f_21[2, 1] + neo_mu_fe[2, 2] * fe_f_21[2, 2]
    neo_mu_f[2, 2] = neo_mu_fe[0,0] * fe_f_22[0,0] + neo_mu_fe[0, 1] * fe_f_22[0, 1] + neo_mu_fe[0, 2] * fe_f_22[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_22[1,0] + neo_mu_fe[1, 1] * fe_f_22[1, 1] + neo_mu_fe[1, 2] * fe_f_22[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_22[2,0] + neo_mu_fe[2, 1] * fe_f_22[2, 1] + neo_mu_fe[2, 2] * fe_f_22[2, 2]

    neo_lam_f = ti.Matrix.zero(float, 3, 3)
    neo_lam_f[0, 0] = neo_lam_fe[0,0] * fe_f_00[0,0] + neo_lam_fe[0, 1] * fe_f_00[0, 1] + neo_lam_fe[0, 2] * fe_f_00[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_00[1,0] + neo_lam_fe[1, 1] * fe_f_00[1, 1] + neo_lam_fe[1, 2] * fe_f_00[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_00[2,0] + neo_lam_fe[2, 1] * fe_f_00[2, 1] + neo_lam_fe[2, 2] * fe_f_00[2, 2]
    neo_lam_f[0, 1] = neo_lam_fe[0,0] * fe_f_01[0,0] + neo_lam_fe[0, 1] * fe_f_01[0, 1] + neo_lam_fe[0, 2] * fe_f_01[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_01[1,0] + neo_lam_fe[1, 1] * fe_f_01[1, 1] + neo_lam_fe[1, 2] * fe_f_01[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_01[2,0] + neo_lam_fe[2, 1] * fe_f_01[2, 1] + neo_lam_fe[2, 2] * fe_f_01[2, 2]
    neo_lam_f[0, 2] = neo_lam_fe[0,0] * fe_f_02[0,0] + neo_lam_fe[0, 1] * fe_f_02[0, 1] + neo_lam_fe[0, 2] * fe_f_02[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_02[1,0] + neo_lam_fe[1, 1] * fe_f_02[1, 1] + neo_lam_fe[1, 2] * fe_f_02[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_02[2,0] + neo_lam_fe[2, 1] * fe_f_02[2, 1] + neo_lam_fe[2, 2] * fe_f_02[2, 2]
    neo_lam_f[1, 0] = neo_lam_fe[0,0] * fe_f_10[0,0] + neo_lam_fe[0, 1] * fe_f_10[0, 1] + neo_lam_fe[0, 2] * fe_f_10[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_10[1,0] + neo_lam_fe[1, 1] * fe_f_10[1, 1] + neo_lam_fe[1, 2] * fe_f_10[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_10[2,0] + neo_lam_fe[2, 1] * fe_f_10[2, 1] + neo_lam_fe[2, 2] * fe_f_10[2, 2]
    neo_lam_f[1, 1] = neo_lam_fe[0,0] * fe_f_11[0,0] + neo_lam_fe[0, 1] * fe_f_11[0, 1] + neo_lam_fe[0, 2] * fe_f_11[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_11[1,0] + neo_lam_fe[1, 1] * fe_f_11[1, 1] + neo_lam_fe[1, 2] * fe_f_11[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_11[2,0] + neo_lam_fe[2, 1] * fe_f_11[2, 1] + neo_lam_fe[2, 2] * fe_f_11[2, 2]
    neo_lam_f[1, 2] = neo_lam_fe[0,0] * fe_f_12[0,0] + neo_lam_fe[0, 1] * fe_f_12[0, 1] + neo_lam_fe[0, 2] * fe_f_12[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_12[1,0] + neo_lam_fe[1, 1] * fe_f_12[1, 1] + neo_lam_fe[1, 2] * fe_f_12[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_12[2,0] + neo_lam_fe[2, 1] * fe_f_12[2, 1] + neo_lam_fe[2, 2] * fe_f_12[2, 2]
    neo_lam_f[2, 0] = neo_lam_fe[0,0] * fe_f_20[0,0] + neo_lam_fe[0, 1] * fe_f_20[0, 1] + neo_lam_fe[0, 2] * fe_f_20[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_20[1,0] + neo_lam_fe[1, 1] * fe_f_20[1, 1] + neo_lam_fe[1, 2] * fe_f_20[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_20[2,0] + neo_lam_fe[2, 1] * fe_f_20[2, 1] + neo_lam_fe[2, 2] * fe_f_20[2, 2]
    neo_lam_f[2, 1] = neo_lam_fe[0,0] * fe_f_21[0,0] + neo_lam_fe[0, 1] * fe_f_21[0, 1] + neo_lam_fe[0, 2] * fe_f_21[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_21[1,0] + neo_lam_fe[1, 1] * fe_f_21[1, 1] + neo_lam_fe[1, 2] * fe_f_21[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_21[2,0] + neo_lam_fe[2, 1] * fe_f_21[2, 1] + neo_lam_fe[2, 2] * fe_f_21[2, 2]
    neo_lam_f[2, 2] = neo_lam_fe[0,0] * fe_f_22[0,0] + neo_lam_fe[0, 1] * fe_f_22[0, 1] + neo_lam_fe[0, 2] * fe_f_22[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_22[1,0] + neo_lam_fe[1, 1] * fe_f_22[1, 1] + neo_lam_fe[1, 2] * fe_f_22[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_22[2,0] + neo_lam_fe[2, 1] * fe_f_22[2, 1] + neo_lam_fe[2, 2] * fe_f_22[2, 2]
    # neo_mu_f = neo_mu_fe
    # neo_lam_f = neo_lam_fe

    stress = ti.Matrix.zero(float, 3, 3)
    if sig_e_p[0, 0] * sig_e_p[1, 1] > 1:
        stress = g_d * (neo_mu_f + neo_lam_f)
    else:
        stress = g_d * neo_mu_f + neo_lam_f
    stress = stress@f_dg_p.transpose()
    return stress, sig_e_p


@ti.func
def cauchy_stress_ori_neo_hookean(f_dg_p, f_e, f_p, u1, u2, u3, u_p, v_r_p, rho_p, sig_p, f_dm_p):
    g_d = (((1 - f_dm_p) ** 2) * (1 - r)) + r
    # g_d = 1

    p_u_1 = f_p @ ti.Vector([u1[0, 0], u1[1, 0], u1[2, 0]])
    p_u_2 = f_p @ ti.Vector([u2[0, 0], u2[1, 0], u2[2, 0]])
    p_u_3 = p_u_1.cross(p_u_2)
    p_u_2 = p_u_3.cross(p_u_1)
    p_u_1 = ti.math.normalize(p_u_1)
    p_u_2 = ti.math.normalize(p_u_2)
    p_u_3 = ti.math.normalize(p_u_3)
    f_e_rho = ti.Matrix.zero(float, 3, 3)
    for i in range(3):
        f_e_rho[i, 0] = p_u_1[i]
        f_e_rho[i, 1] = p_u_2[i]
        f_e_rho[i, 2] = p_u_3[i]
    f_e_rho = f_e_rho.transpose()
    u_e_p, sig_e_p, v_e_p = decompose(f_e, f_e_rho)
    sig_e_p[2, 2] = 1.0
    J = sig_e_p[0, 0] * sig_e_p[1, 1] * sig_e_p[2, 2]
    new_fe = u_e_p @ sig_e_p @ v_e_p.transpose() @ f_e_rho
    new_fe_inv = new_fe.inverse()

    d_sig_mu = ti.Matrix.zero(float, 3, 3)
    d_sig_mu[0, 0] = mu_0 * (sig_e_p[0, 0] - sig_e_p[1, 1])
    d_sig_mu[1, 1] = mu_0 * (sig_e_p[1, 1] - sig_e_p[0, 0])

    d_sig_lambda = ti.Matrix.zero(float, 3, 3)
    d_sig_lambda[0, 0] = lambda_0 * (sig_e_p[0, 0] * sig_e_p[1, 1] - 1) * sig_e_p[1, 1]
    d_sig_lambda[1, 1] = lambda_0 * (sig_e_p[0, 0] * sig_e_p[1, 1] - 1) * sig_e_p[0, 0]

    neo_mu_fe = u_e_p @ d_sig_mu @ v_e_p.transpose() @ f_e_rho
    neo_lam_fe = u_e_p @ d_sig_lambda @ v_e_p.transpose() @ f_e_rho

    f_p_inv = f_p.inverse()
    zeros_00 = ti.Matrix.zero(float, 3, 3)
    zeros_00[0, 0] = zeros_00[0, 0] + 1.0
    fe_f_00 = zeros_00 @ f_p_inv
    zeros_01 = ti.Matrix.zero(float, 3, 3)
    zeros_01[0, 1] = zeros_01[0, 1] + 1.0
    fe_f_01 = zeros_01 @ f_p_inv
    zeros_02 = ti.Matrix.zero(float, 3, 3)
    zeros_02[0, 2] = zeros_02[0, 2] + 1.0
    fe_f_02 = zeros_02 @ f_p_inv
    zeros_10 = ti.Matrix.zero(float, 3, 3)
    zeros_10[1, 0] = zeros_10[1, 0] + 1.0
    fe_f_10 = zeros_10 @ f_p_inv
    zeros_11 = ti.Matrix.zero(float, 3, 3)
    zeros_11[1, 1] = zeros_11[1, 1] + 1.0
    fe_f_11 = zeros_11 @ f_p_inv
    zeros_12 = ti.Matrix.zero(float, 3, 3)
    zeros_12[1, 2] = zeros_12[1, 2] + 1.0
    fe_f_12 = zeros_12 @ f_p_inv
    zeros_20 = ti.Matrix.zero(float, 3, 3)
    zeros_20[2, 0] = zeros_20[2, 0] + 1.0
    fe_f_20 = zeros_20 @ f_p_inv
    zeros_21 = ti.Matrix.zero(float, 3, 3)
    zeros_21[2, 1] = zeros_21[2, 1] + 1.0
    fe_f_21 = zeros_21 @ f_p_inv
    zeros_22 = ti.Matrix.zero(float, 3, 3)
    zeros_22[2, 2] = zeros_22[2, 2] + 1.0
    fe_f_22 = zeros_22 @ f_p_inv

    neo_mu_f = ti.Matrix.zero(float, 3, 3)

    neo_mu_f[0, 0] = neo_mu_fe[0, 0] * fe_f_00[0, 0] + neo_mu_fe[0, 1] * fe_f_00[0, 1] + neo_mu_fe[0, 2] * fe_f_00[
        0, 2] + \
                     neo_mu_fe[1, 0] * fe_f_00[1, 0] + neo_mu_fe[1, 1] * fe_f_00[1, 1] + neo_mu_fe[1, 2] * fe_f_00[
                         1, 2] + \
                     neo_mu_fe[2, 0] * fe_f_00[2, 0] + neo_mu_fe[2, 1] * fe_f_00[2, 1] + neo_mu_fe[2, 2] * fe_f_00[2, 2]
    neo_mu_f[0, 1] = neo_mu_fe[0, 0] * fe_f_01[0, 0] + neo_mu_fe[0, 1] * fe_f_01[0, 1] + neo_mu_fe[0, 2] * fe_f_01[
        0, 2] + \
                     neo_mu_fe[1, 0] * fe_f_01[1, 0] + neo_mu_fe[1, 1] * fe_f_01[1, 1] + neo_mu_fe[1, 2] * fe_f_01[
                         1, 2] + \
                     neo_mu_fe[2, 0] * fe_f_01[2, 0] + neo_mu_fe[2, 1] * fe_f_01[2, 1] + neo_mu_fe[2, 2] * fe_f_01[2, 2]
    neo_mu_f[0, 2] = neo_mu_fe[0, 0] * fe_f_02[0, 0] + neo_mu_fe[0, 1] * fe_f_02[0, 1] + neo_mu_fe[0, 2] * fe_f_02[
        0, 2] + \
                     neo_mu_fe[1, 0] * fe_f_02[1, 0] + neo_mu_fe[1, 1] * fe_f_02[1, 1] + neo_mu_fe[1, 2] * fe_f_02[
                         1, 2] + \
                     neo_mu_fe[2, 0] * fe_f_02[2, 0] + neo_mu_fe[2, 1] * fe_f_02[2, 1] + neo_mu_fe[2, 2] * fe_f_02[2, 2]
    neo_mu_f[1, 0] = neo_mu_fe[0, 0] * fe_f_10[0, 0] + neo_mu_fe[0, 1] * fe_f_10[0, 1] + neo_mu_fe[0, 2] * fe_f_10[
        0, 2] + \
                     neo_mu_fe[1, 0] * fe_f_10[1, 0] + neo_mu_fe[1, 1] * fe_f_10[1, 1] + neo_mu_fe[1, 2] * fe_f_10[
                         1, 2] + \
                     neo_mu_fe[2, 0] * fe_f_10[2, 0] + neo_mu_fe[2, 1] * fe_f_10[2, 1] + neo_mu_fe[2, 2] * fe_f_10[2, 2]
    neo_mu_f[1, 1] = neo_mu_fe[0, 0] * fe_f_11[0, 0] + neo_mu_fe[0, 1] * fe_f_11[0, 1] + neo_mu_fe[0, 2] * fe_f_11[
        0, 2] + \
                     neo_mu_fe[1, 0] * fe_f_11[1, 0] + neo_mu_fe[1, 1] * fe_f_11[1, 1] + neo_mu_fe[1, 2] * fe_f_11[
                         1, 2] + \
                     neo_mu_fe[2, 0] * fe_f_11[2, 0] + neo_mu_fe[2, 1] * fe_f_11[2, 1] + neo_mu_fe[2, 2] * fe_f_11[2, 2]
    neo_mu_f[1, 2] = neo_mu_fe[0, 0] * fe_f_12[0, 0] + neo_mu_fe[0, 1] * fe_f_12[0, 1] + neo_mu_fe[0, 2] * fe_f_12[
        0, 2] + \
                     neo_mu_fe[1, 0] * fe_f_12[1, 0] + neo_mu_fe[1, 1] * fe_f_12[1, 1] + neo_mu_fe[1, 2] * fe_f_12[
                         1, 2] + \
                     neo_mu_fe[2, 0] * fe_f_12[2, 0] + neo_mu_fe[2, 1] * fe_f_12[2, 1] + neo_mu_fe[2, 2] * fe_f_12[2, 2]
    neo_mu_f[2, 0] = neo_mu_fe[0, 0] * fe_f_20[0, 0] + neo_mu_fe[0, 1] * fe_f_20[0, 1] + neo_mu_fe[0, 2] * fe_f_20[
        0, 2] + \
                     neo_mu_fe[1, 0] * fe_f_20[1, 0] + neo_mu_fe[1, 1] * fe_f_20[1, 1] + neo_mu_fe[1, 2] * fe_f_20[
                         1, 2] + \
                     neo_mu_fe[2, 0] * fe_f_20[2, 0] + neo_mu_fe[2, 1] * fe_f_20[2, 1] + neo_mu_fe[2, 2] * fe_f_20[2, 2]
    neo_mu_f[2, 1] = neo_mu_fe[0, 0] * fe_f_21[0, 0] + neo_mu_fe[0, 1] * fe_f_21[0, 1] + neo_mu_fe[0, 2] * fe_f_21[
        0, 2] + \
                     neo_mu_fe[1, 0] * fe_f_21[1, 0] + neo_mu_fe[1, 1] * fe_f_21[1, 1] + neo_mu_fe[1, 2] * fe_f_21[
                         1, 2] + \
                     neo_mu_fe[2, 0] * fe_f_21[2, 0] + neo_mu_fe[2, 1] * fe_f_21[2, 1] + neo_mu_fe[2, 2] * fe_f_21[2, 2]
    neo_mu_f[2, 2] = neo_mu_fe[0, 0] * fe_f_22[0, 0] + neo_mu_fe[0, 1] * fe_f_22[0, 1] + neo_mu_fe[0, 2] * fe_f_22[
        0, 2] + \
                     neo_mu_fe[1, 0] * fe_f_22[1, 0] + neo_mu_fe[1, 1] * fe_f_22[1, 1] + neo_mu_fe[1, 2] * fe_f_22[
                         1, 2] + \
                     neo_mu_fe[2, 0] * fe_f_22[2, 0] + neo_mu_fe[2, 1] * fe_f_22[2, 1] + neo_mu_fe[2, 2] * fe_f_22[2, 2]

    neo_lam_f = ti.Matrix.zero(float, 3, 3)
    neo_lam_f[0, 0] = neo_lam_fe[0, 0] * fe_f_00[0, 0] + neo_lam_fe[0, 1] * fe_f_00[0, 1] + neo_lam_fe[0, 2] * fe_f_00[
        0, 2] + \
                      neo_lam_fe[1, 0] * fe_f_00[1, 0] + neo_lam_fe[1, 1] * fe_f_00[1, 1] + neo_lam_fe[1, 2] * fe_f_00[
                          1, 2] + \
                      neo_lam_fe[2, 0] * fe_f_00[2, 0] + neo_lam_fe[2, 1] * fe_f_00[2, 1] + neo_lam_fe[2, 2] * fe_f_00[
                          2, 2]
    neo_lam_f[0, 1] = neo_lam_fe[0, 0] * fe_f_01[0, 0] + neo_lam_fe[0, 1] * fe_f_01[0, 1] + neo_lam_fe[0, 2] * fe_f_01[
        0, 2] + \
                      neo_lam_fe[1, 0] * fe_f_01[1, 0] + neo_lam_fe[1, 1] * fe_f_01[1, 1] + neo_lam_fe[1, 2] * fe_f_01[
                          1, 2] + \
                      neo_lam_fe[2, 0] * fe_f_01[2, 0] + neo_lam_fe[2, 1] * fe_f_01[2, 1] + neo_lam_fe[2, 2] * fe_f_01[
                          2, 2]
    neo_lam_f[0, 2] = neo_lam_fe[0, 0] * fe_f_02[0, 0] + neo_lam_fe[0, 1] * fe_f_02[0, 1] + neo_lam_fe[0, 2] * fe_f_02[
        0, 2] + \
                      neo_lam_fe[1, 0] * fe_f_02[1, 0] + neo_lam_fe[1, 1] * fe_f_02[1, 1] + neo_lam_fe[1, 2] * fe_f_02[
                          1, 2] + \
                      neo_lam_fe[2, 0] * fe_f_02[2, 0] + neo_lam_fe[2, 1] * fe_f_02[2, 1] + neo_lam_fe[2, 2] * fe_f_02[
                          2, 2]
    neo_lam_f[1, 0] = neo_lam_fe[0, 0] * fe_f_10[0, 0] + neo_lam_fe[0, 1] * fe_f_10[0, 1] + neo_lam_fe[0, 2] * fe_f_10[
        0, 2] + \
                      neo_lam_fe[1, 0] * fe_f_10[1, 0] + neo_lam_fe[1, 1] * fe_f_10[1, 1] + neo_lam_fe[1, 2] * fe_f_10[
                          1, 2] + \
                      neo_lam_fe[2, 0] * fe_f_10[2, 0] + neo_lam_fe[2, 1] * fe_f_10[2, 1] + neo_lam_fe[2, 2] * fe_f_10[
                          2, 2]
    neo_lam_f[1, 1] = neo_lam_fe[0, 0] * fe_f_11[0, 0] + neo_lam_fe[0, 1] * fe_f_11[0, 1] + neo_lam_fe[0, 2] * fe_f_11[
        0, 2] + \
                      neo_lam_fe[1, 0] * fe_f_11[1, 0] + neo_lam_fe[1, 1] * fe_f_11[1, 1] + neo_lam_fe[1, 2] * fe_f_11[
                          1, 2] + \
                      neo_lam_fe[2, 0] * fe_f_11[2, 0] + neo_lam_fe[2, 1] * fe_f_11[2, 1] + neo_lam_fe[2, 2] * fe_f_11[
                          2, 2]
    neo_lam_f[1, 2] = neo_lam_fe[0, 0] * fe_f_12[0, 0] + neo_lam_fe[0, 1] * fe_f_12[0, 1] + neo_lam_fe[0, 2] * fe_f_12[
        0, 2] + \
                      neo_lam_fe[1, 0] * fe_f_12[1, 0] + neo_lam_fe[1, 1] * fe_f_12[1, 1] + neo_lam_fe[1, 2] * fe_f_12[
                          1, 2] + \
                      neo_lam_fe[2, 0] * fe_f_12[2, 0] + neo_lam_fe[2, 1] * fe_f_12[2, 1] + neo_lam_fe[2, 2] * fe_f_12[
                          2, 2]
    neo_lam_f[2, 0] = neo_lam_fe[0, 0] * fe_f_20[0, 0] + neo_lam_fe[0, 1] * fe_f_20[0, 1] + neo_lam_fe[0, 2] * fe_f_20[
        0, 2] + \
                      neo_lam_fe[1, 0] * fe_f_20[1, 0] + neo_lam_fe[1, 1] * fe_f_20[1, 1] + neo_lam_fe[1, 2] * fe_f_20[
                          1, 2] + \
                      neo_lam_fe[2, 0] * fe_f_20[2, 0] + neo_lam_fe[2, 1] * fe_f_20[2, 1] + neo_lam_fe[2, 2] * fe_f_20[
                          2, 2]
    neo_lam_f[2, 1] = neo_lam_fe[0, 0] * fe_f_21[0, 0] + neo_lam_fe[0, 1] * fe_f_21[0, 1] + neo_lam_fe[0, 2] * fe_f_21[
        0, 2] + \
                      neo_lam_fe[1, 0] * fe_f_21[1, 0] + neo_lam_fe[1, 1] * fe_f_21[1, 1] + neo_lam_fe[1, 2] * fe_f_21[
                          1, 2] + \
                      neo_lam_fe[2, 0] * fe_f_21[2, 0] + neo_lam_fe[2, 1] * fe_f_21[2, 1] + neo_lam_fe[2, 2] * fe_f_21[
                          2, 2]
    neo_lam_f[2, 2] = neo_lam_fe[0, 0] * fe_f_22[0, 0] + neo_lam_fe[0, 1] * fe_f_22[0, 1] + neo_lam_fe[0, 2] * fe_f_22[
        0, 2] + \
                      neo_lam_fe[1, 0] * fe_f_22[1, 0] + neo_lam_fe[1, 1] * fe_f_22[1, 1] + neo_lam_fe[1, 2] * fe_f_22[
                          1, 2] + \
                      neo_lam_fe[2, 0] * fe_f_22[2, 0] + neo_lam_fe[2, 1] * fe_f_22[2, 1] + neo_lam_fe[2, 2] * fe_f_22[
                          2, 2]
    # neo_mu_f = neo_mu_fe
    # neo_lam_f = neo_lam_fe

    stress = ti.Matrix.zero(float, 3, 3)
    if sig_e_p[0, 0] * sig_e_p[1, 1] > 1:
        stress = g_d * (neo_mu_f + neo_lam_f)
    else:
        stress = g_d * neo_mu_f + neo_lam_f
    stress = stress @ f_dg_p.transpose()
    return stress/sig_e_p[0,0]/sig_e_p[1,1]



@ti.func
def stress_ori_neo_hookean3d(f_dg_p, f_e, f_p, u1, u2, u3, u_p, v_r_p, rho_p, sig_p, f_dm_p):
    g_d = (((1 - f_dm_p) ** 2) * (1 - r)) + r
    # g_d = 1

    u_e_p, sig_e_p, v_e_p = ti.svd(f_dg_p)
    J = sig_e_p[0, 0] * sig_e_p[1, 1] * sig_e_p[2, 2]
    new_fe = u_e_p @ sig_e_p @ v_e_p.transpose()

    d_sig_mu = ti.Matrix.zero(float, 3, 3)
    d_sig_mu[0,0] = mu_0*(sig_e_p[0, 0] - sig_e_p[1, 1]*sig_e_p[2, 2])
    d_sig_mu[1,1] = mu_0*(sig_e_p[1, 1] - sig_e_p[0, 0]*sig_e_p[2, 2])
    d_sig_mu[2,2] = mu_0*(sig_e_p[2, 2] - sig_e_p[0, 0]*sig_e_p[1, 1])

    d_sig_lambda = ti.Matrix.zero(float, 3, 3)
    d_sig_lambda[0, 0] = lambda_0*(sig_e_p[0, 0]*sig_e_p[1, 1]*sig_e_p[2, 2] - 1)*sig_e_p[1, 1]*sig_e_p[2, 2]
    d_sig_lambda[1, 1] = lambda_0*(sig_e_p[0, 0]*sig_e_p[1, 1]*sig_e_p[2, 2] - 1)*sig_e_p[0, 0]*sig_e_p[2, 2]
    d_sig_lambda[2, 2] = lambda_0*(sig_e_p[0, 0]*sig_e_p[1, 1]*sig_e_p[2, 2] - 1)*sig_e_p[0, 0]*sig_e_p[1, 1]

    neo_mu_fe = u_e_p @ d_sig_mu @ v_e_p.transpose()
    neo_lam_fe = u_e_p @ d_sig_lambda @ v_e_p.transpose()

    f_p_inv = f_p.inverse()
    zeros_00 = ti.Matrix.zero(float, 3, 3)
    zeros_00[0, 0] = zeros_00[0, 0] + 1.0
    fe_f_00 = zeros_00@f_p_inv
    zeros_01 = ti.Matrix.zero(float, 3, 3)
    zeros_01[0, 1] = zeros_01[0, 1] + 1.0
    fe_f_01 = zeros_01@f_p_inv
    zeros_02 = ti.Matrix.zero(float, 3, 3)
    zeros_02[0, 2] = zeros_02[0, 2] + 1.0
    fe_f_02 = zeros_02@f_p_inv
    zeros_10 = ti.Matrix.zero(float, 3, 3)
    zeros_10[1, 0] = zeros_10[1, 0] + 1.0
    fe_f_10 = zeros_10@f_p_inv
    zeros_11 = ti.Matrix.zero(float, 3, 3)
    zeros_11[1, 1] = zeros_11[1, 1] + 1.0
    fe_f_11 = zeros_11@f_p_inv
    zeros_12 = ti.Matrix.zero(float, 3, 3)
    zeros_12[1, 2] = zeros_12[1, 2] + 1.0
    fe_f_12 = zeros_12@f_p_inv
    zeros_20 = ti.Matrix.zero(float, 3, 3)
    zeros_20[2, 0] = zeros_20[2, 0] + 1.0
    fe_f_20 = zeros_20@f_p_inv
    zeros_21 = ti.Matrix.zero(float, 3, 3)
    zeros_21[2, 1] = zeros_21[2, 1] + 1.0
    fe_f_21 = zeros_21@f_p_inv
    zeros_22 = ti.Matrix.zero(float, 3, 3)
    zeros_22[2, 2] = zeros_22[2, 2] + 1.0
    fe_f_22 = zeros_22@f_p_inv

    neo_mu_f = ti.Matrix.zero(float, 3, 3)

    neo_mu_f[0, 0] = neo_mu_fe[0,0] * fe_f_00[0,0] + neo_mu_fe[0, 1] * fe_f_00[0, 1] + neo_mu_fe[0, 2] * fe_f_00[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_00[1,0] + neo_mu_fe[1, 1] * fe_f_00[1, 1] + neo_mu_fe[1, 2] * fe_f_00[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_00[2,0] + neo_mu_fe[2, 1] * fe_f_00[2, 1] + neo_mu_fe[2, 2] * fe_f_00[2, 2]
    neo_mu_f[0, 1] = neo_mu_fe[0,0] * fe_f_01[0,0] + neo_mu_fe[0, 1] * fe_f_01[0, 1] + neo_mu_fe[0, 2] * fe_f_01[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_01[1,0] + neo_mu_fe[1, 1] * fe_f_01[1, 1] + neo_mu_fe[1, 2] * fe_f_01[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_01[2,0] + neo_mu_fe[2, 1] * fe_f_01[2, 1] + neo_mu_fe[2, 2] * fe_f_01[2, 2]
    neo_mu_f[0, 2] = neo_mu_fe[0,0] * fe_f_02[0,0] + neo_mu_fe[0, 1] * fe_f_02[0, 1] + neo_mu_fe[0, 2] * fe_f_02[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_02[1,0] + neo_mu_fe[1, 1] * fe_f_02[1, 1] + neo_mu_fe[1, 2] * fe_f_02[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_02[2,0] + neo_mu_fe[2, 1] * fe_f_02[2, 1] + neo_mu_fe[2, 2] * fe_f_02[2, 2]
    neo_mu_f[1, 0] = neo_mu_fe[0,0] * fe_f_10[0,0] + neo_mu_fe[0, 1] * fe_f_10[0, 1] + neo_mu_fe[0, 2] * fe_f_10[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_10[1,0] + neo_mu_fe[1, 1] * fe_f_10[1, 1] + neo_mu_fe[1, 2] * fe_f_10[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_10[2,0] + neo_mu_fe[2, 1] * fe_f_10[2, 1] + neo_mu_fe[2, 2] * fe_f_10[2, 2]
    neo_mu_f[1, 1] = neo_mu_fe[0,0] * fe_f_11[0,0] + neo_mu_fe[0, 1] * fe_f_11[0, 1] + neo_mu_fe[0, 2] * fe_f_11[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_11[1,0] + neo_mu_fe[1, 1] * fe_f_11[1, 1] + neo_mu_fe[1, 2] * fe_f_11[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_11[2,0] + neo_mu_fe[2, 1] * fe_f_11[2, 1] + neo_mu_fe[2, 2] * fe_f_11[2, 2]
    neo_mu_f[1, 2] = neo_mu_fe[0,0] * fe_f_12[0,0] + neo_mu_fe[0, 1] * fe_f_12[0, 1] + neo_mu_fe[0, 2] * fe_f_12[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_12[1,0] + neo_mu_fe[1, 1] * fe_f_12[1, 1] + neo_mu_fe[1, 2] * fe_f_12[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_12[2,0] + neo_mu_fe[2, 1] * fe_f_12[2, 1] + neo_mu_fe[2, 2] * fe_f_12[2, 2]
    neo_mu_f[2, 0] = neo_mu_fe[0,0] * fe_f_20[0,0] + neo_mu_fe[0, 1] * fe_f_20[0, 1] + neo_mu_fe[0, 2] * fe_f_20[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_20[1,0] + neo_mu_fe[1, 1] * fe_f_20[1, 1] + neo_mu_fe[1, 2] * fe_f_20[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_20[2,0] + neo_mu_fe[2, 1] * fe_f_20[2, 1] + neo_mu_fe[2, 2] * fe_f_20[2, 2]
    neo_mu_f[2, 1] = neo_mu_fe[0,0] * fe_f_21[0,0] + neo_mu_fe[0, 1] * fe_f_21[0, 1] + neo_mu_fe[0, 2] * fe_f_21[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_21[1,0] + neo_mu_fe[1, 1] * fe_f_21[1, 1] + neo_mu_fe[1, 2] * fe_f_21[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_21[2,0] + neo_mu_fe[2, 1] * fe_f_21[2, 1] + neo_mu_fe[2, 2] * fe_f_21[2, 2]
    neo_mu_f[2, 2] = neo_mu_fe[0,0] * fe_f_22[0,0] + neo_mu_fe[0, 1] * fe_f_22[0, 1] + neo_mu_fe[0, 2] * fe_f_22[0, 2] +\
                    neo_mu_fe[1,0] * fe_f_22[1,0] + neo_mu_fe[1, 1] * fe_f_22[1, 1] + neo_mu_fe[1, 2] * fe_f_22[1, 2] +\
                    neo_mu_fe[2,0] * fe_f_22[2,0] + neo_mu_fe[2, 1] * fe_f_22[2, 1] + neo_mu_fe[2, 2] * fe_f_22[2, 2]

    neo_lam_f = ti.Matrix.zero(float, 3, 3)
    neo_lam_f[0, 0] = neo_lam_fe[0,0] * fe_f_00[0,0] + neo_lam_fe[0, 1] * fe_f_00[0, 1] + neo_lam_fe[0, 2] * fe_f_00[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_00[1,0] + neo_lam_fe[1, 1] * fe_f_00[1, 1] + neo_lam_fe[1, 2] * fe_f_00[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_00[2,0] + neo_lam_fe[2, 1] * fe_f_00[2, 1] + neo_lam_fe[2, 2] * fe_f_00[2, 2]
    neo_lam_f[0, 1] = neo_lam_fe[0,0] * fe_f_01[0,0] + neo_lam_fe[0, 1] * fe_f_01[0, 1] + neo_lam_fe[0, 2] * fe_f_01[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_01[1,0] + neo_lam_fe[1, 1] * fe_f_01[1, 1] + neo_lam_fe[1, 2] * fe_f_01[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_01[2,0] + neo_lam_fe[2, 1] * fe_f_01[2, 1] + neo_lam_fe[2, 2] * fe_f_01[2, 2]
    neo_lam_f[0, 2] = neo_lam_fe[0,0] * fe_f_02[0,0] + neo_lam_fe[0, 1] * fe_f_02[0, 1] + neo_lam_fe[0, 2] * fe_f_02[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_02[1,0] + neo_lam_fe[1, 1] * fe_f_02[1, 1] + neo_lam_fe[1, 2] * fe_f_02[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_02[2,0] + neo_lam_fe[2, 1] * fe_f_02[2, 1] + neo_lam_fe[2, 2] * fe_f_02[2, 2]
    neo_lam_f[1, 0] = neo_lam_fe[0,0] * fe_f_10[0,0] + neo_lam_fe[0, 1] * fe_f_10[0, 1] + neo_lam_fe[0, 2] * fe_f_10[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_10[1,0] + neo_lam_fe[1, 1] * fe_f_10[1, 1] + neo_lam_fe[1, 2] * fe_f_10[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_10[2,0] + neo_lam_fe[2, 1] * fe_f_10[2, 1] + neo_lam_fe[2, 2] * fe_f_10[2, 2]
    neo_lam_f[1, 1] = neo_lam_fe[0,0] * fe_f_11[0,0] + neo_lam_fe[0, 1] * fe_f_11[0, 1] + neo_lam_fe[0, 2] * fe_f_11[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_11[1,0] + neo_lam_fe[1, 1] * fe_f_11[1, 1] + neo_lam_fe[1, 2] * fe_f_11[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_11[2,0] + neo_lam_fe[2, 1] * fe_f_11[2, 1] + neo_lam_fe[2, 2] * fe_f_11[2, 2]
    neo_lam_f[1, 2] = neo_lam_fe[0,0] * fe_f_12[0,0] + neo_lam_fe[0, 1] * fe_f_12[0, 1] + neo_lam_fe[0, 2] * fe_f_12[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_12[1,0] + neo_lam_fe[1, 1] * fe_f_12[1, 1] + neo_lam_fe[1, 2] * fe_f_12[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_12[2,0] + neo_lam_fe[2, 1] * fe_f_12[2, 1] + neo_lam_fe[2, 2] * fe_f_12[2, 2]
    neo_lam_f[2, 0] = neo_lam_fe[0,0] * fe_f_20[0,0] + neo_lam_fe[0, 1] * fe_f_20[0, 1] + neo_lam_fe[0, 2] * fe_f_20[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_20[1,0] + neo_lam_fe[1, 1] * fe_f_20[1, 1] + neo_lam_fe[1, 2] * fe_f_20[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_20[2,0] + neo_lam_fe[2, 1] * fe_f_20[2, 1] + neo_lam_fe[2, 2] * fe_f_20[2, 2]
    neo_lam_f[2, 1] = neo_lam_fe[0,0] * fe_f_21[0,0] + neo_lam_fe[0, 1] * fe_f_21[0, 1] + neo_lam_fe[0, 2] * fe_f_21[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_21[1,0] + neo_lam_fe[1, 1] * fe_f_21[1, 1] + neo_lam_fe[1, 2] * fe_f_21[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_21[2,0] + neo_lam_fe[2, 1] * fe_f_21[2, 1] + neo_lam_fe[2, 2] * fe_f_21[2, 2]
    neo_lam_f[2, 2] = neo_lam_fe[0,0] * fe_f_22[0,0] + neo_lam_fe[0, 1] * fe_f_22[0, 1] + neo_lam_fe[0, 2] * fe_f_22[0, 2] +\
                    neo_lam_fe[1,0] * fe_f_22[1,0] + neo_lam_fe[1, 1] * fe_f_22[1, 1] + neo_lam_fe[1, 2] * fe_f_22[1, 2] +\
                    neo_lam_fe[2,0] * fe_f_22[2,0] + neo_lam_fe[2, 1] * fe_f_22[2, 1] + neo_lam_fe[2, 2] * fe_f_22[2, 2]
    neo_mu_f = neo_mu_fe
    neo_lam_f = neo_lam_fe

    stress = ti.Matrix.zero(float, 3, 3)
    if sig_e_p[0, 0] * sig_e_p[1, 1] > 1:
        stress = g_d * (neo_mu_f + neo_lam_f)
    else:
        stress = g_d * neo_mu_f + neo_lam_f
    stress = stress@f_dg_p.transpose()
    return stress
