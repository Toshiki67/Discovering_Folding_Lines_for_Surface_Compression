import numpy as np
import taichi as ti
import math

import readObj
import os

import force as force
import energy_stress as energy_stress
import color
import scipy.linalg as la
import time
import sys

dim, n_grid, steps, dt = 3, 64, 25, 1e-3
n_particles = n_grid**dim // 2 ** (dim - 1)

time_start = None

dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
GRAVITY = [0, -9.8, 0]
bound = 3

p_mass_box = p_mass*100.0

eta = 1
r = 0.001
ti.init(debug=True, arch=ti.cpu)

# get input filename
filename = sys.argv[1] if len(sys.argv) > 1 else None

#("../model/half_hemi_deformed.obj")
# filename = ("../taichi/results/01_17/hemisphere_2/0.obj")
# filename = "./results/08_12/sphere_2/0.obj"
# file = readObj.File(filename, "./results/08_12/sphere_2/sphere_remesh.obj")
file = readObj.File(filename)
C, N, dblA, V, scale, rotation_matrices = file.readObj_original()

n_particles = int(C.shape[0])
n_verticies = int(V.shape[0])
C = C.astype(np.float32)
N = N.astype(np.float32)
dblA = dblA.astype(np.float32)
V = V.astype(np.float32)

max_x = np.max(C[:, 0])
min_x = np.min(C[:, 0])
max_y = np.max(C[:, 1])
min_y = np.min(C[:, 1])
max_z = np.max(C[:, 2])
min_z = np.min(C[:, 2])

box_width = int(((max_x - min_x)+0.2)//0.005)
box_height = int(((max_z - min_z)+0.2)//0.005)
box_depth = 5

print("box_width: ", box_width)
print("box_height: ", box_height)
print("box_depth: ", box_depth)


rotation_matrices = rotation_matrices.astype(np.float32)

print(n_particles)




F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
F_dg = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)  # deformation gradient
F_e = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
F_p = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)

F_box_x = ti.Vector.field(dim, float, box_width * box_height * box_depth)
F_box_v = ti.Vector.field(dim, float, box_width * box_height * box_depth)
F_box_C = ti.Matrix.field(dim, dim, float, box_width * box_height * box_depth)
F_box_dg = ti.Matrix.field(3, 3, dtype=float, shape=box_width * box_height * box_depth)  # deformation gradient
F_colors_box = ti.Vector.field(3, float, box_width * box_height * box_depth)

F_box_sub_x = ti.Vector.field(dim, float, box_width * box_height * box_depth)
F_box_sub_v = ti.Vector.field(dim, float, box_width * box_height * box_depth)
F_box_sub_C = ti.Matrix.field(dim, dim, float, box_width * box_height * box_depth)
F_box_sub_dg = ti.Matrix.field(3, 3, dtype=float, shape=box_width * box_height * box_depth)  # deformation gradient
F_colors_box_sub = ti.Vector.field(3, float, box_width * box_height * box_depth)



log_C = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
kirchhoff_C = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
log_C_np = np.zeros((n_particles, 3, 3), dtype=np.float32)

F_Jp = ti.field(float, n_particles)


F_d1 = ti.Matrix.field(3, 1, float, n_particles)
F_d2 = ti.Matrix.field(3, 1, float, n_particles)
F_N = ti.Matrix.field(3, 1, float, n_particles)
F_M = ti.field(float, n_particles)
F_Vtx = ti.Vector.field(dim, float, n_verticies)
F_colors_Vtx = ti.Vector.field(4, float, n_verticies)

F_dm = ti.field(float, n_particles)
F_dmi = ti.field(float, (n_grid,) * dim)
F_wi = ti.field(float, (n_grid,) * dim)

F_dm_vector = ti.Vector.field(3, float, (n_grid,) * dim)


F_colors = ti.Vector.field(3, float, n_particles)
F_materials = ti.field(int, n_particles)
F_grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)
F_grid_m = ti.field(float, (n_grid,) * dim)
F_used = ti.field(int, n_particles)

theta = ti.Vector.field(2, dtype=float, shape=n_particles)
U = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
V_R = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
Rho = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
Sig = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)






iteration = 0

neighbour = (3,) * dim
print("neighbour: ", neighbour)

WATER = 0
JELLY = 1
SNOW = 2




@ti.func
def decompose(f, rho):
    e1 = ti.Vector([1, 0, 0])
    e2 = ti.Vector([0, 1, 0])
    b1 = f @ rho.transpose() @ e1
    b2 = f @ rho.transpose() @ e2
    b = ti.Matrix.cols([b1, b2, ti.Vector([0.0, 0.0, 0.0])])
    u_d, sig_d, v_d = ti.svd(b)
    return u_d, sig_d, v_d



@ti.kernel
def substep(g_x: float, g_y: float, g_z: float, force: ti.template(), force_box: ti.template(), force_box_sub: ti.template()):
    ti.loop_config(block_dim=n_grid)
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
        F_wi[I] = 0
        F_dmi[I] = 0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_dmi[base + offset] += weight * F_dm[p]
            F_wi[base + offset] += weight
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base

        F_sub = (ti.Matrix.identity(float, 3) + dt * F_C[p]) @ F_dg[p]
        u_p, sig, v_p = decompose(F_sub, Rho[p])
        # if sig[0, 0] < 0:

        U[p] = u_p
        V_R[p] = v_p
        Sig[p] = ti.Matrix([[sig[0, 0], 0, 0], [0, sig[1, 1], 0], [0, 0, 1]])
        new_F = U[p] @ Sig[p] @ V_R[p].transpose() @ Rho[p]
        F_C[p] = (new_F @ F_dg[p].inverse() - ti.Matrix.identity(float, 3)) / dt
        F_dg[p] = new_F



        F_e[p] = energy_stress.plasticity_von_mises(F_dg[p], F_e[p], F_p[p], F_d1[p], F_d2[p], F_N[p], F_dm[p], Sig[p])
        F_p[p] = F_e[p].inverse() @ F_dg[p]

    

        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        F_colors[p] = color.get_color(F_dm[p])

        stress, sig_e_p = energy_stress.stress_ori_neo_hookean(F_dg[p], F_e[p], F_p[p], F_d1[p], F_d2[p], F_N[p], U[p], V_R[p], Rho[p], Sig[p], F_dm[p])
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + F_M[p] * F_C[p]
        g_d = 1.0
        if sig_e_p[0, 0] * sig_e_p[1, 1] > 1:
            g_d = (((1 - F_dm[p]) ** 2) * (1 - r)) + r
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (F_M[p] * F_v[p] + affine @ dpos) * g_d
            F_grid_m[base + offset] += weight * F_M[p] * g_d

    ti.loop_config(block_dim=n_grid)
    for p in F_box_x:
        Xp = F_box_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F_box_dg[p] = (ti.Matrix.identity(float, 3) + dt * F_box_C[p]) @ F_box_dg[p]
        stress = energy_stress.normal_stress(F_box_dg[p])
        stress = (-dt * p_vol * 4) * stress / dx ** 2
        affine = stress + p_mass_box * F_box_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (p_mass_box * F_box_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass_box
    ti.loop_config(block_dim=n_grid)
    for p in F_box_sub_x:
        Xp = F_box_sub_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F_box_sub_dg[p] = (ti.Matrix.identity(float, 3) + dt * F_box_sub_C[p]) @ F_box_sub_dg[p]
        stress = energy_stress.normal_stress(F_box_sub_dg[p])
        stress = (-dt * p_vol * 4) * stress / dx ** 2
        affine = stress + p_mass_box * F_box_sub_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (p_mass_box * F_box_sub_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass_box


    ti.loop_config(block_dim=n_grid)
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
    force_box(F_grid_v, F_grid_m, F_box_x, F_box_dg, F_d1, F_d2, F_M, F_dm)
    force_box_sub(F_grid_v, F_grid_m, F_box_sub_x, F_box_sub_dg, F_d1, F_d2, F_M, F_dm)
    force(F_grid_v, F_grid_m, F_x, F_dg, F_d1, F_d2, F_M, F_dm)
    ti.loop_config(block_dim=n_grid)
    for I in ti.grouped(F_grid_m):
        cond = (I < bound) & (F_grid_v[I] < 0) | (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
        cond = (I*dx + F_grid_v[I]*dt < 0) | (I*dx + F_grid_v[I]*dt > 1.0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            if F_grid_m[base + offset] > 0:
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                g_v = F_grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        F_C[p] = new_C
    ti.loop_config(block_dim=n_grid)
    for p in F_box_x:
        Xp = F_box_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_box_v[p])
        new_C = ti.zero(F_box_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            if F_grid_m[base + offset] > 0:
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                g_v = F_grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        F_box_v[p] = new_v
        F_box_x[p] += dt * F_box_v[p]
        F_box_C[p] = new_C
    ti.loop_config(block_dim=n_grid)
    for p in F_box_sub_x:
        Xp = F_box_sub_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_box_sub_v[p])
        new_C = ti.zero(F_box_sub_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            if F_grid_m[base + offset] > 0:
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                g_v = F_grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        F_box_sub_v[p] = new_v
        F_box_sub_x[p] += dt * F_box_sub_v[p]
        F_box_sub_C[p] = new_C

    ti.loop_config(block_dim=n_grid)
    for p in F_Vtx:
        Xp = F_Vtx[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_Vtx[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            if F_grid_m[base + offset] > 0:
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                g_v = F_grid_v[base + offset]
                new_v += weight * g_v
        F_Vtx[p] += dt * new_v




@ti.kernel
def set_all_unused():
    for p in F_used:
        F_used[p] = 0
        # basically throw them away so they aren't rendered
        F_x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        F_Jp[p] = 1
        F_dg[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        F_v[p] = ti.Vector([0.0, 0.0, 0.0])




@ti.kernel
def set_color_by_material(mat_color: ti.types.ndarray()):
    for i in range(n_particles):
        mat = F_materials[i]
        F_colors[i] = ti.Vector([0,0,1])
    for i in range(n_verticies):
        F_colors_Vtx[i] = ti.Vector([mat_color[0, 0], mat_color[0, 1], mat_color[0, 2], 1.0])
    for i in range(box_width * box_height * box_depth):
        F_colors_box[i] = ti.Vector([0, 1, 0])
    for i in range(box_width * box_height * box_depth):
        F_colors_box_sub[i] = ti.Vector([0, 1, 0])




paused = True

Isometry = False

use_random_colors = False
particles_radius = 0.002

material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0)]

@ti.kernel
def compute_tangent():
    for i in range(n_particles):
        Sig[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        U[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_M[i] = F_M[i] * p_vol
        cross_product = ti.math.cross(ti.Vector([F_N[i][0,0], F_N[i][1,0], F_N[i][2,0]]), ti.Vector([1.0, 0.0, 0.0]))
        if cross_product.norm_sqr() < 1e-4:
            cross_product = ti.math.cross(ti.Vector([F_N[i][0,0], F_N[i][1,0], F_N[i][2,0]]), ti.Vector([0.0, 1.0, 0.0]))
        cross_product = cross_product / ti.sqrt(cross_product.norm_sqr())
        F_d1[i] = ti.Matrix([[cross_product[0]], [cross_product[1]], [cross_product[2]]])
        cross_product_2 = ti.math.cross(ti.Vector([F_N[i][0,0], F_N[i][1,0], F_N[i][2,0]]), ti.Vector([F_d1[i][0,0], F_d1[i][1,0], F_d1[i][2,0]]))
        cross_product_2 = cross_product_2 / ti.sqrt(cross_product_2.norm_sqr())
        F_d2[i] = ti.Matrix([[cross_product_2[0]], [cross_product_2[1]], [cross_product_2[2]]])
        assert ti.math.dot(ti.Vector([F_N[i][0,0], F_N[i][1,0], F_N[i][2,0]]), ti.Vector([F_d1[i][0,0], F_d1[i][1,0], F_d1[i][2,0]])) < 1e-4
        assert ti.math.dot(ti.Vector([F_N[i][0,0], F_N[i][1,0], F_N[i][2,0]]), ti.Vector([F_d2[i][0,0], F_d2[i][1,0], F_d2[i][2,0]])) < 1e-4
    base = ti.Vector([min_x-0.05, max_y+0.01, min_z-0.05])
    height = max_z - min_z + 0.1
    width = max_x - min_x + 0.1
    for i in range(box_width):
        for j in range(box_height):
            for k in range(box_depth):
                I = i * box_height * box_depth + j * box_depth + k
                F_box_x[I] = ti.Vector([base[0] + i *  width/box_width, base[1] + k * 0.005, base[2] + j * height/box_height])
                F_box_dg[I] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    base = ti.Vector([min_x - 0.05, min_y - 0.01, min_z - 0.05])
    for i in range(box_width):
        for j in range(box_height):
            for k in range(box_depth):
                I = i * box_height * box_depth + j * box_depth + k
                F_box_sub_x[I] = ti.Vector([base[0] + i * width/box_width, base[1] + k * 0.005 - box_depth*0.005, base[2] + j * height/box_height])
                F_box_sub_dg[I] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
def init_mesh():
    # read the mesh
    global C, N, dblA
    set_all_unused()
    F_x.from_numpy(C)
    F_Vtx.from_numpy(V)
    F_v.from_numpy(np.zeros_like(C))
    F_N.from_numpy(N)
    F_M.from_numpy(dblA)
    # F_M *= p_vol
    F_materials.from_numpy(np.ones(n_particles, dtype=np.int32) * JELLY)
    F_used.from_numpy(np.ones(n_particles, dtype=np.int32))
    F_Jp.from_numpy(np.ones(n_particles, dtype=np.float32))
    F_dg.from_numpy(np.stack([np.eye(3) for _ in range(n_particles)]).astype(np.float32))
    F_e.from_numpy(np.stack([np.eye(3) for _ in range(n_particles)]).astype(np.float32))
    F_p.from_numpy(np.stack([np.eye(3) for _ in range(n_particles)]).astype(np.float32))
    F_C.from_numpy(np.zeros((n_particles, 3, 3), dtype=np.float32))
    F_box_dg.from_numpy(np.stack([np.eye(3) for _ in range(box_width * box_height * box_depth)]).astype(np.float32))
    F_box_v.from_numpy(np.zeros((box_width * box_height * box_depth, 3), dtype=np.float32))
    F_box_C.from_numpy(np.zeros((box_width * box_height * box_depth, 3, 3), dtype=np.float32))
    F_box_sub_dg.from_numpy(np.stack([np.eye(3) for _ in range(box_width * box_height * box_depth)]).astype(np.float32))
    F_box_sub_v.from_numpy(np.zeros((box_width * box_height * box_depth, 3), dtype=np.float32))
    F_box_sub_C.from_numpy(np.zeros((box_width * box_height * box_depth, 3, 3), dtype=np.float32))
    compute_tangent()
    Rho.from_numpy(rotation_matrices)


def init():
    global paused
    init_mesh()


init()

res = (1080, 720)
window = ti.ui.Window("MPM 3D", res, vsync=True)

canvas = window.get_canvas()
gui = window.get_gui()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)

camera.fov(55)


def show_options():
    global use_random_colors
    global paused
    global particles_radius
    global Isometry


    with gui.sub_window("Options", 0.05, 0.45, 0.2, 0.4) as w:
        set_color_by_material(np.array(material_colors, dtype=np.float32))
        particles_radius = w.slider_float("particles radius ", particles_radius, 0, 0.1)
        if w.button("restart"):
            init()
        if paused:
            if w.button("Continue"):
                paused = False
        else:
            if w.button("Pause"):
                paused = True


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    colors_used = F_colors
    scene.particles(F_x, per_vertex_color=colors_used, radius=particles_radius)
    # scene.particles(F_Vtx, per_vertex_color=F_colors_Vtx, radius=particles_radius)
    scene.particles(F_box_x, per_vertex_color=F_colors_box, radius=particles_radius)
    scene.particles(F_box_sub_x, per_vertex_color=F_colors_box_sub, radius=particles_radius)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

def main():
    frame_id = 0
    iteration = 0

    if iteration % 1 == 0:
        C = F_Vtx.to_numpy()
        file.writeObj_csv(C, iteration)
        x_np = F_x.to_numpy()
        dm_np = F_dm.to_numpy()
        file.writeParticle(x_np, dm_np, iteration)


    while window.running:
        frame_id += 1
        frame_id = frame_id % 256


        if not paused and not Isometry:
            if iteration < 100000:
                for _ in range(steps):
                    substep(*GRAVITY, force.no_force, force.box_gravity, force.box_anti_gravity)
                    # substep(*GRAVITY, force.no_force, force.box_twist, force.box_anti_twist)
                    print("iteration: ", iteration)

            # else:
            #     for _ in range(steps):
            #         substep(*GRAVITY, squash)
            iteration += 1
            if iteration % 10 == 0:
                C = F_Vtx.to_numpy()
                file.writeObj_csv(C, iteration)
                x_np = F_x.to_numpy()
                dm_np = F_dm.to_numpy()
                file.writeParticle(x_np, dm_np, iteration)
                file.write_time(time.time() - time_start, iteration)


        render()
        show_options()
        window.show()


if __name__ == "__main__":
    time_start = time.time()
    main()
