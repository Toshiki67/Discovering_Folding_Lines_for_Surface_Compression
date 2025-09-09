import numpy as np
import taichi as ti
import math
import os




dim, n_grid, steps, dt = 3, 64, 25, 1e-3
neighbour = (3,) * dim
print("neighbour: ", neighbour)
dx = 1 / n_grid
r = 0.0
p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
GRAVITY = [0, -9.8, 0]
bound = 3

p_mass_box = p_mass*100.0
@ti.func
def mean_xz(miny: float, maxy: float, F_x):
    ps_x = 0.0
    ps_z = 0.0
    count = 0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_x[p][1] / dx > miny and F_x[p][1] / dx < maxy:
            ps_x += F_x[p][0]/dx
            ps_z += F_x[p][2]/dx
            count += 1
    # convert to numpy array
    mean_x = ps_x/count
    mean_z = ps_z/count
    return mean_x, mean_z

@ti.func
def tangent_force(F_grid_v, F_grid_m, F_x, F_dg, F_d1, F_d2, F_M, F_dm):
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        d1 = F_dg[p] @ F_d1[p]
        d2 = F_dg[p] @ F_d2[p]
        normal = ti.math.cross(ti.Vector([d1[0, 0], d1[1, 0], d1[2, 0]]), ti.Vector([d2[0, 0], d2[1, 0], d2[2, 0]]))
        # project [0,-1,0] to the plane normal to noraml
        d3 = ti.Vector([0, -1, 0]) - normal * (normal.dot(ti.Vector([0, -1, 0])))
        d3 = d3 / d3.norm()
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += 20 * weight * (F_M[p] * d3)
    max_y = 0.0
    min_y = 10000.0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_x[p][1] / dx > max_y:
            max_y = F_x[p][1] / dx
        if F_x[p][1] / dx < min_y:
            min_y = F_x[p][1] / dx
    for I in ti.grouped(F_grid_m):
        if I[1] < min_y + 2.0:
            # F_grid_v[I] += dt * ti.Vector([-(I[2]-min_mean_z)*dx*1000, 0, (I[0]-min_mean_x)*dx*1000])
            F_grid_v[I][1] = 0
@ti.func
def normal_force(F_grid_v, F_grid_m, F_x, F_dg, F_d1, F_d2, F_M, F_dm):
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        d1 = F_dg[p] @ F_d1[p]
        d2 = F_dg[p] @ F_d2[p]
        normal = ti.math.cross(ti.Vector([d1[0, 0], d1[1, 0], d1[2, 0]]), ti.Vector([d2[0, 0], d2[1, 0], d2[2, 0]]))
        # noramlize
        normal = normal / normal.norm()
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]

            F_grid_v[base + offset] -= 0.03 * weight * (F_M[p] * normal)/F_grid_m[base + offset]

@ti.func
def twist(F_grid_v, F_grid_m, F_x, F_dg, F_d1, F_d2, F_M, F_dm):
    # get the max and min y for particles
    max_y = 0.0
    min_y = 10000.0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_x[p][1] / dx > max_y:
            max_y = F_x[p][1] / dx
        if F_x[p][1] / dx < min_y:
            min_y = F_x[p][1] / dx
    max_mean_x, max_mean_z = mean_xz(max_y - 1.0, max_y, F_x)
    min_mean_x, min_mean_z = mean_xz(min_y, min_y + 1.0, F_x)
    mean_mean_x, mean_mean_z = mean_xz((max_y + min_y) / 2 - 0.5, (max_y + min_y) / 2 + 0.5, F_x)
    for I in ti.grouped(F_grid_m):
        if I[1] > max_y - 2.0:
            # F_grid_v[I] = 10 * ti.Vector([(I[2]-max_mean_z)*dx, 0, -(I[0]-max_mean_x)*dx])
            F_grid_v[I][0] += 1* 0.01 * (I[2] - max_mean_z) * dx
            F_grid_v[I][2] += -1* 0.01 * (I[0] - max_mean_x) * dx
            # F_grid_v[I] = ti.Vector([0, 0, 0])
        elif I[1] < min_y + 2.0:
            # F_grid_v[I] += dt * ti.Vector([-(I[2]-min_mean_z)*dx*1000, 0, (I[0]-min_mean_x)*dx*1000])
            F_grid_v[I][0] += -1* 0.01 * (I[2] - max_mean_z) * dx
            F_grid_v[I][2] += 1 * 0.01 * (I[0] - max_mean_x) * dx
            # F_grid_v[I][1] = 0
            # F_grid_v[I] = ti.Vector([0, 0, 0])
@ti.func
def no_force(F_grid_v, F_grid_m, F_x, F_dg, F_d1, F_d2, F_M, F_dm):
    return
@ti.func
def squash(F_grid_v, F_grid_m, F_x, F_dg, F_d1, F_d2, F_M, F_dm):
    max_y = 0.0
    min_y = 10000.0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_x[p][1] / dx > max_y:
            max_y = F_x[p][1] / dx
        if F_x[p][1] / dx < min_y:
            min_y = F_x[p][1] / dx
    for I in ti.grouped(F_grid_m):
        # F_grid_v[I][1] += -dt*9.8*100
        if I[1] > max_y - 1.0:
            F_grid_v[I][1] = -6.0 * (1e-5/dt)
    for I in ti.grouped(F_grid_m):
        if I[1] < min_y + 1.0:
            # F_grid_v[I] = ti.Vector([0, 0, 0])
            F_grid_v[I][1] = 0

@ti.func
def stretch(F_grid_v, F_grid_m, F_x, F_dg, F_d1, F_d2, F_M, F_dm):
    max_y = 0.0
    min_y = 10000.0
    ti.loop_config(block_dim=n_grid)
    # for p in F_x:
    #     if F_x[p][1] / dx > max_y:
    #         max_y = F_x[p][1] / dx
    #     if F_x[p][1] / dx < min_y:
    #         min_y = F_x[p][1] / dx
    # for I in ti.grouped(F_grid_m):
    #     # F_grid_v[I][1] += -dt*9.8*100
    #     if I[1] > max_y - 1.0:
    #         F_grid_v[I][1] = 10.0
    # for I in ti.grouped(F_grid_m):
    #     if I[1] < min_y + 1.0:
    #         # F_grid_v[I] = ti.Vector([0, 0, 0])
    #         F_grid_v[I][1] = -10.0
    for p in F_x:
        if F_x[p][0] / dx > max_y:
            max_y = F_x[p][0] / dx
        if F_x[p][1] / dx < min_y:
            min_y = F_x[p][0] / dx
    for I in ti.grouped(F_grid_m):
        # F_grid_v[I][1] += -dt*9.8*100
        if I[0] > max_y - 1.0:
            F_grid_v[I][0] = 10.0
    for I in ti.grouped(F_grid_m):
        if I[0] < min_y + 1.0:
            # F_grid_v[I] = ti.Vector([0, 0, 0])
            F_grid_v[I][0] = -10.0

@ti.func
def compress(F_grid_v, F_grid_m, F_x, F_dg, F_d1, F_d2, F_M, F_dm):
    max_y = 0.0
    min_y = 10000.0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_x[p][1] / dx > max_y:
            max_y = F_x[p][1] / dx
        if F_x[p][1] / dx < min_y:
            min_y = F_x[p][1] / dx
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        # if (Xp[1] < (max_y*dx - 10.0/scale)/dx):
        #     continue
        if (Xp[1] < max_y - 3.0) and (Xp[1] < max_y - (max_y - min_y)/5):
            continue
        g_d = (((1 - F_dm[p]) ** 4) * (1 - r)) + r
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            if F_grid_m[base + offset] > 0:
                F_grid_v[base + offset] += (2.0*weight * (F_M[p]*g_d * ti.Vector([0, -9.8, 0]))/F_grid_m[base + offset] * dt) * 1e-5 / dt
                                        #/F_grid_m[base + offset] * dt)
    for I in ti.grouped(F_grid_m):
        if I[1] < min_y + 0.1:
            # F_grid_v[I] = ti.Vector([0, 0, 0])
            F_grid_v[I][1] = 0
        if F_grid_m[I] <= 0:
            F_grid_v[I] = 2.0 * dt * ti.Vector([0, -9.8, 0]) * 1e-5 / dt

@ti.func
def stick_bottom(F_grid_v, F_grid_m, F_x, F_dg, F_d1, F_d2, F_M, F_dm):
    max_y = 0.0
    min_y = 10000.0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_x[p][1] / dx > max_y:
            max_y = F_x[p][1] / dx
        if F_x[p][1] / dx < min_y:
            min_y = F_x[p][1] / dx
    for I in ti.grouped(F_grid_m):
        if I[1] < min_y + 0.1:
            # F_grid_v[I] = ti.Vector([0, 0, 0])
            F_grid_v[I][1] = 0

@ti.func
def box_gravity(F_grid_v, F_grid_m, F_box_x, F_box_dg, F_d1, F_d2, F_M, F_dm):
    ti.loop_config(block_dim=n_grid)
    for p in F_box_x:
        Xp = F_box_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            if F_grid_m[base + offset] > 0:
                F_grid_v[base + offset] += (0.002*weight * ( p_mass_box   * ti.Vector([0, -9.8, 0])) / F_grid_m[
                    base + offset]) * dt
                # /F_grid_m[base + offset] * dt)
@ti.func
def box_anti_gravity(F_grid_v, F_grid_m, F_box_x, F_box_dg, F_d1, F_d2, F_M, F_dm):
    ti.loop_config(block_dim=n_grid)
    for p in F_box_x:
        Xp = F_box_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            if F_grid_m[base + offset] > 0:
                F_grid_v[base + offset] -= (0.002*weight * ( p_mass_box   * ti.Vector([0, -9.8, 0])) / F_grid_m[
                    base + offset]) * dt


@ti.func
def box_twist(F_grid_v, F_grid_m, F_box_x, F_box_dg, F_d1, F_d2, F_M, F_dm):
    # get the max and min y for particles
    max_y = 0.0
    min_y = 10000.0
    ti.loop_config(block_dim=n_grid)
    for p in F_box_x:
        if F_box_x[p][1] / dx > max_y:
            max_y = F_box_x[p][1] / dx
    max_mean_x, max_mean_z = mean_xz(-1, 1000, F_box_x)
    for I in ti.grouped(F_grid_m):
        if I[1] > max_y - 2.0:
            # F_grid_v[I] = 10 * ti.Vector([(I[2]-max_mean_z)*dx, 0, -(I[0]-max_mean_x)*dx])
            F_grid_v[I][0] += 1* 0.001 * (I[2] - max_mean_z) * dx
            F_grid_v[I][2] += -1* 0.001 * (I[0] - max_mean_x) * dx
            # F_grid_v[I] = ti.Vector([0, 0, 0])

@ti.func
def box_anti_twist(F_grid_v, F_grid_m, F_box_x, F_box_dg, F_d1, F_d2, F_M, F_dm):
    # get the max and min y for particles
    max_y = 0.0
    min_y = 10000.0
    ti.loop_config(block_dim=n_grid)
    for p in F_box_x:
        if F_box_x[p][1] / dx < min_y:
            min_y = F_box_x[p][1] / dx
    max_mean_x, max_mean_z = mean_xz(-1, 1000, F_box_x)
    for I in ti.grouped(F_grid_m):
        if I[1] < min_y + 2.0:
            # F_grid_v[I] += dt * ti.Vector([-(I[2]-min_mean_z)*dx*1000, 0, (I[0]-min_mean_x)*dx*1000])
            F_grid_v[I][0] += -1 * 0.001 * (I[2] - max_mean_z) * dx
            F_grid_v[I][2] += 1 * 0.001 * (I[0] - max_mean_x) * dx
            # F_grid_v[I] = ti.Vector([0, 0, 0])