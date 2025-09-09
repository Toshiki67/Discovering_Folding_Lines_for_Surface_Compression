import taichi as ti
import math

import os



@ti.func
def get_color(val):
    tmp_val = (-ti.cos(math.pi * val * 4) / 2.0 + 0.5)
    color = ti.Vector([0.0, 0.0, 0.0])
    if val >= 1.0:
        color = ti.Vector([1, 0, 0])
    elif val >= 0.75:
        color = ti.Vector([1, tmp_val, 0])
    elif val>= 0.5:
        color = ti.Vector([tmp_val, 1, 0])
    elif val >= 0.25:
        color = ti.Vector([0, 1, tmp_val])
    else:
        color = ti.Vector([0, tmp_val, 1])
    return color