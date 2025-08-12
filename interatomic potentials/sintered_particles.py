# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 14:11:43 2025

@author: tom_m
"""

"""
Following script is for a Monte Carlo simulation to ascertain the void fraction
of a cube containing sintered particles, with a graphical representation of the
void fraction from all three axes.
"""

import numpy as np
from random import random
import matplotlib.pyplot as plt
from numba import jit
#determine fraction porosity in a sintered material, and visualize it
n = 6 #number of sintered particles
rp = np.array((0.38, 0.45, 0.64, 0.24, 0.25, 0.08)) #radii of sintered particles
xp = np.array((0.64, 0.21, 0.13, 0.69, 0.45, 0.96)) #x position of sintered particles
yp = np.array((0.37, 0.22, 0.98, 0.53, 0.46, 0.76)) #y position of sintered particles
zp = np.array((0.01, 0.61, 0.55, 0.91, 0.40, 0.06)) #z position of sintered particles
boxedge = 1 #edge of the cubic box in which the particles exist
ns = 1000000 #number of randomly selected points inside the box


@jit
def xval(r, a, b, c, y, z):
    """
    Returns the x coordinates that intersect the sphere for an input set
    of y and z coordinates
    """
    x1 = np.sqrt(r**2 - (y-b)**2 - (z-c)**2) + a
    x2 = -np.sqrt(r**2 - (y-b)**2 - (z-c)**2) + a
    return x1, x2


@jit
def yval(r, a, b, c, x, z):
    """
    Returns the y coordinates that intersect the sphere for an input set
    of x and z coordinates
    """
    y1 = np.sqrt(r**2 - (x-a)**2 - (z-c)**2) + b
    y2 = -np.sqrt(r**2 - (x-a)**2 - (z-c)**2) + b
    return y1, y2


@jit
def zval(r, a, b, c, x, y):
    """
    Returns the z coordinates that intersect the sphere for an input set
    of x and y coordinates
    """
    z1 = np.sqrt(r**2 - (x-a)**2 - (y-b)**2) + c
    z2 = -np.sqrt(r**2 - (x-a)**2 - (y-b)**2) + c
    return z1, z2


pointsx = np.empty(ns)
pointsy = np.empty(ns)
pointsz = np.empty(ns)
copyx = np.copy(pointsx)
copyy = np.copy(pointsy)
copyz = np.copy(pointsz)
Inx = []
Iny = []
Inz = []
Outx = []
Outy = []
Outz = []
intemp = 0
for i in range(ns):
    x = random()
    y = random()
    z = random()
    pointsx[i] = x
    pointsy[i] = y
    pointsz[i] = z
    for j in range(6):
        # x, y and z points that intersect sphere, providing boundaries.
        # If a random point is within all three boundaries, it is in the sphere
        xb1, xb2 = xval(rp[j], xp[j], yp[j], zp[j], y, z)
        yb1, yb2 = yval(rp[j], xp[j], yp[j], zp[j], x, z)
        zb1, zb2 = xval(rp[j], xp[j], yp[j], zp[j], x, y)
        if x >= xb2 and x <= xb1:
            if y >= yb2 and y <= yb1:
                if z >= zb2 and z <= zb1:
                    Inx.append(x)
                    Iny.append(y)
                    Inz.append(z)
                    intemp += 1
                
    if intemp > 0:
        Inx.append(x)
        Iny.append(y)
        Inz.append(z)
        intemp = 0
    else:
        Outx.append(x)
        Outy.append(y)
        Outz.append(z)
        
        
frac = len(Outx) / (len(Inx) + len(Outx))
print('Fraction not in spheres =', frac)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(pointsx, pointsy, pointsz, marker='x', linewidth=0.01)
# ax.set_title('Trajectories of single adatom - 10,000 jumps')
plt.figure()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(Inx, Iny, Inz, marker='x', linewidth=0.01)
ax.set_title('Angle 1')
plt.figure()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(Inz, Inx, Iny, marker='x', linewidth=0.01)
ax.set_title('Angle 2')
plt.figure()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(Iny, Inz, Inx, marker='x', linewidth=0.01)
ax.set_title('Angle 3')
plt.figure()
plt.show()