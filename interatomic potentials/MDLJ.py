from __future__ import division  # make / return a float instead of a truncated int (for python 2)
# from math import sqrt
# import numpy as np
from numpy import zeros, pi, cos, log, sqrt, round, copy
from random import random  # seed
from pylab import figure, plot, show, title
from numba import jit

"""
Below script simulates a system of atoms interacting via the Lennard-Jones 
potential in 3D space using the Velocity Verlet algorithm.
"""

@jit
def fLJsum(a, n, rc, x, y, z):
    """
    Simple lattice sum for force with cutoffs and
    minimum image convention
    Calculates the force (fx, fy, fz) using LJ potential, potential energy (u),
    and part of the pressure (w)
    """
    fx = zeros(n, float)
    fy = zeros(n, float)
    fz = zeros(n, float)
    u = 0
    w = 0
    for i in range(n - 1):  # note limits
        ftx = fty = ftz = 0
        for j in range(i + 1, n):  # note limits
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]
            dx -= round(dx)
            dy -= round(dy)
            dz -= round(dz)
            dist = a * sqrt(dx**2 + dy**2 + dz**2)
            if dist < 1e-5:
                print("dist zero:", dist, i, j)
            if dist <= rc:
                dphi = (2 / dist**12 - 1 / dist**6)
                ffx = dphi * a * dx / dist**2
                ffy = dphi * a * dy / dist**2
                ffz = dphi * a * dz / dist**2
                ftx += ffx
                fty += ffy
                ftz += ffz
                phi = (1 / dist**12 - 1 / dist**6)
                u += phi
                w += dphi
                fx[j] -= ffx
                fy[j] -= ffy
                fz[j] -= ffz
        fx[i] += ftx
        fy[i] += fty
        fz[i] += ftz
    u *= 4
    w *= 24
    fx *= -24
    fy *= -24
    fz *= -24
    return u, w, fx, fy, fz

def initLJMD(nc, tin):
    """
    Initializes atom positions and velocities. Positions are set on a 
    face-centered cubic (FCC) lattice. Velocities are sampled from a 
    Maxwell-Boltzmann distribution and rescaled to match the desired initial 
    temperature (tin)
    """
    ncell = 4
    x = [0, .5, 0, .5]
    y = [0, .5, .5, 0]
    z = [0, 0, .5, .5]
    i1 = 0
    n = ncell * nc**3
    sx = zeros(n, float)
    sy = zeros(n, float)
    sz = zeros(n, float)
    vx = zeros(n, float)
    vy = zeros(n, float)
    vz = zeros(n, float)
    for k in range(nc):
        for l in range(nc):
            for m in range(nc):
                for i in range(ncell):
                    sx[i1] = (x[i] + k - 1) / nc
                    sy[i1] = (y[i] + l - 1) / nc
                    sz[i1] = (z[i] + m - 1) / nc
                    i1 += 1
    px = py = pz = 0
    for i in range(n):
        vx[i] = sqrt(-2 * log(random())) * cos(2 * pi * random())
        vy[i] = sqrt(-2 * log(random())) * cos(2 * pi * random())
        vz[i] = sqrt(-2 * log(random())) * cos(2 * pi * random())
        px += vx[i]
        py += vy[i]
        pz += vz[i]
    px /= n
    py /= n
    pz /= n
    vx -= px
    vy -= py
    vz -= pz
    vx2 = vx**2
    vy2 = vy**2
    vz2 = vz**2
    k = 0.5 * (vx2.sum() + vy2.sum() + vz2.sum())
    kin = 1.5 * n * tin
    sc = sqrt(kin / k)
    vx *= sc
    vy *= sc
    vz *= sc
    return n, sx, sy, sz, vx, vy, vz

def MDLJ(nc, density, tin, nsteps, dt):
    """
    MD code for 3D system of LJ atoms
    Iteratively updates positions and velocities using Velocity Verlet.
    Tracks energy (en, un, kn), temperature (tn), pressure (pn), 
    and velocity autocorrelation (cvv)
    """
    n, x, y, z, vx, vy, vz = initLJMD(nc, tin)
    vol = n / density
    a = vol**(1/3)
    rc = a / 2
    u, w, fx, fy, fz = fLJsum(a, n, rc, x, y, z)
    xold = x - vx * dt / a + 0.5 * fx * dt**2 / a
    yold = y - vy * dt / a + 0.5 * fy * dt**2 / a
    zold = z - vz * dt / a + 0.5 * fz * dt**2 / a
    un = zeros(nsteps, float)
    kn = zeros(nsteps, float)
    en = zeros(nsteps, float)
    tn = zeros(nsteps, float)
    pn = zeros(nsteps, float)
    cvv = zeros((nsteps, n), float)
    for j in range(nsteps):
        vxo = copy(vx)
        vyo = copy(vy)
        vzo = copy(vz)
        xnew = 2 * x - xold + fx * dt**2 / a
        ynew = 2 * y - yold + fy * dt**2 / a
        znew = 2 * z - zold + fz * dt**2 / a
        vx = a * (xnew - xold) / (2 * dt)
        vy = a * (ynew - yold) / (2 * dt)
        vz = a * (znew - zold) / (2 * dt)
        vx2 = vx**2
        vy2 = vy**2
        vz2 = vz**2
        k = 0.5 * (vx2.sum() + vy2.sum() + vz2.sum())
        temp = 2 * k / (3 * n)
        e = k + u
        un[j] = u / n
        kn[j] = k / n
        en[j] = e / n
        tn[j] = temp
        pn[j] = density * temp + w / (3 * vol)
        cvv[j, :] = vxo * vx + vyo * vy + vzo * vz
        xold = x.copy()
        yold = y.copy()
        zold = z.copy()
        x = xnew
        y = ynew
        z = znew
        u, w, fx, fy, fz = fLJsum(a, n, rc, x, y, z)
    return un, kn, en, tn, pn, n, xnew, ynew, znew, cvv

if __name__ == '__main__':
    dt = 1e-5
    nc = 3
    dens = 0.9
    T = 0.2
    nsteps = 100000
    un, kn, en, tn, pn, n, x, y, z, cvv = MDLJ(nc, dens, T, nsteps, dt)
    figure()
    plot(en)
    title("en")
    show()
    figure()
    plot(un)
    title("un")
    show()
    figure()
    plot(kn)
    title("kn")
    show()