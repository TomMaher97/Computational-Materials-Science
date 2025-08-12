# -*- coding: utf-8 -*-
"""
Below script is to analyse the behavior of a liquid argon system over time
using a fixed, optimised time step, found in the MD_timestep_optimisation
script, and outputting the following:

- Plots potential, kinetic, total energy, temperature, and pressure
- Computes fluctuations and variances
- Calculates radial distribution function g(r) 
- Evaluates velocity autocorrelation 

For context and results see 'Argon molecular dynamics' PDF
"""

import numpy as np
from MDLJ import MDLJ
import matplotlib.pyplot as plt

eps = 0.0104  # epsilon of Ar in eV
sig = 3.40  # sigma of Ar in Angstroms
Mr = 39.948  # molecular mass of Ar in g/mol
rho = 1.3  # mass density of liquid Ar in g/cm^3
kb = 1.381 * 10**-23
dtr = 0.01  # optimised reduced time step in seconds, approximated due to fluctuations

eps *= 1.60218 * 10**-19  # conversion to J
sig *= 10**-10  # conversion to m
Mr /= 1000  # conversion to kg/mol
rho *= 10**3  # conversion to kg/m^3
m = Mr / (6.022 * 10**23)  # mass of a single atom in kg
dens = (rho / Mr) * 6.022 * 10**23  # atomic density in atoms/m^3
rdens = dens * sig**3

print('Reduced density of the liquid =', rdens)

t0 = np.sqrt((m * sig**2) / eps)
# dtr = dt / t0  # reduced time step

nc = 7  # number of repeat fcc unit cells
Tin = 273.15 - 187  # initial temperature in K
tin = (kb * Tin) / eps  # reduced temperature
steps = 5000  # number of time steps
cut = 1500

# returns the potential energy, kinetic energy, total energy, temperature
# pressure, number of atoms, x y and z coordinates and velocity autocorrelation function
u, k, e, T, p, n, X, Y, Z, cvv = MDLJ(nc, rdens, tin, steps, dtr)

# x values to plot functions with cutoff
x = np.linspace(cut, steps, (steps - cut) + 1)

plt.figure()
plt.plot(u/n)
plt.title('Potential energy')
plt.ylabel('u* / N')
plt.xlabel('Timestep')

plt.figure()
plt.plot(k/n)
plt.title('Kinetic energy')
plt.ylabel('k* / N')
plt.xlabel('Timestep')

plt.figure()
plt.plot(e/n)
plt.title('Total energy')
plt.ylabel('e* / N')
plt.xlabel('Timestep')

plt.figure()
plt.plot(T)
plt.title('Temperature')
plt.ylabel('T*')
plt.xlabel('Timestep')

plt.figure()
plt.plot(p)
plt.title('Pressure')
plt.ylabel('p*')
plt.xlabel('Timestep')

plt.figure()
plt.plot(x, u[(cut-2):-1]/n)
plt.title('Potential energy')
plt.ylabel('u* / N')
plt.xlabel('Timestep')

plt.figure()
plt.plot(x, k[(cut-2):-1]/n)
plt.title('Kinetic energy')
plt.ylabel('k* / N')
plt.xlabel('Timestep')

plt.figure()
plt.plot(x, e[(cut-2):-1]/n)
plt.title('Total energy')
plt.ylabel('e* / N')
plt.xlabel('Timestep')

plt.figure()
plt.plot(x, T[(cut-2):-1])
plt.title('Temperature')
plt.ylabel('T*')
plt.xlabel('Timestep')

plt.figure()
plt.plot(x, p[(cut-2):-1])
plt.title('Pressure')
plt.ylabel('p*')
plt.xlabel('Timestep')

u_avg = np.mean(u[(cut-2):-1])
u_min = np.min(u[(cut-2):-1])
u_max = np.max(u[(cut-2):-1])
u_fluc = (u_max - u_min) / u_avg  # fluctuation from average u
print('Average potential energy =', u_avg)
print('Fluctuation of potential energy from average =', u_fluc)

k_avg = np.mean(k[(cut-2):-1])
k_min = np.min(k[(cut-2):-1])
k_max = np.max(k[(cut-2):-1])
k_fluc = (k_max - k_min) / k_avg  # fluctuation from average k
print('Average kinetic energy =', k_avg)
print('Fluctuation of kinetic energy from average =', k_fluc)

e_avg = np.mean(e[(cut-2):-1])
e_min = np.min(e[(cut-2):-1])
e_max = np.max(e[(cut-2):-1])
e_fluc = (e_max - e_min) / e_avg  # fluctuation from average e
print('Average total energy =', e_avg)
print('Fluctuation of total energy from average =', e_fluc)

T_avg = np.mean(T[(cut-2):-1])
T_min = np.min(T[(cut-2):-1])
T_max = np.max(T[(cut-2):-1])
T_fluc = (T_max - T_min) / T_avg  # fluctuation from average T
print('Average temperature =', T_avg)
print('Fluctuation of temperature from average =', T_fluc)

p_avg = np.mean(p[(cut-2):-1])
p_min = np.min(p[(cut-2):-1])
p_max = np.max(p[(cut-2):-1])
p_fluc = (p_max - p_min) / p_avg  # fluctuation from average p
print('Average pressure =', p_avg)
print('Fluctuation of pressure from average =', p_fluc)

"""
Below is for question 8
"""

# number of bins for averaging
width = cut // 6
nbins = (steps - cut) // width

binsu = np.zeros((nbins, width))
binsk = np.zeros((nbins, width))
binse = np.zeros((nbins, width))
binsT = np.zeros((nbins, width))
binsp = np.zeros((nbins, width))

# divides the outputs into bins
for i in range(nbins):
    binsu[i, :] = u[width*(i+1):width*(i+2)]
    binsk[i, :] = k[width*(i+1):width*(i+2)]
    binse[i, :] = e[width*(i+1):width*(i+2)]
    binsT[i, :] = T[width*(i+1):width*(i+2)]
    binsp[i, :] = p[width*(i+1):width*(i+2)]

# averages the outputs in each bin
avgbu = np.mean(binsu, axis=1)
avgbk = np.mean(binsk, axis=1)
avgbe = np.mean(binse, axis=1)
avgbT = np.mean(binsT, axis=1)
avgbp = np.mean(binsp, axis=1)

# calculates the variance of the outputs
varu = np.mean(np.square(avgbu)) - np.square(np.mean(avgbu))
vark = np.mean(np.square(avgbk)) - np.square(np.mean(avgbk))
vare = np.mean(np.square(avgbe)) - np.square(np.mean(avgbe))
varT = np.mean(np.square(avgbT)) - np.square(np.mean(avgbT))
varp = np.mean(np.square(avgbp)) - np.square(np.mean(avgbp))

# calculates the average of the outputs
avgu = np.mean(avgbu)
avgk = np.mean(avgbk)
avge = np.mean(avgbe)
avgT = np.mean(avgbT)
avgp = np.mean(avgbp)

print('Variance of potential energy =', varu)
print('Variance of kinetic energy =', vark)
print('Variance of total energy =', vare)
print('Variance of temperature =', varT)
print('Variance of pressure =', varp)

"""
Below is for question 9
"""

def gr(nbin, a, n, x, y, z):
    """
    calculates the radial distribution function
    Parameters
    ----------
    nbin : number of bins
    a : length of unit cell (lattice parameter)
    n : number of atoms
    x : x coordinates of atoms
    y : y coordinates of atoms
    z : z coordinates of atoms
    """
    rc = a / 2
    xb = rc / nbin
    g = np.zeros((nbin, 1))
    bp = np.zeros((nbin, 1))
    ng = np.zeros((nbin, 1))

    # bin for all the distances
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]
            dx = dx - np.round(dx)
            dy = dy - np.round(dy)
            dz = dz - np.round(dz)
            dist = a * np.sqrt(dx**2 + dy**2 + dz**2)
            if dist <= rc:
                ib = np.floor(dist // xb)
                ib = int(ib)
                g[ib] = g[ib] + 1

    factor = 2 * a**3 / (4 * np.pi * n**2 * xb)
    for k in range(nbin):
        bp[k] = (k + 3/2) * xb
        ng[k] = factor * g[k] / ((k + 1) * xb)**2

    return bp, ng

a = 5.486
b, g = gr(20, a, n, X, Y, Z)

plt.figure()
plt.plot(b, g)
plt.ylabel('g(r)')
plt.xlabel('r*')

"""
Below is for question 11
"""

# averages the velocity products for all atoms at each timestep
Cvv = np.mean(cvv, axis=1)
Cvv *= (m / (3 * kb * Tin))

plt.figure()
plt.plot(Cvv)
plt.ylabel('Cvv')
plt.xlabel('Timestep')