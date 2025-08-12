"""
Below code performs a time step optimisation for a Molecular Dynamics (MD) 
simulation of argon atoms using the Lennard-Jones potential, to find the 
largest stable time step for the MD simulation that maintains 
energy conservation and minimises numerical error.


For context and results see 'Argon molecular dynamics' PDF
"""

import numpy as np
from MDLJ import MDLJ

eps = 0.0104  # epsilon of Ar in eV
sig = 3.40  # sigma of Ar in Angstroms
Mr = 39.948  # molecular mass of Ar in g/mol
dt = 5 * 10**-15  # time step in seconds

eps *= 1.60218 * 10**-19  # conversion to J
sig *= 10**-10  # conversion to m
Mr /= 1000  # conversion to kg/mol
m = Mr / (6.022 * 10**23)  # mass of a single atom in kg
t0 = np.sqrt((m * sig**2) / eps)
dtr = dt / t0  # reduced time step

print('Time step in seconds =', dt)
print('Reduced time step =', dtr)

nc = 7  # number of repeat fcc unit cells
dens = 0.9  # density
Tin = 0.2  # initial temperature
steps = 300  # number of time steps
tests = 100  # number of time steps to be tried to find optimal one

e = np.zeros((steps, tests))  # array to store the system energies for every
# time step at each iteration

merit = np.zeros(tests)  # array to store the figure of merit
time = np.linspace(dtr, 0.1, tests)  # timesteps to be tested

# returns the potential energy, kinetic energy, total energy, temperature
# pressure, number of atoms, x y and z coordinates and velocity
# autocorrelation function at each time step
u, k, e[:, 0], T, p, n, x, y, z, cvv = MDLJ(nc, dens, Tin, steps, dtr)

e_avg = np.mean(e[:, 0])
e_min = np.min(e[:, 0])
e_max = np.max(e[:, 0])
merit[0] = (e_max - e_min) / e_avg  # fluctuation from average e

print('Average e for initial timestep =', e_avg)
print('Fluctuation from average e for initial timestep =', merit[0])

merit_f = 0  # value of final accepted fluctuation
dt_f = 0  # value of final accepted timestep

# will run through each timestep and calculate the fluctuation from the average
for i in range(1, tests):
    u, k, e[:, i], T, p, n, x, y, z, cvv = MDLJ(nc, dens, Tin, steps, time[i])
    e_avg = np.mean(e[:, i])
    e_min = np.min(e[:, i])
    e_max = np.max(e[:, i])
    merit[i] = (e_max - e_min) / e_avg
    # sets the final fluctuation and timestep as the ones that are still less
    # than 10**-4, with the condition that if the fluctuation goes above
    # 10**-4 then the iteration stops
    if np.absolute(merit[i]) < 10**-4:
        merit_f = merit[i]
        dt_f = time[i]
    else:
        break

print('Optimised reduced timestep (s) =', dt_f)
print('Fluctuation from average e for optimised timestep =', merit_f)

dto = dt_f * t0
print('Optimised timestep (s) =', dto)
