# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 15:16:09 2025

@author: tom_m
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import random
import random as rand
from fcc_neighbours import neighbours
from potentials import LJ, Mie_n6, Morse


def quadratic(x, a, b, c):
    y = a * x**2 + b * x + c
    return y


r0_exp = 3.75  # experimental nearest neighbour distance, which in an fcc
# is equivalent to the lattice parameter, in Angstroms
E0_exp = -0.08  # cohesive energy, in eV/atom
B0_exp = 2.7    # bulk modulus, in GPa

print('Experimental a =', r0_exp)
print('Experimental B =', B0_exp)
print('Experimental E =', E0_exp)


def parameters(rm, E_fun, arg1, arg2):
    """
    rm : position of potential minimum
    E_fun : potential energy function
    arg1 : input 1 for energy function
    arg2 : input 2 for energy function
    returns the equilibrium volume per atom, lattice parameter, bulk
    modulus
    and cohesive energy
    """
    cnt = 100  # number of datapoints
    n_shells = 17  # number of shells included

    # number of atoms per shell, relative shell distance in units of nearest neighbour distance
    shell_at, shell_r = neighbours()

    # vectors to store the nearest neighbour distances and the energies corresponding to them
    E = np.zeros(cnt)
    r = np.zeros(cnt)

    for i in range(cnt):
        # vector of the nearest neighbour distances from rm
        r[i] = rm * (i + cnt) * 4 / (5 * cnt)
        for j in range(n_shells):
            # atom distance from rm is the relative shell distance multiplied by the nearest neighbour distance from rm
            dist = r[i] * shell_r[j]
            # potential energy for an atom at that distance
            pot = E_fun(dist, rm, arg1, arg2)
            # Energy for each nearest neighbour distance is the potential for each atom in the shells divided by 2
            E[i] += pot * shell_at[j] / 2

    # volume per atom
    v = (r * np.sqrt(2)) ** (3 / 4)

    # sorts the energies in ascending order
    ascend_E = np.sort(E)

    # number of datapoints around the minimum for which a parabola will be fit
    minima = 5

    # empty array to store the energies, NN distances and volumes per atom
    min_E = []
    min_r = []
    min_v = []

    for k in range(minima):
        for l in range(len(E)):
            if E[l] == ascend_E[k]:
                min_E.append(E[l])
                min_r.append(r[l])
                min_v.append(v[l])

    # returns the coefficients of the 2nd order polynomial fitted to the 'minima' number of datapoints
    poly, poly_cov = curve_fit(quadratic, min_v, min_E)

    # returns the coefficients of the 2nd order derivative of the polynomial
    d2E_dv2 = np.polyder(poly, 2)

    # abscissa and ordinates of polynomial around minimum
    x = np.linspace(min(min_v), max(min_v), 100)
    y = quadratic(x, *poly)

    # equilibrium volume per atom
    eq_v = x[np.argmin(y)]

    # equilibrium lattice constant
    eq_a = (eq_v ** (4 / 3)) / np.sqrt(2)

    # equilibrium bulk modulus
    eq_B = eq_v * np.polyval(d2E_dv2, eq_v)

    # equilibrium cohesive energy
    eq_E = min(y)

    return eq_v, eq_a, eq_B, eq_E


rm = 3.816  # initial value for position of potential minimum
sig = rm / (2**(1 / 6))  # corresponding distance at which potential is 0 for LJ
eps = 0.0104  # absolute value of potential well
alp = np.log(2) / (rm - sig)  # alpha, governs steepness of repulsive wall for Morse potential

print('Initial rm =', rm)
print('Initial eps =', eps)
print('Initial alpha =', alp)

# equilibrium v, a, B and E for LJ using initial values of rm and eps
LJ_v, LJ_a, LJ_B, LJ_E = parameters(rm, LJ, eps, 1)
print('Initial LJ v =', LJ_v)
print('Initial LJ a =', LJ_a)
print('Initial LJ B =', LJ_B)
print('Initial LJ E =', LJ_E)

# equilibrium v, a, B and E for Mie using initial values of rm, eps and m
Mie_v, Mie_a, Mie_B, Mie_E = parameters(rm, Mie_n6, 10, eps)
print('Initial Mie v =', Mie_v)
print('Initial Mie a =', Mie_a)
print('Initial Mie B =', Mie_B)
print('Initial Mie E =', Mie_E)

# equilibrium v, a, B and E for Morse using initial values of rm, ep and alpha
Mo_v, Mo_a, Mo_B, Mo_E = parameters(rm, Morse, alp, eps)
print('Initial Morse v =', Mo_v)
print('Initial Morse a =', Mo_a)
print('Initial Morse B =', Mo_B)
print('Initial Morse E =', Mo_E)


def optimise_LJ(rm0, eps0, r_exp, B_exp, E_exp):
    """
    Parameters
    ----------
    rm0 : initial value for rm
    eps0 : initial value for epsilon
    r_exp : experimental value for r (or a)
    B_exp : experimental value for B
    E_exp : experimental value for E
    Function optimises the parameters rm and epsilon for LJ potential
    around given experimental values for a, B and E
    """
    # calculates initial values for atomic volume, NN distance, bulk modulus and system energy
    v0, a0, B0, E0 = parameters(rm0, LJ, eps0, 1)

    def Q_calc(a, B, E):
        # optimisation function
        Q = (a / r_exp - 1)**2 + (B / B_exp - 1)**2 + (E / E_exp - 1)**2
        return Q

    n_opt = 1000
    Q = np.empty(n_opt)
    r_m = np.empty(n_opt)
    ep = np.empty(n_opt)
    a = np.empty(n_opt)
    B = np.empty(n_opt)
    E = np.empty(n_opt)

    Q[0] = Q_calc(a0, B0, E0)
    r_m[0] = rm0
    ep[0] = eps0
    a[0] = a0
    B[0] = B0
    E[0] = E0
    Q_start = Q[0]

    changes = np.array([0.01, 0.001, 0.0001])

    for h in range(2):
        for i in range(1, n_opt):
            r_m[i] = r_m[i - 1]
            ep[i] = ep[i - 1]
            a[i] = a[i - 1]
            B[i] = B[i - 1]
            E[i] = E[i - 1]

            var = np.array([r_m[i], ep[i]])
            ran = rand.randint(0, 1)
            maxchange = var[ran] * changes[h]
            change = (random() - 0.5) * maxchange
            var[ran] += change

            r_m_c = var[0]
            ep_c = var[1]
            v_c, a_c, B_c, E_c = parameters(r_m_c, LJ, ep_c, 1)
            Q[i] = Q_calc(a_c, B_c, E_c)

            if Q[i] < Q[i - 1]:
                r_m[i] = r_m_c
                ep[i] = ep_c
                a[i] = a_c
                B[i] = B_c
                E[i] = E_c
                Q_start = Q[i]

        Q[0] = Q_start
        r_m[0] = r_m[-1]
        ep[0] = ep[-1]
        a[0] = a[-1]
        B[0] = B[-1]
        E[0] = E[-1]

    return r_m[-1], ep[-1], a[-1], B[-1], E[-1]


LJ_rm, LJ_ep, LJ_a2, LJ_B2, LJ_E2 = optimise_LJ(rm, eps, r0_exp, B0_exp, E0_exp)
print('Improved LJ rm =', LJ_rm)
print('Improved LJ eps =', LJ_ep)
print('Improved LJ a =', LJ_a2)
print('Improved LJ B =', LJ_B2)
print('Improved LJ E =', LJ_E2)

def optimise_Mie(rm0, eps0, m0, r_exp, B_exp, E_exp):
    """
    Parameters
    ----------
    rm0 : initial value for rm
    eps0 : initial value for epsilon
    m0 : initial value for m
    r_exp : experimental value for r (or a)
    B_exp : experimental value for B
    E_exp : experimental value for E
    Same optimisation function but now for Mie potential
    """
    v0, a0, B0, E0 = parameters(rm0, Mie_n6, m0, eps0)

    def Q_calc(a, B, E):
        Q = (a/r_exp - 1)**2 + (B/B_exp - 1)**2 + (E/E_exp - 1)**2
        return Q

    n_opt = 1000
    Q = np.empty(n_opt)
    r_m = np.empty(n_opt)
    ep = np.empty(n_opt)
    m = np.empty(n_opt)
    a = np.empty(n_opt)
    B = np.empty(n_opt)
    E = np.empty(n_opt)
    Q[0] = Q_calc(a0, B0, E0)
    r_m[0] = rm0
    ep[0] = eps0
    m[0] = m0
    a[0] = a0
    B[0] = B0
    E[0] = E0
    Q_start = Q[0]
    changes = np.array([0.01, 0.001, 0.0001])

    for h in range(2):
        for i in range(1, n_opt):
            r_m[i] = r_m[i-1]
            ep[i] = ep[i-1]
            m[i] = m[i-1]
            a[i] = a[i-1]
            B[i] = B[i-1]
            E[i] = E[i-1]

            var = np.array([r_m[i], ep[i], m[i]])
            ran = rand.randint(0, 2)
            maxchange = var[ran] * changes[h]
            change = (random() - 0.5) * maxchange
            var[ran] += change

            r_m_c = var[0]
            ep_c = var[1]
            m_c = var[2]

            v_c, a_c, B_c, E_c = parameters(r_m_c, Mie_n6, m_c, ep_c)
            Q[i] = Q_calc(a_c, B_c, E_c)

            if Q[i] < Q[i-1]:
                r_m[i] = r_m_c
                ep[i] = ep_c
                m[i] = m_c
                a[i] = a_c
                B[i] = B_c
                E[i] = E_c
                Q_start = Q[i]

        Q[0] = Q_start
        r_m[0] = r_m[-1]
        ep[0] = ep[-1]
        m[0] = m[-1]
        a[0] = a[-1]
        B[0] = B[-1]
        E[0] = E[-1]

    return r_m[-1], ep[-1], m[-1], a[-1], B[-1], E[-1]


Mie_rm, Mie_ep, Mie_m, Mie_a2, Mie_B2, Mie_E2 = optimise_Mie(
    rm, eps, 10, r0_exp, B0_exp, E0_exp
)
print('Improved Mie rm =', Mie_rm)
print('Improved Mie eps =', Mie_ep)
print('Improved Mie m =', Mie_m)
print('Improved Mie a =', Mie_a2)
print('Improved Mie B =', Mie_B2)
print('Improved Mie E =', Mie_E2)


def optimise_Morse(rm0, eps0, alp0, r_exp, B_exp, E_exp):
    """
    Parameters
    ----------
    rm0 : initial value for rm
    eps0 : initial value for epsilon
    alp0 : initial value for alpha
    r_exp : experimental value for r (or a)
    B_exp : experimental value for B
    E_exp : experimental value for E
    Same optimisation function but now for Morse potential
    """
    v0, a0, B0, E0 = parameters(rm0, Morse, alp0, eps0)

    def Q_calc(a, B, E):
        Q = (a/r_exp - 1)**2 + (B/B_exp - 1)**2 + (E/E_exp - 1)**2
        return Q

    n_opt = 1000
    Q = np.empty(n_opt)
    r_m = np.empty(n_opt)
    ep = np.empty(n_opt)
    al = np.empty(n_opt)
    a = np.empty(n_opt)
    B = np.empty(n_opt)
    E = np.empty(n_opt)
    Q[0] = Q_calc(a0, B0, E0)
    r_m[0] = rm0
    ep[0] = eps0
    al[0] = alp0
    a[0] = a0
    B[0] = B0
    E[0] = E0
    Q_start = Q[0]
    changes = np.array([0.01, 0.001, 0.0001])

    for h in range(2):
        for i in range(1, n_opt):
            r_m[i] = r_m[i-1]
            ep[i] = ep[i-1]
            al[i] = al[i-1]
            a[i] = a[i-1]
            B[i] = B[i-1]
            E[i] = E[i-1]

            var = np.array([r_m[i], ep[i], al[i]])
            ran = rand.randint(0, 2)
            maxchange = var[ran] * changes[h]
            change = (random() - 0.5) * maxchange
            var[ran] += change

            r_m_c = var[0]
            ep_c = var[1]
            al_c = var[2]

            v_c, a_c, B_c, E_c = parameters(r_m_c, Morse, al_c, ep_c)
            Q[i] = Q_calc(a_c, B_c, E_c)

            if Q[i] < Q[i-1]:
                r_m[i] = r_m_c
                ep[i] = ep_c
                al[i] = al_c
                a[i] = a_c
                B[i] = B_c
                E[i] = E_c
                Q_start = Q[i]

        Q[0] = Q_start
        r_m[0] = r_m[-1]
        ep[0] = ep[-1]
        al[0] = al[-1]
        a[0] = a[-1]
        B[0] = B[-1]
        E[0] = E[-1]

    return r_m[-1], ep[-1], al[-1], a[-1], B[-1], E[-1]


Mo_rm, Mo_ep, Mo_al, Mo_a2, Mo_B2, Mo_E2 = optimise_Morse(
    rm, eps, alp, r0_exp, B0_exp, E0_exp
)
print('Improved Morse rm =', Mo_rm)
print('Improved Morse eps =', Mo_ep)
print('Improved Morsee alpha =', Mo_al)
print('Improved Morse a =', Mo_a2)
print('Improved Morse B =', Mo_B2)
print('Improved Morse E =', Mo_E2)


"""
With equilibrium parameters for each potential calculated and optimised,
the below code will plot them for comparison
"""
cnt = 100  # number of datapoints
n_shells = 17  # number of shells included
# number of atoms per shell, relative shell distance in units of
# nearest neighbour distance
shell_at, shell_r = neighbours()
# vectors to store the nearest neighbour distances and the energies
# corresponding to them for each potential
E_LJ = np.zeros(cnt)
E_Mie = np.zeros(cnt)
E_Mo = np.zeros(cnt)
r = np.zeros(cnt)

for i in range(cnt):
    # vector of all the nearest neighbour distances from rm
    r[i] = rm * (i + cnt) * 4 / (5 * cnt)
    for j in range(n_shells):
        # atom distance from rm is the relative shell distance
        # multiplied by the nearest neighbour distance from rm
        dist = r[i] * shell_r[j]
        # potential energies for an atom at that distance
        pot_LJ = LJ(dist, LJ_rm, LJ_ep, 1)
        pot_Mie = Mie_n6(dist, Mie_rm, Mie_m, Mie_ep)
        pot_Mo = Morse(dist, Mo_rm, Mo_al, Mo_ep)
        # Energy for each nearest neighbour distance is the potential for
        # each atom in the shells divided by 2, so interactions aren't
        # counted twice
        E_LJ[i] += pot_LJ * shell_at[j] / 2
        E_Mie[i] += pot_Mie * shell_at[j] / 2
        E_Mo[i] += pot_Mo * shell_at[j] / 2

# volume per atom
v = (r * np.sqrt(2)) ** (3 / 4)
plt.figure()
plt.plot(r, E_LJ, label='LJ')
plt.plot(r, E_Mie, label='Mie')
plt.plot(r, E_Mo, label='Morse')
plt.xlabel('Lattice parameter (Angstroms)')
plt.ylabel('System energy (eV/atom)')
plt.legend()

plt.figure()
plt.plot(v, E_LJ, label='LJ')
plt.plot(v, E_Mie, label='Mie')
plt.plot(v, E_Mo, label='Morse')
plt.xlabel('Volume per atom (Angstroms^3)')
plt.ylabel('System energy (eV/atom)')
plt.legend()

plt.show()
