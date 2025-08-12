# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 15:12:11 2025

@author: tom_m
"""

import numpy as np

def phi_LJ(dist, eps, sig):
    """
    Returns the Lennard-Jones (12-6) pair potential interaction
    energy as a function of the distance (dist) between the atoms
    eps, sigma are set to one
    """
    phi = 4 * eps * ((sig / dist)**12 - (sig / dist)**6)
    return phi

def LJ(dist, r_m, eps, one):
    """
    Lennard-Jones potential
    dist : distance
    r_m : position of potential minimum
    """
    phi = eps * ((r_m / dist)**12 - 2 * (r_m / dist)**6) * one
    return phi

def Mie_n6(dist, rm, m, eps):
    """
    Mie potential
    dist : distance
    rm : position of potential minimum
    m : parameter 1
    eps : depth of potential well
    """
    A = ((m / 6)**(m / (6 - m))) * (rm / dist)**m
    B = ((m / 6)**(6 / (6 - m))) * (rm / dist)**6
    phi = (eps / (m - 6)) * ((m**m) / (6**6))**(1 / (m - 6)) * (A - B)
    return phi

def Morse(dist, rm, a, eps):
    """
    dist : distance
    r_m : position of potential minimum
    a : alpha, governs steepness of repulsive wall
    eps : depth of potential well
    """
    phi = eps * (np.exp(-2 * a * (dist - rm)) - 2 * np.exp(-a * (dist - rm)))
    return phi