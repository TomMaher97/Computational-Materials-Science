# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 15:12:37 2025

@author: tom_m
"""
import numpy as np

def neighbours():
    nshells = 17
    shell_at = np.empty(nshells)
    shell_r = np.empty(nshells)
    shell_at[0] = 12
    shell_r[0] = 1
    shell_at[1] = 6
    shell_r[1] = np.sqrt(2)
    shell_at[2] = 24
    shell_r[2] = np.sqrt(3)
    shell_at[3] = 12
    shell_r[3] = 2
    shell_at[4] = 24
    shell_r[4] = np.sqrt(5)
    shell_at[5] = 8
    shell_r[5] = np.sqrt(6)
    shell_at[6] = 48
    shell_r[6] = np.sqrt(7)
    shell_at[7] = 6
    shell_r[7] = np.sqrt(8)
    shell_at[8] = 24
    shell_r[8] = 3
    shell_at[9] = 12
    shell_r[9] = 3
    shell_at[10] = 24
    shell_r[10] = np.sqrt(10)
    shell_at[11] = 24
    shell_r[11] = np.sqrt(11)
    shell_at[12] = 24
    shell_r[12] = np.sqrt(12)
    shell_at[13] = 48
    shell_r[13] = np.sqrt(13)
    shell_at[14] = 24
    shell_r[14] = np.sqrt(13)
    shell_at[15] = 48
    shell_r[15] = np.sqrt(15)
    shell_at[16] = 24
    shell_r[16] = 4
    return shell_at, shell_r