# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 00:53:46 2025

@author: tom_m
"""

from __future__ import division
import numpy as np
from random import randrange, random
import matplotlib.pyplot as plt
from numba import jit

#vTask: To find the path of quickest descent from point A[1,0] to B[0,1], 
# i.e. the Brachistochrone curve

@jit
def time(x, y):
    # Returns the time it would take for a ball to travel down the total
    # length of the trajectory
    t = 0
    v = 0
    g = 1
    n = len(x)-1
    for i in range(1, n):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        v_f = np.sqrt(v**2 - 2*g*dy)
        v_avg = (v + v_f)/2
        dist = np.sqrt(dx**2 + dy**2)
        dt = dist / v_avg
        v = v_f
        t = t + dt
    return t

def brachistochrone(A, B, N):
    n_opt = 10000 # Number of optimisation trials per segment
    n_trials = N * n_opt # Total number of optimisation trials run
    times = np.empty(n_trials, float) # Array to store the times for each
    # trajectory
    x = np.linspace(A[0], B[0], N) # x-values for the segment boundaries
    y_line = np.linspace(A1[1], B1[1], N) # y-values if a straight line were drawn between A and B
    y = np.zeros(len(x), float) # Array to store the calculated y-values
    t_line = time(x, y_line) # Time taken for ball to travel down a
    # straight line from A to B
    if A[1] == B[0]:
        for j in range(0, N):
            y[j] = -np.sqrt((A[1]-A[0])**2 - (x[j] - A[1])**2) + B[0]
            # Sets the condition that the initial shape of the curve joining
            # A and B can be the quarter sector of a circle if A[y]=B[x]
    else:
        y = y_line
        # Otherwise the initial guess is simply a straight line, for ease
    x_start = np.copy(x) # Copy of x and y to store as initial guess
    y_start = np.copy(y)
    t_start = time(x_start, y_start)
    times[0] = t_start # First element in times array is time for initial
    # shape guess
    for k in range(1, n_trials):
        ran = randrange(1, N-2) # Selects a random number which will be
        # the index of the y-value to change. Location of A and B are
        # constant so index must be within the inner coordinates
        if y[ran] != 0:
            maxchange = y[ran] * 0.01
            # If the y-value is non-zero then the max it can change from
            # its current position is 1% of its currrent value
        else:
            maxchange = random() * 0.01
            # If it is zero, then it can change up to 1% of a random
            # number between 0 and 1
        changey = (random()-0.5)*maxchange
        # Amount that y-value changes is between -0.5 and 0.5 multiplied
        # by the maximum amount its allowed to change
        yt = np.copy(y) # Stores copy of the new y-values
        yt[ran] = yt[ran] + changey # Adds that change to the random yvalue
        times[k] = time(x, yt) # Calculates time for new coordinates
        if times[k] < times[k-1]:
            y = yt
            t_start = times[k]
            # If the current time is less than the time at the previous
            # iteration, then the current set of y-values is made the new set
        
    print('t for straight line:', t_line, 'Brachistochrone t: ', t_start,
    't for initial guess', times[0])
    plt.plot(x, y, label = 'Brachistochrone')
    plt.plot(x_start, y_start, label = 'Initial guess')
    plt.plot((A[0],B[0]), (A[1],B[1]), '--')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.legend()
    plt.show()
    plt.figure()
    x_axis = np.linspace(1, n_trials, n_trials)
    plt.plot(x_axis, times)
    plt.ylabel('Time')
    plt.xlabel('Iteration')


A1 = [0, 1]
B1 = [1, 0]
N = 100

brachistochrone(A1, B1, N)