"""
file in which all the constant parameters can be found
"""

particles  = 1000           # amount of particles
dimensions = 2
dt = 0.0001
Lx, Ly = 28.9, 28.9         # box dimensions
A = Lx * Ly                 # box area

rmax = 5
dr = 0.02

r_cutoff_yp = 1.06          # r cutoff for a young and passive glass
r_cutoff_op = 0             # r cutoff for an old and passive glass
r_cutoff_ya = 0             # r cutoff for a young and active glass
r_cutoff_oa = 0             # r cutoff for an old and active glass