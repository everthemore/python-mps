#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import dmrg
import scipy as sp
import scipy.sparse

#-----
# Groundstate of AKLT model using DMRG
#-----
L = 20
d = 2

# Define spin-1 matrices
S0 = np.array( [[1, 0], [0, 1]], dtype=np.complex128 )
Sx = np.array( [[0, 1], [1, 0]], dtype=np.complex128 )
Sy = np.array( [[0,-1j], [1j, 0]], dtype=np.complex128 )
Sz = np.array( [[1, 0], [0, -1]], dtype=np.complex128 )

# Define Hamiltonian for AKLT model
# We define H for every bond, and for every bond we have a list of 2-body or maximally 3-body terms
J = -0.5
H = {}
for s in range(L-1):
    H[s] = np.array([ [J*Sx, Sx], [J*Sy, Sy], [J*Sz, Sz] ] )

# Set bond dimension D and initialize and MPS and simulation controller
D = 4
mps = dmrg.mps(L,d,D,threshold=1e-8)
sim = dmrg.dmrgSim(mps, H)

# Set random state for mps, or alternatively use set_product_state( list_of_initial_values )
np.random.seed(1990)
mps.set_random_state()

# Perform sweeps and store energies
E = []
for sweep in range(10):
    E.append( sim.full_sweep(k=1, ncv=10)[-1] )
    print(mps.D)
    print(E[-1]/(L-1))

print("Groundstate energy: ", E[-1]/(L-1))

# Optionally, measure other things
#spinprofile = []
#for s in xrange(L):
#        spinprofile.append( np.real(mps.measure( Sz, s )) )
#        mps.move_gauge_right(s)

