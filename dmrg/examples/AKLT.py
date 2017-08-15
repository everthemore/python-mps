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
d = 3

# Define spin-1 matrices
S0 = np.array( [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.complex128 )
Sx = 1/np.sqrt(2)*np.array( [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.complex128 )
Sy = 1/np.sqrt(2)*np.array( [[0,-1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=np.complex128 )
Sz = np.array( [[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.complex128 )

# Define Hamiltonian for AKLT model
# We define H for every bond, and for every bond we have a list of 2-body or maximally 3-body terms
H = {}
for s in xrange(L-1):
    H[s] = np.array([ [Sx, Sx], [Sy, Sy], [Sz, Sz], \
                        [1/3*np.dot(Sx,Sx),np.dot(Sx,Sx)], [1/3*np.dot(Sx,Sy),np.dot(Sx,Sy)], [1/3*np.dot(Sx,Sz),np.dot(Sx,Sz)],
                        [1/3*np.dot(Sy,Sx),np.dot(Sy,Sx)], [1/3*np.dot(Sy,Sy),np.dot(Sy,Sy)], [1/3*np.dot(Sy,Sz),np.dot(Sy,Sz)],
                        [1/3*np.dot(Sz,Sx),np.dot(Sz,Sx)], [1/3*np.dot(Sz,Sy),np.dot(Sz,Sy)], [1/3*np.dot(Sz,Sz),np.dot(Sz,Sz)]])

# Set bond dimension D and initialize and MPS and simulation controller
D = 4
mps = dmrg.mps(L,d,D)
sim = dmrg.dmrgSim(mps, H)

# Set random state for mps, or alternatively use set_product_state( list_of_initial_values )
np.random.seed(1990)
mps.set_random_state()

# Perform sweeps and store energies
E = []
for sweep in xrange(10):
    E.append( sim.full_sweep(k=1, ncv=10)[-1] )
    print E[-1]/(L-1)

print "Groundstate energy: ", E[-1]/(L-1)

# Optionally, measure other things
#spinprofile = []
#for s in xrange(L):
#        spinprofile.append( np.real(mps.measure( Sz, s )) )
#        mps.move_gauge_right(s)

