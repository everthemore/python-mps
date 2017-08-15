#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy as sp, scipy.linalg
import matplotlib.pyplot as plt
import sys

# Import pure state MPS
import iMPS
import iTEBD

# Set system size
L = 50
# Set local Hilbert space dimension (i.e. how many particles per site = d - 1)
d = 3
# Create a new MPS
qwMPS = iMPS.iMPS(L, d, 10)

# Create an array of zeros...
sites = np.zeros(L)
# ...and set the middle element to 1
sites[int(L/2)] = 1

# Use this array as a configuration of how many particles are on each site
qwMPS.set_product_state(sites)

# Define operators
b    = np.diag(np.sqrt(range(d-1)+np.ones(d-1)),1)  # Annihilation
bdag = np.diag(np.sqrt(range(d-1)+np.ones(d-1)),-1) # Creation
dens = np.dot(bdag, b)                              # Density operator

# Construct the Hamiltonian
# Only hopping
t = 1
H = -t*(np.kron( bdag, b) + np.kron( b, bdag ))

# Create the propagation operator and reshape it for the MPS
delta = 0.01
U = np.reshape( sp.linalg.expm2(-1j*delta*H), (d,d,d,d) )

densityplot = np.zeros((50, L))
for n in xrange(500):
    for i in xrange(0,L-1,2):
        iTEBD.update_bond(qwMPS, U, i)
    for i in xrange(1,L,2):
        iTEBD.update_bond(qwMPS, U, i)

    # Every 10 updates we compute the density distribution and update the drawing
    if n % 10 == 0:
        occ = [np.real(qwMPS.measure([dens], [s])) for s in range(L)]
        densityplot[n/10] = occ
#        print( "Total density: ", np.sum(occ) )
#        print( "Bond dim: ", qwMPS.Chi )

# Create a new plot window and make sure it is visible
fig, ax = plt.subplots()
# Show density plot
ax.imshow(densityplot, origin='lower')
plt.show()
