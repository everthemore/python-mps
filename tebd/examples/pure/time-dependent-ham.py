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

def time_evolve(MPS, H, T, dt, order=1):
    # Create the propagation operator and reshape it for the MPS
    U = np.reshape( sp.linalg.expm2(-1j*dt*H), (d,d,d,d) )

    # Build evo operator for half-time step
    U2 = np.reshape( sp.linalg.expm2(-1j*dt/2*H), (d,d,d,d) )

    t = 0
    while t <= T:
        # FIRST ORDER TROTTER
        if order == 1:
            for i in xrange(0,L-1,2):
                iTEBD.update_bond(MPS, U, i)
            for i in xrange(1,L,2):
                iTEBD.update_bond(MPS, U, i)

        elif order == 2:
            # SECOND ORDER TROTTER
            for i in xrange(0,L-1,2):
                iTEBD.update_bond(MPS, U2, i)
            for i in xrange(1,L,2):
                iTEBD.update_bond(MPS, U, i)
            for i in xrange(0,L-1,2):
                iTEBD.update_bond(MPS, U2, i)
        else:
            print "Trotter order {0} not supported!".format(order)
            sys.exit()

        t += dt

    return MPS

densities = []
occ = [np.real(qwMPS.measure(dens, s)) for s in range(L)]
densities.append(occ)

J = 1
for t in np.arange(0, 2.1, 0.1):
    H = J*(np.kron( bdag, b) + np.kron( b, bdag ))

    qwMPS = time_evolve(qwMPS, H, t, 0.01)
    occ = [np.real(qwMPS.measure(dens, s)) for s in range(L)]
    densities.append(occ)

    print "g2(15,25) = {0}".format(np.real(qwMPS.measure(np.array([dens, dens]), np.array([15, 25]))) )

# Create a new plot window and make sure it is visible
fig, ax = plt.subplots()
# Show density plot
ax.imshow(densities, extent=[0,1,0,1], origin='lower')
plt.show()
