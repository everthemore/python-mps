#!/usr/bin/env python
# Demo file for pure state MPS
# Compute groundstate of spin-1/2 heisenberg chain

from __future__ import division
import numpy as np
import scipy as sp
import sys
# Import pure state MPS
import iMPS
import iTEBD

# Define the Pauli spin matrices
Sx = 1/2*np.array([[0, 1], [1, 0]], dtype=np.complex128)
Sy = 1/2*np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Sz = 1/2*np.array([[1, 0], [0,-1]], dtype=np.complex128)

# Create a new MPS instance. 
# Set system size
L = 20
# The local Hilbert space is that of a spin-1/2, so 2-dimensional
d = 2
# The bond-dimension is the most important parameter of the MPS, it sets the 'complexity'
# of the wavefunction. The larger the better, but at a price of MUCH slower numerics.
# Best is to start with a small value (~10) and slowly increase. If the final result
# of the numerics no longer changes drastically with increasing D, we are safe.
D = 30
testMPS = iMPS.iMPS(L, d, D)
# We will be performing imaginary time evolution, so we want to have some overlap
#  with the actual groundstate. Using a random initial state, we will most likely
#  have such overlap. 
# !! For L != 2, the code does not know how to bring the state into canonical form
# automatically. The evolution to the groundstate takes care of this.
testMPS.set_random_state(ortho=False)


# Heisenberg model for spin-1/2
#  Since we are considering the infinite system, we consider only a single
#  type of bond. Hence all we need to define is the Hamiltonian for this bond.
#  For illustrative purposes however, we will explicitly construct it per bond.
H = []
U = []

# We include the periodic bond
for site in range(0,L):
    # This is the bond Hamiltonian, notice that it can depend on 'site' if we want,
    # to add e.g. disorder
    Hforthisbond = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz,Sz)


    # Imaginary time propagator
    #  The trotter step-size dt is an important parameter. If it is too large,
    #   we will not converge to the groundstate (and/or build up large errors).
    dt = 0.01

    # Operators that act on bonds of the MPS need to have the shape (d,d,d,d)
    Uforthisbond = np.reshape( sp.linalg.expm( -dt*Hforthisbond ), (d,d,d,d) )
    U.append(Uforthisbond)

    # Add it to the list
    H.append( np.reshape(Hforthisbond, (d,d,d,d)) )

# All that is left is to actually perform the imaginary time evolution
# For the purpose of this demo, we simply perform 5000 steps. 
#  One may try to automatize this more, for example by keeping track of the 
#  variance of the energy, which should be below some threshold when the 
#  state has converged to the groundstate.
number_of_steps = 5000
for n in xrange(number_of_steps):

    # Update both bonds (notice that this is a first order trotter step, and
    #  accuracy may improve drastically when using a second order)
    
    # First the even bonds
    for i in xrange(0,L,2):
        iTEBD.update_bond(testMPS, U[i], i)
    # Then the odd bonds
    # !!!!
    # !! Notice that we go up to L-1 here, i.e. we do NOT include the periodic bond.
    # !! Also, notice that the periodic bond would have been EVEN if L was ODD
    # !!!!
    for i in xrange(1,L-1,2):
        iTEBD.update_bond(testMPS, U[i], i)

    # Measure and output the energy
    print np.mean( [np.real( testMPS.measure( [H[i]], [i] ) ) for i in range(L)] )

# Save the MPS to a file
testMPS.saveToFile("finite-spin-half-heisenberg-GS.mps")
