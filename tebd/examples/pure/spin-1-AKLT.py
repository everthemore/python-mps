#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy as sp
import sys

# Import pure state MPS
import iMPS
import iTEBD

# Make spin matrices
Sx = 1/np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.complex128)
Sy = 1/np.sqrt(2)*np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.complex128)

L = 2
d = 3
D = 50
testMPS = iMPS.iMPS(L, d, D)
testMPS.set_random_state()

# Heisenberg model for spin-1
Heisenberg = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz,Sz)
H = Heisenberg + 1/3*np.dot(Heisenberg, Heisenberg)

# Imaginary time propagator
U = np.reshape( sp.linalg.expm(-0.01*H), (d,d,d,d) )
H = np.reshape( H, (d,d,d,d) )

# Get the groundstate
for n in xrange(10000):
    # Update both bonds
    for i in xrange(L):
        iTEBD.update_bond(testMPS, U, i)

    # Print energy
    if n % 100 == 0:
        print (np.real( testMPS.measure( [H], [0] ) ) + np.real( testMPS.measure( [H], [1] ) ))/2

# Quench to a different Hamiltonian
H = Heisenberg

# Imaginary time propagator
dt = 0.001
# Define two separate time-evolution operators, one for half-time steps and one for full
U = np.reshape( sp.linalg.expm(-dt*H*1j), (d,d,d,d) )
U2 = np.reshape( sp.linalg.expm(-dt*H*1j/2), (d,d,d,d) )
H = np.reshape( H, (d,d,d,d) )

# Perform 5000 time steps
energy = []
ESvst  = []
for n in xrange(5000):
    # Second order trotter update: half a step on even bonds, full on odd, then half on even again

    # Update both bonds
    for i in xrange(0,L,2):
        iTEBD.update_bond(testMPS, U2, i)
    # Update both bonds
    for i in xrange(1,L,2):
        iTEBD.update_bond(testMPS, U, i)
    # Update both bonds
    for i in xrange(0,L,2):
        iTEBD.update_bond(testMPS, U2, i)

    if n % 100 == 0:
        energy.append( (np.real( testMPS.measure( [H], [0] ) ) + np.real( testMPS.measure( [H], [1] ) ))/2 )
        print energy[-1]
        ESvst.append( -2*np.log(testMPS.Lambda[0]) )

np.save("resultsHeisenbergToLargeD2-BD50.npy", {'energy':np.array(energy), 'ES':np.array(ESvst)})
