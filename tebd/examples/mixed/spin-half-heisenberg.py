#!/usr/bin/env python

# Demo file for mixed state MPS
# Computes the ground state of the spin-1 heisenberg model
from __future__ import division
import numpy as np
import scipy as sp
import sys

# Import mixed state library
from iMPS import *
from iTEBD import *
from superOps import *

L = 40
testMPS = iMPS(L, 2, 10, pure=False)

np.random.seed(1990)
Sx = 0.5*np.array([[0, 1], [1, 0]])
Sy = 0.5*np.array([[0, -1j], [1j, 0]])
Sz = 0.5*np.array([[1, 0], [0, -1]])

H = np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz,Sz)

U = superOperator( sp.linalg.expm2(-0.01*H), 2 )
H = superOperator( H, 2 )

Is = superOperator( np.eye(4), 2 )

# Set infinite temperature state
testMPS.set_infinite_temperature_state()

energies = [ np.real( testMPS.measure( [H], [i] ) ) for i in range(L) ]
print energies

# For 1000 iterations, update the MPS
for n in range(3000):
    err = 0

    # Periodic boundaries
    for i in range(0,L,2):
        err += update_bond(testMPS, U, i)
    for i in range(1,L,2):
        err += update_bond(testMPS, U, i)

    energies = [ np.real( testMPS.measure( [H], [i] ) ) for i in range(L) ]
#    print energies
    print np.mean(energies)
