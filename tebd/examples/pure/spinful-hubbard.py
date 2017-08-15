#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy as sp
import sys
# Import pure state MPS
import iMPS
import iTEBD

# (0,0,0,1) = empty        3
# (0,0,1,0) = spin up      2
# (0,1,0,0) = spin down    1
# (1,0,0,0) = both         0

# Construct local operators
d = 4
cdagup      = np.array([[0,1,0,0], [0,0,0,0], [0,0,0,1], [0,0,0,0]])
cdagdown    = np.array([[0,1,0,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]])
cup         = cdagup.T
cdown       = cdagdown.T

densup      = np.dot( cdagup, cup )
densdown    = np.dot( cdagdown, cdown )

# Heisenberg model for spin-1/2
t = 1
H = t * ( np.kron( cdagup, cup ) + np.kron( cup, cdagup ) + np.kron( cdagdown, cdown ) + np.kron( cdown, cdagdown ) ) + np.kron( np.dot( cdagup, cdagup) , identity )

# Imaginary time propagator
dt = 0.01
U = np.reshape( sp.linalg.expm(-dt*1j*H), (d,d,d,d) )
H = np.reshape( H, (d,d,d,d) )

L = 20
testMPS = iMPS.iMPS(L, d, 10)

sites = 3*np.ones( L )
sites[4] = 2
sites[16]  = 1
testMPS.set_product_state(sites)

# Perform 5000 time steps
total_dens_up = []
total_dens_down = []
updowncorr = []

for n in xrange(500):

    # Update both bonds
    for i in xrange(0,L-1,2):
        iTEBD.update_bond(testMPS, U, i)
    for i in xrange(1,L-1,2):
        iTEBD.update_bond(testMPS, U, i)

    # Print energy
    total_dens_up.append( [ np.real(testMPS.measure( densup, i )) for i in range(L) ] )
    total_dens_down.append( [ np.real(testMPS.measure( densdown, i )) for i in range(L) ] )

    updowncorr.append( [ np.real(testMPS.measure( [densup, densdown], [4, i] )) for i in range(4,16) ] )

total_dens_up = np.array(total_dens_up)
total_dens_down = np.array(total_dens_down)
updowncorr = np.array(updowncorr)

np.savetxt("densup", total_dens_up)
np.savetxt("densdown", total_dens_down)
np.savetxt("dens", total_dens_up + total_dens_down)
np.savetxt("updowncorr", updowncorr)
