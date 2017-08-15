#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy as sp, scipy.linalg
import matplotlib.pyplot as plt
import sys
import copy

# Import pure state MPS
import iMPS
import iTEBD
from superOps import *

#
# In this simulation, we perform time evolution of a spin-1/2 XY model with 
# external field in the presence of two baths, coupled to the ends of the system. 
# At each end, we have Lindblad operators sigma-plus and sigma-minus, so that we 
# can set temperature baths or magnetization baths. 
#
# Additionally, the Hamiltonian for the system itself is time-dependent,
# namely it has an additional square-wave driving term. The external field
# given by 'mu' is driven from mu + s1 to mu + s2 every period. The parameter
# alpha below sets the point within one period at which the field switches.
#

# Set system size
L = 4
# Set local Hilbert space dimension (i.e. how many particles per site = d - 1)
d = 2
# Maximum bond dimension
D = 120

#parameters
gamma   = 1
delta   = float(sys.argv[1])
fixmu   = float(sys.argv[2])
period  = float(sys.argv[3])
alpha   = float(sys.argv[4])
s1     = 0
s2      = float(sys.argv[5])

# Spin coupling
Jx       = (gamma - delta)/2
Jy       = (gamma + delta)/2

# Define operators
s0   = np.array([[1,0],[0,1]], dtype=np.complex128)
sx   = np.array([[0,1],[1,0]], dtype=np.complex128)
sy   = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
sz   = np.array([[1,0],[0,-1]], dtype=np.complex128)
smin = np.array([[0,0],[1,0]], dtype=np.complex128)
splus= np.array([[0,1],[0,0]], dtype=np.complex128)

# Define separate superoperators for the spins, so that we may use them
# in measurements
ssx   = superOperator(sx,d)
ssy   = superOperator(sy,d)
ssz   = superOperator(sz,d)

# Drive
gammaLplus = 1 #0.3
gammaLminus = 0 #0.5
gammaRplus = 0 #0.1
gammaRminus = 1 #0.5

# The initial state is a simple product state
MPS = iMPS.iMPS(4, 2, D, pure=False)
MPS.set_product_state([1,0]*2)

# Function to compute H and the time evolution operator U for the various values of parameters
def compute_H_and_U( Jx, Jy, mu, dt ):
    """Compute Hamiltonian and time evolution operators"""

    u  = np.zeros( (L, d**2, d**2, d**2, d**2), dtype=np.complex128 )
    u2 = np.zeros( (L, d**2, d**2, d**2, d**2), dtype=np.complex128 )
    for site in range(L):
        # Define Hamiltonian
        h = -Jx*np.kron( sx, sx ) - Jy*np.kron( sy, sy ) - mu*0.5*( np.kron(sz, s0) + np.kron( s0, sz ) )

        if site == 0:
            # Construct superoperator
            h += -mu*0.5*np.kron(sz, s0)
            L1 = superOperator([h, np.sqrt(gammaLplus)*np.kron( splus, s0 ), np.sqrt(gammaLminus)*np.kron( smin, s0 )], d, func=lindblad)
        elif site == L-2:
            h += -mu*0.5*np.kron(s0, sz)
            L1 = superOperator([h, np.sqrt(gammaRplus)*np.kron( s0, splus ), np.sqrt(gammaRminus)*np.kron( s0, smin )], d, func=lindblad)
        else:
            L1 = superOperator([h], d, func=lindblad)

        u[site] = np.reshape( sp.linalg.expm2( dt*np.reshape( L1, (d**2*d**2, d**2*d**2) ) ), (d**2,d**2,d**2,d**2) )
        u2[site] = np.reshape( sp.linalg.expm2( 0.5*dt*np.reshape( L1, (d**2*d**2, d**2*d**2) ) ), (d**2,d**2,d**2,d**2) )

    # Reshape h 
    H = superOperator( h, d )

    return h, H, u, u2

# Measure the spin current in the current state
def measure_spin_current(MPS):
    Sm = []
    for i in range(L-1):
        Sm.append( np.real( MPS.measure( [ssx,ssy], [i,i+1] ) - MPS.measure( [ssy,ssx], [i,i+1] ) ) )
    return Sm

# We compute two sets of Hamiltonians, one for lower half of square pulse and one for upper
dt = 0.01
hlower, Hlower, Ulower, U2lower  = compute_H_and_U( Jx, Jy, fixmu + s1, dt )
hupper, Hupper, Uupper, U2upper  = compute_H_and_U( Jx, Jy, fixmu + s2, dt )

# We want open boundary conditions, so depending on L being even or odd
# the last bond is even or odd.
l_end_even = L if L % 2 == 0 else L - 1
l_end_odd  = L if L % 2 == 1 else L - 1

# New lists for storing the evolution times and mu
Lambdas = []
times   = []
mus     = []
energies= []
spincurrents = []
heatcurrents = []
spinprofile  = []

How_many_periods = 2000

outputcounter = 0
for t in np.arange(0, How_many_periods*period + dt, dt):
    try:
        # Store runtimes
        times.append(t)

        if t % period >= alpha * period:
            # Set current hamiltonian and U to be the one with kick
            h, H, U, U2 = hupper, Hupper, Uupper, U2upper
            # Add mu to list
            mus.append( fixmu + s2 )

        else:
            # Update without kick
            h, H, U, U2 = hlower, Hlower, Ulower, U2lower
            # Add mu to list
            mus.append( fixmu + s1 )

        # Store the entanglement spectrum
        tmp = np.zeros(D)
        tmp[:len(MPS.Lambda[int(L/2)])] = MPS.Lambda[int(L/2)]
        Lambdas.append( tmp )
   
        # Measure the average energy of the instantaneous Hamiltonian
#        energies_for_bonds = [ np.real(MPS.measure( [H], [i] )) for i in range(L-1) ]
#        mean_energy = np.mean( energies_for_bonds )
#        energies.append( mean_energy )

#        S = measure_spin_current(MPS)
#        spincurrents.append( S )

#        currentspinprofile = [ np.real(MPS.measure( [ssz], [i] )) for i in range(L) ]
#        spinprofile.append( currentspinprofile )

        outputcounter += 1
        if outputcounter == 1:
            currentspinprofile = [ np.real(MPS.measure( [ssz], [i] )) for i in range(L) ]
            print currentspinprofile
            outputcounter = 0

#        if t % 20 == 0:
#            data = {'times':times, 'energies':energies, 'heatcurrents':heatcurrents, 'spincurrents':spincurrents, 'entanglement_spectrum':Lambdas, 'mu':mus, 'spinprofile':spinprofile}
            #np.save("./data/raw/progress/L-{0}-D-{1}-delta-{2}-mu-{3}/open-period-{4}-alpha-{5}-s2-{6}.npy".format(L,D,delta,fixmu,period,alpha,s2), data )

        # SECOND ORDER TROTTER
        for i in xrange(0, l_end_even, 2):  # 0 2 
            iTEBD.update_bond(MPS, U2[i], i)
        for i in xrange(1, l_end_odd, 2):   # 1 
            iTEBD.update_bond(MPS, U[i], i)
        for i in xrange(0, l_end_even, 2):  # 0 2
            iTEBD.update_bond(MPS, U2[i], i)

    except Exception as e:
        print "Exception caught in evolving: ", e
        print "It happened at t = {0}".format(t)
        exit(-1)

# Save data to file
data = {'times':times, 'energies':energies, 'heatcurrents':heatcurrents, 'spincurrents':spincurrents, 'entanglement_spectrum':Lambdas, 'mu':mus, 'spinprofile':spinprofile}
np.save("./data/raw/L-{0}-D-{1}-delta-{2}-mu-{3}/open-period-{4}-alpha-{5}-s2-{6}.npy".format(L,D,delta,fixmu,period,alpha,s2), data )
