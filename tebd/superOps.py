################################################################################
# superOps.py: Construct superoperators
################################################################################
"""superOps.py: Main module for performing pure or mixed iTEBD simulations."""
__author__      =   "Evert van Nieuwenburg"
__copyright__   =   "Copyright 2015, CMT-QO group ETH Zurich"
__version__     =   "1.0.3"

#--------------------------------------------------------------------------------
# Import dependencies
#--------------------------------------------------------------------------------
import numpy as np
import scipy as sp
import scipy.linalg, scipy.sparse, scipy.sparse.linalg
import pickle
import itertools

#--------------------------------------------------------------------------------
# Matrix Direct Sum
#--------------------------------------------------------------------------------
def directsum(A, B):
    C = np.zeros( (A.shape[0]+B.shape[0], A.shape[1]+B.shape[1]) )
    C[0:A.shape[0], 0:A.shape[1]] = A
    C[A.shape[0]:, A.shape[1]:]   = B
    return C

def buildBasis(d):
    # Physical basis for d = d
    basisdD = []
    basisdD.append(np.eye(d, dtype=np.complex128))
    # Compute other diagonal matrices
    for i in range(1, d):
        diag = [1 for n in range(d-i)] + [1-d+(i-1)] + [0 for n in range(i-1)]
        mat = np.zeros( (d,d), dtype = np.complex128 )
        for j in range(d):
            mat[j,j] = diag[j]
        basisdD.append( mat )

    # Add the other symmetric & anti-symmetric matrices
    for i in range(1, d):
        for j in range(0, i):
            mat = np.zeros( (d,d), dtype = np.complex128 )
            mat[i,j] = 1; mat[j,i] = 1
            basisdD.append(mat)

            mat = np.zeros( (d,d), dtype = np.complex128 )
            mat[i,j] = 1j; mat[j,i] = -1j
            basisdD.append(mat)

    return basisdD

#--------------------------------------------------------------------------------
# Define the inner product between two matrices (matrixInnerProduct)
#--------------------------------------------------------------------------------
def matrixInnerProduct(A, B):
    """Return the inner product between two matrices.

    The inner product is defined as
        <A|B> = Tr( A^dagger * B )/dim(A)
    """
    return np.trace(np.dot(A.conj().T, B))/A.shape[0]

#--------------------------------------------------------------------------------
# Orthonormalize the bases wrt the matrixInnerProduct
#--------------------------------------------------------------------------------
bases = {}
bases[2] = buildBasis(2)
bases[3] = buildBasis(3)
bases[4] = buildBasis(4)

for b in bases.keys():
    for element in bases[b]:
        element /= np.sqrt(matrixInnerProduct(element, element))

#--------------------------------------------------------------------------------
# Define default functions for superoperator behaviour
#--------------------------------------------------------------------------------
def mult(A, B):
    """Return dot product between A and B."""
    #return np.dot(A,B)
    return 0.5*(np.dot(A,B) + np.dot(B,A))

def apply(A, B):
    """ Apply operator A to B.
        Applying means A.B.A.conj().T
    """
    return np.dot(np.dot(A, B), A.conj().T)

def commutator(A, B):
    return np.dot(A, B) - np.dot(B, A)

def dissipator(Op, B):
    LdagL = np.dot(Op.conj().T, Op)
    return np.dot(np.dot(Op, B), Op.conj().T) - 0.5*( np.dot(LdagL, B) + np.dot(B, LdagL) )
    
def master(Ops,B):
    L = dissipator(Ops[0], B)
    for i in range(1, len(Ops)):
        L = L + dissipator(Ops[i], B)
    return L

def lindblad(Ops, B): #Liouvillian, Ops[0] = H, Ops[1:] = Jump ops
    L = -1j*commutator(Ops[0], B) 
    for i in range(1, len(Ops)):
        L = L + dissipator(Ops[i], B)
    return L

#--------------------------------------------------------------------------------
# Make a superoperator
#--------------------------------------------------------------------------------
def superOperator(Op, d, func = mult):
    """Make a superoperator representing [func] out of Op."""

    if isinstance(d, list) or isinstance(d, np.ndarray):
        return np.array([[[[matrixInnerProduct(np.kron(e, c), func(Op, np.kron(b,a))) for a in bases[d[1]] ] for b in bases[d[0]] ] for c in bases[d[1]]] for e in bases[d[0]]])

    # Determine which basis to use
    basis = bases[d]

    # Determine site vs bond operator
    if( type(Op) == list ):
        opsize = int(np.log(Op[0].shape[0])/np.log(d))
    else:
        opsize = int(np.log(Op.shape[0])/np.log(d))
    
    if( opsize == 1 ):
        return np.array([[matrixInnerProduct(b, func(Op, c)) for b in bases[d]] for c in bases[d]])
    if( opsize == 2 ):
        return np.array([[[[matrixInnerProduct(np.kron(e, c), func(Op, np.kron(b,a))) for a in bases[d]] for b in bases[d]] for c in bases[d]] for e in bases[d]])
    if( opsize == 3 ):
        superOp = np.zeros( (d**2,d**2,d**2,d**2,d**2,d**2) ,dtype=np.complex128)
        for conf in itertools.product( *[range(d**2),range(d**2),range(d**2),range(d**2),range(d**2),range(d**2)] ):
            superOp[conf] = matrixInnerProduct( np.kron(np.kron(bases[d][conf[0]], bases[d][conf[1]]), bases[d][conf[2]]),
             func(Op, np.kron(np.kron(bases[d][conf[3]], bases[d][conf[4]]), bases[d][conf[5]])))
        return superOp    
