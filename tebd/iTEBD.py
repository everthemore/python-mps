################################################################################
# TEBD_core.py: Main module for performing pure or mixed state iTEBD simulations   
################################################################################
from __future__ import division
""" 
Main module for performing pure or mixed (i)TEBD simulations. 
This file contains the 'update' routines for (i)TEBD, both with and without
the 'Hastings' improvement. It always assumes a right-canonical MPS.

"""
__author__      =   "Evert van Nieuwenburg"
__copyright__   =   "Copyright 2015, CMT-QO group ETH Zurich"
__version__     =   "1.0.1"

#--------------------------------------------------------------------------------
# Import dependencies
#--------------------------------------------------------------------------------
import numpy as np
import scipy as sp
import scipy.linalg, scipy.sparse, scipy.sparse.linalg
import pickle
import itertools
import svd_dgesvd, svd_zgesvd

#--------------------------------------------------------------------------------
# Local SVD routine for stability
#--------------------------------------------------------------------------------
def svd(A, chi = 0, full_matrices = False, compute_uv = True):
    """
    Perform the SVD of matrix A.

    Performs the SVD of a matrix A, and returns U, S, Vh just as the scipy version does.
    First and foremost this is a wrapper for the scipy.linalg SVD function, but if it
    fails to compute, the function resorts to a manual SVD.

    Parameters:
    A (matrix):  The matrix to perform the SVD on.
    full_matrices (bool): If False, truncates the matrices to match the number of singular values.
    compute_uv (bool): If True, computes the U and V.H matrices.

    Returns:
    U (matrix): Left orthonormal singular matrix
    S (array):  Array of singular values
    V.H (matrix): Hermitian conjugate of right singular matrix

    TODO: the sparse svd is much faster and much more memory efficient. But it needs
    an explicit number of how many values/vectors to compute. Can we set it to
    the 'chi' first values? What if the user doesn't want to limit chi but
    limit the error?
    """
    # Try using the normal svd
    try:
        #if A.dtype != np.complex128:
        #    return svd_dgesvd.svd_dgesvd(A, full_matrices = full_matrices, compute_uv = compute_uv)
        #else:
        #    return svd_zgesvd.svd_zgesvd(A, full_matrices = full_matrices, compute_uv = compute_uv)
        return sp.linalg.svd(A, compute_uv = compute_uv, full_matrices = full_matrices, overwrite_a = True)

    # Do manual if it failed
    except Exception as e:
        print("Canonical SVD failed: ", e )

        # Try making it square
        try:
            print("Trying to SVD the square version")
            shape = A.shape
            dim = np.max( A.shape )
            squareA = np.zeros( (dim, dim), dtype=np.complex128 )
            squareA[:A.shape[0], :A.shape[1]] = A
            return sp.linalg.svd(squareA[:shape[0], :shape[1]])

        except:
            # Try sparse
            try:
                print("\t \t Resorting to MANUAL SVD", "red")

                # Compute AA\dagger
                AAt = np.dot(A, np.conjugate(np.transpose(A)))
                # Make sure symmetric/hermitian
                AAt = 0.5*(AAt + np.conjugate(np.transpose(AAt)))

                # Diagonalize and sort
                S1,U = np.linalg.eigh(AAt)
                idx = S1.argsort()[::-1]  # Sort descending
                S1 = np.sqrt(np.abs(S1[idx])); U = U[:,idx]

                # Compute A\daggerA
                AtA = np.dot(np.conjugate(np.transpose(A)), A)
                # Make sure symmetric/hermitian
                AtA = 0.5*(AtA + np.conjugate(np.transpose(AtA)))

                # Diagonalize and sort
                S2,V = np.linalg.eigh(AtA)
                idx = S2.argsort()[::-1]  # Sort descending
                S2 = np.sqrt(np.abs(S2[idx])); V = V[:,idx]

                # Return SVD
                return U, 0.5*(S1+S2), np.conjugate(np.transpose(V))

            except:
                print("\t Trying sparse", "yellow")
                if chi == 0:
                    chi = int(A.shape[0]/2)
                U, S, V = sp.sparse.linalg.svds(squareA, k=chi)#A.shape[0])
                S = np.real(S); idx = S.argsort()[::-1]  # Sort descending
                return U[:,idx], S, V[idx,:]


#---------------------------------------------------------------------------
# TEBD update without Hastings improvement
#---------------------------------------------------------------------------
def update_bond_inverse(MPS, Op, bond, tol = 1e-10):
    """
    Apply an operator to the MPS at given bond and update it. 
    This version does NOT use the Hastings improvement, and so has to 
    compute inverse lambda matrices.

    Parameters:
        MPS:    MPS to be acted upon.
        Op:     Operator to act on the specified bond. Is assumed to be in 
                the form U.shape = (d1,d2,d1,d2), where d1 and d2 are the
                local Hilbert dimensions of the sites connected by the
                bond.
        bond:   Specifies the bond number of the MPS to act on.        

    Returns:
        err (float): The truncation error (want this to be 0). If this is closer
                     to one, you may want to increase the bond dimension!

    """
    # Convenience
    left    = (bond-1)%MPS.L
    this    =   (bond)%MPS.L
    right   = (bond+1)%MPS.L

    theta = MPS.getTheta(bond) #(c d d c)
    theta = np.tensordot( theta, Op, axes=([1,2], [2,3]) ) # (cddc)(dddd) -> (ccdd)
    theta = np.transpose( theta, (2, 0, 3, 1) ) #(d c d c)
    theta = np.reshape(theta, ( MPS.d[this]*MPS.Chi[left], MPS.d[right]*MPS.Chi[right] ) )
 
    # Perform SVD
    U, S, V = svd(theta, full_matrices=0); V = V.T
    
    # Truncate
    MPS.Chi[this]                    = np.max([np.min([np.sum( S > tol ), MPS.D[this]]),1])
    norm                             = np.linalg.norm( S[:MPS.Chi[this]], ord = 2 )
    err                              = np.linalg.norm( S[MPS.Chi[this]:], ord = 2 )
    MPS.Lambda[this][:MPS.Chi[this]] = S[:MPS.Chi[this]]/norm

    # U will be the new left B, V the new Bright
    U = np.reshape( U[:MPS.d[this]*MPS.Chi[left], :MPS.Chi[this]], (MPS.d[this], MPS.Chi[left], MPS.Chi[this]))
    V = np.transpose( np.reshape( V[:MPS.d[right]*MPS.Chi[right], :MPS.Chi[this]], (MPS.d[right],MPS.Chi[right],MPS.Chi[this])),(0,2,1))

    MPS.B[this] = np.transpose(np.tensordot(np.diag(MPS.Lambda[left][:MPS.Chi[left]]**(-1)), U, axes=(1,1)), (1,0,2))
    MPS.B[this] = np.dot(MPS.B[this][:, :MPS.Chi[left], :MPS.Chi[this]], np.diag(MPS.Lambda[this][:MPS.Chi[this]]))
    MPS.B[right]= V

    return err

#---------------------------------------------------------------------------
# (i)TEBD update with Hastings improvement
#---------------------------------------------------------------------------
def update_bond(MPS, Op, bond, truncate = True, tol = 1e-10):
    """
    Apply an operator to the MPS at given bond and update it. 
    This version DOES use the Hastings improvement.

    Parameters:
        MPS:    MPS to be acted upon.
        Op:     Operator to act on the specified bond. Is assumed to be in 
                the form U.shape = (d1,d2,d1,d2), where d1 and d2 are the
                local Hilbert dimensions of the sites connected by the
                bond.
        bond:   Specifies the bond number of the MPS to act on.        

    Returns:
        err (float): The truncation error (want this to be 0). If this is closer
                     to one, you may want to increase the bond dimension!

    """
    # Convenience
    left    = (bond-1)%MPS.L
    this    =   (bond)%MPS.L
    right   = (bond+1)%MPS.L

    leftLambda  = np.diag(MPS.Lambda[left])
    leftB       = MPS.getTensor( this ) #MPS.B[this][:MPS.d[this], :MPS.Chi[left], :MPS.Chi[this]] 
    rightB      = MPS.getTensor( right ) #MPS.B[right][:MPS.d[right], :MPS.Chi[this], :MPS.Chi[right]] 

    # Get C matrix
    C       = np.tensordot( leftB, rightB, axes=(-1,1) ) #(dcc)(dcc) -> (dcdc)
    C       = np.tensordot(     C,     Op, axes=([0, 2], [2, 3])) #(dcdc)(dddd) -> (ccdd) 
    # Get theta and reshape to combine legs
    theta   = np.tensordot( leftLambda, C, axes=(-1,0) ) #(cc)(ccdd) -> (ccdd)
    theta   = np.reshape( np.transpose((theta), (2,0,3,1)), (MPS.d[this]*MPS.Chi[left], MPS.d[right]*MPS.Chi[right]))

    # SVD
    U, S, V = svd(theta, full_matrices=0); V = V.T
  
    # Construct W
    C = np.reshape( np.transpose(C, (2,0,3,1)), (MPS.d[this]*MPS.Chi[left], MPS.d[right]*MPS.Chi[right]))
    W = np.dot(C, V.conj())

    # Truncate
    MPS.Chi[this]    = np.max( [np.min([np.sum( S > tol ), MPS.D[this]]), 1] ) if truncate else np.min([MPS.Chi[this], W.shape[1]])
    norm             = np.linalg.norm( S[:MPS.Chi[this]], ord = 2 )
    err              = np.linalg.norm( S[MPS.Chi[this]:], ord = 2 )
    MPS.Lambda[this] = S[:MPS.Chi[this]]/norm

    # Update B matrices
    MPS.B[this]   = np.reshape(W[:, :MPS.Chi[this]], (MPS.d[this], MPS.Chi[left], MPS.Chi[this]))/norm
    MPS.B[right]  = np.transpose(np.reshape(V[:, :MPS.Chi[this]], (MPS.d[right], MPS.Chi[right], MPS.Chi[this])), (0, 2, 1))

    # Return truncation error
    return err

#-------------------------------------------------------------------------------
# Update bonds while conserving quantum numbers
#-------------------------------------------------------------------------------
def update_bond_conserve(MPS, Op, bond, truncate = True, tol = 1e-10):
    '''
    Update bond with conserving numbers
    '''
    # first, we should truncate all the matrices to reduce overhead
    if bond == 0:
        chi_left = 1
    else:
        chi_left   = MPS.Chi[bond-1]
    chi_middle = MPS.Chi[bond]
    chi_right  = MPS.Chi[bond+1]

    # Get the quantum numbers to the left of each bons
    # The q are dictionaries, that have as keys the q-numbers,
    #  and a list of indices as corresponding values. 
    #  The number of indices corresponds to the bond dimension.
    if bond == 0:
        q_left = {0:[0]}
    else:
        q_left   = MPS.Q[bond-1]
    q_middle = MPS.Q[bond]
    q_right  = MPS.Q[bond+1]
    
    #after SVD, the matrices should be dimension mxm and nxn
    if bond == 0:
        L_left = [1]
    else:
        L_left   = MPS.Lambda[bond-1][:chi_left]
    B_left   = MPS.B[bond][:,:chi_left,:chi_middle]
    B_right  = MPS.B[bond+1][:,:chi_middle,:chi_right]
    
    # Construct theta
    theta=np.tensordot(np.diag(L_left),B_left,axes=(1,1)) #(c c)(d c c) -> (c d c)
    theta=np.tensordot(theta,B_right,axes=(2,1)) # (c d c)(d c c) -> (c d d c)
    theta = np.tensordot(theta,Op,axes=([1,2],[0,1])) #(c d d c)(d d d d) -> (c c d d)
    theta = np.reshape(np.transpose(theta,(2,0,3,1)),(MPS.d[bond]*chi_left,MPS.d[bond]*chi_right))

    # Compute quantum numbers for reshaped theta
    # First the quantum numbers associated with the left merged bond of theta
    # We have the current q-numbers from q_left (e.g. { 0:[0,2], 1:[1] } )
    # and we know the numbers associated to the physical leg (e.g. {0:[0], 1:[1]} )
    # 
    # As an example, this would lead to the following associated values before reshaping.
    #
    # The 'corresponds to #' column has all possible particle numbers, which will
    #  serve as the new keys. The indices in the leftmost columns still need to be combined,
    #  but we know how reshaping does that:
    #    first theta is of shape (d_left, chi_left, d_right, chi_right)
    #     with indices that we call a, b, c, d
    #    then the indices in the reshaped theta are:
    #     i = (dim b)*a + b
    #     j = (dim d)*c + d
    #
    #  dim b = len ([0,2,1]) = 3
    #  index physical (a)   index left (b)   combined index   corresponds to #
    #         0                 0                 0                 0
    #         0                 1                 1                 1
    #         0                 2                 2                 0
    #         1                 0                 3                 1
    #         1                 1                 4                 2
    #         1                 2                 5                 1
    #
    #  This gives: Q_left_new = { 0:[0,2], 1:[1,3,5], 2:[4]}

    # Possible new charge values:
    Q_left = {}
    for k in q_left.keys():
        for d in np.arange(MPS.d[bond]):
            newq = d + k  #mps.add_qn(mps.pqn[d],k,1)
            Q_left[newq] = Q_left.get( newq, [] ) + list(np.array(q_left.get(k))+ d*int(chi_left))

    # Now let's do the same for the right. The slight difference here, is that 
    # the allowed q_values need to take into account that the rightmost leg is an outgoing one.
    
    # q_right (e.g. { 0:[0,2], 1:[1] } )
    # physical e.g. {0:[0], 1:[1]} )
    #  dim d = len ([0,2,1]) = 3
    #  index physical (c)   index left (d)   combined index   corresponds to #
    #         0                 0                 0                 0
    #         0                 1                 1                 1
    #         0                 2                 2                 0
    #         1                 0                 3                 -1
    #         1                 1                 4                 0
    #         1                 2                 5                 -1
    #
    #  This gives: Q_right_new = { -1:[3,5], 0:[0,2,4], 1:[1]}
    Q_right = {}
    for k in q_right.keys():
        for d in np.arange(MPS.d[bond]):
            newq = k-d #mps.add_qn(k,mps.pqn[d],-1)
            Q_right[newq] = Q_right.get( newq, [] ) + list(np.array(q_right.get(k))+ d*int(chi_right))

    # Create empty matrices for us to put the blocks in
    U = np.zeros((chi_left*MPS.d[bond], min(chi_left, chi_right) * MPS.d[bond]), dtype=np.complex128)
    S = np.zeros( min(chi_left, chi_right) * MPS.d[bond], dtype=np.complex128 )
    V = np.zeros((min(chi_left, chi_right) * MPS.d[bond], chi_right*MPS.d[bond]), dtype=np.complex128)

    Qlist = []
    current_size = 0
    for ql in Q_left.keys():
        # Get the indices for the entries corresponding to these elements
        left_indices = [int(a) for a in Q_left.get(ql, [])]
        right_indices = [int(a) for a in Q_right.get(ql, [])]

        # If either of the two lists is empty, the corresponding legs
        # do not allow for the q_number, and so we can skip it. 
        if left_indices == [] or right_indices == []:
            continue

        # Extract the block
        theta_Q = theta[np.ix_(left_indices,right_indices)]

        # We can now SVD the block
        if theta_Q.dtype != np.complex128 :
            UQ,SQ,VQ = svd_dgesvd.svd_dgesvd(theta_Q, full_matrices = 0,compute_uv = 1)
        else:
            UQ,SQ,VQ = svd_zgesvd.svd_zgesvd(theta_Q, full_matrices = 0,compute_uv = 1)

        # And assign them back to the larger matrices, as if we had SVD'ed it in one go
        U[np.array(left_indices), current_size:current_size+len(SQ)] = UQ[:, :len(SQ)]
        S[current_size:current_size+len(SQ)] = SQ
        V[current_size:current_size+len(SQ), np.array(right_indices)] = VQ[:len(SQ), :]
        
        # Keep track of the quantum number at a given index
        Qlist.append( [ql]*len(SQ) )
        
        # Update size
        current_size += len(SQ)

    # Truncate trailing zeros on Y
    S = S[:current_size]
    # Flatten Qlist (can find more elegant way?)
    Qlist = list( itertools.chain.from_iterable(Qlist) )

    # Get indices for sorting singular values
    sorted_idx = np.argsort( S )[::-1]
    # Actually sort them
    S = S[sorted_idx]
    # But we needed the indices so that we can also sort Qlist
    #  since the corresponding indices for the q-sectors have moved
    Qlist = np.array(Qlist)[sorted_idx]
    # Sort the other SVD matrices
    U = U[:, sorted_idx]
    V = V[sorted_idx, :].T

    # Truncate
    MPS.Chi[bond]                    = np.max([np.min([np.sum( S > tol ), MPS.D[bond]]),1])
    # Compute new normalization and error
    norm                             = np.linalg.norm( S[:MPS.Chi[bond]], ord = 2 )
    err                              = np.linalg.norm( S[MPS.Chi[bond]:], ord = 2 )
    # Update Lambda
    MPS.Lambda[bond]                 = S[:MPS.Chi[bond]]/norm

    # Compute the new quantum numbers for the middle bond that we have
    #  just updated. This is why we used Qlist in the first place.
    q_middle_new = {}
    for i,q in enumerate(Qlist[:MPS.Chi[bond]]):
        q_middle_new[q] = q_middle_new.get(q,[]) + [i]
    MPS.Q[bond] = q_middle_new

    # U will be the new left B, V the new Bright
    U = np.reshape( U[:MPS.d[bond]*chi_left, :MPS.Chi[bond]], (MPS.d[bond], chi_left, MPS.Chi[bond] ))
    MPS.B[bond] = np.transpose(np.tensordot(np.diag(np.array(L_left)[:chi_left]**(-1)), U, axes=(1,1)), (1,0,2))
    MPS.B[bond] = np.dot(MPS.B[bond][:, :chi_left, :MPS.Chi[bond]], np.diag(MPS.Lambda[bond][:MPS.Chi[bond]]))

    V = np.transpose( np.reshape( V[:MPS.d[bond]*chi_right, :MPS.Chi[bond]], (MPS.d[bond],chi_right,MPS.Chi[bond])),(0,2,1)) #MPS.Chi[this])),(0,2,1))
    MPS.B[bond+1]= V
    
    return err


#-------------------------------------------------------------------------------
# Update with 3-site operator
#-------------------------------------------------------------------------------
def update_three_site_bond(MPS, Op, left_most_site, verbose = 0, ops=None):
    """
    Apply an operator to the MPS at given bond and update it. 
    This version does NOT use the Hastings improvement.

    Parameters:
        MPS:              MPS to be acted upon.
        U:                Operator to act on the three sites.
        left_most_site:   Specifies the left most site of the three to be updated

    Returns:
        err (float): The truncation error (want this to be 0). If this is closer
                     to one, you may want to increase the bond dimension!

    """
    # Convenience: label sites a b c d e where, b c and d are the three sites we consider
    a = (left_most_site - 1)%MPS.L
    b = (left_most_site + 0)%MPS.L
    c = (left_most_site + 1)%MPS.L
    d = (left_most_site + 2)%MPS.L
    e = (left_most_site + 3)%MPS.L

    # Construct fully contracted network over operator
    Lambda_a = np.diag( MPS.Lambda[a][:MPS.Chi[a]] )
    B_b      = MPS.B[b][:, :MPS.Chi[a], :MPS.Chi[b]]
    B_c      = MPS.B[c][:, :MPS.Chi[b], :MPS.Chi[c]]
    B_d      = MPS.B[d][:, :MPS.Chi[c], :MPS.Chi[d]]

    theta = np.tensordot( Lambda_a, B_b, axes=(-1,1) )  #(ab)(d1 b c) -> (a d1 c)
    theta = np.tensordot( theta, B_c, axes=(-1,1) )     #(a d1 c)(d2 c d) -> (a d1 d2 d)
    theta = np.tensordot( theta, B_d, axes=(-1,1) )     #(a d1 d2 d)(d3 d e) -> (a d1 d2 d3 e)
    
    # Apply operator
    theta = np.tensordot( theta, Op, axes=([1,2,3], [3,4,5]) ) # (a d1 d2 d3 e)(d1 d2 d3 ddd) -> (ae d1 d2 d3)
    # Reshape
    theta = np.transpose( theta, (0,2,3, 1,4) ) #(ad1d2 ed3)
    theta = np.reshape(theta, ( MPS.Chi[a]*MPS.d[b]*MPS.d[c], MPS.Chi[d]*MPS.d[d] ) )

    # SVD into site (12) and (3)
    U, S, V = svd(theta, full_matrices=False)

    # Truncate (allow for squared bond dim since two sites?)
    #MPS.Chi[c]                 = np.max([np.min([np.sum( S > MPS.tol ), MPS.maxchi[c]]),1])
    MPS.Chi[c]                 = np.max([np.min([np.sum( S > MPS.tol ), MPS.D[c]]),1])
    MPS.Lambda[c][:MPS.Chi[c]] = S[:MPS.Chi[c]]/np.linalg.norm( S[:MPS.Chi[c]], ord = 2 )

    # We can now reshape V into the new tensor for site 3
    V = np.transpose(np.reshape( V[:MPS.Chi[c], :MPS.Chi[d]*MPS.d[d]], (MPS.Chi[c], MPS.Chi[d], MPS.d[d])),(2,0,1))
    MPS.B[d][:, :MPS.Chi[c], :MPS.Chi[d]] = V

    # take lambda[c] into account on 2-site theta
    U = np.dot( U[:,:MPS.Chi[c]], np.diag(MPS.Lambda[c][:MPS.Chi[c]]) )
    # U is still an object ~ what theta is for the 2 site case, but it needs some reshaping to
    # make its legs correspond to that case
    U = np.reshape( U[:MPS.Chi[a]*MPS.d[b]*MPS.d[c], :MPS.Chi[c]], (MPS.Chi[a], MPS.d[b], MPS.d[c], MPS.Chi[c]))
    U = np.reshape(np.transpose( U, (0,1,3,2) ), (MPS.Chi[a]*MPS.d[b], MPS.Chi[c]*MPS.d[c]))

    # SVD again
    U, S, V = svd(U, full_matrices=False)

    # Truncate
    #MPS.Chi[b]                 = np.max([np.min([np.sum( S > MPS.tol ), MPS.maxchi[b]]),1])
    MPS.Chi[b]                 = np.max([np.min([np.sum( S > MPS.tol ), MPS.D[b]]),1])
    MPS.Lambda[b][:MPS.Chi[b]] = S[:MPS.Chi[b]]/np.linalg.norm( S[:MPS.Chi[b]], ord = 2 )
    err                        = np.linalg.norm( S[MPS.Chi[b]:], ord=2 )

    # We can now reshape V into the new tensor for site 2
    V = np.transpose(np.reshape( V[:MPS.Chi[b], :MPS.Chi[c]*MPS.d[c]], (MPS.Chi[b], MPS.Chi[c], MPS.d[c])),(2,0,1))
    MPS.B[c][:, :MPS.Chi[b], :MPS.Chi[c]] = V

    # And finally, the tensor for site 1
    U = np.dot( U[:,:MPS.Chi[b]], np.diag( MPS.Lambda[b][:MPS.Chi[b]] ) )
    U = np.reshape(U[:MPS.Chi[a]*MPS.d[b], :MPS.Chi[b]], (MPS.Chi[a], MPS.d[b], MPS.Chi[b]))
    MPS.B[b][:, :MPS.Chi[a], :MPS.Chi[b]]   = np.transpose(np.tensordot(np.diag(MPS.Lambda[a][:MPS.Chi[a]]**(-1)), U, axes=(1,0)), (1,0,2))

    return err

def swap(mps, bond, threshold = 1e-10):
    """ Swap sites connected by bond. """
    
    # Convenience
    left    = (bond-1)%mps.L
    this    =   (bond)%mps.L
    right   = (bond+1)%mps.L

    leftLambda  = np.diag(mps.Lambda[left][:mps.Chi[left]])
    leftB       = mps.getTensor(this)
    rightB      = mps.getTensor(right)

    # Get C matrix
    C       = np.tensordot( leftB, rightB, axes=(-1,1) ) #(dcc)(dcc) -> (dcdc)
    # Swap!
    C       = np.transpose( C, [1,3,2,0] )

    # Get theta and reshape to combine legs
    theta   = np.tensordot( leftLambda, C, axes=(-1,0) ) #(cc)(ccdd) -> (ccdd)
    theta   = np.reshape( np.transpose((theta), (2,0,3,1)), (mps.d[this]*mps.Chi[left], mps.d[right]*mps.Chi[right]))

    # SVD
    U, S, V = svd(theta, full_matrices=0); V = V.T
   
    # Construct W
    C = np.reshape( np.transpose(C, (2,0,3,1)), (mps.d[this]*mps.Chi[left], mps.d[right]*mps.Chi[right]))
    W = np.dot(C, V.conj())

    # Truncate
    mps.Chi[this]      = np.max( [np.min([np.sum( S > threshold ), mps.D[this]]), 1] )
    err                 = np.linalg.norm( S[:mps.Chi[this]] )
    mps.Lambda[this]   = S[:mps.Chi[this]]/err

    # Update B matrices
    mps.B[this]    = np.reshape( W[:, :mps.Chi[this]], (mps.d[this], mps.Chi[left], mps.Chi[this]))/err
    mps.B[right]   = np.transpose(np.reshape( V[:, :mps.Chi[this]], (mps.d[right], mps.Chi[right], mps.Chi[this])), (0, 2, 1))

    # Return truncation error
    return 1 - err

