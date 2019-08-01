###############################################################################
# iMPS.py: Object representing an infinite MPS
################################################################################
from __future__ import division

"""
Implements class for infinite Matrix Product State

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
from scipy.sparse.linalg import LinearOperator
import pickle as cPickle
import copy
from superOps import *
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
        print("Canonical SVD failed: ", e)

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


#--------------------------------------------------------------------------------
# Pure state infinite Matrix Product State class
#--------------------------------------------------------------------------------
class iMPS:
    """ Infinite Matrix Product State

    Description:
        Object for representing an Infinite Matrix Product State.
        Stores the MPS matrices B[i], Lambda[i] and Chi[i] (bond dimension).

    Conventions:
        Lambda[i] is the bond between site i and i+1.
    """

    def __init__(self, L, d, D, tol=1e-8, pure=True):
        """ Initialize an empty infinite Matrix Product State

            Parameters
            ----------
                - L (int)
                    Number of MPS matrices
                - d (int) or (list_of_ints)
                    Hilbert space dimension(s).
                    If a single number is specified, the code assumes that all
                    of the sites have an identical Hilbert space. If a list is
                    specified, it needs to indicate the Hilbert space per site.
                - D (int) or (list_of_ints)
                    Maximal bond dimension(s).
                    If a single number is specified, all bonds will have this
                    maximal bond dimension. If a list is given, it needs to
                    specify the maximal bond dimension per bond. By convention,
                    bond number L is the periodic bond.
        """

        # Size of MPS
        self.L = L

        # Parse local Hilbert space dimension
        if isinstance(d, (int, float)):
            self.d = d*np.ones(L, dtype=int)
        elif isinstance(d, (list, np.ndarray)):
            if len(d) != self.L:
                raise RuntimeError("Length of specified local Hilbert space does not match length of MPS")
            self.d = np.array(d, dtype=int)
        else:
            raise RuntimeError("Local Hilbert space dimension(s) not a number, nor a list/array")

        # Parse bond dimension
        if isinstance(D, (int, float)):
            self.D = D*np.ones(L, dtype=int)
        elif isinstance(D, (list, np.ndarray)):
            if len(D) != self.L:
                raise RuntimeError("Length of specified bond dimensions does not match length of MPS")
            self.D = np.array(D, dtype=int)
        else:
            raise RuntimeError("Bond dimension(s) incorrectly specified")

        # MPS type
        self.type = 'infinite'
        self.tol  = tol
        self.pure = pure
        # Square size of Hilbert space if not a pure state
        if not self.pure:
            self.d = self.d**2

        self.d = np.array( [int(a) for a in self.d] )
        self.D = np.array( [int(a) for a in self.D] )

        # Set matrices to empty ones
        self.B      = {}
        self.Lambda = {}
        self.Chi    = {}
        for s in np.arange(self.L):
            self.B[s]      = np.zeros( (int(self.d[s]), int(self.D[s-1]), int(self.D[s])), dtype=np.complex128 )
            self.Lambda[s] = np.zeros( int(self.D[s]) )
            self.Chi[s]    = self.D[s]

    def set_random_state(self, ortho=True):
        """
        Set the state of the MPS to a random one
        """
        for s in range(self.L):
            # Set random, normalized Lambda[s]
            self.Lambda[s] = np.random.rand(self.D[s]); self.Lambda[s] /= np.linalg.norm(self.Lambda[s], 2); self.Lambda[s] = np.sort(self.Lambda[s])[::-1]
            # Set random B
            self.B[s] = np.random.rand(self.d[s], self.D[s], self.D[s]) + 1j*np.random.rand(self.d[s], self.D[s], self.D[s])
            self.B[s] = np.tensordot( self.B[s], np.diag(self.Lambda[s]), axes=[2,0] )
            # Set current bond dimes
            self.Chi[s] = self.D[s]

        if ortho:
            # Make sure state is in canonical form
            self.orthogonalize(truncate=False)
            # Twice makes it a bit better (threshold)
            self.orthogonalize(truncate=False)

    def set_product_state(self, state = None):
        """
        Set a product state
        """
        # Make sure we have an array for the sites
        state = np.array([int(s) for s in state])

        # Reset B's and Lambda's
        self.B      = {}
        self.Lambda = {}
        self.Chi    = {}
        self.Q      = {}

        for s in np.arange(self.L):
            self.B[s]      = np.zeros( (self.d[s], self.D[s-1], self.D[s]), dtype=np.complex128 )
            self.Lambda[s] = np.zeros( self.D[s] )
            self.Chi[s]    = self.D[s]

        q_left = 0
        self.Q[0] = {q_left : [0]}
        if self.pure:
            # Set the i'th B to have s particles
            for i,s in enumerate(state):
                self.B[i]           = np.zeros( (self.d[s],1,1) )
                self.B[i][s]        = 1
                self.Chi[i]         = 1
                self.Lambda[i]      = np.array([1])

                q_left              = q_left + int(state[i])
                self.Q[i+1]           = {q_left : [0]}

        else:
            # The linear combinations we need to take are now those with the columns of
            # trafo as the coefficients.
            for i,s in enumerate(state):
                # Here we need to know which linear combinations of basis matrices give us the
                # standard basis. To figure this out, we first gather all the diagonal matrices.
                # And we know which ones are diagonal, namely the first N of them.
#                print("Setting site %d as a %d"%(i,s))

                num = int(np.sqrt(self.d[i]))
#                print("The local hilbert space here is %d"%num)

                trafo = np.zeros( (num,num), dtype=np.complex128 )
                for j in range(num):
                    trafo[:,j] = np.diag( bases[num][j] ).T
                trafo = np.linalg.inv(trafo)

                self.B[i] = np.zeros( (self.d[i],1,1), dtype=np.complex128 )
                self.B[i][:self.d[i],0,0] = np.concatenate([trafo[:,s], np.array([0 for n in range(num**2 - num)])])
                self.Chi[i]         = 1
                self.Lambda[i]   = np.array([1])

    def set_infinite_temperature_state( self ):
        """ Set infinite temperature state - only for non-pure! """
        if self.pure:
            raise RuntimeError("Cannot set pure state to infinite temperature - Forgot to indicate non-pure?")
            return -1

        # Select the identity matrix on every site
        state = np.zeros( self.L )

        # Reset B's and Lambda's
        self.B      = {}
        self.Lambda = {}
        self.Chi    = {}
        for s in np.arange(self.L):
            self.B[s]      = np.zeros( (self.d[s], self.D[s-1], self.D[s]), dtype=np.complex128 )
            self.Lambda[s] = np.zeros( self.D[s] )
            self.Chi[s]    = self.D[s]

        # Set the i'th B to have s particles
        for i,s in enumerate(state):
            self.B[i]           = np.zeros( (self.d[s],1,1) )
            self.B[i][s]        = 1
            self.Chi[i]         = 1
            self.Lambda[i]      = np.array([1])

    def getTensor(self, site, full = False, form='R'):
        """ Return MPS tensor for given site. If not full, return only up to current bond dimension. """
        # TODO: If we have Gamma and Lambda, we return Gamma*L if form = 'R', otherwise L*Gamma.
        # For now, we only have B = Gamma*L. Could do L*G*Linv for L

        if form != 'R':
            raise RuntimeError("getTensor: Only R-form supported at the moment!")
        return self.B[site]

    def getTheta(self, bond):
        """ Construct theta for bond """
        # For convenient indexing and readability
        leftbond    = np.mod( (bond-1), self.L)
        bond        = np.mod( (bond)  , self.L)
        rightbond   = np.mod( (bond+1), self.L)

        # Convenient abbreviations
        leftLambda  = np.diag(self.Lambda[leftbond])
        leftB       = self.getTensor(bond)
        rightB      = self.getTensor(rightbond)

        # Construct theta
        theta = np.tensordot( leftLambda, leftB, axes=(-1,1)) # (cc)(dcc)  -> (cdc)
        theta = np.tensordot( theta, rightB, axes=(-1,1))     # (cdc)(dcc) -> (cddc)
        return theta

    def swap(self, bond):
        """ Swap sites connected by bond. """

        # Convenience
        left    = (bond-1)%self.L
        this    =   (bond)%self.L
        right   = (bond+1)%self.L

        leftLambda  = np.diag(self.Lambda[left][:self.Chi[left]])
        leftB       = self.getTensor(this)
        rightB      = self.getTensor(right)

        chi_left = leftB.shape[1]
        chi_this = leftB.shape[2]
        if rightB.shape[1] != chi_this:
            return None
        chi_right = rightB.shape[2]

        # Get C matrix
        C       = np.tensordot( leftB, rightB, axes=(-1,1) ) #(dcc)(dcc) -> (dcdc)  e.g. (246)(368) -> (2438)
        # Swap! (only the physical indices
#        C       = np.transpose( C, [1,3,2,0] )  #e.g. (4832)
        C       = np.transpose( C, [1,3,2,0] )  #e.g. (4832)

        # Get theta and reshape to combine legs
        #theta   = np.tensordot( leftLambda, C, axes=(-1,0) ) #(cc)(ccdd) -> (ccdd)   e.g.(4832)
        theta   = np.tensordot( leftLambda, C, axes=(-1,0) ) #(cc)(ccdd) -> (ccdd)   e.g.(4832)
        #theta   = np.reshape( np.transpose((theta), (2,0,3,1)), (self.d[this]*self.Chi[left], self.d[right]*self.Chi[right]))   # e.g. (3428)
        theta   = np.reshape( np.transpose((theta), (2,0,3,1)), (self.d[right]*chi_left, self.d[this]*chi_right))   # e.g. (4832)->(3428)

        # SVD
        # (3*4, 2*8) -> (3*4, s), (s,s), (s, 2*8), (2*8, s)
        U, S, V = sp.linalg.svd(theta, full_matrices=0); V = V.T

        # Construct W
        #C = np.reshape( np.transpose(C, (2,0,3,1)), (self.d[this]*self.Chi[left], self.d[right]*self.Chi[right]))  # e.g. (4432) -> (3428)
        C = np.reshape( np.transpose(C, (2,0,3,1)), (self.d[right]*chi_left, self.d[this]*chi_right))  # e.g. (4832) -> (3428)
        W = np.dot(C, V.conj())  #(3*4, 2*8)*(2*8,s)

        # Truncate
        self.Chi[this]    = np.max( [np.min([np.sum( S > 1e-8 ), self.D[this]]), 1] )
        norm             = np.linalg.norm( S[:self.Chi[this]], ord = 2 )
        err              = np.linalg.norm( S[self.Chi[this]:], ord = 2 )
        self.Lambda[this] = S[:self.Chi[this]]/norm

        # Update B matrices
        self.d[this], self.d[right] = self.d[right], self.d[this]
        self.D[this], self.D[right] = self.D[right], self.D[this]
        self.Chi[left], self.Chi[right] =  chi_left, chi_right

        # (3*4, s) -> (3, 4->8, s)
        self.B[this]   = np.reshape(W[:, :self.Chi[this]], (self.d[this], self.Chi[left], self.Chi[this]))/norm
        # (2*4, s) -> (3->2, chiright->chileft, chithis) -> (reshape)
        self.B[right]  = np.transpose(np.reshape(V[:, :self.Chi[this]], (self.d[right], self.Chi[right], self.Chi[this])), (0, 2, 1))

        # Return truncation error
        return err

    def getEntEntr(self):
        """ Return the entanglement entropy for each of the bonds """
        S = []
        for bond in range(self.L):
            x = self.Lambda[bond]**2
            S.append( -np.inner( x, np.log(x) ) )
        return S

    def saveToFile(self, filename):
        """ Save MPS to a file """
        with open(filename, "wb") as f:
            cPickle.dump(self, f)

    @classmethod
    def loadFromFile(cls, filename):
        with open(filename, "rb") as f:
            return cPickle.load(f)

    def __add__(self, other):
        """ Add two MPS objects and truncate result """
        # Sanity checks
        if self.L != other.L:
            raise RuntimeError("Cannot sum two MPS's of different length!")
        if not np.all(self.d == other.d):
            raise RuntimeError("Cannot sum two MPS's with different local spaces!")

        A   = copy.deepcopy(self)
        B   = copy.deepcopy(other)

        # New sizes for the resulting matrix
        bond_dimensions = [x+y for (x,y) in zip(A.D,B.D)]
        # Now construct an empty MPS
        res = iMPS( A.L, A.d, bond_dimensions )

        # For the first row vector
        for s in np.arange(res.L):
            res.Lambda[s][:len(A.Lambda[s])] = A.Lambda[s]
            res.Lambda[s][len(A.Lambda[s]):] = B.Lambda[s]
            res.Lambda[s] /= np.linalg.norm(res.Lambda[s])

            for l in np.arange(res.d[s]):
                res.B[s][l][:A.B[s][l].shape[0], :A.B[s][l].shape[1]] = A.B[s][l]
                res.B[s][l][A.B[s][l].shape[0]:, A.B[s][l].shape[1]:] = B.B[s][l]

        # Bring into canonical form
        res.orthogonalize()
        res.orthogonalize()
        return res

    #---------------------------------------------------------------------------
    # Orthogonality functions start here
    #---------------------------------------------------------------------------
    def isCanonical(self, threshold = 10**(-8)):
        """
        Check if iMPS is in canonical form by checking threshold on each bond.

        Parameters:
        threshold: Sets threshold for what is considered canonical.
        Defaults to 10^(-8)

        """
        # Reset error; will contain sqrt of squared sum of errors on l/r bonds
        err = 0.

        # Check every site
        for site in range(self.L):
            # For convenient indexing and readability
            leftbond    = np.mod( (site-1), self.L)
            bond        = np.mod( (site)  , self.L)

            # Convenient abbreviations
            leftLambda  = np.diag(self.Lambda[leftbond])
            B           = self.getTensor(bond)
            rightLambda = np.diag(self.Lambda[bond])

            # Contract on the right
            R       = np.tensordot(B, B.conj(), axes=([0, 2], [0, 2]))
            Rerr    = np.linalg.norm( R/R[0,0] - np.eye(R.shape[0]))

            # Contract on the left
            B       = np.tensordot( B, np.linalg.inv(rightLambda), axes=(2,0) )
            B       = np.tensordot( leftLambda, B, axes=(1,1) )
            L       = np.tensordot(B, B.conj(), axes=([0, 1], [0, 1]))
            Lerr    = np.linalg.norm( L/L[0,0] - np.eye(L.shape[0]) )

            if( abs(Rerr) > threshold or abs(Lerr) > threshold ):
                print("\n *** Error for canonical form exceeds threshold")
                print("\t On site %d: %f, %f (threshold: %g)"%(site, Lerr, Rerr, threshold))

            err += np.sqrt(Rerr**2 + Lerr**2)
        return err

    def orthogonalize(self, tol = 1e-12, truncate=True ):
        """
        Orthogonalize the whole MPS.

        Parameters:
            threshold: Threshold up to which the MPS has to be orthogonal.

        """
        if self.L != 2:
            raise RuntimeError("Explicit orthogonalization for L != 2 not supported yet! Try updating with identities.")

        verbose = 0

        def __left_eigvec( A ):
            """ Compute dominant left eigenvector of transfermatrix with tensor A.
            """
            def __apply_left( vec ):
                # Reshape vec
                vec = np.reshape( vec, (A[0].shape[1], A[0].shape[1]) )

                # Contract as if transfer matrix
                vec = np.tensordot( vec, A[0], axes=(0,1) ) # (lt lb)(d lt rt) -> (lb d rt)
                vec = np.tensordot( vec, np.conjugate(A[0]), axes=((0,1),(1,0)) ) #(lb d rt)(d lb rb) -> (rt rb)

                if len(A) > 1:
                    for s in range(1,len(A)):
                        vec = np.tensordot( vec, A[s], axes=(0,1) )
                        vec = np.tensordot( vec, np.conjugate(A[s]), axes=((0,1),(1,0)) )

                return np.reshape( vec, A[-1].shape[2]*A[-1].shape[2] )

            E = LinearOperator( (A[-1].shape[2]*A[-1].shape[2], A[0].shape[1]*A[0].shape[1]), matvec = __apply_left, dtype=np.complex128 )

            # Hermitian initial guess!
            init = np.random.rand( A[0].shape[1], A[0].shape[1] ) + 1j*np.random.rand( A[0].shape[1], A[0].shape[1] )
            init = 0.5*(init + np.conjugate(np.transpose(init)))
            init = np.reshape( init, A[0].shape[1]*A[0].shape[1] )

            ev, eigvec = sp.sparse.linalg.eigs(E, k=1, which='LM', v0=init, maxiter=1e4)
            return ev, np.array(np.reshape(eigvec, (A[-1].shape[2], A[-1].shape[2])))

        def __right_eigvec( A ):
            """ Compute dominant right eigenvector of transfermatrix with tensors A.
            """

            def __apply_right( vec ):
                # Reshape vec
                vec = np.reshape( vec, (A[-1].shape[2],A[-1].shape[2]) )

                # Contract as if transfer matrix
                vec = np.tensordot( vec, A[-1], axes=(0,2) )  # (rt rb)(d lt rt) -> (rb d lt)
                vec = np.tensordot( vec, np.conjugate(A[-1]), axes=( (0,1),(2,0) ) )  # (rb d lt)(d lb rb) -> (lt lb)

                if len(A) > 1:
                    for s in range(len(A)-2,-1,-1):
                        vec = np.tensordot( vec, A[s], axes=(0,2) )
                        vec = np.tensordot( vec, np.conjugate(A[s]), axes=( (0,1),(2,0)) )

                return np.reshape( vec, A[0].shape[1]*A[0].shape[1] )

            E = LinearOperator( (A[0].shape[1]*A[0].shape[1], A[-1].shape[2]*A[-1].shape[2]), matvec = __apply_right, dtype=np.complex128 )

            # Hermitian initial guess!
            init = np.random.rand( A[-1].shape[2], A[-1].shape[2] ) + 1j*np.random.rand( A[-1].shape[2], A[-1].shape[2] )
            init = 0.5*(init + np.conjugate(np.transpose(init)))
            init = np.reshape( init, A[-1].shape[2]*A[-1].shape[2] )

            ev, eigvec = sp.sparse.linalg.eigs(E, k=1, which='LM', v0=init, maxiter=1e4)
            return ev, np.array(np.reshape(eigvec, (A[0].shape[1], A[0].shape[1])))

        # Find left and right dominant eigenvectors
        LbGa = np.tensordot( np.diag(self.Lambda[1]), np.tensordot( self.getTensor(0), np.diag( self.Lambda[0]**(-1)), axes=(2,0)), axes=(1,1) )
        LbGa = np.transpose( LbGa, (1,0,2) )
        LaGb = np.tensordot( np.diag(self.Lambda[0]), np.tensordot( self.getTensor(1), np.diag( self.Lambda[1]**(-1)), axes=(2,0)), axes=(1,1) )
        LaGb = np.transpose( LaGb, (1,0,2) )

        ev, vL = __left_eigvec( [ LbGa, LaGb ] )
        if verbose > 0:
            print("Dominant left ev: ", ev)
            print(vL)
        ev, vR = __right_eigvec( [ self.getTensor(0), self.getTensor(1) ] )
        if verbose > 0:
            print("Dominant right ev: ", ev)
            print(vR)

        # Decompose them as squares
        vLdiag, vLM = np.linalg.eigh(vL)
        Y = np.dot(np.diag(np.lib.scimath.sqrt(vLdiag)), np.conjugate(np.transpose(vLM)))
        if verbose > 0:
            print("Ydag Y = ", np.dot( np.conjugate(np.transpose(Y)), Y ))
            print("Ydag Y - vL", np.dot( np.conjugate(np.transpose(Y)), Y ) - vL)

        vRdiag, vRM = np.linalg.eigh(vR)
        X = np.dot(vRM, np.diag(np.lib.scimath.sqrt(vRdiag)))
        if verbose > 0:
            print("X Xdag = ", np.dot( X, np.conjugate(np.transpose(X))))
            print("X Xdag - vR ", np.dot( X, np.conjugate(np.transpose(X))) - vR)

        # Test resolutions of identity
        if verbose > 0:
            print("X.Xinv = ", np.dot(X, np.linalg.inv(X)))
            print("YTinv.YT = ", np.dot(np.linalg.inv(np.transpose(Y)), np.transpose(Y)))

        Y = np.conjugate(np.transpose(Y))
        U, lb, V = sp.linalg.svd( np.dot( np.dot( np.transpose(Y), np.diag(self.Lambda[1])), X ) )

        oldGamma = np.tensordot( self.getTensor(0), np.tensordot( self.getTensor(1), np.diag(self.Lambda[1]**(-1)), axes=(2,0) ), axes=(2,1) )  #(dlr)(dlr)->(dldr)
        newGamma = np.tensordot( np.dot(V, np.linalg.inv(X)), oldGamma, axes=(1,1) ) #(xx)(dldr) -> (xddr)
        newGamma = np.tensordot( newGamma, np.dot(np.linalg.inv(np.transpose(Y)), U), axes=(3,0) ) #(xddr)(yy) -> (xddy)

        # New Lambda
        self.Chi[1]    = np.max([np.min([np.sum( lb > tol ), self.D[1]]), 1]) if truncate else self.D[1]
        norm           = np.linalg.norm( lb[:self.Chi[1]], 2 )
        self.Lambda[1] = lb[:self.Chi[1]] / norm

        # Extract new Gamma's from current combined Gamma
        newGamma = np.tensordot( np.diag(self.Lambda[1]), np.tensordot( newGamma[:self.Chi[1], :, :, :self.Chi[1]], np.diag(self.Lambda[1]), axes=(3,0)), axes=(1,0) ) #(ll)(xddy)(rr)->(lddr)
        newGamma = np.reshape( newGamma, (self.Chi[1]*self.d[0], self.Chi[1]*self.d[1]) )

        # SVD
        P, la, Q = sp.linalg.svd( newGamma, full_matrices=False )

        # New Lambda
        self.Chi[0]    = np.max([np.min([np.sum( la > tol ), self.D[0]]), 1]) if truncate else self.D[0]
        norm           = np.linalg.norm( la[:self.Chi[0]], 2 )
        self.Lambda[0] = la[:self.Chi[0]] / norm

        P = np.reshape(P[:,:self.Chi[0]], (self.Chi[1], self.d[0], self.Chi[0]) )
        P = np.transpose( P, (1,0,2) )

        Q = np.reshape(Q[:self.Chi[0],:], (self.Chi[0], self.d[1], self.Chi[1]) )
        Q = np.transpose( Q, (1,0,2) )

        self.B[0] = np.tensordot( np.diag( self.Lambda[1]**(-1) ), np.tensordot(P, np.diag(self.Lambda[0]), axes=(2,0) ), (1,1) )
        self.B[0] = np.transpose( self.B[0], (1,0,2) )
        self.B[1] = Q

    #---------------------------------------------------------------------------
    # Measurement
    #---------------------------------------------------------------------------
    def measure( self, ops, sites ):
        # Check inputs
        if not isinstance(ops, (list, np.ndarray)):
            raise RuntimeError("Please specify operators as a python list (even if only one!)")
        if not isinstance(sites, (list, np.ndarray)):
            raise RuntimeError("Please specify sites as a python list (even if only one!)")

        ops = np.array(ops)
        sites = np.array(sites)

        if len(ops) != len(sites):
            raise RuntimeError("Trying to measure more ops than sites or vice versa!")

        # Delegate to underlying functions depending on whether we are measuring
        # a pure state or a mixed state
        if self.pure:
            return self.measure_pure( ops, sites )
        else:
            return self.measure_mixed( ops, sites )

    def get_coefficient( self, sites ):

        #print(self.Lambda[0].shape)
        #print(self.B[0][sites[0]].shape)
        #print(self.Chi[0], self.Chi[1])
        tmp = np.dot(np.diag(self.Lambda[0][:1]), self.B[0][sites[0]][:self.Chi[0], :self.Chi[1]])
        #tmp = np.dot( np.diag(self.Lambda[(current_site-1)%self.L]), np.diag(self.Lambda[(current_site-1)%self.L]) )
        for i in range(1,self.L):
            # Add one 'transfer matrix'
        #    print(tmp.shape)
        #    print(self.B[i][sites[i]].shape)
        #    print(self.Chi[i-1], self.Chi[i])

            tmp = np.dot( tmp, self.B[i][sites[i]][:self.Chi[i-1], :self.Chi[i]] )

        # Contract on right to close contraction
        ans = np.trace( tmp )
        return ans



    # Measurement on a pure state
    def overlap( self, mps2 ):
        """ Compute expectation value of (list of) operator(s).

            Assumes that the isometric gauge is centered on the left-most site, and
            that all other matrices are right-normalized.

            Parameters
            ----------
                - Ops (numpy array/matrix) or (list of numpy arrays/matrices)
                    If a single operator is specified, a single site has to be specified.
                    If a list of operators is given, then also an equal length list with
                    different sites has to be given.

                - sites (int) or (list of ints)
                    Sites on which corresponding operators act
        """

        tmp = np.dot( np.diag(self.Lambda[0]), np.diag(mps2.Lambda[0]) )
        #tmp = np.dot( np.diag(self.Lambda[(current_site-1)%self.L]), np.diag(self.Lambda[(current_site-1)%self.L]) )
        for i in range(self.L):
            # Add one 'transfer matrix'
            tmp = np.tensordot( tmp, self.B[i].conj(), axes=(0,1) ) #(u d)(p l r) -> (d p r)
            # Contract with identity
            tmp = np.transpose( np.tensordot( tmp, np.eye(self.d[i]), axes=(1,0) ), (0,2,1) ) #(d p r)(a b) -> (d r b) -> (d b r)
            # Contract on bottom
            tmp = np.tensordot( tmp, mps2.B[i], axes=((0,1),(1,0)) ) #(d b r)(p lbot rbot) -> (rtop rbot)

        # Contract on right to close contraction
        ans = np.trace( tmp )
        return ans


    # Measurement on a pure state
    def measure_pure( self, ops, sites ):
        """ Compute expectation value of (list of) operator(s).

            Assumes that the isometric gauge is centered on the left-most site, and
            that all other matrices are right-normalized.

            Parameters
            ----------
                - Ops (numpy array/matrix) or (list of numpy arrays/matrices)
                    If a single operator is specified, a single site has to be specified.
                    If a list of operators is given, then also an equal length list with
                    different sites has to be given.

                - sites (int) or (list of ints)
                    Sites on which corresponding operators act
        """

        # Sort arguments from leftmost site to right
        modulo_sites    = [a%self.L for a in sites]
        sorted_indices  = np.argsort( modulo_sites )
        sites           = np.array(modulo_sites)[sorted_indices]
        ops             = np.array(ops)[sorted_indices]

        # Keep track of current_site and current_op, so that we know if we should
        # insert an operator or just contract as if identity operator was specified.
        current_site = sites[0]
        current_op   = 0

        tmp = np.dot( np.diag(self.Lambda[(current_site-1)%self.L]), np.diag(self.Lambda[(current_site-1)%self.L]) )
        while current_site < sites[-1] + 1:
            # Add one 'transfer matrix'
            tmp = np.tensordot( tmp, self.B[(current_site)%self.L].conj(), axes=(0,1) ) #(u d)(p l r) -> (d p r)

            # Insert operator if there is one
            if current_site in sites:

                # Check size of current operator
                if len(ops[current_op].shape) == 2:
                    # Contract with operator
                    tmp = np.transpose( np.tensordot( tmp, ops[current_op], axes=(1,0) ), (0,2,1) ) #(d p r)(a b) -> (d r b) -> (d b r)
                    # Contract on bottom
                    tmp = np.tensordot( tmp, self.B[current_site%self.L], axes=((0,1),(1,0)) ) #(d b r)(p lbot rbot) -> (rtop rbot)

                    current_site += 1

                # Bond operator
                elif len(ops[current_op].shape) == 4:
                    # Add one more site to transfer matrix
                    tmp = np.tensordot( tmp, self.B[(current_site+1)%self.L].conj(), axes=(2,1) ) #(d p r)(p2 l r) -> (d p p2 r)
                    # Contract with operator
                    tmp = np.tensordot( tmp, ops[current_op], axes=((1,2),(0,1))) #(d p1 p2 r)(a b a b) -> (d r a b)
                    # Contract on bottom
                    tmp = np.tensordot( tmp, self.B[current_site%self.L], axes=((0,2),(1,0)) ) # (d r a b)(p1 l r) -> (r b rbottom)
                    tmp = np.tensordot( tmp, self.B[(current_site+1)%self.L], axes=((2,1),(1,0)) ) #(r b rbottom)(p2 l r) -> (r rbottom)

                    current_site += 2

                # Bond operator
                elif len(ops[current_op].shape) == 6:
                    # Add two more sites to transfer matrix
                    tmp = np.tensordot( tmp, self.B[(current_site+1)%self.L].conj(), axes=(2,1) ) #(d p r)(p2 l r) -> (d p p2 r)
                    tmp = np.tensordot( tmp, self.B[(current_site+2)%self.L].conj(), axes=(3,1) ) #(d p p2 r)(p3 l r) -> (d p p2 p3 r)

                    # Contract with operator
                    tmp = np.tensordot( tmp, ops[current_op], axes=((1,2,3),(0,1,2))) #(d p1 p2 p3 r)(a b c a b c) -> (d r a b c)

                    # Contract on bottom
                    tmp = np.tensordot( tmp, self.B[current_site%self.L], axes=((0,2),(1,0)) ) # (d r a b c)(p1 l rbottom) -> (r b c rbottom)
                    tmp = np.tensordot( tmp, self.B[(current_site+1)%self.L], axes=((3,1),(1,0)) ) #(r b c rbottom)(p2 l r) -> (r c r)
                    tmp = np.tensordot( tmp, self.B[(current_site+2)%self.L], axes=((1,2),(0,1)) ) #(r b r)(p2 l r) -> (r rbottom)

                    current_site += 3

                else:
                    raise RuntimeError("Operator type not supported!")

                current_op += 1
            else:
                # Contract with identity
                tmp = np.transpose( np.tensordot( tmp, np.eye(self.d[current_site]), axes=(1,0) ), (0,2,1) ) #(d p r)(a b) -> (d r b) -> (d b r)
                # Contract on bottom
                tmp = np.tensordot( tmp, self.B[current_site%self.L], axes=((0,1),(1,0)) ) #(d b r)(p lbot rbot) -> (rtop rbot)
                current_site += 1

        # Contract on right to close contraction
        ans = np.trace( tmp )
        return ans

    def measure_mixed( self, ops, sites ):
        """ Compute expectation value of (list of) operator(s).

            Assumes that the isometric gauge is centered on the left-most site, and
            that all other matrices are right-normalized.

            Parameters
            ----------
                - Ops (numpy array/matrix) or (list of numpy arrays/matrices)
                    If a single operator is specified, a single site has to be specified.
                    If a list of operators is given, then also an equal length list with
                    different sites has to be given.

                - sites (int) or (list of ints)
                    Sites on which corresponding operators act
        """

        # Sort arguments from leftmost site to right
        modulo_sites    = [a%self.L for a in sites]
        sorted_indices  = np.argsort( modulo_sites )
        sites           = np.array(modulo_sites)[sorted_indices]
        ops             = np.array(ops)[sorted_indices]

        # Keep track of current_site and current_op, so that we know if we should
        # insert an operator or just contract as if identity operator was specified.
        current_site = 0
        current_op   = 0

        tmp = np.diag(self.Lambda[(current_site-1)%self.L])
        while current_site < self.L: #sites[-1]:
            # Add one 'transfer matrix'
            tmp = np.tensordot( tmp, self.B[(current_site)%self.L].conj(), axes=(1,1) ) #(u d)(p l r) -> (u p r)

            # Insert operator if there is one
            if current_site in sites:
                # Check size of current operator
                if len(ops[current_op].shape) == 2:
                    # Contract with operator
                    tmp = np.tensordot( tmp, ops[current_op][:,0], axes=(1,0) ) #(u p r)(b) -> (u r)
                    current_site += 1

                # Bond operator
                elif len(ops[current_op].shape) == 4:
                    # Add one more site to transfer matrix
                    tmp = np.tensordot( tmp, self.B[(current_site+1)%self.L].conj(), axes=(2,1) ) #(u p r)(p2 l r) -> (d p p2 r)
                    # Contract with operator
                    tmp = np.tensordot( tmp, ops[current_op][:,:,0,0], axes=((1,2),(0,1))) #(d p1 p2 r)(a b) -> (d r)
                    current_site += 2

                # Bond operator
                elif len(ops[current_op].shape) == 6:
                    # Add two more sites to transfer matrix
                    tmp = np.tensordot( tmp, self.B[(current_site+1)%self.L].conj(), axes=(2,1) ) #(d p r)(p2 l r) -> (d p p2 r)
                    tmp = np.tensordot( tmp, self.B[(current_site+2)%self.L].conj(), axes=(3,1) ) #(d p p2 r)(p3 l r) -> (d p p2 p3 r)

                    # Contract with operator
                    tmp = np.tensordot( tmp, ops[current_op][:,:,:,0,0,0], axes=((1,2,3),(0,1,2))) #(d p1 p2 p3 r)(a b c a b c) -> (d r a b c)
                    current_site += 3

                else:
                    raise RuntimeError("Operator type not supported!")

                current_op += 1
            else:
                # Current site does not have an operator applied to it; select 0 leg
                tmp = tmp[:,0,:] # (u r)
                current_site += 1

        # Store unnormalized answer
        ans = np.trace( tmp )

        #---------------------------------------------------------------------------
        # Normalize
        #---------------------------------------------------------------------------
        tmp = np.diag(self.Lambda[-1%self.L])
        for i in range(self.L): #range(sites[-1]):
            tmp = np.tensordot( tmp, self.B[i%self.L][0].conj(), axes=(1,0) ) #(u d)(l r) -> (d r)

        # Store norm
        norm = np.trace(tmp)

        # Normalize answer
        return ans / norm


class transferMat:
    def __init__(self, mps1, mps2 = None):
        # Store MPS matrices
        self.mps1 = mps1
        self.mps2 = mps2
        if mps2 == None:
            self.mps2 = mps1

    def left_eigvec( self, k = 1, tol=1e-10, init=None ):
        """ Compute dominant left eigenvector of transfermatrix with tensor A.
        """

        # Build list of correct matrices
        A = [ np.tensordot( np.diag(self.mps1.Lambda[(i-1)%self.mps1.L]), np.tensordot( self.mps1.B[i], np.diag(self.mps1.Lambda[i]**(-1)) )) for i in range(self.mps1.L) ]
        B = [ np.tensordot( np.diag(self.mps2.Lambda[(i-1)%self.mps2.L]), np.tensordot( self.mps2.B[i], np.diag(self.mps2.Lambda[i]**(-1)) )) for i in range(self.mps2.L) ]

        def __apply_left( vec ):
            # Reshape vec
            vec = np.reshape( vec, (A[0].shape[1], B[0].shape[1]) )

            # Contract as if transfer matrix
            vec = np.tensordot( vec, A[0], axes=(0,1) ) # (lt lb)(d lt rt) -> (lb d rt)
            vec = np.tensordot( vec, np.conjugate(B[0]), axes=((0,1),(1,0)) ) #(lb d rt)(d lb rb) -> (rt rb)

            if len(A) > 1:
                for s in range(1,len(A)):
                    vec = np.tensordot( vec, A[s], axes=(0,1) )
                    vec = np.tensordot( vec, np.conjugate(B[s]), axes=((0,1),(1,0)) )

            return np.reshape( vec, A[-1].shape[2]*B[-1].shape[2] )

        E = LinearOperator( (A[-1].shape[2]*B[-1].shape[2], A[0].shape[1]*B[0].shape[1]), matvec = __apply_left, dtype=np.complex128 )

        if init == None:
            init = np.random.rand( A[0].shape[1], B[0].shape[1] ) + 1j*np.random.rand( A[0].shape[1], B[0].shape[1] )
            # Hermitian initial guess!
            if A[0].shape[0] == B[0].shape[1]:
                init = 0.5*(init + np.conjugate(np.transpose(init)))
            init = np.reshape( init, A[0].shape[1]*B[0].shape[1] )

        if E.shape[0] == 1 and E.shape[1] == 1:
            return E.matvec( np.array([1]) ), np.array([1])

        ev, eigvec = sp.sparse.linalg.eigs(E, k=k, which='LM', v0=init, maxiter=1e4, tol=tol)
        return ev, np.array([ np.array(np.reshape(eigvec[:,i], (A[-1].shape[2], B[-1].shape[2]))) for i in range(k) ])

    def right_eigvec( self, k = 1, tol=1e-10, init=None ):
        """ Compute dominant right eigenvector of transfermatrix with tensors A.
        """

        # Build list of correct matrices
        A = [ self.mps1.B[i] for i in range(self.mps1.L) ]
        B = [ self.mps2.B[i] for i in range(self.mps2.L) ]

        def __apply_right( vec ):
            # Reshape vec
            vec = np.reshape( vec, (A[-1].shape[2],B[-1].shape[2]) )

            # Contract as if transfer matrix
            vec = np.tensordot( vec, A[-1], axes=(0,2) )  # (rt rb)(d lt rt) -> (rb d lt)
            vec = np.tensordot( vec, np.conjugate(B[-1]), axes=( (0,1),(2,0) ) )  # (rb d lt)(d lb rb) -> (lt lb)

            if len(A) > 1:
                for s in range(len(A)-2,-1,-1):
                    vec = np.tensordot( vec, A[s], axes=(0,2) )
                    vec = np.tensordot( vec, np.conjugate(B[s]), axes=( (0,1),(2,0)) )

            return np.reshape( vec, A[0].shape[1]*B[0].shape[1] )

        E = LinearOperator( (A[0].shape[1]*B[0].shape[1], A[-1].shape[2]*B[-1].shape[2]), matvec = __apply_right, dtype=np.complex128 )

        if init == None:
            newinit = np.random.rand( A[-1].shape[2], B[-1].shape[2] ) + 1j*np.random.rand( A[-1].shape[2], B[-1].shape[2] )
            if A[-1].shape[2] == B[-1].shape[2]:
                newinit = 0.5*(newinit + np.conjugate(np.transpose(newinit)))
        else:
            if init.shape[0] != A[-1].shape[2] or init.shape[1] != B[-1].shape[2]:
                newinit = np.zeros( (A[-1].shape[2], B[-1].shape[2]), dtype=np.complex128 )
                newinit[:init.shape[0], :init.shape[1]] = init
            else:
                newinit = init

            init = np.reshape( newinit, A[-1].shape[2]*B[-1].shape[2] )

        if E.shape[0] == 1 and E.shape[1] == 1:
            return E.matvec( np.array([1]) ), np.array([1])

        ev, eigvec = sp.sparse.linalg.eigs(E, k=k, which='LM', v0=init, maxiter=1e4, tol=tol )
        return ev, np.array([ np.array(np.reshape(eigvec[:,i], (A[0].shape[1], B[0].shape[1]))) for i in range(k) ])
