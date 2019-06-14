from __future__ import division
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import copy

def svd(A, full_matrices=False):
    try:
        return np.linalg.svd(A, full_matrices)
    except Exception as e:
        print(e)
        #print A

class mps:
    def __init__(self, L, d, D, type='finite', threshold=1e-12):
        """ Initialize a Matrix Product State 

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
            self.d = d*np.ones(L)
        elif isinstance(d, (list, np.ndarray)):
            if len(d) != self.L:
                raise RuntimeError("Length of specified local Hilbert space does not match length of MPS")
            self.d = np.array(d)
        else:
            raise RuntimeError("Local Hilbert space dimension(s) not a number, nor a list/array")

        # Parse bond dimension
        if isinstance(D, (int, float)):
            self.D = D*np.ones(L)
        elif isinstance(D, (list, np.ndarray)):
            if len(D) != self.L:
                raise RuntimeError("Length of specified bond dimensions does not match length of MPS")
            self.D = np.array(D)
        else:
            raise RuntimeError("Bond dimension(s) incorrectly specified")

        # MPS type
        self.type = type
        if type=='finite':
            self.D[-1] = 1

        # Truncation threshold
        self.threshold = threshold

        self.d = np.array([int(d) for d in self.d])
        self.D = np.array([int(d) for d in self.D])
        # Set matrices to empty ones
        self.M = {}
        self.Lambda = {}
        for s in range(self.L):
            self.M[s]      = np.zeros( (self.d[s], self.D[s-1], self.D[s]), dtype=np.complex128 )
            self.Lambda[s] = np.zeros( self.D[s] )

        # Default gauge location is the rightmost edge, so that reset will bring the whole
        # MPS into canonical form
        self.gauge_location = self.L - 1

    def orthogonalize(self, to = "left", truncate = False):
        """ Move gauge to either end of system 
            So far, only finite system.
        """

        if to == "left":
            # Move isometric gauge to first site
            for s in range(self.gauge_location,0,-1):
                self.move_gauge_left(s, truncate)

        if to == "right":
            # Move isometric gauge to last site
            for s in range(self.gauge_location,self.L):
                self.move_gauge_right(s, truncate)

    def set_random_state(self):
        """ Randomize MPS matrices, and fully left-orthogonalize """
         # Set matrices
        self.M = {}
        self.Lambda = {}
        self.d = np.array([int(d) for d in self.d])
        self.D = np.array([int(d) for d in self.D])

        for s in range(self.L):
            self.M[s]      = np.random.rand( self.d[s], self.D[s-1], self.D[s] ) + 1j*np.random.rand( self.d[s], self.D[s-1], self.D[s] )
            self.Lambda[s] = np.zeros( self.D[s] )

        # Normalize
        for s in range(self.L):
            norm = np.sqrt( np.tensordot( self.M[s].conj(), self.M[s], axes=((0,1,2),(0,1,2)) ) )
            self.M[s] /= norm

        self.orthogonalize()
  
    def set_product_state(self, state):
        """ Set MPS matrices to a product state, and fully left-orthogonalize """

        self.D = np.ones( self.L )
        # Set matrices
        self.M = {}
        self.Lambda = {}

        self.d = np.array([int(d) for d in self.d])
        self.D = np.array([int(d) for d in self.D])

        for s in range(self.L):
            self.M[s]      = np.zeros( (self.d[s], self.D[s-1], self.D[s]), dtype = np.complex128 )
            self.Lambda[s] = np.zeros( self.D[s] )

        # Set the i'th Gamma to have s particles
        for i,s in enumerate(state):
            self.M[i][int(s),0,0]  = 1
            self.D[i]              = 1
            self.Lambda[i]         = 1

        self.orthogonalize()

    def __add__(self, other):
        """ Add two MPS objects and truncate result """
        # Sanity checks
        if self.L != other.L:
            raise RuntimeError("Cannot sum two MPS's of different length!")
        if not np.all(self.d == other.d):
            raise RuntimeError("Cannot sum two MPS's with different local spaces!")

        # make new MPS that will store the result (but will have twice as large bond dim)
        A   = copy.deepcopy(self); A.orthogonalize('right')
        B   = copy.deepcopy(other); B.orthogonalize('right')

        # New sizes for the resulting matrix
        bond_dimensions = [x+y for (x,y) in zip(A.D,B.D)]
        # These bond dimensions need a small modification, since the first and last matrices
        # are actually row and column vectors
        bond_dimensions[-1] = 1
        # Now construct an empty MPS
        res = mps( A.L, A.d, bond_dimensions, A.type, A.threshold )

        # For the first row vector
        for l in np.arange(A.d[0]):
            res.M[0][l] = np.concatenate( [A.M[0][l], B.M[0][l]], axis=1 )
        # Last site as column vector
        for l in np.arange(A.d[A.L-1]):
            res.M[res.L-1][l] = np.concatenate( [A.M[A.L-1][l], B.M[B.L-1][l]], axis=0 )
        # All other sides as direct sum
        for s in range(1,A.L-1):
            for l in np.arange(A.d[s]):
                res.M[s][l][:A.M[s][l].shape[0], :A.M[s][l].shape[1]] = A.M[s][l]
                res.M[s][l][A.M[s][l].shape[0]:, A.M[s][l].shape[1]:] = B.M[s][l]
    
        # Bring into canonical form
        res.orthogonalize( 'left', truncate = True ) 
        return res

    def move_gauge_left(self, s, truncate=False):
        """ Move the isometric gauge point to site s-1. """
        # Group physical leg with right leg
        tmp = np.reshape( np.transpose( self.M[s], (1,0,2) ), (self.M[s].shape[1], self.M[s].shape[0]*self.M[s].shape[2]) )
        # SVD and store new bond dimension
        U, S, V = svd( tmp, full_matrices=False )
        
        # Possibly truncate and normalize
        self.D[s-1] = len(S)
        if truncate:
            self.D[s-1] = np.sum( S > self.threshold )
            S = S[:self.D[s-1]] / np.linalg.norm( S[:self.D[s-1]] )
            V = V[:self.D[s-1], :]
            U = U[:, :self.D[s-1]]

        # Store singular values
        self.Lambda[s-1] = S
        # Reshape V into new local M
        self.M[s] = np.transpose(np.reshape( V, (len(S), self.M[s].shape[0], self.M[s].shape[2]) ), (1,0,2))

        # Prepare prevous matrix
        if s > 0:
            self.M[s-1] = np.tensordot( self.M[s-1][:,:,:], np.dot(U, np.diag(S)), axes=(2,0) )

            # Update gauge center
            self.gauge_location -= 1

    def move_gauge_right(self, s, truncate=False):
        """ Move the isometric gauge point to site s+1. """
        # Group physical leg with left leg
        tmp = np.reshape( self.M[s], (self.M[s].shape[0]*self.M[s].shape[1], self.M[s].shape[2]) )
        # SVD and store new bond dimension
        U, S, V = svd( tmp, full_matrices=False )

        # Possibly truncate and normalize
        self.D[s] = len(S)
        if truncate:
            self.D[s] = np.sum( S > self.threshold )
            S = S[:self.D[s]] / np.linalg.norm( S[:self.D[s]] )
            V = V[:self.D[s], :]
            U = U[:, :self.D[s]]

        # Store singular values
        self.Lambda[s] = S
        # Reshape U into new local M
        self.M[s] = np.reshape( U, (self.M[s].shape[0], self.M[s].shape[1], len(S)) )

        # Prepare next matrix
        if s < self.L-1:
            self.M[s+1] = np.transpose( np.tensordot( np.dot(np.diag(S),V), self.M[s+1], axes=(1,1) ), (1,0,2) )

            # Update gauge center
            self.gauge_location += 1

    def measure( self, ops, sites ):
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
        # Turn Ops and sites into a list for convenience
        if len(np.array(ops).shape) != 3:
            ops = np.array([ops])
        if not isinstance(sites, (list, np.ndarray)):
            sites = np.array([sites])

        if len(ops) != len(sites):
            raise RuntimeError("Trying to measure more ops than sites or vice versa!")

        # Sort arguments from leftmost site to right
        sorted_indices  = np.argsort( sites )
        sites           = np.array(sites)[sorted_indices]
        ops             = np.array(ops)[sorted_indices]

        # Keep track of current_site and current_op, so that we know if we should
        # insert an operator or just contract as if identity operator was specified.
        current_site = sites[0]
        current_op   = 0

        # Gauge check
        if self.gauge_location != current_site:
            raise Exception("Measuring with gauge center at different location!")

        tmp = np.eye( self.M[current_site].shape[1] )
        while current_site != sites[-1] + 1:
            # Add one 'transfer matrix'
            tmp = np.tensordot( tmp, self.M[current_site].conj(), axes=(0,1) ) #(u d)(p l r) -> (d p r)

            # Insert operator if there is one
            if current_site in sites:
                tmp = np.transpose( np.tensordot( tmp, ops[current_op], axes=(1,0) ), (0,2,1) ) #(d p r)(a b) -> (d r b) -> (d b r)
                current_op += 1
    
            # Contract on bottom
            tmp = np.tensordot( tmp, self.M[current_site], axes=((0,1),(1,0)) )
            current_site += 1

        # Contract on right to close contraction
        tmp = np.trace( tmp )
        return tmp

class dmrgSim:
    def __init__(self, mps, hamiltonian):
        self.mps = mps
        self.ham = hamiltonian

        # Initialize left environment
        self.Lenv       = {}
        self.Lenv[0]    = np.zeros( (self.mps.M[0].shape[1], self.mps.M[0].shape[1]), dtype=np.complex128 )
        self.Lenv[1]    = np.zeros( (self.mps.M[1].shape[1], self.mps.M[1].shape[1]), dtype=np.complex128 )
        # Initialize right environment
        self.Renv       = {}
        self.Renv[self.mps.L-1]  = np.zeros( (self.mps.M[self.mps.L-1].shape[2], self.mps.M[self.mps.L-1].shape[2]), dtype=np.complex128 )
        self.Renv[self.mps.L-2]  = np.zeros( (self.mps.M[self.mps.L-2].shape[2], self.mps.M[self.mps.L-2].shape[2]), dtype=np.complex128 )

        # Update environments once
        for s in range(self.mps.L-1):
            self.mps.move_gauge_right( s )
            self.update_left_environment( s+1 )
        for s in range(self.mps.L-1, 0, -1):
            self.mps.move_gauge_left( s )
            self.update_right_environment( s-1 )

    def update_left_environment( self, s ):
        """ Update the left environment of site s. """
        self.Lenv[s] = np.tensordot( self.Lenv[s-1], self.mps.M[s-1].conj(), axes=(0,1) )
        self.Lenv[s] = np.tensordot( self.Lenv[s], self.mps.M[s-1], axes=((0,1),(1,0)) )

        # Two-site operators
        if s > 1:
            for t in range( self.ham[s-2].shape[0] ):
                if len( self.ham[s-2][t] ) == 2:
                    tmp = np.tensordot( self.mps.M[s-2].conj(), self.ham[s-2][t][0], axes=(0,0) )
                    tmp = np.tensordot( tmp, self.mps.M[s-2], axes=((0,2),(1,0)) ) 
                    tmp = np.tensordot( tmp, self.mps.M[s-1].conj(), axes=(0, 1) )
                    tmp = np.tensordot( tmp, self.ham[s-2][t][1], axes=(1,0) )
                    tmp = np.tensordot( tmp, self.mps.M[s-1], axes=((0,2),(1,0)) )
                    self.Lenv[s] += tmp

        # Three-site operators
        if s > 2:
            for t in range( self.ham[s-3].shape[0] ):
                if len( self.ham[s-3][t] ) == 3:
                    # Operator on first site
                    tmp = np.tensordot( self.mps.M[s-3].conj(), self.ham[s-3][t][0], axes=(0,0) )
                    tmp = np.tensordot( tmp, self.mps.M[s-3], axes=((0,2),(1,0)) ) 
                    # Second site operator
                    tmp = np.tensordot( tmp, self.mps.M[s-2].conj(), axes=(0, 1) )
                    tmp = np.tensordot( tmp, self.ham[s-3][t][1], axes=(1,0) )
                    tmp = np.tensordot( tmp, self.mps.M[s-2], axes=((0,2),(1,0)) )
                    # Third site operator
                    tmp = np.tensordot( tmp, self.mps.M[s-1].conj(), axes=(0, 1) )
                    tmp = np.tensordot( tmp, self.ham[s-3][t][2], axes=(1,0) )
                    tmp = np.tensordot( tmp, self.mps.M[s-1], axes=((0,2),(1,0)) )
                    # Add to left environment of site s
                    self.Lenv[s] += tmp

    def update_right_environment( self, s ):
        """ Update the right environment of site s. """
        self.Renv[s] = np.tensordot( self.mps.M[s+1].conj(), self.Renv[s+1], axes=(2,0) ) # (d lu ru)(up dwn) -> (d lu dwn)
        self.Renv[s] = np.tensordot( self.mps.M[s+1], self.Renv[s], axes=( (0,2), (0,2) ) ) #(d ld rd)(d lu dwn) -> (ld lu)
        self.Renv[s] = np.transpose( self.Renv[s] ) #(up down)

        # Two site operators
        if s < self.mps.L-2:
            for t in range(self.ham[s+1].shape[0]):
                if len( self.ham[s+1][t] ) == 2:
                    tmp = np.tensordot( self.ham[s+1][t][1], self.mps.M[s+2].conj(), axes=(0,0) ) #(a b)(a lu ru)
                    tmp = np.tensordot( tmp, self.mps.M[s+2], axes=( (0,2), (0,2) )) #(b lu ru)(d ld rd) -> (lu ld)
                    tmp = np.tensordot( self.mps.M[s+1].conj(), tmp, axes=(2,0) ) #(d lu ru)(lu ld) -> (d lu ld)
                    tmp = np.tensordot( tmp, self.ham[s+1][t][0], axes=(0,0) ) #(d lu ld)(a b) -> (lu ld b)
                    tmp = np.tensordot( tmp, self.mps.M[s+1], axes=( (1,2), (2,0) ) ) #(lu ld b)(d ld rd) -> (lu ld)
                    self.Renv[s] += tmp
           
        # Three site operators
        if s < self.mps.L-3:
            for t in range(self.ham[s+1].shape[0]):
                if len( self.ham[s+1][t] ) == 3:
                    # Operator on first site
                    tmp = np.tensordot( self.ham[s+1][t][2], self.mps.M[s+3].conj(), axes=(0,0) ) #(a b)(a lu ru)
                    tmp = np.tensordot( tmp, self.mps.M[s+3], axes=( (0,2), (0,2) )) #(b lu ru)(d ld rd) -> (lu ld)
                    # Second site operator
                    tmp = np.tensordot( self.mps.M[s+2].conj(), tmp, axes=(2,0) ) #(d lu ru)(lu ld) -> (d lu ld)
                    tmp = np.tensordot( tmp, self.ham[s+1][t][1], axes=(0,0) ) #(d lu ld)(a b) -> (lu ld b)
                    tmp = np.tensordot( tmp, self.mps.M[s+2], axes=( (1,2), (2,0) ) ) #(lu ld b)(d ld rd) -> (lu ld)
                    # Third site operator
                    tmp = np.tensordot( self.mps.M[s+1].conj(), tmp, axes=(2,0) )
                    tmp = np.tensordot( tmp, self.ham[s+1][t][0], axes=(0,0) )
                    tmp = np.tensordot( tmp, self.mps.M[s+1], axes=( (1,2), (2,0) ) )
                    # Add to right environment of site s
                    self.Renv[s] += tmp

    def minimize_energy( self, s, k, ncv ):
        """ Minimize the energy by optimizing the tensor on site s. 
        
            TODO: Efficiently encode action of 'M' on matrix, never compute M at all 
        """

        def matvec( x ):
            # Reshape back into tensor
            x = np.reshape( x, (self.mps.M[s].shape[0], self.mps.M[s].shape[1], self.mps.M[s].shape[2]) )

            # Pure environments, with no part of the Hamiltonian acting on site s
            v  = np.tensordot( self.Lenv[s], x, axes=(1,1) ) #(alpha, alpha')(d alpha' beta) -> (alpha d beta)
            v  = np.transpose( v, (1, 0, 2) ) # (d alpha beta)
            v += np.tensordot( x, self.Renv[s], axes=(2,1) ) #(d alpha beta')(beta beta') -> (d alpha beta)

            #----------------------------------------------------------------------
            # Mixed environment terms (only occur for 3 body-operators)
            #----------------------------------------------------------------------
            if s > 0 and s < self.mps.L-1:
                # Only if there is a three site term on one site to the left
                for t in range( self.ham[s-1].shape[0] ):
                    if len( self.ham[s-1][t] ) == 3:
                        # Transfermatrix with A
                        tmpL = np.tensordot( self.mps.M[s-1].conj(), self.ham[s-1][t][0], axes=(0,0) ) #(d alpha beta)(Aup Adown) -> (alpha beta Adown)
                        tmpL = np.tensordot( tmpL, self.mps.M[s-1], axes=( (0,2),(1,0) ) ) #(alpha beta Adown)(d alpha' beta') -> (beta beta')
                        # Transfermatrix with C
                        tmpR = np.tensordot( self.ham[s-1][t][2], self.mps.M[s+1].conj(), axes=(0,0) )
                        tmpR = np.tensordot( tmpR, self.mps.M[s+1], axes=((0,2),(0,2)) ) #(delta delta')
                        # Combine 
                        tmp = np.tensordot( tmpL, x, axes=(1,1) ) #(beta beta')(d beta' gamma') -> (beta d gamma')
                        tmp = np.tensordot( tmp, self.ham[s-1][t][1], axes=(1,1) ) #(beta d gamma')(Bup Bdown) -> (beta gamma' Bup)
                        tmp = np.tensordot( tmp, tmpR, axes=(1,1) ) #(beta gamma' Bup)(delta delta') -> (beta Bup delta)
                        v  += np.transpose( tmp, (1,0,2) )

            # Two-site operators with one operator on site s-1 and one on site s
            if s > 0:
                for t in range( self.ham[s-1].shape[0] ):
                    if len( self.ham[s-1][t] ) == 2:
                        # Add A
                        tmp = np.tensordot( self.mps.M[s-1], self.ham[s-1][t][0], axes=(0,1) ) #(d alpha beta)(u d) -> (alpha beta u)
                        tmp = np.tensordot( tmp, self.mps.M[s-1].conj(), axes=((0,2),(1,0)) ) #(alpha beta u)(d alpha' beta') -> (beta beta')
                        tmp = np.tensordot( tmp, x, axes=(0,1) ) #(beta beta')(d alpha' beta) -> (beta' d beta)
                        tmp = np.tensordot( tmp, self.ham[s-1][t][1], axes=(1,1) ) #(beta' d beta)(u d) -> (beta' beta u)
                        v += np.transpose( tmp, (2,0,1) )

            # Three-site operators with two operators on site s-2 and s-1, and one on site s
            if s > 1:
                for t in range( self.ham[s-2].shape[0] ):
                    if len( self.ham[s-1][t] ) == 3:
                        # Add A
                        tmp = np.tensordot( self.mps.M[s-2].conj(), self.ham[s-2][t][0], axes=(0,0) ) #(p l r)(u d) -> (l r d)
                        tmp = np.tensordot( tmp, self.mps.M[s-2], axes=( (0,2),(1,0) ) ) # (r rbottom)

                        # Add B
                        tmp = np.tensordot( tmp, self.mps.M[s-1].conj(), axes=(0,1) ) #(r b)(d l r) -> (b d r)
                        tmp = np.tensordot( tmp, self.ham[s-2][t][1], axes=(1,0) ) #(b d rup)(x y) -> (b rup y)
                        tmp = np.tensordot( tmp, self.mps.M[s-1], axes=( (0,2), (1,0) ) ) #(b rup y)(d l rbottom) -> (rup rbottom)
    
                        # Add x and C
                        tmp = np.tensordot( tmp, x, axes=(1,1) ) #(rup rbottom)(d l r) -> (rup d r)
                        tmp = np.tensordot( tmp, self.ham[s-2][t][2], axes=(1,1) ) #(rup d r)(Cup Cdown) -> (rup r Cup)
                        v  += np.transpose( tmp, (2,0,1) )

            # Two-site operators with one operator on site s and one on site s+1
            if s < self.mps.L-1:
                for t in range(self.ham[s].shape[0]):
                    if len( self.ham[s][t] ) == 2:
                        # Add C
                        tmp = np.tensordot( self.ham[s][t][1], self.mps.M[s+1], axes=(1, 0) ) #(u d)(d alpha beta) -> (u alpha beta)
                        tmp = np.tensordot( tmp, self.mps.M[s+1].conj(), axes=( (0,2),(0,2) ) ) #(u alpha beta)(d alpha' beta') -> (alpha alpha')
                        tmp = np.tensordot( x, tmp, axes=(2,0) ) #(d alpha beta')(alpha alpha') -> (d alpha alpha')
                        tmp = np.tensordot( self.ham[s][t][0], tmp, axes=(1,0) ) #(u d)(d alpha alpha') -> (u alpha alpha')
                        v += tmp

            # Three-site operators with two operators on site s+2 and s+1, and one on site s
            if s < self.mps.L-2:
                for t in range(self.ham[s].shape[0]):
                    if len( self.ham[s][t] ) == 3:
                        # Add C
                        tmp = np.tensordot( self.ham[s][t][2], self.mps.M[s+2].conj(), axes=(0, 0) )
                        tmp = np.tensordot( tmp, self.mps.M[s+2], axes=((0,2),(0,2))) 
                        # Add B
                        tmp = np.tensordot( self.mps.M[s+1].conj(), tmp, axes=(2,0) ) #(d l r)(t b) -> (d lup b)
                        tmp = np.tensordot( tmp, self.ham[s][t][1], axes=(0,0) ) #(d lup b)(x y) -> (lup b y)
                        tmp = np.tensordot( tmp, self.mps.M[s+1], axes=( (1,2), (2,0) ) ) #(lup b y)(d lbot rb) -> (lup lbot)
                        # Add x and A
                        tmp = np.tensordot( x, tmp, axes=(2,1) ) #(d l r)(lup lbot) -> (d l lup)
                        tmp = np.tensordot( self.ham[s][t][0], tmp, axes=(1,0) ) #(Aup Adown)(d l lup) -> (Aup l lup)
                        v  += tmp

            return np.reshape( v, (self.mps.M[s].shape[0] * self.mps.M[s].shape[1] * self.mps.M[s].shape[2]) )

        # Construct linear operator to be used for Lanczos diag
        dim = self.mps.d[s] * self.mps.M[s].shape[1] * self.mps.M[s].shape[2]
        A = sp.sparse.linalg.LinearOperator( (dim,dim), matvec, dtype=np.complex128 )

        # Make sure we don't ask for more than possible
        k = np.max( [np.min( [k, self.mps.M[s].shape[1] - 1, self.mps.M[s].shape[2] - 1] ), 1] )

        # If we are only dealing with a 2x2 matrix, sparse diag with k = 1 will complain about rank
        # So we do full
        if dim <= 2:
            vecs = np.array([ [1,0], [0,1] ])
            entry = np.array([ [np.dot( vecs[i], A.matvec( vecs[j] ) ) for i in range(2)] for j in range(2) ])
            eigval, eigvec = np.linalg.eigh(entry)
        else:
            eigval, eigvec = sp.sparse.linalg.eigsh( A, k=k, which='SA', ncv=ncv, v0 = np.reshape( self.mps.M[s], dim ) )
       
        # Sort eigenvalues and corresponding eigenvectors so we can pick the lowest
        sorted_indices = np.argsort( eigval )
        eigval = eigval[sorted_indices]
        eigvec = eigvec[:, sorted_indices]

        # Reshape lowest eigenvector into new A[s] tensor
        self.mps.M[s] = np.reshape( eigvec[:,0], (self.mps.M[s].shape[0], self.mps.M[s].shape[1], self.mps.M[s].shape[2]) )
        return eigval[0]

    def full_sweep( self, k=1, ncv=None ):
        """ Perform full sweep, assuming we start from left end. """

        # Empty list to keep track of energy during sweep
        E       = []

        # Sweep right
        for s in range(self.mps.L-1):
            E.append( self.minimize_energy( s, k=k, ncv=ncv ) )
            self.mps.move_gauge_right( s )
            self.update_left_environment( s+1 )

        # Sweep left
        for s in range(self.mps.L-1, 0, -1):
            E.append( self.minimize_energy( s, k=k, ncv=ncv ) )
            self.mps.move_gauge_left( s )
            self.update_right_environment( s-1 )

        return E
