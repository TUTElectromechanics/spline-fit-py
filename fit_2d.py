#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:45:36 2017

Python port of fit_2d.m by Paavo Rasilo.

Tests with bivariate B-spline fitting and differentiation.

The aim is to fit the spline against data that is available only for the partial derivatives of the function.

This effectively finds, in the least-squares sense, the best scalar potential that generates the given partial derivatives.

@author: Juha Jeronen, juha.jeronen@tut.fi
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.linalg

import scipy.io
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

# pip install bspline
# or  https://github.com/johntfoster/bspline
#
import bspline
import bspline.splinelab as splinelab

import util.index
import util.plot


def main():
    ######################
    # Config
    ######################

    # Choose least-squares solver to use:
    #
#    lsq_solver = "dense"  # LAPACK DGELSD, direct, good for small problems
#    lsq_solver = "sparse"  # SciPy LSQR, iterative, asymptotically faster, good for large problems
#    lsq_solver = "optimize"  # general nonlinear optimizer using Trust Region Reflective (trf) algorithm
#    lsq_solver = "qr"
#    lsq_solver = "cholesky"
#    lsq_solver = "sparse_qr"
    lsq_solver = "sparse_qr_solve"

    ######################
    # Load multiscale data
    ######################

    print( "Loading measurement data..." )

    # measurements are provided on a meshgrid over (Hx, sigxx)

    # data2.mat contains virtual measurements, generated from a multiscale model.

#    data2 = scipy.io.loadmat("data2.mat")
#    Hx    = np.squeeze(data2["Hx"])     # 1D array, (M,)
#    sigxx = np.squeeze(data2["sigxx"])  # 1D array, (N,)
#    Bx    = data2["Bx"]                 # 2D array, (M, N)
#    lamxx = data2["lamxx"]              #       --"--
##    lamyy = data2["lamyy"]              #       --"--
##    lamzz = data2["lamzz"]              #       --"--

    data2 = scipy.io.loadmat("umair_gal_denoised.mat")
    sigxx = -1e6*np.array( [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80][::-1], dtype=np.float64 )
    assert sigxx.shape[0] == 18
    Hx    = data2["Hval"][0,:]  # same for all sigma, just take the first row
    Bx    = data2["Bval"].T
    lamxx = data2["LHval"].T * 1e-6
    Bx    = Bx[::-1,:]
    lamxx = lamxx[::-1,:]

    # HACK, fix later (must decouple number of knots from number of data sites)
    ii     = np.arange(Hx.shape[0])
    n_newi = 401
    newii  = np.linspace(0, ii[-1], n_newi)
    nsigma = sigxx.shape[0]
    fH     = scipy.interpolate.interp1d( ii, Hx )
    newB   = np.empty( (n_newi, Bx.shape[1]),    dtype=np.float64 )
    newlam = np.empty( (n_newi, lamxx.shape[1]), dtype=np.float64 )
    for j in range(nsigma):
        fB          = scipy.interpolate.interp1d( ii, Bx[:,j] )
        newB[:,j]   = fB(newii)

        flam        = scipy.interpolate.interp1d( ii, lamxx[:,j] )
        newlam[:,j] = flam(newii)
    Hx    = fH(newii)
    Bx    = newB
    lamxx = newlam


    # Order of spline (as-is! 3 = cubic)
    ordr = 3

    # Auxiliary variables (H, sig_xx, sig_xy)
    Hscale = np.max(Hx)
    sscale = np.max(np.abs(sigxx))
    x      = Hx / Hscale
    y      = sigxx / sscale
    nx     = x.shape[0]  # number of grid points, x axis
    ny     = y.shape[0]  # number of grid points, y axis

    # Partial derivatives (B, lam_xx, lam_xy) from multiscale model
    #
    # In the magnetostriction components, the multiscale model produces nonzero lamxx at zero stress.
    # We normalize this away for purposes of performing the curve fit.
    #
    dpsi_dx = Bx * Hscale
    dpsi_dy = (lamxx - lamxx[0,:]) * sscale

    ######################
    # Set up splines
    ######################

    print( "Setting up splines..." )

    # The evaluation algorithm used in bspline.py uses half-open intervals  t_i <= x < t_{i+1}.
    #
    # This causes havoc for evaluation at the end of each interval, because it is actually the start
    # of the next interval.
    #
    # Especially, the end of the last interval is the start of the next (non-existent) interval.
    #
    # We work around this by using a small epsilon to avoid evaluation exactly at t_{i+1} (for the last interval).
    #
    def marginize_end(x):
        out      = x.copy()
        out[-1] += 1e-10 * (x[-1] - x[0])
        return out

    # create knots and spline basis
    xknots = splinelab.aptknt( marginize_end(x), ordr )
    yknots = splinelab.aptknt( marginize_end(y), ordr )
    splx   = bspline.Bspline(xknots, ordr)
    sply   = bspline.Bspline(yknots, ordr)

    # get number of basis functions (perform dummy evaluation and count)
    nxb = len( splx(0.) )
    nyb = len( sply(0.) )

    # TODO Check if we need to convert input Bx and sigxx to u,v (what is actually stored in the data files?)

    # Create collocation matrices:
    #
    #   A[i,j] = d**deriv_order B_j(tau[i])
    #
    # where d denotes differentiation and B_j is the jth basis function.
    #
    # We place the collocation sites at the points where we have measurements.
    #
    Au = splx.collmat(x)
    Av = sply.collmat(y)
    Du = splx.collmat(x, deriv_order=1)
    Dv = sply.collmat(y, deriv_order=1)

    ######################
    # Assemble system
    ######################

    print( "Assembling system..." )

    # Assemble the equation system for fitting against data on the partial derivatives of psi.
    #
    # By writing psi in the spline basis,
    #
    #   psi_{ij}       = A^{u}_{ik} A^{v}_{jl} c_{kl}
    #
    # the quantities to be fitted, which are the partial derivatives of psi, become
    #
    #   B_{ij}         = D^{u}_{ik} A^{v}_{jl} c_{kl}
    #   lambda_{xx,ij} = A^{u}_{ik} D^{v}_{jl} c_{kl}
    #
    # Repeated indices are summed over.
    #
    # Column: kl converted to linear index (k = 0,1,...,nxb-1,  l = 0,1,...,nyb-1)
    # Row:    ij converted to linear index (i = 0,1,...,nx-1,   j = 0,1,...,ny-1)
    #
    # (Paavo's notes, Stresses4.pdf)

    nf = 2      # number of unknown fields
    nr = nx*ny  # equation system rows per unknown field
    A  = np.empty( (nf*nr, nxb*nyb), dtype=np.float64 )  # global matrix
    b  = np.empty( (nf*nr),          dtype=np.float64 )  # global RHS

    # zero array element detection tolerance
    tol = 1e-6

    I,J,IJ = util.index.genidx( (nx, ny)  )
    K,L,KL = util.index.genidx( (nxb,nyb) )

    # loop only over rows of the equation system
    for i,j,ij in zip(I,J,IJ):
        A[nf*ij,  KL] = Du[i,K] * Av[j,L]
        A[nf*ij+1,KL] = Au[i,K] * Dv[j,L]

    b[nf*IJ]   = dpsi_dx[I,J]  # RHS for B_x
    b[nf*IJ+1] = dpsi_dy[I,J]  # RHS for lambda_xx

#    # the above is equivalent to this much slower version:
#    #
#    # equation system row
#    for j in range(ny):
#        for i in range(nx):
#            ij = np.ravel_multi_index( (i,j), (nx,ny) )
#
#            # equation system column
#            for l in range(nyb):
#                for k in range(nxb):
#                    kl = np.ravel_multi_index( (k,l), (nxb,nyb) )
#                    A[nf*ij,  kl] = Du[i,k] * Av[j,l]
#                    A[nf*ij+1,kl] = Au[i,k] * Dv[j,l]
#
#            b[nf*ij]   = dpsi_dx[i,j] if abs(dpsi_dx[i,j]) > tol else 0.  # RHS for B_x
#            b[nf*ij+1] = dpsi_dy[i,j] if abs(dpsi_dy[i,j]) > tol else 0.  # RHS for lambda_xx

    ######################
    # Solve
    ######################

    # Solve the optimal coefficients.

    # Note that we are constructing a potential function from partial derivatives only,
    # so the solution is unique only up to a global additive shift term.
    #
    # Under the hood, numpy.linalg.lstsq uses LAPACK DGELSD:
    #
    #   http://stackoverflow.com/questions/29372559/what-is-the-difference-between-numpy-linalg-lstsq-and-scipy-linalg-lstsq
    #
    # DGELSD accepts also rank-deficient input (rank(A) < min(nrows,ncols)), returning  arg min( ||x||_2 ) ,
    # so we don't need to do anything special to account for this.
    #
    # Same goes for the sparse LSQR.

    # equilibrate row and column norms
    #
    # See documentation of  scipy.sparse.linalg.lsqr,  it requires this to work properly.
    #
    # https://github.com/Technologicat/python-wlsqm
    #
    print( "Equilibrating..." )
    S = A.copy(order='F')  # the rescaler requires Fortran memory layout
    A = scipy.sparse.csr_matrix(A)  # save memory (dense "A" no longer needed)

#    eps = 7./3. - 4./3. - 1  # http://stackoverflow.com/questions/19141432/python-numpy-machine-epsilon
#    print( S.max() * max(S.shape) * eps )  # default zero singular value detection tolerance in np.linalg.matrix_rank()

#    import wlsqm.utils.lapackdrivers as wul
#    rs,cs = wul.do_rescale( S, wul.ScalingAlgo.ALGO_DGEEQU )

#    # row scaling only (for weighting)
#    with np.errstate(divide='ignore', invalid='ignore'):
#        rs = np.where( np.abs(b) > tol, 1./b, 1. )
#    for i in range(S.shape[0]):
#        S[i,:] *= rs[i]
#    cs = 1.

    # scale rows corresponding to Bx
    #
    rs = np.ones_like(b)
    rs[nf*IJ] = 2
    for i in range(S.shape[0]):
        S[i,:] *= rs[i]
    cs = 1.

#    # It seems this is not needed in the 2D problem (fitting error is slightly smaller without it).
#
#    # Additional row scaling.
#    #
#    # This equilibrates equation weights, but deteriorates the condition number of the matrix.
#    #
#    # Note that in a least-squares problem the row weighting *does* matter, because it affects
#    # the fitting error contribution from the rows.
#    #
#    with np.errstate(divide='ignore', invalid='ignore'):
#        rs2 = np.where( np.abs(b) > tol, 1./b, 1. )
#    for i in range(S.shape[0]):
#        S[i,:] *= rs2[i]
#    rs *= rs2

#    a = np.abs(rs2)
#    print( np.min(a), np.mean(a), np.max(a) )

#    rs = np.asanyarray(rs)
#    cs = np.asanyarray(cs)
#    a = np.abs(rs)
#    print( np.min(a), np.mean(a), np.max(a) )

    b *= rs  # scale RHS accordingly

#    colnorms = np.linalg.norm(S, ord=np.inf, axis=0)  # sum over rows    -> column norms
#    rownorms = np.linalg.norm(S, ord=np.inf, axis=1)  # sum over columns -> row    norms
#    print( "    rescaled column norms min = %g, avg = %g, max = %g" % (np.min(colnorms), np.mean(colnorms), np.max(colnorms)) )
#    print( "    rescaled row    norms min = %g, avg = %g, max = %g" % (np.min(rownorms), np.mean(rownorms), np.max(rownorms)) )

    print( "Solving with algorithm = '%s'..." % (lsq_solver) )
    if lsq_solver == "dense":
        print( "    matrix shape %s = %d elements" % (S.shape, np.prod(S.shape)) )
        ret = numpy.linalg.lstsq(S, b)  # c,residuals,rank,singvals
        c = ret[0]

    elif lsq_solver == "sparse":
        S = scipy.sparse.coo_matrix(S)
        print( "    matrix shape %s = %d elements; %d nonzeros (%g%%)" % (S.shape, np.prod(S.shape), S.nnz, 100. * S.nnz / np.prod(S.shape) ) )

        ret = scipy.sparse.linalg.lsqr( S, b )
        c,exit_reason,iters = ret[:3]
        if exit_reason != 2:  # 2 = least-squares solution found
            print( "WARNING: solver did not converge (exit_reason = %d)" % (exit_reason) )
        print( "    sparse solver iterations taken: %d" % (iters) )

    elif lsq_solver == "optimize":
        # make sparse matrix (faster for dot products)
        S = scipy.sparse.coo_matrix(S)
        print( "    matrix shape %s = %d elements; %d nonzeros (%g%%)" % (S.shape, np.prod(S.shape), S.nnz, 100. * S.nnz / np.prod(S.shape) ) )

        def fitting_error(c):
            return S.dot(c) - b
        ret = scipy.optimize.least_squares( fitting_error, np.ones(S.shape[1], dtype=np.float64), method="trf", loss="linear" )

        c = ret.x
        if ret.status < 1:
            # status codes: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.least_squares.html
            print( "WARNING: solver did not converge (status = %d)" % (ret.status) )

    elif lsq_solver == "qr":
        print( "    matrix shape %s = %d elements" % (S.shape, np.prod(S.shape)) )
        # http://glowingpython.blogspot.fi/2012/03/solving-overdetermined-systems-with-qr.html
        Q,R = np.linalg.qr(S) # qr decomposition of A
        Qb = (Q.T).dot(b) # computing Q^T*b (project b onto the range of A)
#        c = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
        c = scipy.linalg.solve_triangular(R, Qb, check_finite=False)

    elif lsq_solver == "cholesky":
        # S is rank-deficient by one, because we are solving a potential based on data on its partial derivatives.
        #
        # Before solving, force S to have full rank by fixing one coefficient.
        #
        S[0,:] = 0.
        S[0,0] = 1.
        b[0]   = 1.
        rs[0]  = 1.
        S = scipy.sparse.csr_matrix(S)
        print( "    matrix shape %s = %d elements; %d nonzeros (%g%%)" % (S.shape, np.prod(S.shape), S.nnz, 100. * S.nnz / np.prod(S.shape) ) )

        # Be sure to use the new sksparse from
        #
        #   https://github.com/scikit-sparse/scikit-sparse
        #
        # instead of the old scikits.sparse (which will fail with an error).
        #
        # Requires libsuitesparse-dev for CHOLMOD headers.
        #
        from sksparse.cholmod import cholesky_AAt
        # Notice that CHOLMOD computes AA' and we want M'M, so we must set A = M'!
        factor = cholesky_AAt(S.T)
        c = factor.solve_A(S.T * b)

    elif lsq_solver == "sparse_qr":
        # S is rank-deficient by one, because we are solving a potential based on data on its partial derivatives.
        #
        # Before solving, force S to have full rank by fixing one coefficient;
        # otherwise the linear solve step will fail because R will be exactly singular.
        #
        S[0,:] = 0.
        S[0,0] = 1.
        b[0]   = 1.
        rs[0]  = 1.
        S = scipy.sparse.coo_matrix(S)
        print( "    matrix shape %s = %d elements; %d nonzeros (%g%%)" % (S.shape, np.prod(S.shape), S.nnz, 100. * S.nnz / np.prod(S.shape) ) )

        # pip install sparseqr
        # or https://github.com/yig/PySPQR
        #
        # Works like MATLAB's [Q,R,e] = qr(...):
        #
        # https://se.mathworks.com/help/matlab/ref/qr.html
        #
        # [Q,R,E] = qr(A) or [Q,R,E] = qr(A,'matrix') produces unitary Q, upper triangular R and a permutation matrix E
        # so that A*E = Q*R. The column permutation E is chosen to reduce fill-in in R.
        #
        # [Q,R,e] = qr(A,'vector') returns the permutation information as a vector instead of a matrix.
        # That is, e is a row vector such that A(:,e) = Q*R.
        #
        import sparseqr
        print( "    performing sparse QR decomposition..." )
        Q, R, E, rank = sparseqr.qr( S )

        # produce reduced QR (for least-squares fitting)
        #
        # - cut away bottom part of R (zeros!)
        # - cut away the corresponding far-right part of Q
        #
        # see
        #    np.linalg.qr
        #    https://andreask.cs.illinois.edu/cs357-s15/public/demos/06-qr-applications/Solving%20Least-Squares%20Problems.html
        #
#        # inefficient way:
#        k = min(S.shape)
#        R = scipy.sparse.csr_matrix( R.A[:k,:] )
#        Q = scipy.sparse.csr_matrix( Q.A[:,:k] )

        print( "    reducing matrices..." )
        # somewhat more efficient way:
        k = min(S.shape)
        R = R.tocsr()[:k,:]
        Q = Q.tocsc()[:,:k]

#        # maybe somewhat efficient way: manipulate data vectors, create new coo matrix
#        #
#        # (incomplete, needs work; need to shift indices of rows/cols after the removed ones)
#        #
#        k    = min(S.shape)
#        mask = np.nonzero( R.row < k )[0]
#        R = scipy.sparse.coo_matrix( ( R.data[mask], (R.row[mask], R.col[mask]) ), shape=(k,k) )
#        mask = np.nonzero( Q.col < k )[0]
#        Q = scipy.sparse.coo_matrix( ( Q.data[mask], (Q.row[mask], Q.col[mask]) ), shape=(k,k) )

        print( "    solving..." )
        Qb = (Q.T).dot(b)
        x = scipy.sparse.linalg.spsolve(R, Qb)
        c = np.empty_like(x)
        c[E] = x[:]  # apply inverse permutation

    elif lsq_solver == "sparse_qr_solve":
        S[0,:] = 0.
        S[0,0] = 1.
        b[0]   = 1.
        rs[0]  = 1.
        S = scipy.sparse.coo_matrix(S)
        print( "    matrix shape %s = %d elements; %d nonzeros (%g%%)" % (S.shape, np.prod(S.shape), S.nnz, 100. * S.nnz / np.prod(S.shape) ) )

        import sparseqr
        c = sparseqr.solve( S, b )

    else:
        raise ValueError("unknown solver '%s'; valid: 'dense', 'sparse'" % (lsq_solver))

    c *= cs  # undo column scaling in solution

    # now c contains the spline coefficients, c_{kl}, where kl has been raveled into a linear index.

    ######################
    # Save
    ######################

    filename = "tmp_s2d.mat"
    L = locals()
    data = { key: L[key] for key in ["ordr", "xknots", "yknots", "c", "Hscale", "sscale"] }
    scipy.io.savemat(filename, data, format='5', oned_as='row')

    ######################
    # Plot
    ######################

    print( "Visualizing..." )

    # unpack results onto meshgrid
    #
    fitted  = A.dot(c)  # function values corresponding to each row in the global equation system
    X,Y     = np.meshgrid(Hx,sigxx, indexing='ij')   # indexed like X[i,j]  (i is x index, j is y index)
    Z_Bx    = np.empty_like(X)
    Z_lamxx = np.empty_like(X)

    Z_Bx[I,J]    = fitted[nf*IJ]
    Z_lamxx[I,J] = fitted[nf*IJ+1]

#    # the above is equivalent to:
#    for ij in range(nr):
#        i,j = np.unravel_index( ij, (nx,ny) )
#        Z_Bx[i,j]    = fitted[nf*ij]
#        Z_lamxx[i,j] = fitted[nf*ij+1]

    data_Bx = { "x" : (X, r"$H_{x}$"),
                "y" : (Y, r"$\sigma_{xx}$"),
                "z" : (Z_Bx / Hscale, r"$B_{x}$")
              }

    data_lamxx = { "x" : (X, r"$H_{x}$"),
                   "y" : (Y, r"$\sigma_{xx}$"),
                   "z" : (Z_lamxx / sscale, r"$\lambda_{xx}$")
                 }

    def relerr(data, refdata):
        refdata_linview = refdata.reshape(-1)
        return 100. * np.linalg.norm(refdata_linview - data.reshape(-1)) / np.linalg.norm(refdata_linview)

    plt.figure(1)
    plt.clf()
    ax = util.plot.plot_wireframe(data_Bx, legend_label="Spline", figno=1)
    ax.plot_wireframe( X, Y, dpsi_dx / Hscale, label="Multiscale", color="r" )
    plt.legend(loc="best")
    print( "B_x relative error %g%%" % ( relerr(Z_Bx, dpsi_dx) ) )

    plt.figure(2)
    plt.clf()
    ax = util.plot.plot_wireframe(data_lamxx, legend_label="Spline", figno=2)
    ax.plot_wireframe( X, Y, dpsi_dy / sscale, label="Multiscale", color="r" )
    plt.legend(loc="best")
    print( "lambda_{xx} relative error %g%%" % ( relerr(Z_lamxx, dpsi_dy) ) )

    # match the grid point numbering used in MATLAB version of this script
    #
    def t(A):
        return np.transpose(A, [1,0])
    dpsi_dx = t(dpsi_dx)
    Z_Bx    = t(Z_Bx)
    dpsi_dy = t(dpsi_dy)
    Z_lamxx = t(Z_lamxx)

    plt.figure(3)
    plt.clf()
    ax = plt.subplot(1,1, 1)
    ax.plot( dpsi_dx.reshape(-1) / Hscale, 'ro', markersize='2', label="Multiscale" )
    ax.plot( Z_Bx.reshape(-1) / Hscale,    'ko', markersize='2', label="Spline" )
    ax.set_xlabel("Grid point number")
    ax.set_ylabel(r"$B_{x}$")
    plt.legend(loc="best")

    plt.figure(4)
    plt.clf()
    ax = plt.subplot(1,1, 1)
    ax.plot( dpsi_dy.reshape(-1) / sscale, 'ro', markersize='2', label="Multiscale" )
    ax.plot( Z_lamxx.reshape(-1) / sscale, 'ko', markersize='2', label="Spline" )
    ax.set_xlabel("Grid point number")
    ax.set_ylabel(r"$\lambda_{xx}$")
    plt.legend(loc="best")

    print( "All done." )


if __name__ == '__main__':
    main()
    plt.show()
