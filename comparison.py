#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:29:47 2017

Python port of comparison.m by Paavo Rasilo.

@author: Juha Jeronen, juha.jeronen@tut.fi
"""

from __future__ import division, print_function, absolute_import

import os

try:
    import cPickle as pickle  # Python 2.7
except ImportError:
    import pickle  # Python 3.x+

import numpy as np

import scipy.io
import scipy.sparse

import sympy as sy

from freya.symutil import recursive_collect as collect

import bspline
import util.index


# Symbolic derivation of the required functions
Hx, Hy, Hz, sxx, syy, szz, sxy, syz, szx = sy.symbols( "Hx, Hy, Hz, sxx, syy, szz, sxy, syz, szx", real=True )
fargs = (Hx, Hy, Hz, sxx, syy, szz, sxy, syz, szx)  # parameters for callables (which are generated later)

symfile = "invariants.pickle"
if not os.path.isfile(symfile):  # slow, so we cache this
    H   = sy.Matrix( [Hx, Hy, Hz] )
    sig = sy.Matrix( [[sxx, sxy, szx],
                      [sxy, syy, syz],
                      [szx, syz, szz]] )

    s = sig - sy.ones( (3,3) ) * sig.trace() / sy.sympify("3")
    #s = s.applyfunc(sy.factor)

    I4 = (H.T * H)[0,0]  # extract scalar from matrix wrapper
    I5 = (H.T * s * H)[0,0]
    I6 = (H.T * s * s * H)[0,0]

    #I4 = collect(I4)
    #I5 = collect(I5)
    #I6 = collect(I6)

    I4 = sy.simplify(I4)
    I5 = sy.simplify(I5)
    I6 = sy.simplify(I6)

#    print( I4 )
#    print( I5 )
#    print( I6 )

#    vis = False
#    print( sy.count_ops(I4, visual=vis) )
#    print( sy.count_ops(I5, visual=vis) )
#    print( sy.count_ops(I6, visual=vis) )

    u = sy.sqrt(I4)
    v = collect( sy.sympify("3/2") * I5   / I4 )
    w = collect( sy.sqrt( I6*I4 - I5**2 ) / I4 )

#    print( sy.count_ops(u, visual=vis) )
#    print( sy.count_ops(v, visual=vis) )
#    print( sy.count_ops(w, visual=vis) )

    # save the SymPy expression objects into a datafile
    #
    L = locals()
    data = { key: L[key] for key in ["I4", "I5", "I6", "u", "v", "w"] }
    with open(symfile, 'wb') as f:
        pickle.dump( data, f, protocol=2 )

else:  # datafile exists, load it
    with open(symfile, 'rb') as f:
        data = pickle.load(f)
    I4 = data["I4"]
    I5 = data["I5"]
    I6 = data["I6"]
    u  = data["u"]
    v  = data["v"]
    w  = data["w"]

# convert to callables
#
fI4 = sy.lambdify( fargs, I4, modules="numpy" )
fI5 = sy.lambdify( fargs, I5, modules="numpy" )
fI6 = sy.lambdify( fargs, I6, modules="numpy" )
fu  = sy.lambdify( fargs, u,  modules="numpy" )
fv  = sy.lambdify( fargs, v,  modules="numpy" )
fw  = sy.lambdify( fargs, w,  modules="numpy" )


# load spline fit data
s2d = scipy.io.loadmat("tmp_s2d.mat")
#s3d = scipy.io.loadmat("tmp_s3d.mat")

ordr   = np.squeeze(s2d["ordr"])
xknots = np.squeeze(s2d["xknots"])
yknots = np.squeeze(s2d["yknots"])
c      = np.squeeze(s2d["c"])  # spline fit coefficients
Hscale = np.squeeze(s2d["Hscale"])
sscale = np.squeeze(s2d["sscale"])


# Variation ranges
#
Hmin = np.min(xknots) * Hscale
Hmax = np.max(xknots) * Hscale
smin = np.min(yknots) * sscale
smax = np.max(yknots) * sscale

# How many vectors to generate
#
n = 1000

# Function for generating random numbers in given range
rrand = lambda n, xmin, xmax: (xmax - xmin) * np.random.rand(n) + xmin

# Generate random H vector in given range
#
# We do this by generating uniform random directions (points on the unit sphere).
#
# One needs to be careful here due to the pole singularities;
# a uniform distribution in the (theta,phi) plane does not
# map into a uniform distribution on the sphere.
#
# To create a uniform distribution on the sphere, we use this transformation trick:
#     http://www.cognitive-antics.net/uniform-random-orientation/
#
tmp    = np.random.rand(n,2)
Hphi   = 2. * np.pi * tmp[:,0]          # azimuth
Htheta = np.arcsin(2. * tmp[:,1] - 1.)  # elevation (actually its complement i.e. measured from +vertical)
Hamp   = rrand(n, Hmin, Hmax)

Hx     = Hamp * np.sin(Htheta) * np.cos(Hphi)
Hy     = Hamp * np.sin(Htheta) * np.sin(Hphi)
Hz     = Hamp * np.cos(Htheta)

# TODO visualize generated H vectors

# Generate random sigma
#
# FIXME This causes some test data to be out of the model range (-1, 1) on the v axis.
#
# Maybe this is because sigma is a tensor, and setting the scale of each component
# to smax does not limit the matrix norm sufficiently?
#
sxx    = rrand(n, smin, smax)
syy    = rrand(n, smin, smax)
szz    = rrand(n, smin, smax)
sxy    = rrand(n, smin, smax)
syz    = rrand(n, smin, smax)
szx    = rrand(n, smin, smax)

# Compute invariants and auxiliary variables
#
I4 = fI4(Hx, Hy, Hz, sxx, syy, szz, sxy, syz, szx)
I5 = fI5(Hx, Hy, Hz, sxx, syy, szz, sxy, syz, szx)
#I6 = fI6(Hx, Hy, Hz, sxx, syy, szz, sxy, syz, szx)

u  = fu(Hx, Hy, Hz, sxx, syy, szz, sxy, syz, szx) / Hscale
v  = fv(Hx, Hy, Hz, sxx, syy, szz, sxy, syz, szx) / sscale
#w  = fw(Hx, Hy, Hz, sxx, syy, szz, sxy, syz, szx) / sscale

nx   = len(u)
ny   = len(v)
assert nx == ny, "sequences (u,v) describing the input points must be of the same length"


# Set up splines for evaluation
#
splx = bspline.Bspline(xknots, ordr)
sply = bspline.Bspline(yknots, ordr)

# get number of basis functions (perform dummy evaluation and count)
nxb  = len( splx(0.) )
nyb  = len( sply(0.) )

# NOTE: we are evaluating the model on a sequence of points (u[j], v[j]), not on a meshgrid (outer(u,v)).
#
# However, the *spline basis* is a meshgrid in the sense that
# it is the outer product of the x and y basis functions.
#
Au   = splx.collmat(u)
Av   = sply.collmat(v)
Du   = splx.collmat(u, deriv_order=1)
Dv   = sply.collmat(v, deriv_order=1)

nf = 2  # number of unknown fields
nr = n  # matrix rows per field (B_x, lambda_xx, ...) (each row corresponds to a collocation point)
A  = np.empty( (nf*nr, nxb*nyb), dtype=np.float64 )  # collocation matrix

K,L,KL = util.index.genidx( (nxb,nyb) )

# loop only over rows of the matrix
for j in range(n):
    A[nf*j,  KL] = Du[j,K] * Av[j,L]
    A[nf*j+1,KL] = Au[j,K] * Dv[j,L]

A = scipy.sparse.csr_matrix(A)  # done, sparsify for faster dot products

# evaluate the fit at the collocation points
#
fitted    = A.dot(c)  # function values corresponding to each row in the global equation system

out_Bx    = fitted[::2]
out_lamxx = fitted[1::2]

print( xknots )
print( yknots )
print( u )
print( v )
print( np.max(np.abs(u)), np.max(np.abs(v)) )
print( out_Bx )
print( out_lamxx )

