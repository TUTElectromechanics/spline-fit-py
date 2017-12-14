#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:45:36 2017

Python port of bfunc.m by Paavo Rasilo.

[1] L. Daniel, O. Hubert, M. Rekik, A Simplified 3D Constitutive Law for Magneto-Mechanical Behavior,
    IEEE Trans. Magn. 51(3) (2015), 7300704.

@author: Juha Jeronen, juha.jeronen@tut.fi
"""

from __future__ import division

import numpy as np

import scipy.integrate


# FIXME Inefficient cubature, based on nested quad.
#
# Better to use true 2D integration schemes for quadrilaterals.
# Software e.g.  https://github.com/saullocastro/cubature
#     based on   http://ab-initio.mit.edu/wiki/index.php/Cubature

def _quad2d( f, ranges, opts=None ):
    ret = scipy.integrate.nquad( f, ranges, opts=opts )
    return ret[0]  # value of integral


def bfunc(H, sig):
    mu0   = 4.*np.pi*1e-7
    Ms    = 1.45e6
    lams  = 10e-6
    As    = 1.8e-3
#    Bs    = As #2.2e-3
    l_111 = lams

    # Functions for each direction, Equations (1)-(6), no anistropy
    f_Malfa1 = lambda theta,phi: Ms * np.sin(phi) * np.cos(theta)
    f_Malfa2 = lambda theta,phi: Ms * np.sin(phi) * np.sin(theta)
    f_Malfa3 = lambda theta,phi: Ms * np.cos(phi)

    f_lalfa1 = lambda theta,phi: lams  * ((np.sin(phi) * np.cos(theta))**2 - 1./3.)
    f_lalfa2 = lambda theta,phi: l_111 * ((np.sin(phi) * np.cos(theta)) * (np.sin(phi) * np.sin(theta)))
    f_lalfa3 = lambda theta,phi: l_111 * ((np.sin(phi) * np.cos(theta)) * (np.cos(phi)))
    f_lalfa4 = lambda theta,phi: l_111 * ((np.sin(phi) * np.cos(theta)) * (np.sin(phi) * np.sin(theta)))
    f_lalfa5 = lambda theta,phi: lams  * ((np.sin(phi) * np.sin(theta))**2 - 1./3.)
    f_lalfa6 = lambda theta,phi: l_111 * ((np.sin(phi) * np.sin(theta)) * (np.cos(phi)))
    f_lalfa7 = lambda theta,phi: l_111 * ((np.sin(phi) * np.cos(theta)) * (np.cos(phi)))
    f_lalfa8 = lambda theta,phi: l_111 * ((np.sin(phi) * np.sin(theta)) * (np.cos(phi)))
    f_lalfa9 = lambda theta,phi: lams  * ((np.cos(phi))**2 - 1./3.)

    # equ (2)
    f_Wmag   = lambda theta,phi: -mu0 * (H[0] * f_Malfa1(theta,phi) + H[1] * f_Malfa2(theta,phi) + H[2] * f_Malfa3(theta,phi))

    # equ (3)
    f_Wmek   = lambda theta,phi: -(  sig[0,0] * f_lalfa1(theta,phi) + sig[0,1] * f_lalfa2(theta,phi) + sig[0,2] * f_lalfa3(theta,phi) \
                                   + sig[1,0] * f_lalfa4(theta,phi) + sig[1,1] * f_lalfa5(theta,phi) + sig[1,2] * f_lalfa6(theta,phi) \
                                   + sig[2,0] * f_lalfa7(theta,phi) + sig[2,1] * f_lalfa8(theta,phi) + sig[2,2] * f_lalfa9(theta,phi))

    # equ (1)
    f_Walfa  = lambda theta,phi:  (f_Wmag(theta,phi) + f_Wmek(theta,phi))


    # Probability, this already requires numerical integration, Equation (7)
    funcint   = lambda theta,phi:  np.sin(phi) * np.exp(-As * f_Walfa(theta,phi))
    limsint   = ( (0., 2.*np.pi), (0., np.pi) )
    intfuncin = _quad2d( funcint, limsint )
    falfa     = lambda theta,phi:  np.exp(-As * f_Walfa(theta,phi)) / intfuncin

    f1 = lambda theta,phi: np.sin(phi) * f_Malfa1(theta,phi) * falfa(theta,phi)
    f2 = lambda theta,phi: np.sin(phi) * f_Malfa2(theta,phi) * falfa(theta,phi)
    f3 = lambda theta,phi: np.sin(phi) * f_Malfa3(theta,phi) * falfa(theta,phi)

    Mx = _quad2d( f1, limsint )
    My = _quad2d( f2, limsint )
    Mz = _quad2d( f3, limsint )
    M  = np.array( [Mx, My, Mz] )

    fe1 = lambda theta,phi:  np.sin(phi) * f_lalfa1(theta,phi) * falfa(theta,phi)
    fe2 = lambda theta,phi:  np.sin(phi) * f_lalfa2(theta,phi) * falfa(theta,phi)
    fe3 = lambda theta,phi:  np.sin(phi) * f_lalfa3(theta,phi) * falfa(theta,phi)
#    fe4 = lambda theta,phi:  np.sin(phi) * f_lalfa4(theta,phi) * falfa(theta,phi)
    fe5 = lambda theta,phi:  np.sin(phi) * f_lalfa5(theta,phi) * falfa(theta,phi)
    fe6 = lambda theta,phi:  np.sin(phi) * f_lalfa6(theta,phi) * falfa(theta,phi)
#    fe7 = lambda theta,phi:  np.sin(phi) * f_lalfa7(theta,phi) * falfa(theta,phi)
#    fe8 = lambda theta,phi:  np.sin(phi) * f_lalfa8(theta,phi) * falfa(theta,phi)
    fe9 = lambda theta,phi:  np.sin(phi) * f_lalfa9(theta,phi) * falfa(theta,phi)

    opts = { "epsabs" : 1e-12,
             "epsrel" : 1e-6 }

    lam1 = _quad2d(fe1, limsint, opts)
    lam2 = _quad2d(fe2, limsint, opts)
    lam3 = _quad2d(fe3, limsint, opts)
    lam4 = lam2; #_quad2d(fe4, limsint, opts)
    lam5 = _quad2d(fe5, limsint, opts)
    lam6 = _quad2d(fe6, limsint, opts)
    lam7 = lam3; #_quad2d(fe7, limsint, opts)
    lam8 = lam6; #_quad2d(fe8, limsint, opts)
    lam9 = _quad2d(fe9, limsint, opts)
    lam = np.array( [[lam1, lam2, lam3],
                     [lam4, lam5, lam6],
                     [lam7, lam8, lam9]] )

    B = mu0 * (H + M)

    return (B, lam)
