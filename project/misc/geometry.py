# You will find here a bunch of useful and various geometrical tools.

import numpy as np
import sympy as sp

def get_radius(a):
    return np.sqrt(a/np.pi)

def get_distance_regarding_intersection(a1, a2, a3):
    # parameters
    r1 = get_radius(a1)
    r2 = get_radius(a2)
    
    # variable
    d  = sp.symbols('d')

    # expression of the intersecting area A3
    # between two disks of radius r1, r2 
    # and such that the distance between their centers is d.
    #
    # 1) solve the system {
    #        h**2 + d1**2 = r1**2 ; 
    #        h**2 + d2**2 = r2**2 ;
    #        d1   + d2    = d     ;
    #    } with respect to d1 and h (and implicitly d2).
    d1 = (d**2 + r1**2 - r2**2)/(2*d)
    h  = sp.sqrt(r1**2 - d1**2)

    # 2) split A3 as the sections of the two disks 
    #    minus the area of some triangles
    A3  = sp.asin(h/r1)*r1**2 + sp.asin(h/r2)*r2**2 - h*d

    # solve
    d_ = sp.nsolve(A3 - a3, r1+r2)
    return float(sp.re(d_))
