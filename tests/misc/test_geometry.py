import numpy as np

from xmlot.misc.geometry import get_distance_regarding_intersection

def test_get_distance_regarding_intersection():
    r   = 1
    a   = np.pi * r**2
    eps = 1e-3

    assert np.abs(get_distance_regarding_intersection(a1=a, a2=a, a3=a)) < eps
    assert np.abs(get_distance_regarding_intersection(a1=a, a2=a, a3=0)) > r
