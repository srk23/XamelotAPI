from project.misc.lists import *

a, b, c, d, e = 'a', 'b', 'c', 'd', 'e'

L1 = [a, b, d]
L2 = [d, b, c, e]

def test_intersection():
    assert intersection(L1, L2) == [b, d]

def test_union():
    assert union(L1, L2) == [a, b, d, c, e]

def test_difference():
    assert difference(L1, L2) == [a]
