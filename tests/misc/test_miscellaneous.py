from   numpy.random               import randn
from   xmlot.misc.misc import *

def test_identity():
    """
    Another late evening at work...
    """
    x = randn()
    assert x == identity(x)

def test_string_autotype():
    s = ("42", 42)
    adjust_type, _ = string_autotype(s[0])
    assert adjust_type(s[0]) == s[1]

    s = ("3.14", 3.14)
    adjust_type, _ = string_autotype(s[0])
    assert adjust_type(s[0]) == s[1]

    s = ("teapot", "teapot")
    adjust_type, _ = string_autotype(s[0])
    assert adjust_type(s[0]) == s[1]

    s = ("3.0", 3)
    adjust_type, _ = string_autotype(s[0])
    assert type(adjust_type(s[0])) is float

def test_set_seed():
    set_seed(42)
    a = randn()

    set_seed(42)
    b = randn()

    assert a == b
