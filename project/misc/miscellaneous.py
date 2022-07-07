# You will find here a bunch of useful but diverse functions.

import re

def identity(x):
    """
    A simple function for identity
    """
    return x

def string_autotype(s):
    if re.fullmatch(r"-?\d+", s):
        adjust_type = int
        cast_type   = "Int64"
    elif re.fullmatch(r"-?\d+\.\d+", s):
        adjust_type = float
        cast_type   = "float64"
    else:
        adjust_type = identity
        cast_type   = "object"
    return adjust_type, cast_type
