# You will find here a bunch of useful but diverse functions.

import numpy as np

import inspect
import re
import torch
import random


def identity(x):
    """
    A simple function for identity
    """
    return x


def gandalf(msg="You shall not pass!"):
    """
    A silly function that raises an AssertionError, preventing code to run after a given point.
    """
    assert False, msg


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    else:
        return None

class SeedGenerator:
    def __init__(self, seed=None):
        self.m_seed = seed
        self.m_rng  = random.Random(seed)

    def __call__(self):
        if self.m_seed is not None:
            return self.m_rng.randint(0, 1000000)
        else:
            return None

def get_var_name(var, depth=1):
    """
    Returns the list of variable names corresponding to a given value.

    Source: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523#18425523

    Args:
         - var  : the variable (in the sense of its value)
         - depth: since variable names change when in a function, specifies at which composition degree to focus
                  (0 being the last function being used, 1 the function above, etc.).

    Returns:
          - a list of strings (variable names)
    """
    callers_local_vars = inspect.currentframe().f_back
    for _ in range(depth):
        callers_local_vars = callers_local_vars.f_back
    callers_local_vars = callers_local_vars.f_locals.items()

    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def string_autotype(s):
    """
    Based on the shape of a given string, provides the tools to retype it correctly.

    Args:
        - s: a string

    Return:
        - adjust_type: a function to retype the input
        - cast_type  : the string Pandas reference of the appropriate type
    """
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
