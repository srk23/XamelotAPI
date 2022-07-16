# You will find here a bunch of useful but diverse functions.

import inspect
import re

def identity(x):
    """
    A simple function for identity
    """
    return x

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
