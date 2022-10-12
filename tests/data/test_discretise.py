import pandas as pd

from project.data.discretise import *
from project.data.accessor   import build_survival_accessor

ACCESSOR_CODE = "code"
build_survival_accessor(
    event="k",
    duration="t",
    accessor_code=ACCESSOR_CODE
)

def test_discretise():
    discretiser = Discretiser(
        [0, 1],
        ACCESSOR_CODE
    )

    df = pd.DataFrame({
        'k' : [0,  0, 0],
        't' : [0, .5, 1]
    })

    assert discretiser(df).equals(pd.DataFrame({
        'k' : [0, 0, 0],
        't' : [0, 0, 0]
    }))

    df = pd.DataFrame({
        'k' : [1,  1, 1],
        't' : [0, .5, 1]
    })

    assert discretiser(df).equals(pd.DataFrame({
        'k' : [1, 1, 1],
        't' : [0, 1, 1]
    }))

def test_equidistant():
    df = pd.DataFrame({
        'k' : [1,  1, 1],
        't' : [0, .5, 1]
    })

    # Home-made
    discretiser = EquidistantDiscretiser(
        df,
        accessor_code=ACCESSOR_CODE,
        size_grid=2
    )

    assert discretiser.grid == [0, 1]

    # Pycox
    discretiser = PycoxDiscretiser(
        df,
        accessor_code=ACCESSOR_CODE,
        size_grid=2,
        scheme="equidistant"
    )

    assert discretiser.grid == [0, 1]

def test_quantiles():
    df = pd.DataFrame({
        'k': [1, 0, 1, 0, 0],
        't': [0, 0, 1, 2, 3]
    })

    # Home-made
    discretiser = QuantileDiscretiser(
        df,
        accessor_code=ACCESSOR_CODE,
        size_grid=3
    )

    assert discretiser.grid == [0.0, 1.0]

    # Pycox
    discretiser = PycoxDiscretiser(
        df,
        accessor_code=ACCESSOR_CODE,
        size_grid=3,
        scheme="quantiles"
    )

    assert discretiser.grid == [0.0, 3.0]

def test_balanced():
    df = pd.DataFrame({
        'k' : [1 ,  0, 1, 0, 0],
        't' : [-1, -2, 1, 2, 3]
    })

    discretiser = BalancedDiscretiser(
        df,
        accessor_code=ACCESSOR_CODE,
        size_grid=3
    )

    assert discretiser.grid == [-2, 1, 3]
