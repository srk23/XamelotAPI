import pandas as pd
from project.misc.dataframes import build_empty_mask, density, intersect_columns

DF1 = pd.DataFrame({"a": [0, 0], "b": [pd.NA, 0], "c": [pd.NA, pd.NA]})
DF2 = pd.DataFrame({"a": [], "b": [], "c": []})


def test_build_enpty_mask():
    assert build_empty_mask(DF1).equals(pd.DataFrame(
        {
            "a": [False, False],
            "b": [False, False],
            "c": [False, False]
        }
    ))


def test_density():
    assert density(DF1, "a") == 1
    assert density(DF1, "b") == 1 / 2
    assert density(DF1, "c") == 0


def test_intersect_columns():
    l  = ["a", "c", "e"]
    assert intersect_columns(l, DF2) == ["a", "c"]
