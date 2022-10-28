from project.data.dataframes import *

DF1 = pd.DataFrame({"a": [0, 0], "b": [pd.NA, 0], "c": [pd.NA, pd.NA]})
DF2 = pd.DataFrame({"a": [], "b": [], "c": []})
DF3 = pd.DataFrame({'A': [1, 0, pd.NA], 'B': [1, pd.NA, pd.NA], 'C': [pd.NA, pd.NA, pd.NA]})


def test_build_empty_mask():
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


def test_get_constant_columns():
    assert get_constant_columns(DF3) == ['B', 'C']

def test_get_sparse_columns():
    assert get_sparse_columns(DF3, .5) == ['B', 'C']