import pandas as pd
from project.misc.dataframes import intersect_columns


def test_intersect_columns():
    df = pd.DataFrame({"a": [], "b": [], "c": []})
    l  = ["a", "c", "e"]

    assert intersect_columns(l, df) == ["a", "c"]
