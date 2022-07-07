import pandas as pd

from project.data.augment import build_min,\
                                 build_max,\
                                 build_easy_trend

DF = pd.DataFrame(
    {
        'col_10': [1, 2, 5],
        'col_20': [2, 1, 5],
        'col_30': [5, 0, 1],
    })

def test_build_easy_trend():
    s = build_easy_trend(DF, ['col_10', 'col_20'])
    assert s.equals(pd.Series([1, -1, 0]))

def test_build_min():
    s = build_min(DF, ['col_10', 'col_20'])
    assert s.equals(pd.Series([1, 1, 5]))


def test_build_max():
    s = build_max(DF, ['col_10', 'col_20'])
    assert s.equals(pd.Series([2, 2, 5]))
