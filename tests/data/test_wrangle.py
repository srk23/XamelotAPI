import numpy  as np
import pandas as pd

from project.data.describe      import Entry, Descriptor
from project.data.wrangle       import build_min,\
                                       build_max,\
                                       build_easy_trend,\
                                       build_binary_code,\
                                       get_constant_columns,\
                                       get_irrelevant_columns,\
                                       get_sparse_columns
from project.misc.miscellaneous import identity

DESCRIPTOR = Descriptor(
    [
        Entry("A", tags="feature"),
        Entry("B", tags="not relevant"),
        Entry("C", tags="target")
    ]
)

DF1 = pd.DataFrame(
    {
        'col_10': [1, 2, 5],
        'col_20': [2, 1, 5],
        'col_30': [5, 0, 1],
    })
DF2 = pd.DataFrame({'A': [1, 0, pd.NA], 'B': [1, pd.NA, pd.NA], 'C': [pd.NA, pd.NA, pd.NA]})
DF3 = pd.DataFrame(
    {
        'A': [pd.NA,    -2,     5, pd.NA, -1, 42, pd.NA, -8, 0],
        'B': [pd.NA, pd.NA, pd.NA,     1,  1,  1,     0,  0, 0],
        'C': np.random.randn(9),
    })


def test_build_easy_trend():
    s = build_easy_trend(DF1, ['col_10', 'col_20'])
    assert s.equals(pd.Series([1, -1, 0]))

def test_build_min():
    s = build_min(DF1, ['col_10', 'col_20'])
    assert s.equals(pd.Series([1, 1, 5]))


def test_build_max():
    s = build_max(DF1, ['col_10', 'col_20'])
    assert s.equals(pd.Series([2, 2, 5]))


def test_build_binary_code():
    s_ = pd.Series(range(9))
    is_positive_or_negative = {
        'A': lambda x: 1 if x < 0 else 0,
        'B': identity,
        'C': identity
    }
    s = build_binary_code(DF3, ['A', 'B'], is_positive_or_negative)

    print(s)
    print(s_)
    assert s.equals(s_)


def test_get_constant_columns():
    assert get_constant_columns(DF2) == ['B', 'C']


def test_get_irrelevant_columns():
    assert get_irrelevant_columns(DF2, DESCRIPTOR) == ['B']


def test_get_sparse_columns():
    assert get_sparse_columns(DF2, .5) == ['B', 'C']
