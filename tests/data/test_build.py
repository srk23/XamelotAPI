from project.data.build         import *
from project.misc.miscellaneous import identity


DF4 = pd.DataFrame(
    {
        'col_10': [1, 2, 5],
        'col_20': [2, 1, 5],
        'col_30': [5, 0, 1],
    })

DF6 = pd.DataFrame(
    {
        'A': [pd.NA, -2, 5, pd.NA, -1, 42, pd.NA, -8, 0],
        'B': [pd.NA, pd.NA, pd.NA, 1, 1, 1, 0, 0, 0],
        'C': np.random.randn(9),
    })


def test_build_easy_trend():
    s = build_easy_trend(DF4, ['col_10', 'col_20'])
    assert s.equals(pd.Series([1, -1, 0]))


def test_build_min():
    s = build_min(DF4, ['col_10', 'col_20'])
    assert s.equals(pd.Series([1, 1, 5]))


def test_build_max():
    s = build_max(DF4, ['col_10', 'col_20'])
    assert s.equals(pd.Series([2, 2, 5]))


def test_build_binary_code():
    s_ = pd.Series(range(9))
    is_positive_or_negative = {
        'A': lambda x: 1 if x < 0 else 0,
        'B': identity,
        'C': identity
    }
    s = build_binary_code(DF6, ['A', 'B'], is_positive_or_negative)

    print(s)
    print(s_)
    assert s.equals(s_)