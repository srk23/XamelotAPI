from project.data.build         import *
from project.misc.misc import identity


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

DF7 = pd.DataFrame({
    "pcens": [True , True, False, True, False, False, True ],
    "psurv": [1    , 1   , 0    , 0   , 0    , 0    , 0    ],
    "gcens": [False, True, True , True, False, True , False],
    "gsurv": [0    , 0   , 1    , 1   , 0    , 0    , 0    ]
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

def test_build_mcens():
    s_target = pd.Series(
        [
            "Censored",
            "Graft failure"     ,
            "Censored",
            "Deceased"                     ,
            "Censored",
            "Graft failure"     ,
            "Deceased"
        ]
    )
    s_obtained = DF7.apply(lambda s: build_mcens(s, pcens="pcens"), axis=1)

    assert s_obtained.equals(s_target)

def test_build_msurv():
    s_target = pd.Series([0] * 7)
    s_obtained = DF7.apply(build_msurv, axis=1)

    assert s_obtained.equals(s_target)

