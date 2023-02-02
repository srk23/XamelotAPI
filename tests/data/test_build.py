import pandas as pd

from xmlot.data.build      import *
from xmlot.misc.misc       import identity


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
    "pcens": ["Event"   , "Event"  , "Censored", "Event"  , "Censored", "Censored", "Event"   ],
    "psurv": [1    , 1   , 0    , 0   , 0    , 0    , 0    ],
    "gcens": ["Censored", "Event"  , "Event"   , "Event"  , "Censored", "Event"   , "Censored"],
    "gsurv": [0    , 0   , 1    , 1   , 0    , 0    , 0    ]
})

DF8 = pd.DataFrame({
    "gcens": ["Censored", "Censored", "Event", "Event"   , "Censored"],
    "gsurv": [0         , 1         , 1      , 1         , 1         ],
    "pcens": ["Censored", "Censored", "Event", "Censored", "Event"   ],
    "psurv": [1         , 0         , 0      , 0         , 0         ]
})

###############
#   TARGETS   #
###############


def test_build_pcens():
    s_target   = pd.Series([-1, -1, -1, 1, 1, 1, 0, 0, 0])
    s_obtained = build_pcens(DF6, pcens="B", censored=-1)

    assert s_obtained.equals(s_target)


def test_build_psurv():
    s_target   = pd.Series([pd.NA, -2, 5, 1, 1, 1, 0, 0, 0])
    s_obtained = build_psurv(DF6, gsurv="A", psurv="B")

    assert s_obtained.equals(s_target)


def test_build_tcens():
    s_target = pd.Series([
        "Event", "Event", "Event", "Event", "Censored", "Event", "Event"
    ])
    s_obtained = DF7.apply(lambda s: build_tcens(s, event="Event"), axis=1)

    assert s_obtained.equals(s_target)


def test_build_tsurv():
    s_target = pd.Series([0, 0, 1, 1, 0])
    s_obtained = DF8.apply(build_tsurv, axis=1)

    assert s_obtained.equals(s_target)


def test_build_mcens():
    s_target = pd.Series(
        [
            "Censored",
            "Graft failure",
            "Censored",
            "Deceased",
            "Censored",
            "Graft failure",
            "Deceased"
        ]
    )
    s_obtained = DF7.apply(lambda s: build_mcens(s, pcens="pcens"), axis=1)

    assert s_obtained.equals(s_target)


def test_build_msurv():
    s_target = pd.Series([0] * 7)
    s_obtained = DF7.apply(build_msurv, axis=1)

    assert s_obtained.equals(s_target)

def test_build_single_classification_from_survival():

    censored = 2.
    event    = 1.
    alive    = 0.

    s_target = pd.Series([
        censored,
        event,
        alive,
        alive,
        censored,
        event,
        censored
    ])
    s_obtained = build_single_classification_from_survival(
        DF7,
        0.5,
        xcens    = "gcens",
        xsurv    = "gsurv",
        unknown  = "Unknown",
        censored = "Censored"
    )

    assert s_obtained.equals(s_target)

    s_target = pd.Series([
        censored,
        event,
        alive,
        alive,
        censored,
        event,
        censored
    ])

    assert s_obtained.equals(s_target)


def test_build_multi_classification_from_survival():
    censored                     = 3
    deceased                     = 2
    alive_with_failed_graft      = 1
    alive_with_functioning_graft = 0

    # TEST DF7
    s_target = pd.Series([
        censored,
        alive_with_failed_graft,
        censored,
        deceased,
        censored,
        censored,
        deceased
    ])

    s_obtained = build_multi_classification_from_survival(
            DF7,
            .5,
            gcens="gcens",
            gsurv="gsurv",
            pcens="pcens",
            psurv="psurv",
            censored="Censored"
    )

    assert s_obtained.equals(s_target)

    # TEST DF8
    s_target = pd.Series([
        censored,
        censored,
        deceased,
        censored,
        deceased
    ])

    s_obtained = build_multi_classification_from_survival(
            DF8,
            .5,
            gcens="gcens",
            gsurv="gsurv",
            pcens="pcens",
            psurv="psurv",
            censored="Censored"
    )

    assert s_obtained.equals(s_target)

    # TEST Alive with functioning graft (DF8 and t<0)
    s_target = pd.Series([
        alive_with_functioning_graft,
        alive_with_functioning_graft,
        alive_with_functioning_graft,
        alive_with_functioning_graft,
        alive_with_functioning_graft
    ])

    s_obtained = build_multi_classification_from_survival(
            DF8,
            -.5,
            gcens="gcens",
            gsurv="gsurv",
            pcens="pcens",
            psurv="psurv",
            censored="Censored"
    )

    print("target  :", s_target)
    print("obtained:", s_obtained)

    assert s_obtained.equals(s_target)


################
#   FEATURES   #
################


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

