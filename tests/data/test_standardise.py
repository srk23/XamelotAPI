import numpy  as np
import pandas as pd

from project.data.standardise import *
from project.data.describe    import Entry, Descriptor
from project.data.datamanager import SurvivalDataManager

DF = pd.DataFrame({
    "x1": [4., 2.],
    "x2": [50., 100.],
    "c" : [1, 0],
    "t" : [10., 20.]
})

DESCRIPTOR = Descriptor({
    Entry(
        "x1",
        is_categorical=False
    ),
    Entry(
        "x2",
        is_categorical=False
    ),
    Entry(
        "c",
        is_categorical=True
    ),
    Entry(
        "t",
        is_categorical=False
    )
})

class FakeOHE:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.separator  = "#"


OHE = FakeOHE(DESCRIPTOR)
SDM = SurvivalDataManager(DF, event="c", duration="t", ohe=OHE)


def test_nothing():
    standardiser = Standardiser(SDM)
    standardised_sdm = standardiser(SDM)

    assert standardised_sdm.equals(SDM)

def test_standardisation():
    s = np.sqrt(1 / 2)
    sdm_ = SurvivalDataManager(pd.DataFrame(
        {
            "x1": [s  , -s ],
            "x2": [-s , s  ],
            "c" : [1  , 0  ],
            "t" : [10., 20.]
        }
    ), event="c", duration="t")

    standardiser = Standardiser(SDM, **get_standardisation(SDM))  # , standardise_target_duration=False)
    standardised_sdm = standardiser(SDM)

    decimals = 15  # floats are badly handled, so we can't use `equals`.

    assert standardised_sdm.df.round(decimals).equals(sdm_.df.round(decimals)) \
           and standardised_sdm.event_name    == sdm_.event_name               \
           and standardised_sdm.duration_name == sdm_.duration_name


def test_normalisation():
    sdm_ = SurvivalDataManager(pd.DataFrame(
        {
            "x1": [1. , 0. ],
            "x2": [0. , 1. ],
            "c" : [1  , 0  ],
            "t" : [10., 20.]
        }
    ), event="c", duration="t", ohe=OHE)

    standardiser = Standardiser(SDM, **get_normalisation(SDM))  # , standardise_target_duration=False)
    standardised_sdm = standardiser(SDM)

    assert standardised_sdm.equals(sdm_)
