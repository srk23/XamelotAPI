import pandas as pd

from project.data.bin import *
from project.data.datamanager import DataManager

from project.misc.miscellaneous import set_seed
def test_bin1():
    b  = Binner([0, 1])
    dm = DataManager(
        df = pd.DataFrame({
            'x' : [0, 1],
            'y' : [1, 0]
        }),
        targets_list=['y'],
        reference_target='y'
    )

    binned_dm   = b(dm, target='y')
    expected_dm = dm.copy()
    assert binned_dm.equals(expected_dm)

def test_bin2():
    b  = Binner([0, 2, 4, 6, 8, 10])
    dm = DataManager(
        df = pd.DataFrame({
            'x' : [1 , 1,   1,   1,  1],
            'y' : [-1, 5, 4.1, 6.9, 11]
        }),
        targets_list=['y'],
        reference_target='y'
    )

    binned_dm        = b(dm, target='y')
    expected_dm      = dm.copy()
    expected_dm.m_df = pd.DataFrame({
            'x' : [1, 1, 1, 1, 1],
            'y' : [0, 3, 3, 4, 6]
    })
    assert binned_dm.equals(expected_dm)

def test_bin3():
    b  = Binner([0, 2, 12])
    dm = DataManager(
        df = pd.DataFrame({
            'x' : [1 ,   1, 1,   1,  1],
            'y' : [-1, 1.1, 3, 7.1, 15]
        }),
        targets_list=['y'],
        reference_target='y'
    )

    binned_dm        = b(dm, target='y')
    expected_dm      = dm.copy()
    expected_dm.m_df = pd.DataFrame({
            'x' : [1, 1, 1, 1, 1],
            'y' : [0, 1, 2, 2, 3]
    })
    assert binned_dm.equals(expected_dm)

def test_equidistant_bin():
    dm = DataManager(
        df=pd.DataFrame({
            'x': [0, 1],
            'y': [-10, 10]
        }),
        targets_list=['y'],
        reference_target='y'
    )

    eb = EquidistantBinner(dm, target='y', num_bins=10)
    assert eb.cuts == [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]

def test_equidistant_cut():
    dm = DataManager(
        df=pd.DataFrame({
            'x': [1, 1, 1, 1, 1],
            'y': [0, 0, .5, 1, 10]
        }),
        targets_list=['y'],
        reference_target='y'
    )

    b = EquidistantBinner(dm, target='y', num_bins=3)

    assert b.num_bins == 3
    assert b.cuts == [10/3, 20/3]

def test_equidistant_pycox_cut():
    dm = SurvivalDataManager(
        df=pd.DataFrame({
            'x': [1, 1, 1, 1, 1],
            'y': [0, 0, .5, 1, 10]
        }),
        event='x',
        duration='y'
    )
    b = EquidistantBinner(dm, target='y', num_bins=3, pycox_style=True)

    assert b.cuts == [0, 5, 10]

def test_pycox_equidistant_cut():
    dm = SurvivalDataManager(
        df = pd.DataFrame({
            'x' : [1 , 1,  1, 1,  1],
            'y' : [0, 0, .5, 1, 10]
        }),
        event='x',
        duration='y'
    )
    b  = PycoxBinner(dm, num_bins=3)

    assert list(b.m_labtrans.cuts) == [0, 5, 10]

def test_quantile_cut():
    from lifelines.datasets import load_dd
    set_seed(42)
    df = load_dd()
    sdm = SurvivalDataManager(df, duration='duration', event='observed')

    num_bins = 10

    b0 = QuantileBinner(sdm, num_bins=num_bins, pycox_style=False)
    cut0 = b0.cuts

    b1 = QuantileBinner(sdm, num_bins=num_bins, pycox_style=True)
    cut1 = b1.cuts

    b2 = PycoxBinner(sdm, num_bins=num_bins, scheme='quantiles')
    cut2 = list(b2.m_labtrans.cuts)

    assert cut0 == [1.0, 2.0, 4.0, 5.0, 6.0, 9.0, 16.0]
    assert cut1 == [0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 14.0, 47.0]
    assert cut2 == [0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 14.0, 47.0]
