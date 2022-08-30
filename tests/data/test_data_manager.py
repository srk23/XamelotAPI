from project.data.datamanager   import *
from project.misc.miscellaneous import set_seed
DF1 = pd.DataFrame({
    'x1' : [1, 1, 1, 1],
    'x2' : [1, 1, 1, 1],
    'x3' : [1, 1, 1, 1],
    'y1' : [1, 1, 1, 1],
    'y2' : [1, 1, 1, 1]
})

DF2 = pd.DataFrame({
    'x1' : [2, 2],
    'x2' : [2, 2],
    'x3' : [2, 2],
    'y1' : [2, 2],
    'y2' : [2, 2]
})
DF2 = DF2.rename(index={0: 4, 1: 5})

DF3 = pd.DataFrame({
    'x1' : [1, 1, 1, 1],
    'x2' : [1, 1, 1, 1],
    'x3' : [1, 2, 1, 1],
    'y1' : [1, 1, 1, 1],
    'y2' : [1, 1, 1, 1]
})

DFC = pd.DataFrame({
    'x1' : [1, 1, 1, 1, 2, 2],
    'x2' : [1, 1, 1, 1, 2, 2],
    'x3' : [1, 1, 1, 1, 2, 2],
    'y1' : [1, 1, 1, 1, 2, 2],
    'y2' : [1, 1, 1, 1, 2, 2]
})

S1 = pd.DataFrame({
    'x1': [1, 2, 1],
    'x2': [1, 2, 1],
    'x3': [1, 2, 1],
    'y1': [1, 2, 1],
    'y2': [1, 2, 1]

})
S1 = S1.rename(index={0: 3, 1: 5, 2: 1})

S2 = pd.DataFrame({
    'x1': [2, 1, 1],
    'x2': [2, 1, 1],
    'x3': [2, 1, 1],
    'y1': [2, 1, 1],
    'y2': [2, 1, 1]

})
S2 = S2.rename(index={0: 4, 1: 2, 2: 0})

def test_equals():
    dm1 = DataManager(DF1, targets_list=['y1', 'y2'], reference_target='y2')
    dm2 = DataManager(DF1, targets_list=['y1', 'y2'], reference_target='y2')
    dm3 = DataManager(DF1, targets_list=['y1', 'y2'], reference_target='y1')
    dm4 = DataManager(DF1, targets_list=['y2'], reference_target='y2')
    dm5 = DataManager(DF3, targets_list=['y1', 'y2'], reference_target='y2')

    assert     dm1.equals(dm2)
    assert not dm1.equals(dm3)
    assert not dm1.equals(dm4)
    assert not dm1.equals(dm5)

def test_copy():
    dm1 = DataManager(DF1, targets_list=['y1', 'y2'], reference_target='y2')
    dm2 = SurvivalDataManager(DF1, event='y1', duration='y2')
    dm3 = SingleTargetDataManager(DF1, 'y2')

    assert dm1.equals(dm1.copy())
    assert dm2.equals(dm2.copy())
    assert dm3.equals(dm3.copy())

def test_concat():
    dm11 = DataManager(DF1, targets_list=['y1', 'y2'], reference_target='y2')
    dm12 = SurvivalDataManager(DF1, event='y1', duration='y2')
    dm13 = SingleTargetDataManager(DF1, 'y2')

    dm21 = DataManager(DF2, targets_list=['y1', 'y2'], reference_target='y2')
    dm22 = SurvivalDataManager(DF2, event='y1', duration='y2')
    dm23 = SingleTargetDataManager(DF2, 'y2')

    dmc1 = DataManager(DFC, targets_list=['y1', 'y2'], reference_target='y2')
    dmc2 = SurvivalDataManager(DFC, event='y1', duration='y2')
    dmc3 = SingleTargetDataManager(DFC, 'y2')

    assert dmc1.equals(dm11.concat(dm21))
    assert dmc2.equals(dm12.concat(dm22))
    assert dmc3.equals(dm13.concat(dm23))

def test_split():
    dmc1 = DataManager(DFC, targets_list=['y1', 'y2'], reference_target='y2')
    set_seed(42)
    splits = dmc1.split([.5, .5])

    assert splits[0].equals(
        DataManager(S1, targets_list=['y1', 'y2'], reference_target='y2')
    )
    assert splits[1].equals(
        DataManager(S2, targets_list=['y1', 'y2'], reference_target='y2')
    )

    dmc2 = SurvivalDataManager(DFC, event='y1', duration='y2')
    set_seed(42)
    splits = dmc2.split([.5, .5])

    assert splits[0].equals(
        SurvivalDataManager(S1, event='y1', duration='y2')
    )
    assert splits[1].equals(
        SurvivalDataManager(S2, event='y1', duration='y2')
    )

    dmc3 = SingleTargetDataManager(DFC, 'y2')
    set_seed(42)
    splits = dmc3.split([.5, .5])

    assert splits[0].equals(
        SingleTargetDataManager(S1, 'y2')
    )
    assert splits[1].equals(
        SingleTargetDataManager(S2, 'y2')
    )





