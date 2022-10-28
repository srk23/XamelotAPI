from xmlot.data.split         import *
from xmlot.misc.misc import set_seed

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

def test_split():
    set_seed(42)
    df = pd.DataFrame({
        'x1': [1, 1, 1, 1, 2, 2],
        'x2': [1, 1, 1, 1, 2, 2],
        'x3': [1, 1, 1, 1, 2, 2],
        'y1': [1, 1, 1, 1, 2, 2],
        'y2': [1, 1, 1, 1, 2, 2]
    })

    splits = split_dataset(df, [.5, .5], 'y2')

    assert splits[0].equals(S1)
    assert splits[1].equals(S2)
