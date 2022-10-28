import pandas as pd

from xmlot.data.embed import extract_dense_dataframe

DF1 = pd.DataFrame({'A': [1, 0, pd.NA], 'B': [1, pd.NA, pd.NA], 'C': [pd.NA, pd.NA, pd.NA]})

def test_extract_dense_dataframe():
    df = extract_dense_dataframe(DF1, .5)
    assert df.equals(pd.DataFrame({'A' : [1, 0]}, dtype='object'))
