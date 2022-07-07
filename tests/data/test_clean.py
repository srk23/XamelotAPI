import pandas as pd

from project.data.clean    import clean_data, get_constant_columns, get_irrelevant_columns, get_sparse_columns
from project.data.describe import Entry, Descriptor

DESCRIPTOR = Descriptor(
    [
        Entry("A", tags="feature"),
        Entry("B", tags="not relevant"),
        Entry("C", tags="target")
    ]
)

DF1 = pd.DataFrame({'uppER': [1, 2], 'LoWEr': [3, 4.5]})
DF2 = pd.DataFrame({'col1' : [1, 2], 'col2' : [1, 2]})
DF3 = pd.DataFrame({'A': [1, 0, pd.NA], 'B': [1, pd.NA, pd.NA], 'C': [pd.NA, pd.NA, pd.NA]})

def test_clean_data_columns_to_lower_case():
    assert (clean_data(DF1).columns == ['upper', 'lower']).all()
    
def test_clean_data_int64():
    L = ['Int64', 'float64']
    L_  = clean_data(DF1).dtypes.to_list()

    assert len(L) == len(L_)
    for l, l_ in zip(L, L_):
        assert l == l_

def test_clean_data_type_uniformity():
    DF   = pd.DataFrame({
            'col1': pd.Series([1, 2], dtype="Int64"),
            'col2': pd.Series(["1", "2"], dtype="object")
        })

    hc  = {"col2", "col3"}
    DF_ = clean_data(DF2, heterogeneous_columns=hc)
    
    assert (DF_.equals(DF))
    
def test_clean_data_type_unknown_values():
    DF  = pd.DataFrame({'col1': [pd.NA, pd.NA], 'col2': [1, pd.NA]}, dtype="Int64")

    gu  = {2}
    su  = {'col1' : {1}}
    DF_ = clean_data(DF2, generic_unknowns=gu, specific_unknowns=su)
    
    assert DF.equals(DF_)

def test_clean_data_abnormal_values():
    limits = [({'col1'}, (0, 1.5))]
    
    DF   = pd.DataFrame({
            'col1': pd.Series([1, pd.NA], dtype="Int64"), 
            'col2': pd.Series([1, 2], dtype="Int64")
        })

    DF_  = clean_data(DF2, limits=limits)
    assert (DF_.equals(DF))

def test_clean_data_categories():
    refs = [
        (
            {'col1'},
            {
                1 : 'A',
                2 : 'B',
                3 : 'C'
            }),
        (
            {'col2'},
            {
                1 : 'D'
            }
        )
    ]
    
    DF   = pd.DataFrame({
            'col1': ['A', 'B'], 
            'col2': ['D',  2 ]
        })

    DF_  = clean_data(DF2, references=refs)
    assert (DF_.equals(DF))


def test_get_constant_columns():
    assert get_constant_columns(DF3) == ['B', 'C']


def test_get_irrelevant_columns():
    assert get_irrelevant_columns(DF3, DESCRIPTOR) == ['B']


def test_get_sparse_columns():
    assert get_sparse_columns(DF3, .5) == ['B', 'C']
    
