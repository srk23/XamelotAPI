import pandas as pd

from project.data.clean import *
from project.data.describe import Entry, Descriptor
from project.misc.dataframes import build_empty_mask

DESCRIPTOR = Descriptor(
    [
        Entry("A", tags="feature"),
        Entry("B", tags="not relevant"),
        Entry("C", tags="target")
    ]
)

DF1 = pd.DataFrame({'uppER': [1, 2], 'LoWEr': [3, 4.5]})
DF2 = pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]})
DF3 = pd.DataFrame({'A': [1, 0, pd.NA], 'B': [1, pd.NA, pd.NA], 'C': [pd.NA, pd.NA, pd.NA]})


def test_clean_data_columns_to_lower_case():
    assert (set_columns_to_lower_case(DF1).columns == ['upper', 'lower']).all()


def test_clean_data_int64():
    L = ['Int64', 'float64']
    L_ = change_int64(DF1).dtypes.to_list()

    assert len(L) == len(L_)
    for l, l_ in zip(L, L_):
        assert l == l_


def test_clean_data_type_uniformity():
    DF = pd.DataFrame({
        'col1': pd.Series([1, 2]),
        'col2': pd.Series(["1", "2"], dtype="object")
    })

    hc = {"col2", "col3"}
    DF_ = ensure_type_uniformity(DF2.copy(), heterogeneous_columns=hc)

    assert (DF_.dtypes == DF.dtypes).all()


def test_clean_data_type_unknown_values():
    mask = pd.DataFrame({'col1': [True, True], 'col2': [False, True]})

    gu = {2}
    su = {'col1': {1}}
    _, mask_ = correct_unknown_values(DF2.copy(), build_empty_mask(DF2), generic_unknowns=gu, specific_unknowns=su)

    assert mask.equals(mask_)


def test_clean_data_abnormal_values():
    limits = [({'col1'}, (0, 1.5))]

    mask = pd.DataFrame({'col1': [False, True], 'col2': [False, False]})

    _, mask_ = remove_abnormal_values(DF2.copy(), build_empty_mask(DF2), limits=limits)

    assert (mask_.equals(mask))


def test_clean_data_categories():
    refs = [
        (
            {'col1'},
            {
                1: 'A',
                2: 'B',
                3: 'C'
            }),
        (
            {'col2'},
            {
                1: 'D'
            }
        )
    ]

    DF = pd.DataFrame({
        'col1': ['A', 'B'],
        'col2': ['D', 2]
    })

    DF_ = use_category_names(DF2.copy(), references=refs)
    assert (DF_.equals(DF))

