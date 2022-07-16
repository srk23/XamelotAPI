import numpy as np
import pandas as pd

from project.data.describe import Entry, Descriptor
from project.data.wrangle import *
from project.misc.clinical import compute_egfr
from project.misc.miscellaneous import identity

DF1 = pd.DataFrame(
    {
        'col_10': [1, 2, 5],
        'col_20': [2, 1, 5],
        'col_30': [5, 0, 1],
    })
DF2 = pd.DataFrame({'A': [1, 0, pd.NA], 'B': [1, pd.NA, pd.NA], 'C': [pd.NA, pd.NA, pd.NA]})
DF3 = pd.DataFrame(
    {
        'A': [pd.NA, -2, 5, pd.NA, -1, 42, pd.NA, -8, 0],
        'B': [pd.NA, pd.NA, pd.NA, 1, 1, 1, 0, 0, 0],
        'C': np.random.randn(9),
    })

DESCRIPTOR1 = Descriptor(
    [
        Entry("A", tags="feature"),
        Entry("B", tags="not relevant"),
        Entry("C", tags="target")
    ]
)


###############################
#      BUILD NEW COLUMNS      #
###############################


def test_build_easy_trend():
    s = build_easy_trend(DF1, ['col_10', 'col_20'])
    assert s.equals(pd.Series([1, -1, 0]))


def test_build_min():
    s = build_min(DF1, ['col_10', 'col_20'])
    assert s.equals(pd.Series([1, 1, 5]))


def test_build_max():
    s = build_max(DF1, ['col_10', 'col_20'])
    assert s.equals(pd.Series([2, 2, 5]))


def test_build_binary_code():
    s_ = pd.Series(range(9))
    is_positive_or_negative = {
        'A': lambda x: 1 if x < 0 else 0,
        'B': identity,
        'C': identity
    }
    s = build_binary_code(DF3, ['A', 'B'], is_positive_or_negative)

    print(s)
    print(s_)
    assert s.equals(s_)


#####################################
#      SELECT SPECIFIC COLUMNS      #
#####################################


def test_get_constant_columns():
    assert get_constant_columns(DF2) == ['B', 'C']


def test_get_irrelevant_columns():
    assert get_irrelevant_columns(DF2, DESCRIPTOR1) == ['B']


def test_get_sparse_columns():
    assert get_sparse_columns(DF2, .5) == ['B', 'C']


#####################
#      WRANGLE      #
#####################

DF4 = pd.DataFrame({
    'irrelevant': [53147, 36278, 14526],
    'constant1': [11111, 11111, 11111],
    'constant2': [11111, pd.NA, 11111],
    'constant3': [pd.NA, pd.NA, pd.NA],
    'dage': [18.0, 18.0, 18.0],
    'dsex': ["Male", "Male", "Male"],
    'dweight': [0.0, 1.0, 100.0],
    'dheight': [1.0, 100.0, 200.0],
    'dbmi': [np.nan, np.nan, 25.0],
    'rage': [18.0, 18.0, 18.0],
    'rsex': ["Male", "Male", "Male"],
    'rweight': [1.0, 100.0, -1.0],
    'rheight': [100.0, 200.0, 1.0],
    'rbmi': [np.nan, np.nan, np.nan],
    'dial_at_reg': [pd.NA, "Not on dialysis", "B"],
    'dial_at_tx': [pd.NA, "No", "Yes"],
    'dial_at_tx_type': [pd.NA, "Not on dialysis", "A"],
    'days_on_dial_tx': [np.nan, np.nan, 42.0],
    'creatinine_11': [np.nan, np.nan, 90.0],
    'degfr_11': [42.0, np.nan, 100.0],
    'alt_11': [np.nan, 1.0, 2.0],
    'alt_12': [1.0, 2.0, 0.0],
    'ast_11': [1.0, 2.0, np.nan],
    'ast_12': [1.0, 1.0, 5.0],
    'ast_81': [2.0, np.nan, 0.0],
    'amylase_11': [1.0, 1.0, 1.0],
    'amylase_81': [0.0, 5.0, 1.0]
})

# DF4_ = pd.DataFrame({
#     'dweight': [np.nan, 1.0, 100.0],
#     'dheight': [np.nan, 100.0, 200.0],
#     'dbmi': [np.nan, 1.0, 25.0],
#     'rweight': [1.0, 100.0, np.nan],
#     'rheight': [100.0, 200.0, np.nan],
#     'rbmi': [1.0, 25, np.nan],
#     'dial_type': [pd.NA, "Not on dialysis", "A"],
#     'dial_days': [np.nan, 0.0, 42.0],
#     'alt_trend': [0, 1, -1],
#     'alt_min': [1.0, 1.0, 0.0],
#     'alt_max': [1.0, 2.0, 2.0],
#     'ast_trend': [0, -1, 0],
#     'ast_min': [1.0, 1.0, 0.0],
#     'ast_max': [2.0, 2.0, 5.0],
#     'amylase_min': [0.0, 1.0, 1.0],
#     'amylase_max': [1.0, 5.0, 1.0],
#     'degfr_min': [42.0, np.nan, compute_egfr(18, 90, "Male", 200)],
#     'degfr_max': [42.0, np.nan, compute_egfr(18, 90, "Male", 200)]
# })

DESCRIPTOR2 = Descriptor(
    [
        Entry(col, tags="feature")
        for col in DF4.columns
    ]
)
DESCRIPTOR2.set_entry(Entry("irrelevant", tags="irrelevant"))
DESCRIPTOR2.set_entry(Entry("dial_at_tx", tags="feature", binary_keys={"Yes": 0, "No": 1}))


def test_wrangle_data_impute_bmi():
    input_df = pd.DataFrame({
        'dweight': [0.0, 1.0, 100.0],
        'dheight': [1.0, 100.0, 200.0],
        'dbmi': [np.nan, np.nan, 25.0],
        'rweight': [1.0, 100.0, -1.0],
        'rheight': [100.0, 200.0, 1.0],
        'rbmi': [np.nan, np.nan, np.nan]
    })

    target_df = pd.DataFrame({
        'dweight': [np.nan, 1.0, 100.0],
        'dheight': [np.nan, 100.0, 200.0],
        'dbmi': [np.nan, 1.0, 25.0],
        'rweight': [1.0, 100.0, np.nan],
        'rheight': [100.0, 200.0, np.nan],
        'rbmi': [1.0, 25, np.nan]
    })

    output_df = impute_bmi(input_df, limits_bmi=(0, 100))

    assert target_df.equals(output_df)


def test_wrangle_data_transform_dialysis_columns():
    input_df = pd.DataFrame({
        'dial_at_reg': [pd.NA, "Not on dialysis", "B"],
        'dial_at_tx': [pd.NA, "No", "Yes"],
        'dial_at_tx_type': [pd.NA, "Not on dialysis", "A"],
        'days_on_dial_tx': [np.nan, np.nan, 42.0]
    })

    target_df = pd.DataFrame({
        'dial_type': [pd.NA, "Not on dialysis", "A"],
        'dial_days': [np.nan, 0.0, 42.0]
    })

    output_df = transform_dialysis_columns(input_df, DESCRIPTOR2)

    assert target_df.equals(output_df)


def test_wrangle_data_recompute_egfr():
    input_df = pd.DataFrame({
        'dsex': ["Male", "Male", "Male"],
        'dheight': [np.nan, 100.0, 200.0],
        'dage': [18.0, 18.0, 18.0],
        'creatinine_11': [np.nan, np.nan, 90.0],
        'degfr_11': [42.0, np.nan, 100.0]
    })

    target_df = pd.DataFrame({
        'dsex': ["Male", "Male", "Male"],
        'dheight': [np.nan, 100.0, 200.0],
        'dage': [18.0, 18.0, 18.0],
        'creatinine_11': [np.nan, np.nan, 90.0],
        'degfr_11': [42.0, np.nan, compute_egfr(18, 90, "Male", 200)]
    })

    output_df = recompute_egfr(input_df)

    assert target_df.equals(output_df)


def test_wrangle_data_impute_biolevels():
    input_df = pd.DataFrame({
        'creatinine_11': [np.nan, np.nan, 90.0],
        'degfr_11': [42.0, np.nan, 100.0],
        'alt_11': [np.nan, 1.0, 2.0],
        'alt_12': [1.0, 2.0, 0.0],
        'ast_11': [1.0, 2.0, np.nan],
        'ast_12': [1.0, 1.0, 5.0],
        'ast_81': [2.0, np.nan, 0.0],
        'amylase_11': [1.0, 1.0, 1.0],
        'amylase_81': [0.0, 5.0, 1.0]
    })

    target_df = pd.DataFrame({
        'alt_trend': [0, 1, -1],
        'alt_min': [1.0, 1.0, 0.0],
        'alt_max': [1.0, 2.0, 2.0],
        'ast_trend': [0, -1, 0],
        'ast_min': [1.0, 1.0, 0.0],
        'ast_max': [2.0, 2.0, 5.0],
        'amylase_trend': [0, 0, 0],
        'amylase_min': [0.0, 1.0, 1.0],
        'amylase_max': [1.0, 5.0, 1.0],
        'creatinine_trend': [np.nan, np.nan, 0],
        'creatinine_min': [np.nan, np.nan, 90.0],
        'creatinine_max': [np.nan, np.nan, 90.0],
        'degfr_trend': [0, np.nan, 0],
        'degfr_min': [42.0, np.nan, 100.0],
        'degfr_max': [42.0, np.nan, 100.0]
    })

    output_df = impute_biolevels(input_df)

    assert target_df.equals(output_df)


def test_wrangle_data_remove_irrelevant_columns():
    input_df = pd.DataFrame({
        'irrelevant': [53147, 36278, 14526],
        'dage': [18.0, 18.0, 18.0],
        'dsex': ["Male", "Male", "Male"],
        'dweight': [0.0, 1.0, 100.0],
        'dheight': [1.0, 100.0, 200.0],
        'dbmi': [np.nan, np.nan, 25.0],
    })

    target_df = pd.DataFrame({
        'dage': [18.0, 18.0, 18.0],
        'dsex': ["Male", "Male", "Male"],
        'dweight': [0.0, 1.0, 100.0],
        'dheight': [1.0, 100.0, 200.0],
        'dbmi': [np.nan, np.nan, 25.0],
    })

    output_df = remove_irrelevant_columns(input_df, DESCRIPTOR2)

    assert target_df.equals(output_df)


def test_wrangle_data_remove_constant_columns():
    input_df = pd.DataFrame({
        'constant1': [11111, 11111, 11111],
        'constant2': [11111, pd.NA, 11111],
        'constant3': [pd.NA, pd.NA, pd.NA],
        'dweight': [0.0, 1.0, 100.0],
        'dheight': [1.0, 100.0, 200.0],
    })

    target_df = pd.DataFrame({
        'dweight': [0.0, 1.0, 100.0],
        'dheight': [1.0, 100.0, 200.0],
    })

    output_df = remove_constant_columns(input_df)

    assert target_df.equals(output_df)
