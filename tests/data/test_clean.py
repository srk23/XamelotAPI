from xmlot.data.clean    import *
from xmlot.data.describe import Entry, Descriptor
from xmlot.misc.clinical import compute_egfr


DF1 = pd.DataFrame({'uppER': [1, 2], 'LoWEr': [3, 4.5]})
DF2 = pd.DataFrame({'col1': [1, 2], 'col2': [1, 2]})
DF3 = pd.DataFrame({'A': [1, 0, pd.NA], 'B': [1, pd.NA, pd.NA], 'C': [pd.NA, pd.NA, pd.NA]})
DF4 = pd.DataFrame({'A': [1, 0, pd.NA], 'B': [1, pd.NA, pd.NA], 'C': [pd.NA, pd.NA, pd.NA]})
DF5 = pd.DataFrame({
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
DF6 = pd.DataFrame({
    "rdeath": ["Event"   , "Event", "Censored", "Event", "Censored", "Censored", "Event"   ],
    "psurv" : [1         , 1      , 0         , 0      , 0         , 0         , 0         ],
    "gcens" : ["Censored", "Event", "Event"   , "Event", "Censored", "Event"   , "Censored"],
    "gsurv" : [0         , 0      , 1         , 1      , 0         , 0         , 0         ]
})

DESCRIPTOR1 = Descriptor(
    [
        Entry("A", tags="feature"),
        Entry("B", tags="not relevant"),
        Entry("C", tags="target")
    ]
)
DESCRIPTOR2 = Descriptor(
    [
        Entry(col, tags="feature")
        for col in DF5.columns
    ]
)
DESCRIPTOR2.set_entry(Entry("irrelevant", tags="irrelevant"))
DESCRIPTOR2.set_entry(Entry("dial_at_tx", tags="feature", categorical_keys={"Yes": 0, "No": 1}))


###################
#      TESTS      #
###################


def test_columns_to_lower_case():
    assert (set_columns_to_lower_case(DF1).columns == ['upper', 'lower']).all()


def test_int64():
    L = ['Int64', 'float64']
    L_ = change_int64(DF1).dtypes.to_list()

    assert len(L) == len(L_)
    for l, l_ in zip(L, L_):
        assert l == l_


def test_type_uniformity():
    DF = pd.DataFrame({
        'col1': pd.Series([1, 2]),
        'col2': pd.Series(["1", "2"], dtype="object")
    })

    cleaning_parameters  = {"heterogeneous_columns" : {"col2", "col3"}}
    DF_ = ensure_type_uniformity(DF2.copy(), **cleaning_parameters)

    assert (DF_.dtypes == DF.dtypes).all()


def test_type_unknown_values():
    df2  = DF2.copy()
    mask = pd.DataFrame({'col1': [True, True], 'col2': [False, True]})
    df   = df2.mask(mask)

    cleaning_parameters = {
        "generic_unknowns"  : {2},
        "specific_unknowns" : {'col1': {1}}
    }
    df_ = correct_unknown_values(df2, **cleaning_parameters)

    assert df.equals(df_)


def test_abnormal_values():
    df2  = DF2.copy()
    mask = pd.DataFrame({'col1': [False, True], 'col2': [False, False]})
    df   = df2.mask(mask)

    cleaning_parameters = {"limits" : [({'col1'}, (0, 1.5))]}
    df_ = remove_abnormal_values(df2, **cleaning_parameters)

    assert (df.equals(df_))


def test_categories():
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
    cleaning_parameters = {"references" : refs}
    DF_ = use_category_names(DF2.copy(), **cleaning_parameters)
    assert (DF_.equals(DF))


#####################
#      WRANGLE      #
#####################


def test_impute_bmi():
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

    output_df = impute_bmi(input_df, **{"bmi_limits" : (0, 100)})

    assert target_df.equals(output_df)


def test_transform_dialysis_columns():
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

    output_df = transform_dialysis_columns(input_df, **{"descriptor" : DESCRIPTOR2})

    assert target_df.equals(output_df)


def test_recompute_egfr():
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


def test_impute_biolevels():
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

def test_replace():
    cleaning_parameters = {"replacement_pairs" : [['LoWEr', 'uppER']]}

    df_target   = pd.DataFrame({'LoWEr': [1, 2]})
    df_obtained = replace(DF1, **cleaning_parameters)

    assert df_obtained.equals(df_target)

def test_categorise():
    cleaning_parameters = {"columns_to_categorise" : {"rweight": [0, 5]}}

    df_target = pd.DataFrame({
        'rweight': [
            "from 0 to 5",
            "after 5",
            "before 0"
        ]
    })
    df_obtained = categorise(DF5, **cleaning_parameters)

    assert df_obtained["cat_rweight"].equals(df_target["rweight"])

def test_impute_multirisk():
    df_target = pd.DataFrame({
        'mcens':
            [
                "Censored",
                "Graft failure",
                "Censored",
                "Deceased",
                "Censored",
                "Graft failure",
                "Deceased"
            ],
        'msurv':
            [0] * 7
    })
    df_obtained = impute_multirisk(DF6)

    assert df_obtained[['mcens', 'msurv']].equals(df_target)

def test_add_unknown_category():
    input_df = pd.DataFrame({"a": ["A", pd.NA], "b": [pd.NA, "B"]})
    output_df = add_unknown_category(input_df, **{"columns_with_unknowns" : ['a'], "unknown" : 42})
    target_df = pd.DataFrame({"a": ["A", 42], "b": [pd.NA, "B"]})

    assert output_df.equals(target_df)

def test_remove_irrelevant_categories():
    irrelevant_categories = {
        'a': ["B"],
        'c': ["A"]
    }

    cleaning_parameters = {"irrelevant_categories" : irrelevant_categories}

    input_df  = pd.DataFrame({"a": ["A", "B"], "b": ["A", "B"]})
    output_df = remove_irrelevant_categories(input_df, **cleaning_parameters)
    target_df = pd.DataFrame({"a": ["A"], "b": ["A"]})

    assert output_df.equals(target_df)

def test_remove_irrelevant_columns():
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
    })

    cleaning_parameters = {"descriptor" : DESCRIPTOR2, "irrelevant_columns" : ['dbmi']}
    output_df = remove_irrelevant_columns(input_df, **cleaning_parameters)

    assert target_df.equals(output_df)


def test_remove_constant_columns():
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
