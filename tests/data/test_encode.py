import pandas as pd
import numpy  as np

from xmlot.data.describe import Entry, Descriptor
from xmlot.data.encode   import OneHotEncoder


X    = np.nan

DESCRIPTOR = Descriptor(
    [
        Entry("animal", is_categorical=True),
        Entry("legs"  , is_categorical=True, categorical_keys={2: 0, 4: 1}),
        Entry("weight", is_categorical=False),
        Entry(
            "col1",
            is_categorical=True,
            categorical_keys={"A": 0, "B": 1}
        ),
        Entry(
            "col2",
            is_categorical=True
        )
    ]
)

DF1 = pd.DataFrame({"animal": ["Happy chicken", "Cow"], "legs": [2, 4], "weight": [1.7, 1000]})
DF1 = DF1.astype({"legs": "Int64", "weight": "float32"}, errors="raise")
# Note: Doing float64 -> float32 -> float64 may induce some precision errors.

DF2 = pd.DataFrame({
    "animal#Happy chicken": [1, 0],
    "legs": [0, 1],
    "weight": [1.7, 1000]
}).astype('float32')

DF3  = pd.DataFrame({
    "col1#A": [X, 0, 0, X, 1, X, 1, 0],
    "col1#B": [X, 0, X, 0, X, 1, 0, 1]}
)

DF4 = pd.DataFrame({"col2": [X, 0, 1, 2]})


def test_encode_decode_without_nan():
    original_df = DF1

    ohe = OneHotEncoder(DESCRIPTOR, separator="#", default_categories={"animal": "Cow"})

    encoded_df = ohe.encode(original_df)

    assert encoded_df.equals(DF2)

    decoded_df = ohe.decode(encoded_df)

    assert decoded_df.equals(original_df)

def test_decode_binary():
    ohe = OneHotEncoder(
        descriptor=DESCRIPTOR
    )
    ohe.m_dtypes = pd.DataFrame({"col1": ["A"]}).dtypes
    assert ohe.decode(DF3).equals(
        pd.DataFrame({"col1": [X, X, X, X, "A", "B", "A", "B"]})
    )

def test_encode_decode_dummy():
    # Without dummy column
    ohe = OneHotEncoder(descriptor=DESCRIPTOR)
    encoded = ohe.encode(DF4, with_dummy_columns=False)
    assert encoded.equals(pd.DataFrame({
        "col2#1.0": [X, 0., 1., 0.],
        "col2#2.0": [X, 0., 0., 1.]
    }).astype('float32'))

    decoded = ohe.decode(encoded)
    assert decoded.equals(DF4)

    # With dummy column
    ohe = OneHotEncoder(descriptor=DESCRIPTOR)
    encoded = ohe.encode(DF4, with_dummy_columns=True)

    print(pd.concat([DF4, encoded], axis=1))

    assert encoded.equals(pd.DataFrame({
        "col2#0.0": [X, 1., 0., 0.],
        "col2#1.0": [X, 0., 1., 0.],
        "col2#2.0": [X, 0., 0., 1.]
    }).astype('float32'))

    decoded = ohe.decode(encoded)
    assert decoded.equals(DF4)

# 1)
#
# x   = np.nan
# df  = pd.DataFrame({"rsex#Male":[x, 1, x, 0, x, 0, 0, 1, 1], "rsex#Female": [x, x, 1, x, 0, 0, 1, 0, 1]})
#
# ohe = OneHotEncoder(
#     descriptor=DESCRIPTOR
# )
# ohe.m_dtypes = DATAFRAME[["rsex"]].dtypes
# pd.concat([df, ohe.decode(df)], axis=1)
#
#
# 2)
#
# x   = np.nan
# df  = pd.DataFrame({"alt_trend": [x, 0, 1, -1]})
#
# ohe = OneHotEncoder(
#     descriptor=DESCRIPTOR
# )
#
# encoded = ohe.encode(df, with_dummy_columns=False)
# decoded = ohe.decode(encoded)
# pd.concat([df, decoded, encoded], axis=1)
