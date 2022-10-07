import pandas as pd

from project.data.describe import Entry, Descriptor
from project.data.encode   import OneHotEncoder


DESCRIPTOR = Descriptor(
    [
        Entry("animal", is_categorical=True),
        Entry("legs"  , is_categorical=True, categorical_keys={2: 0, 4: 1}),
        Entry("weight", is_categorical=False)
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


def test_encode_decode():
    original_df = DF1

    ohe = OneHotEncoder(DESCRIPTOR, separator="#", default_categories={"animal": "Cow"})

    encoded_df = ohe.encode(original_df)

    assert encoded_df.equals(DF2)

    decoded_df = ohe.decode(encoded_df)

    assert decoded_df.equals(original_df)
