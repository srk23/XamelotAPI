import pandas as pd

from project.data.describe import Entry, Descriptor
from project.data.encode   import OneHotEncoder

DESCRIPTOR = Descriptor(
    [
        Entry("animal", is_categorical=True),
        Entry("legs", is_categorical=True, categorical_keys={2: 0, 4: 1}),
        Entry("weight", is_categorical=False)
    ]
)

DF1 = pd.DataFrame({"animal": ["Happy chicken", "Cow"], "legs": [2, 4], "weight": [1.7, 1000]})
DF1["legs"] = DF1["legs"].astype("Int64", errors="raise")

DF2 = pd.DataFrame({
    "animal#Happy chicken": [1, 0],
    "legs": [0, 1],
    "weight": [1.7, 1000]
})


def test_encode_decode():
    original_df = DF1

    print(original_df)

    ohe = OneHotEncoder(DESCRIPTOR, separator="#", default_categories={"animal": "Cow"})

    encoded_df = ohe.encode(original_df)

    print()
    print(encoded_df)

    assert encoded_df.equals(DF2)

    decoded_df = ohe.decode(encoded_df)

    print()
    print(decoded_df)
    print(decoded_df.dtypes)

    print(DF1.dtypes)

    assert decoded_df.equals(original_df)

test_encode_decode()