import pandas as pd

from project.data.describe import Entry, Descriptor
from project.data.encode   import OneHotEncoder

DESCRIPTOR = Descriptor(
    [
        Entry("animal", is_categorical=True),
        Entry("legs", is_categorical=True, binary_keys={2: 0, 4: 1}),
        Entry("weight", is_categorical=False)
    ]
)

DF1 = pd.DataFrame({"animal": ["Cow", "Happy chicken"], "legs": [4, 2], "weight": [1000, 1.7]})
DF1["legs"] = DF1["legs"].astype("Int64", errors="raise")

DF2 = pd.DataFrame({
    "animal#Happy chicken": [0, 1],
    "legs": [1, 0],
    "weight": [1000, 1.7]
})


def test_encode_decode():
    original_df = DF1

    print(original_df)

    ohe = OneHotEncoder(DESCRIPTOR, separator="#")

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
