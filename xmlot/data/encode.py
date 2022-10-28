# Allow to one hot encode categorical data.

import pandas as pd
import re

from xmlot.data.describe      import Descriptor

class OneHotEncoder:
    def __init__(self, descriptor: Descriptor, separator="#", exceptions=(), default_categories=None):
        self.m_descriptor         = descriptor
        self.m_sep                = separator
        self.m_exceptions         = exceptions
        self.m_default_categories = default_categories if default_categories is not None else dict()
        self.m_categorical_values = dict()
        self.m_dtypes             = None

    @property
    def descriptor(self):
        return self.m_descriptor

    @property
    def separator(self):
        return self.m_sep

    @property
    def exceptions(self):
        return self.m_exceptions

    def is_encoded(self, column):
        return self.separator in column

    def separate(self, column):
        return re.split(self.separator, column)

    def encode(self, df):
        encoded_df = df.copy()
        self.m_dtypes = encoded_df.dtypes

        idx_cols = list()
        new_dfs = list()

        for column in encoded_df.columns:

            entry = self.descriptor.get_entry(column)
            if entry.is_categorical:
                if column not in self.exceptions and not entry.is_binary:  # Then, one-hot encode.
                    # If not specified, the first observed category will be considered as a "default value".
                    # A pure one hot encoding would indeed induce some singularity in the data.
                    # In other words, the corresponding matrix would not be invertible in that case.
                    # Indeed, the first column can be seen as a linear combination of the other ones.
                    # Therefore, we need to prevent its construction.
                    if column not in self.m_default_categories.keys():
                        default_category_not_defined_yet = True
                    else:
                        default_category_not_defined_yet = False

                    for category in encoded_df[column].value_counts().index:
                        if default_category_not_defined_yet:
                            self.m_default_categories[column] = str(category)
                            default_category_not_defined_yet  = False
                        elif str(category) != self.m_default_categories[column]:
                            new_column = column + self.separator + str(category)
                            # To prevent performance issues, we do not add encoded columns
                            # into encoded_df one by one;
                            # instead, we build separate DataFrames that we concatenate
                            # all at once at the end of this function.
                            new_df = pd.DataFrame([], columns=[], index=encoded_df.index)
                            new_df[new_column] = 0
                            new_df[new_column].mask(encoded_df[column] == category, other=1, inplace=True)
                            new_dfs.append(new_df)
                            idx_cols.append(new_column)
                    # Drop the old categorical column once processed
                    encoded_df = encoded_df.drop(columns=[column])
                else:  # If the column is an exception or binary, turn it into a numerical column
                    if entry.categorical_keys is not None:
                        categorical_keys = entry.categorical_keys
                    else:
                        categorical_keys = {k: v for (v, k) in enumerate(encoded_df[column].value_counts().index)}
                        self.m_categorical_values[column] = {
                            v: k for (v, k) in enumerate(encoded_df[column].value_counts().index)
                        }

                    def _replace_categories_(s):
                        try:
                            return categorical_keys[s[column]]
                        except KeyError:
                            return pd.NA
                            # v = max(categorical_keys.values()) + 1
                            # k = s[column]
                            # categorical_keys[k] = v
                            # self.m_categorical_values[column][v] = k
                            # return v

                    encoded_df[column] = encoded_df[[column]].apply(_replace_categories_, axis=1)
                    idx_cols.append(column)
            else:
                idx_cols.append(column)

        encoded_df = pd.concat([encoded_df, *new_dfs], axis=1)

        return encoded_df.reindex(idx_cols, axis=1).astype('float32')

    def decode(self, df):
        decoded_df = df.copy()
        columns_to_drop = list()
        idx_cols = list()

        for column in decoded_df.columns:
            if self.is_encoded(column):
                old_column, category = self.separate(column)

                # If it is the first time the column is observed, we place the default value everywhere.
                if old_column not in decoded_df.columns:
                    decoded_df[old_column] = self.m_default_categories[old_column]
                    idx_cols.append(old_column)

                decoded_df[old_column].mask(decoded_df[column] == 1, other=category, inplace=True)
                columns_to_drop.append(column)

            else:  # the column has not been one hot encoded: it is either numerical, binary, or an exception
                entry = self.m_descriptor.get_entry(column)
                if entry.is_numerical:  # Nothing is required
                    pass
                else:  # The column is either binary or an exception
                    if entry.categorical_values is not None:
                        categorical_values = entry.categorical_values
                    else:
                        categorical_values = self.m_categorical_values[column]

                    def _restore_categories_(s):
                        try:
                            return categorical_values[s[column]]
                        except KeyError:
                            return pd.NA

                    decoded_df[column] = decoded_df[[column]].apply(_restore_categories_, axis=1)
                idx_cols.append(column)

        decoded_df = decoded_df.drop(columns=columns_to_drop)
        return decoded_df.reindex(idx_cols, axis=1).astype(self.m_dtypes)
