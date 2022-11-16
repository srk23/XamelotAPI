# Allow to one hot encode categorical data.

import numpy  as np
import pandas as pd
import re

from xmlot.data.dataframes import build_empty_mask
from xmlot.data.describe   import Descriptor
from xmlot.misc.lists      import difference


class OneHotEncoder:
    def __init__(
            self,
            descriptor: Descriptor,
            separator="#",
            exceptions=(),
            default_categories=None
    ):
        """
        Args:
            - descriptor         : a Descriptor to hold meta-data about the DataFrame to encode
            - separator          : a string token to separate column names and categories in one-hot encoded columns.
            - exceptions         : a list of columns to not encode.
            - default_categories : a dictionary that maps a column to its default category
                                   (if not specified, that default category will be the first observed).
                                   We need to remember it during decoding without dummy columns.
        """
        self.m_descriptor = descriptor
        self.m_sep = separator
        self.m_exceptions = exceptions
        self.m_default_categories = default_categories if default_categories is not None else dict()

        # Store the mapping between categories (especially binary ones) and their order.
        # Categorical keys  : category -> order    : needed in encode only, no need to be stored.
        # Categorical values: order    -> category : needed in decode and thus must be stored.
        self.m_categorical_values = dict()

        # Store column types for further decoding
        self.m_dtypes = None

    # Getters / Setters #

    @property
    def descriptor(self):
        return self.m_descriptor

    @property
    def separator(self):
        return self.m_sep

    @property
    def exceptions(self):
        return self.m_exceptions

    @property
    def default_categories(self):
        return self.m_default_categories

    # Miscellaneous #

    def is_encoded(self, column):
        return self.separator in column

    def split_column_name(self, column):
        return re.split(self.separator, column)

    # Encode / Decode #

    def encode(self, df, with_dummy_columns=False):

        # Remark: dealing with dummy columns.
        #
        # In general, it is a good idea to omit one category in the encoding.
        # Indeed, a pure one hot encoding would induce some singularity in the data.
        # In other words, the corresponding matrix would not be invertible in that case.
        # Then, a given column can be seen as a linear combination of the other ones.

        encoded_df = df.copy().replace(pd.NA, np.nan)
        self.m_dtypes = df.dtypes

        idx_cols = list()
        new_dfs = list()
        encoded_columns_lists = dict()

        for column in encoded_df.columns:
            entry = self.descriptor.get_entry(column)

            # One-hot encode if...
            if entry.is_categorical \
                    and column not in self.exceptions \
                    and (not entry.is_binary or with_dummy_columns):

                encoded_columns_lists[column] = list()

                # Ensure there is a default category (if not, build it):
                # it will be dropped if removing dummy columns is required.
                if column not in self.default_categories.keys():
                    first_category = df[column].value_counts().index[0]
                    self.default_categories[column] = str(first_category)

                # If required, add a dummy column: use the default category for that.
                dummy_column = column + self.separator + self.default_categories[column]
                if with_dummy_columns:
                    encoded_df[dummy_column] = 1
                    idx_cols.append(dummy_column)
                    encoded_columns_lists[column].append(dummy_column)

                # Process each category, in order of importance.
                for category in df[column].value_counts().index:
                    # If the category is not default.
                    if str(category) != self.default_categories[column]:
                        new_column = column + self.separator + str(category)

                        # To prevent performance issues, we do not add encoded columns
                        # into encoded_df one by one;
                        # instead, we build separate DataFrames that we concatenate
                        # all at once at the end of this function.

                        new_df = pd.DataFrame([], columns=[], index=encoded_df.index)
                        new_df[new_column] = 0
                        new_df[new_column] = new_df[new_column].mask(df[column] == category, other=1)
                        new_dfs.append(new_df)
                        idx_cols.append(new_column)
                        encoded_columns_lists[column].append(new_column)

                        # Update encoded_df[dummy_column] if considering dummy columns.
                        if with_dummy_columns:
                            encoded_df[dummy_column] = encoded_df[dummy_column] - new_df[new_column]

                # Drop the old categorical column once processed
                encoded_df = encoded_df.drop(columns=[column])

            # If the column is categorical but do not require to be encoded:
            elif entry.is_categorical \
                    and (column in self.exceptions or (entry.is_binary and not with_dummy_columns)):

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
                        return np.nan

                encoded_df[column] = encoded_df[[column]].apply(_replace_categories_, axis=1)
                idx_cols.append(column)

            else:  # ... do not encode.
                idx_cols.append(column)

        # Concat the columns all together
        encoded_df = pd.concat([encoded_df, *new_dfs], axis=1)

        # Check for forgotten Nan and adjust.
        condition = build_empty_mask(encoded_df)
        for column, encoded_columns in encoded_columns_lists.items():
            condition_per_column = df[column].isna()
            for encoded_column in encoded_columns:
                condition[encoded_column] = condition_per_column
        encoded_df = encoded_df.mask(condition, other=np.nan)

        # Finally, re-order them, and uniformise types to `float32`.
        return encoded_df.reindex(idx_cols, axis=1).astype('float32')

    def decode(self, df):
        decoded_df = df.copy()

        columns_to_drop = list()
        idx_cols = list()

        for column in df.columns:
            if self.is_encoded(column):
                old_column, category = self.split_column_name(column)

                # If it is the first time the column is observed, we place an unknown value everywhere.
                if old_column not in decoded_df.columns:
                    decoded_df[old_column] = np.nan  # self.default_categories[old_column]
                    idx_cols.append(old_column)

                # Replace unknown values based on the encoded columns
                decoded_df[old_column] = decoded_df[old_column].mask(df[column] == 1, other=category)
                columns_to_drop.append(column)

                # If the dummy columns are not here, we need to add the default values for the categorical columns.
                entry = self.descriptor.get_entry(old_column)
                if entry.is_categorical and not entry.is_binary:
                    # Get the list of all the encoded columns related to old_column
                    r = re.compile("^" + old_column)
                    encoded_columns_list = list(filter(r.match, df.columns))

                    # Mask rows with only zeros with default category.
                    decoded_df[old_column] = decoded_df[old_column].mask(
                        (df[encoded_columns_list].sum(axis=1) == 0) & ~df[encoded_columns_list].isna().any(axis=1),
                        other=self.default_categories[old_column]
                    )

            else:  # ... the column has not been one hot encoded:
                   # it is either numerical, binary (in a "no dummy" case), or an exception.
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
                            return np.nan

                    decoded_df[column] = decoded_df[[column]].apply(_restore_categories_, axis=1)
                idx_cols.append(column)

        decoded_df    = decoded_df.drop(columns=columns_to_drop)
        decoded_df    = decoded_df.reindex(idx_cols, axis=1)
        index_to_drop = difference(pd.DataFrame(self.m_dtypes).transpose().columns, decoded_df.columns)
        return decoded_df.astype(self.m_dtypes.drop(index=index_to_drop))
