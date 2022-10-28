# Functions to standardise (or normalise) numerical data.

import re

def get_standardisation(df):
    return {
        "get_center" : lambda col: df[col].mean(),
        "get_scale"  : lambda col: df[col].std()
    }

def get_normalisation(df):
    return {
        "get_center" : lambda col: df[col].min(),
        "get_scale"  : lambda col: df[col].max() - df[col].min()
    }

class Standardiser:
    def __init__(
            self,
            df,
            accessor_code,
            ohe,
            get_center=lambda col: 0,
            get_scale =lambda col: 1,
            # standardise_target_duration=False
    ):
        descriptor = ohe.descriptor
        separator  = ohe.separator

        self.m_columns_to_standardise = list()
        for column in getattr(df, accessor_code).features_list:
            if not re.findall(separator, column):
                if descriptor.get_entry(column).is_numerical:
                    self.m_columns_to_standardise.append(column)

        self.m_centers = {column: get_center(column) for column in self.m_columns_to_standardise}
        self.m_scales  = {column: get_scale(column)  for column in self.m_columns_to_standardise}

        assert 0 not in self.m_scales.values(), "One or more columns are constant, inducing a division by zero."

    def __call__(self, df):
        def _standardise_(s):
            return (s - self.m_centers[s.name]) / self.m_scales[s.name]

        output_df = df.copy()
        output_df[self.m_columns_to_standardise] = output_df[self.m_columns_to_standardise].apply(_standardise_)

        return output_df

    def undo(self, df):
        """
        From a standardised DataFrame, returns its unstandardised version.
        """
        def _undo_(s):
            return (s * self.m_scales[s.name]) + self.m_centers[s.name]

        output_df = df.copy()
        output_df[self.m_columns_to_standardise] = output_df[self.m_columns_to_standardise].apply(_undo_)
        return output_df
