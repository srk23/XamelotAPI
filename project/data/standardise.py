# Functions to standardise numerical data
import re

def get_standardisation(sdm):
    return {
        "get_center" : lambda col: sdm.df[col].mean(),
        "get_scale"  : lambda col: sdm.df[col].std()
    }

def get_normalisation(sdm):
    return {
        "get_center" : lambda col: sdm.df[col].min(),
        "get_scale"  : lambda col: sdm.df[col].max() - sdm.df[col].min()
    }

class Standardiser:
    def __init__(
            self,
            sdm,
            get_center=lambda col: 0,
            get_scale =lambda col: 1,
            standardise_target_duration=False
    ):
        descriptor = sdm.ohe.descriptor
        separator  = sdm.ohe.separator

        self.m_columns_to_standardise = list()
        for column in sdm.df.columns:
            if not re.findall(separator, column):
                if descriptor.get_entry(column).is_numerical:
                    self.m_columns_to_standardise.append(column)

        if not standardise_target_duration:
            self.m_columns_to_standardise = list(set(self.m_columns_to_standardise) - {sdm.duration_name})

        self.m_centers = {column: get_center(column) for column in self.m_columns_to_standardise}
        self.m_scales  = {column: get_scale(column)  for column in self.m_columns_to_standardise}

    def __call__(self, sdm):
        def _standardise_(s):
            return (s - self.m_centers[s.name]) / self.m_scales[s.name]

        output_sdm = sdm.copy()
        output_sdm.df[self.m_columns_to_standardise] = output_sdm.df[self.m_columns_to_standardise].apply(_standardise_)
        return output_sdm
