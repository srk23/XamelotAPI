import pandas as pd

from project.data.split       import split_dataset

def get_covariates(df, event, duration):
    return [column for column in df.columns if column not in {event, duration}]

class SurvivalDataManager:
    def __init__(self, df, event, duration, ohe=None):
        if event not in df.columns:
            raise ValueError("event")

        if duration not in df.columns:
            raise ValueError("duration")

        self.m_df         = df
        self.m_covariates = get_covariates(df, event, duration)
        self.m_event      = event
        self.m_duration   = duration
        self.m_ohe        = ohe

    @property
    def df(self):
        return self.m_df

    @property
    def covariates_list(self):
        return self.m_covariates

    @property
    def event_name(self):
        return self.m_event

    @property
    def duration_name(self):
        return self.m_duration

    @property
    def covariates(self):
        return self.m_df[self.m_covariates]

    @property
    def events(self):
        return self.m_df[self.m_event]

    @property
    def durations(self):
        return self.m_df[self.m_duration]

    @property
    def ohe(self):
        return self.m_ohe

    def copy(self):
        return SurvivalDataManager(
            self.m_df.copy(),
            self.m_event,
            self.m_duration
        )

    def equals(self, sdm):
        return self.df.equals(sdm.df)            \
           and self.event_name == sdm.event_name \
           and self.duration_name == sdm.duration_name


def split_sdm(sdm, fracs):
    splits = split_dataset(sdm.df, fracs, sdm.event_name)
    return [SurvivalDataManager(splitted_df, sdm.event_name, sdm.duration_name, sdm.ohe) for splitted_df in splits]


def concat_sdms(list_of_sdms):
    event    = list_of_sdms[0].event_name
    duration = list_of_sdms[0].duration_name
    ohe      = list_of_sdms[0].ohe
    df = pd.concat(map(lambda sdm: sdm.df, list_of_sdms))
    return SurvivalDataManager(df, event, duration, ohe)
