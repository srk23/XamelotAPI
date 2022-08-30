import pandas as pd

from project.data.split       import split_dataset


##########################
#      DATA MANAGER      #
##########################

class DataManager:
    """
    Wraps a DataFrame to add the notion of input features and targets.
    """
    # That static attribute is used to address copy and split operations.
    # Subclasses will have their own version of that attribute.
    # It is simply
    s_attribute_matcher = {
        'm_targets': 'targets_list',
        'm_target': 'reference_target',
        'm_ohe': 'ohe'
    }

    def __init__(self, df, targets_list, reference_target=None, ohe=None):
        """
        Attributes:
            - df               : the wrapped DataFrame
            - targets          : the list of columns corresponding to targets
            - reference_target : a default target used for stratification during split operations
                                 (shall be a set of columns instead?)
            - ohe              : the OneHotEncoder that can decode the data towards a more explicit format.
        """
        self.m_df = df
        self.m_targets = targets_list
        self.m_target = reference_target
        self.m_ohe = ohe

    @property
    def df(self):
        return self.m_df

    @property
    def targets_list(self):
        return list(self.m_targets)

    @property
    def targets(self):
        return self.df[self.targets_list]

    @property
    def features_list(self):
        return list(
            set(self.df.columns) - set(self.targets_list)
        )

    @property
    def features(self):
        return self.df[self.features_list]

    @property
    def reference_target(self):
        return self.m_target

    @property
    def ohe(self):
        return self.m_ohe

    def copy(self):
        matcher = type(self).s_attribute_matcher
        inputs = {'df': self.m_df.copy()}
        for attribute, input_name in matcher.items():
            inputs[input_name] = getattr(self, attribute)
        return type(self)(**inputs)

    def equals(self, dm):
        if type(self) is type(dm):
            result = self.df.equals(dm.df)

            for attribute in type(self).s_attribute_matcher.keys():
                result &= (getattr(self, attribute) == getattr(dm, attribute))
            return result
        else:
            return False

    def concat(self, dm):
        if not self.df.index.intersection(dm.df.index).empty:
            raise ValueError("Ensure that indexes do not overlap before concatenation.")

        matcher = type(self).s_attribute_matcher
        inputs = {'df': pd.concat([self.df, dm.df])}
        for attribute, input_name in matcher.items():
            inputs[input_name] = getattr(self, attribute)

        return type(self)(**inputs)

    def split(self, fracs, target=None):
        reference_target = target if target else self.reference_target
        if reference_target is None:
            raise AttributeError("Please specify a reference target for stratification.")
        matcher = type(self).s_attribute_matcher

        splits = split_dataset(self.df, fracs, reference_target)

        inputs = {
            input_name: getattr(self, attribute)
            for attribute, input_name in matcher.items()
        }
        return [type(self)(splitted_df, **inputs) for splitted_df in splits]


class SurvivalDataManager(DataManager):
    """
    Targets become a pair (event occurence, censoring time).
    """
    s_attribute_matcher = {
        'm_event': 'event',
        'm_duration': 'duration',
        'm_ohe': 'ohe'
    }

    def __init__(self, df, event, duration, ohe=None):
        if event not in df.columns:
            raise ValueError("event")

        if duration not in df.columns:
            raise ValueError("duration")

        super().__init__(df, targets_list=[event, duration], reference_target=event, ohe=ohe)

        self.m_event = event
        self.m_duration = duration

    @property
    def covariates_list(self):
        return self.features_list

    @property
    def covariates(self):
        return self.features

    @property
    def event_name(self):
        return self.m_event

    @property
    def events(self):
        return self.m_df[self.m_event]

    @property
    def duration_name(self):
        return self.m_duration

    @property
    def durations(self):
        return self.m_df[self.m_duration]


class SingleTargetDataManager(DataManager):
    """
    Targets is a singleton.
    """
    s_attribute_matcher = {
        'm_target': 'target',
        'm_ohe': 'ohe'
    }

    def __init__(self, df, target, ohe=None):
        if target not in df.columns:
            raise ValueError("label")

        super().__init__(df, [target], reference_target=target, ohe=ohe)

    @property
    def covariates_list(self):
        return self.features_list

    @property
    def covariates(self):
        return self.features

    @property
    def target_name(self):
        return self.reference_target


#####################################
#      MISCELLANEOUS FUNCTIONS      #
#####################################


def get_covariates(df, event, duration):
    return [column for column in df.columns if column not in {event, duration}]

def concat_dms(list_of_dms):
    result = list_of_dms[0]
    for dm in list_of_dms[1:]:
        result = result.concat(dm)
    return result
