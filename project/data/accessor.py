# Build pandas' accessors to easily access features, targets, events, durations, in a DataFrame
# More information about accessors at:
# https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors

import pandas as pd

from warnings   import catch_warnings, simplefilter
from contextlib import nullcontext

from project.misc.list_operation import difference


def build_survival_accessor(event, duration, accessor_code="surv", disable_warning=True):
    """
    Build an accessor dedicated to the management of survival data.

    Args:
        - event           : event name;
        - duration        : duration name;
        - accessor_code   : code to access extended properties (for example: `df.surv.event`);
        - disable_warning : remove warnings linked to several uses of that function.

    Returns: the corresponding accessor class;
             instances are built when called from the DataFrame (for example: `df.surv`)
    """
    context_manager = catch_warnings() if disable_warning else nullcontext()
    with context_manager:
        if disable_warning:
            simplefilter("ignore")

        @pd.api.extensions.register_dataframe_accessor(accessor_code)
        class SurvivalAccessor:
            def __init__(self, pandas_obj):
                self._validate(pandas_obj)
                self._obj = pandas_obj

                self.m_event = event
                self.m_duration = duration

            @staticmethod
            def _validate(obj):
                # verify there is a column latitude and a column longitude
                if event not in obj.columns or duration not in obj.columns:
                    raise AttributeError("Must have {0} and {1}.".format(event, duration))

            @property
            def event(self):
                return self.m_event

            @property
            def events(self):
                return self._obj[self.event]

            @property
            def duration(self):
                return self.m_duration

            @property
            def durations(self):
                return self._obj[self.duration]

            @property
            def target(self):
                return [self.event, self.duration]

            @property
            def targets(self):
                return self._obj[self.target]

            @property
            def features_list(self):
                return difference(self._obj.columns, self.target)

            @property
            def features(self):
                return self._obj[self.features_list]

    return SurvivalAccessor


def build_classification_accessor(target, accessor_code="class", disable_warning=True):
    """
    Build an accessor dedicated to the management of data for classification.

    Args:
        - target          : target name;
        - accessor_code   : code to access extended properties (for example: `df.class.event`);
        - disable_warning : remove warnings linked to several uses of that function.

    Returns: the corresponding accessor class;
             instances are built when called from the DataFrame (for example: `df.class`)
    """
    context_manager = catch_warnings() if disable_warning else nullcontext()
    with context_manager:
        if disable_warning:
            simplefilter("ignore")

        @pd.api.extensions.register_dataframe_accessor(accessor_code)
        class ClassificationAccessor:
            def __init__(self, pandas_obj):
                self._validate(pandas_obj)
                self._obj = pandas_obj

                self.m_target = target

            @staticmethod
            def _validate(obj):
                # verify there is a column latitude and a column longitude
                if target not in obj.columns:
                    raise AttributeError("Must have {0}.".format(target))

            @property
            def target(self):
                return [self.m_target]

            @property
            def targets(self):
                return self._obj[self.target]

            @property
            def features_list(self):
                return difference(self._obj.columns, self.target)

            @property
            def features(self):
                return self._obj[self.features_list]

    return ClassificationAccessor
