# Transform a continuous-time survival analysis problem into a discrete one.
# This transformation is sometimes required by models (e.g. DeepHit) where time values belong to a grid.
import warnings

import numpy        as     np

from   contextlib   import nullcontext
from   lifelines    import KaplanMeierFitter
from   pycox.models import DeepHitSingle


class Discretiser:
    def __init__(self, grid, accessor_code):
        """
        Args:
            - cuts          : a list of values that define a partition of the real axis;
            - accessor_code : a code pointing to a SurvivalAccessor
        """
        self.m_grid          = sorted(grid)
        self.m_accessor_code = accessor_code

    @property
    def grid(self):
        return self.m_grid

    @property
    def size_grid(self):
        return len(self.grid)

    def __call__(self, df):
        def _get_interval_(t):
            """
            A dichotomy to know in which interval lie a given time point.
            """
            i_min = 0
            i_max = len(self.grid) - 1
            i = (i_max + i_min) // 2

            if t <= self.grid[i_min]:
                return i_min, i_min
            if t > self.grid[i_max]:
                return i_max, i_max
            else:
                while True:
                    c = self.grid[i]
                    if t - c <= 0:
                        i_min = i_min
                        i_max = i
                        i = (i_max + i_min) // 2
                    else:
                        i_min = i
                        i_max = i_max
                        i = (i_max + i_min) // 2
                    if i_max - i_min <= 1:
                        break

            return i_min, i_max

        def _get_position_on_grid_(s, e, d):
            """
            Depending on whether the event has been censored or not, survival time will not be recorded the same way:
                - Uncensored events are recorded at the next available time point on the grid.
                - When censored, information of previous time point on the grid are available, but not next ones.
            """
            i_min, i_max = _get_interval_(s[d])

            if s[e] == 0:
                return i_min
            else:
                return i_max

        df_ = df.copy()

        event, duration = getattr(df, self.m_accessor_code).target
        df_[getattr(df_, self.m_accessor_code).duration] = df_.apply(
            lambda s: _get_position_on_grid_(s, e=event, d=duration), axis=1
        )

        return df_


class EquidistantDiscretiser(Discretiser):
    """
    Equivalent to discretising with PyCox under the `equidistant` scheme.
    """
    def __init__(self, df, accessor_code, size_grid):
        min_value = getattr(df, accessor_code).durations.min()
        max_value = getattr(df, accessor_code).durations.max()

        grid = np.linspace(min_value, max_value, size_grid)

        super().__init__(grid, accessor_code)


class QuantileDiscretiser(Discretiser):
    """
    Equivalent to discretising with PyCox under the `quantiles` scheme.
    """
    def __init__(self, df, accessor_code, size_grid, disabled_warnings=True):
        context_manager = warnings.catch_warnings() if disabled_warnings else nullcontext()
        with context_manager:
            warnings.simplefilter("ignore")

            kmf = KaplanMeierFitter()
            kmf.fit(
                getattr(df, accessor_code).durations,
                event_observed=getattr(df, accessor_code).events
            )

            kmf_min = kmf.survival_function_.min()["KM_estimate"]
            kmf_max = kmf.survival_function_.max()["KM_estimate"]

            survival_thresholds = np.linspace(kmf_min, kmf_max, size_grid)

            p = list()
            for s in survival_thresholds:
                p.append(kmf.percentile(s))

        grid = np.unique(p)

        super().__init__(grid, accessor_code)


class PycoxDiscretiser(Discretiser):
    """
    Wrap Pycox discretisation methods.
    """
    def _get_target_(self, df):
        accessor = getattr(df, self.m_accessor_code)
        return accessor.durations.values, accessor.events.values

    def __init__(self, df, accessor_code, size_grid, scheme='equidistant', disable_warning=True):
        self.m_accessor_code = accessor_code

        context_manager = warnings.catch_warnings() if disable_warning else nullcontext()
        with context_manager:
            if disable_warning:
                warnings.simplefilter("ignore")

            self.m_labtrans = DeepHitSingle.label_transform(size_grid, scheme=scheme)
            self.m_labtrans.fit_transform(*self._get_target_(df))

        super().__init__(self.m_labtrans.cuts, self.m_accessor_code)

    def __call__(self, df):
        df_ = df.copy()

        new_durations, new_events = self.m_labtrans.transform(*self._get_target_(df_))
        df_[getattr(df_, self.m_accessor_code).duration] = new_durations

        return df_


class BalancedDiscretiser(Discretiser):
    def __init__(self, df, accessor_code, size_grid):
        def _find_cut_(targetted_value, values, times):
            """
            Given
            """
            i_min = 0
            i_max = len(values) - 1
            i = (i_max + i_min) // 2

            while i_max - i_min > 1:
                if targetted_value == values[i]:
                    return times[i]
                if targetted_value < values[i]:
                    i_max = i
                else:
                    i_min = i
                i = (i_max + i_min) // 2

            if values[i_max] - targetted_value <= targetted_value - values[i_min]:
                return times[i_max]
            else:
                return times[i_min]

        value_counts = getattr(df, accessor_code).durations.value_counts()  # DATAFRAME['msurv'].value_counts()

        t_axis = list(value_counts.index)
        x_axis = list(np.squeeze(value_counts.values))
        t_axis, x_axis = zip(*sorted(zip(t_axis, x_axis)))
        x_axis = np.cumsum(x_axis)

        x_bins = np.linspace(x_axis[0], x_axis[-1], size_grid)
        grid = [_find_cut_(x, x_axis, t_axis) for x in x_bins]

        super().__init__(grid, accessor_code)
