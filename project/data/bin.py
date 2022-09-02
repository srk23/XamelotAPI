import numpy        as     np
import warnings
from   lifelines    import KaplanMeierFitter
from   pycox.models import DeepHitSingle
from   contextlib   import nullcontext

from project.data.datamanager import SurvivalDataManager


class Binner:
    def __init__(self, cuts):
        self.m_cuts = sorted(cuts)

    @property
    def cuts(self):
        return self.m_cuts

    @property
    def num_bins(self):
        return len(self.cuts) + 1

    def __call__(self, dm, target=None):
        def _get_bin_(t):
            i_min = 0
            i_max = len(self.m_cuts) - 1
            i = int(np.round((i_max + i_min) / 2))

            if t <= self.cuts[i_min]:
                return i_min
            if t > self.cuts[i_max]:
                return i_max + 1
            else:
                while True:
                    c = self.cuts[i]
                    if t - c < 0:
                        i_min = i_min
                        i_max = i
                        i = int(np.floor((i_max + i_min) / 2))
                    else:
                        i_min = i
                        i_max = i_max
                        i = int(np.ceil((i_max + i_min) / 2))
                    if i_max - i_min <= 1:
                        break
            return i_max

        target = target if target else dm.reference_target
        df = dm.df.copy()
        df[target] = df[target].apply(_get_bin_)

        dm = dm.copy()
        dm.df = df

        return dm


class EquidistantBinner(Binner):
    def __init__(self, dm, target=None, num_bins=None, pycox_style=False):

        target = target if target else dm.reference_target

        self.m_min_value = dm.df[target].min()
        self.m_max_value = dm.df[target].max()
        self.m_num_bins = num_bins if num_bins else int(self.m_max_value - self.m_min_value)

        if pycox_style:
            cuts = np.linspace(self.m_min_value, self.m_max_value, self.m_num_bins)
        else:
            step = (self.m_max_value - self.m_min_value) / self.m_num_bins
            cuts = [self.m_min_value + step * i for i in range(1, self.m_num_bins)]

        super().__init__(cuts)


class QuantileBinner(Binner):
    def __init__(self, sdm, num_bins=None, pycox_style=False, enabled_warnings=False):
        context_manager = warnings.catch_warnings() if enabled_warnings else nullcontext()
        with context_manager:
            warnings.simplefilter("ignore")

            self.m_kmf = KaplanMeierFitter()
            self.m_kmf.fit(sdm.durations, event_observed=sdm.events)

            kmf_min = self.m_kmf.survival_function_.min()["KM_estimate"]
            kmf_max = self.m_kmf.survival_function_.max()["KM_estimate"]

            if pycox_style:
                survival_thresholds = np.linspace(kmf_min, kmf_max, num_bins - 1)
            else:
                step = (kmf_max - kmf_min) / num_bins
                survival_thresholds = [kmf_max - i * step for i in range(1, num_bins)]

            p = list()

            for s in survival_thresholds:
                p.append(self.m_kmf.percentile(s))

        super().__init__(np.unique(p))


class PycoxBinner:
    @staticmethod
    def _get_target_(sdm_):
        return sdm_.durations.values, sdm_.events.values

    def __init__(self, sdm, num_bins=None, scheme='equidistant', enabled_warnings=False):
        context_manager = warnings.catch_warnings() if enabled_warnings else nullcontext()
        with context_manager:
            warnings.simplefilter("ignore")
            self.m_labtrans = DeepHitSingle.label_transform(num_bins, scheme=scheme)
            self.m_labtrans.fit_transform(*self._get_target_(sdm))

    def __call__(self, sdm):
        df = sdm.df.copy()
        new_durations, new_events = self.m_labtrans.transform(*self._get_target_(sdm))
        df[sdm.duration_name] = new_durations
        df[sdm.event_name]    = new_events

        return SurvivalDataManager(df, sdm.event_name, sdm.duration_name, ohe=sdm.ohe)


#
# class Binner:
#     def __init__(
#             self,
#             centers,
#
#     ):
#         self.m_centers = sorted(centers)
#
#     def __call__(self, dm, target=None):
#         def _get_bin_(t):
#             i_min = 0
#             i_max = len(self.m_centers) - 1
#             i = int(np.round((i_max + i_min) / 2))
#
#             while True:
#                 c = self.m_centers[i]
#                 if t - c == 0:
#                     return i
#                 if t - c > 0:
#                     i_min = i
#                     i = int(np.ceil((i_max + i_min) / 2))
#
#                 if t - c < 0:
#                     i_max = i
#                     i = int(np.floor((i_max + i_min) / 2))
#
#                 if i_max - i_min <= 1:
#                     break
#
#             if np.abs(t - self.m_centers[i_min]) < np.abs(t - self.m_centers[i_max]):
#                 return i_min
#             else:
#                 return i_max
#
#         target = target if target else dm.reference_target
#         df = dm.df.copy()
#         df[target] = df[target].apply(_get_bin_)
#
#         output    = dm.copy()
#         output.df = df
#         return output
#
#
# class EquidistantBinner(Binner):
#     def __init__(self, dm, target=None, num_bins=None):
#         self.m_min_value = dm.df[target].min()
#         self.m_max_value = dm.df[target].max()
#         self.m_num_bins  = num_bins if num_bins else int(self.m_max_value - self.m_min_value)
#
#         step = (self.m_max_value - self.m_min_value) / self.m_num_bins
#         self.m_cuts = [self.m_min_value + step * i for i in range(self.m_num_bins + 1)]
#
#         centers = [(self.m_cuts[i + 1] + self.m_cuts[i]) / 2 for i in range(self.m_num_bins)]
#         super().__init__(centers)
#
#     @property
#     def cuts(self):
#         return self.m_cuts
#
#     @property
#     def num_bins(self):
#         return self.m_num_bins
#
#
# from pycox.models import DeepHitSingle
#
#
# def _get_target_(sdm_):
#     return sdm_.durations.values, sdm_.events.values
#
# class PycoxBinner:
#     def __init__(self, sdm, num_bins=None):
#         self.m_labtrans = DeepHitSingle.label_transform(num_bins)
#         self.m_labtrans.fit_transform(*_get_target_(sdm))
#
#     def __call__(self, sdm):
#         df = sdm.df.copy()
#         new_durations, new_events = self.m_labtrans.transform(*_get_target_(sdm))
#         df[sdm.duration_name] = new_durations
#         df[sdm.event_name]    = new_events
#
#         return SurvivalDataManager(df, sdm.event_name, sdm.duration_name, ohe=sdm.ohe)
#
#
