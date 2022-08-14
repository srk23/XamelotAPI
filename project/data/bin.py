import numpy as np

from project.data.survival_data_manager import SurvivalDataManager

class Binner:
    def __init__(
            self,
            centers,

    ):
        self.m_centers = sorted(centers)

    def __call__(self, sdm):
        def _get_bin_(t):
            i_min = 0
            i_max = len(self.m_centers) - 1
            i     = int(np.round((i_max + i_min) / 2))

            while i_max - i_min > 1:
                c = self.m_centers[i]

                if t - c == 0:
                    return i
                if t - c > 0:
                    i_min = i
                if t - c < 0:
                    i_max = i
                i = int(np.round((i_max + i_min) / 2))

            # print(
            #     "i_min:", i_min, "\n",
            #     "i    :", i, "\n",
            #     "i_max:", i_max, "\n",
            #     "t    :", t, "\n",
            #     "c_min:", self.m_centers[max(i-1, 0)], "\n",
            #     "c    :", self.m_centers[i], "\n",
            #     "c_max:", self.m_centers[min(i+1, len(self.m_centers)-1)], "\n\n"
            # )
            return i

        df = sdm.df.copy()
        df[sdm.duration_name] = df[sdm.duration_name].apply(_get_bin_)
        return SurvivalDataManager(df, sdm.event_name, sdm.duration_name, ohe=sdm.ohe)


class EquidistantBinner(Binner):
    def __init__(self, sdm, num_bins=None):
        self.m_min_value = sdm.durations.min()
        self.m_max_value = sdm.durations.max()
        self.m_num_bins  = num_bins if num_bins else int(self.m_max_value - self.m_min_value)

        step = (self.m_max_value - self.m_min_value) / self.m_num_bins
        self.m_cuts = [self.m_min_value + step * i for i in range(self.m_num_bins + 1)]

        centers = [(self.m_cuts[i + 1] + self.m_cuts[i]) / 2 for i in range(self.m_num_bins)]
        super().__init__(centers)

    @property
    def cuts(self):
        return self.m_cuts

    @property
    def num_bins(self):
        return self.m_num_bins
