# A set of clinical formulas.

import numpy  as np


def compute_bmi(weight, height):
    return np.round(1e4 * weight / height ** 2, decimals=1)

def compute_egfr(age, creatinine, sex, height):
    creatinine = 113.1222 * 1e-4 * creatinine

    # https://www.kidney.org/content/ckd-epi-creatinine-equation-2021
    if age > 16:
        if sex == "Female":
            kappa = 0.7
            alpha = -0.241
            scoef = 1.012
        else:
            kappa = 0.9
            alpha = -0.302
            scoef = 1

        min_ = min(creatinine / kappa, 1)
        max_ = max(creatinine / kappa, 1)

        return 142 * (min_ ** alpha) * (max_ ** -1.2) * (0.9938 ** age) * scoef
    # https://qxmd.com/calculate/calculator_281/schwartz-pediatric-bedside-egfr-2009
    else:
        return 0.413 * (height / creatinine)
