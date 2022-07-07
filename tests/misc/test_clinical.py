import numpy                 as     np
from   project.misc.clinical import compute_bmi, compute_egfr

def test_compute_bmi():
    height = 200
    weight = 100
    assert compute_bmi(weight, height) == 25

def test_compute_egfr():
    age        = 45
    creatinine = 0.7126698600000001 / 113.1222 * 10000
    sex        = "Female"
    height     = 158

    assert np.abs(compute_egfr(age, creatinine, sex, height) - 106.31023857221831) < 1e-3

    age        = 8
    creatinine = 0.59954766 / 113.1222 * 10000
    sex        = "Male"
    height     = 132

    assert np.abs(compute_egfr(age, creatinine, sex, height) - 90.928551034625) < 1e-3
