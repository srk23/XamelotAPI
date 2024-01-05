# Wrap various classifiers from Lifelines.
# Everything follows Model's design.
#
# More details on: https://lifelines.readthedocs.io/en/latest/

import lifelines

from xmlot.models.model import FromTheShelfModel


class LifelinesCoxModel(FromTheShelfModel):
    """
    cf. https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html

    Hyperparameters:
        - alpha                      : float, optional (default=0.05) - the level in the confidence intervals.
        - baseline_estimation_method : string, optional - specify how the fitter should estimate the baseline.
                                       "breslow", "spline", or "piecewise"
        - penalizer                  : float or array, optional (default=0.0) - attach a penalty to the size of the
                                       coefficients during regression. This improves stability of the estimates and
                                       controls for high correlation between covariates.
        - l1_ratio                   : float, optional (default=0.0) - tell what ratio to assign to a L1 vs L2 penalty.
                                       Same as scikit-learn. See penalizer above.
        - strata                     : list, optional - specify a list of columns to use in stratification.
                                       This is useful if a categorical covariate does not obey the proportional hazard
                                       assumption. This is used similar to the strata expression in R.
                                       See http://courses.washington.edu/b515/l17.pdf.
        - n_baseline_knots           : int - used when baseline_estimation_method="spline".
                                       Set the number of knots (interior & exterior) in the baseline hazard,
                                       which will be placed evenly along the time axis. Should be at least 2.
                                       Royston et al., the authors of this model, suggest 4 to start, but any values
                                       between 2 and 8 are reasonable. If you need to customize the timestamps used to
                                       calculate the curve, use the knots parameter instead.
        - knots                      : list, optional - when baseline_estimation_method="spline", this allows
                                       customizing the points in the time axis for the baseline hazard curve. To use
                                       evenly-spaced points in time, the n_baseline_knots parameter can be employed
                                       instead.
        - breakpoints                : int - used when baseline_estimation_method="piecewise".
                                       Set the positions of the baseline hazard breakpoints.
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        self.m_model = lifelines.CoxPHFitter(**self.hyperparameters)

    def fit(self, data_train, parameters=None):
        accessor = getattr(data_train, self.accessor_code)
        event    = accessor.event
        duration = accessor.duration

        unwanted_keys = ["val_data", "seed"]
        for key in unwanted_keys:
            if key in parameters.keys():
                parameters.pop(key, None)

        self.m_model = self.model.fit(
            data_train,
            duration,
            event,
            **parameters if parameters is not None else dict()
        )
        return self

    def predict(self, x, parameters=None):
        """
        Returns: negative partial (no baseline) hazard function: -dot(betas, X).
        """
        return -self.model.predict_partial_hazard(x).to_numpy()
