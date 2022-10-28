# This file provides a set of existing solutions from various Python libraries.
# They are wrapped here in a harmonised way, so that they all respect the same design.

import pandas as pd

from xmlot.models.model import FromTheShelfModel

# Shelf:
import lifelines

from   sksurv.datasets     import get_x_y
import sksurv.ensemble     as skens
import sksurv.linear_model as sklin

import pycox.models as pycox
import torch
import torchtuples  as tt


##########################
#     MISCELLANEOUS      #
##########################


class SKSurvModel(FromTheShelfModel):
    def __init__(self, accessor_code, sksurv_model, hyperparameters=None):
        super().__init__(accessor_code, hyperparameters)
        self.m_model = sksurv_model(**self.hyperparameters)

    def fit(self, x_train, y_train, parameters=None):
        df       = pd.concat([x_train, y_train], axis=1)
        event    = getattr(y_train, self.accessor_code).event
        duration = getattr(y_train, self.accessor_code).duration

        x, y = get_x_y(data_frame=df, attr_labels=[event, duration], pos_label=1, survival=True)
        self.m_model = self.model.fit(x, y)

        return self

    def predict(self, x):
        return self.model.predict(x)


class PyCoxModel(FromTheShelfModel):
    def __init__(self, accessor_code, hyperparameters=None):
        super().__init__(accessor_code=accessor_code, hyperparameters=hyperparameters)
        self.m_net = None
        self.m_log = None

    @property
    def net(self):
        return self.m_net

    @property
    def log(self):
        return self.m_log


def _turn_df_into_pycox_xy_(df, accessor_code):
    """
    Extract features and targets from a DataFrame into the intended PyCox fromat.
    """
    x = getattr(df, accessor_code).features.to_numpy()
    y = (
        getattr(df, accessor_code).durations.values,
        getattr(df, accessor_code).events.values
    )
    return x, y


##################
#      COX       #
##################


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
        super().__init__(accessor_code, hyperparameters)
        self.m_model = lifelines.CoxPHFitter(**self.hyperparameters)

    def fit(self, x_train, y_train, parameters=None):
        df = pd.concat(objs=[x_train, y_train], axis=1)
        event = getattr(df, self.accessor_code).event
        duration = getattr(df, self.accessor_code).duration

        self.m_model = self.model.fit(
            df,
            duration,
            event,
            **parameters if parameters is not None else dict()
        )
        return self

    def predict(self, x):
        """
        Returns: negative partial (no baseline) hazard function: -dot(betas, X).
        """
        return -self.model.predict_partial_hazard(x).to_numpy()


class ScikitCoxModel(SKSurvModel):
    """
    cf. https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html

    Args:
        - alpha   : float, ndarray of shape (n_features,), optional, default: 0) - regularization parameter for ridge
                    regression penalty.
                    If a single float, the same penalty is used for all features.
                    If an array, there must be one penalty for each feature.
        - ties    : "breslow" | "efron", optional, default: "breslow" - the method to handle tied event times.
                    If there are no tied event times all the methods are equivalent.
        - n_iter  : int, optional, default: 100 - maximum number of iterations.
        - tol     : float, optional, default: 1e-9 - convergence criteria.
                    Convergence is based on the negative log-likelihood.
        - verbose : int, optional, default: 0 - tells the amount of additional debug information during optimization.
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            accessor_code,
            sksurv_model=sklin.CoxPHSurvivalAnalysis,
            hyperparameters=hyperparameters
        )


#######################
#      DEEPSURV       #
#######################


class DeepSurv(PyCoxModel):
    """
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/cox-ph.ipynb

    Args:
        - in_features : dimension of a feature vector as input.
        - num_nodes   : sizes for intermediate layers.
        - batch_norm  : boolean enabling batch normalisation
        - dropout     : drop out rate.
        - output_bias : if set to False, no additive bias will be learnt.
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(accessor_code=accessor_code)
        in_features = hyperparameters["in_features"]
        num_nodes = hyperparameters["num_nodes"]  # [32, 32]
        out_features = 1
        batch_norm = hyperparameters["batch_norm"]  # True
        dropout = hyperparameters["dropout"]  # 0.1
        output_bias = hyperparameters["output_bias"]  # False

        self.m_net = tt.practical.MLPVanilla(
            in_features,
            num_nodes,
            out_features,
            batch_norm,
            dropout,
            output_bias=output_bias)

        self.m_model = pycox.CoxPH(self.m_net, tt.optim.Adam)

    def fit(self, x_train, y_train, parameters=None):
        x = x_train.to_numpy()
        y = (
            getattr(y_train, self.accessor_code).durations.values,
            getattr(y_train, self.accessor_code).events.values
        )

        # Compute learning rate
        if parameters['lr']:
            self.model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.model.lr_finder(
                x,
                y,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.model.optimizer.set_lr(lr)

        # Train
        self.m_log = self.model.fit(
            x,
            y,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=_turn_df_into_pycox_xy_(
                parameters["val_data"],
                self.accessor_code
            ),
            val_batch_size=parameters['batch_size']
        )

        _ = self.model.compute_baseline_hazards()

        return self

    def predict(self, x):
        input_tensor = torch.tensor(x.values).cuda()
        output = self.net(input_tensor).cpu().detach().numpy()
        output = output.reshape((output.shape[0],))
        return output


#######################
#       DEEPHIT       #
#######################


class DeepHit(PyCoxModel):
    """
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit.ipynb (single risk)
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit_competing_risks.ipynb (competing risks)

    Args:
        - in_features  : dimension of a feature vector as input.
        - num_nodes    : sizes for intermediate layers.
        - out_features : matches the size of the grid on which time has been discretised
        - batch_norm   : boolean enabling batch normalisation
        - dropout      : drop out rate.
        - alpha        :
        - sigma        :
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(accessor_code=accessor_code)

        in_features = hyperparameters["in_features"]
        num_nodes = hyperparameters["num_nodes"]  # [32, 32]
        out_features = hyperparameters["out_features"]
        batch_norm = hyperparameters["batch_norm"]  # True
        dropout = hyperparameters["dropout"]  # 0.1

        self.m_net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

        self.m_model = pycox.DeepHitSingle(
            self.m_net,
            tt.optim.Adam,
            alpha=hyperparameters["alpha"],  # 0.2,
            sigma=hyperparameters["sigma"],  # 0.1,
        )

    def fit(self, x_train, y_train, parameters=None):
        x = x_train.to_numpy()
        y = (
            getattr(y_train, self.accessor_code).durations.values,
            getattr(y_train, self.accessor_code).events.values
        )

        # Compute learning rate
        if parameters['lr']:
            self.m_model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.m_model.lr_finder(
                x,
                y,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.m_model.optimizer.set_lr(lr)

        # Train
        self.m_log = self.m_model.fit(
            x,
            y,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=_turn_df_into_pycox_xy_(
                parameters["val_data"],
                self.accessor_code
            ),
            val_batch_size=parameters['batch_size']
        )

        return self

    def predict(self, x):
        pass  # TODO


#############################
#     ENSEMBLE METHODS      #
#############################


class RandomSurvivalForest(SKSurvModel):
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(accessor_code, sksurv_model=skens.RandomSurvivalForest, hyperparameters=hyperparameters)


class XGBoost(SKSurvModel):
    """
    Loss functions include coxph (default) , squared , ipcwls (Only default works on our dataset)
    """
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            accessor_code,
            sksurv_model=skens.GradientBoostingSurvivalAnalysis,
            hyperparameters=hyperparameters
        )
