# Wrap various classifiers from SciKit Learn and SciKit Survival.
# Everything follows Model's design.
#
# More details on: https://scikit-learn.org/stable/ ; https://scikit-survival.readthedocs.io/en/stable/

from sklearn.naive_bayes    import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble       import RandomForestClassifier, AdaBoostClassifier

from   sksurv.datasets      import get_x_y
import sksurv.ensemble      as skens
import sksurv.linear_model  as sklin

from xmlot.models.model     import FromTheShelfModel


########################
#     SCIKIT LEARN     #
########################


class ScikitClassificationModel(FromTheShelfModel):
    def __init__(self, scikit_model, accessor_code, hyperparameters=None):
        """
        Args:
            - scikit_model    : A model from SciKitLearn
            - accessor_code   : ScikitClassificationModel are supervised models: `accessor_code` is no longer optional.
            - hyperparameters : a dict of parameters that defines the model (e.g. neural architecture, etc.).
        """
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        self.m_model = scikit_model(**self.hyperparameters)

    def fit(self, data_train, parameters=None):
        accessor = getattr(data_train, self.accessor_code)
        x_train  = accessor.features

        x_train = x_train.values
        y_train  = accessor.targets.values.ravel()

        self.m_model.fit(x_train, y_train)
        return self

    def predict(self, x, parameters=None):
        return self.model.predict_proba(x.values)


class NaiveBayes(ScikitClassificationModel):
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            scikit_model    = GaussianNB,
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )


class NeuralNet(ScikitClassificationModel):
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            scikit_model    = MLPClassifier,
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )


class RandomForest(ScikitClassificationModel):
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            scikit_model    = RandomForestClassifier,
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )


class AdaBoost(ScikitClassificationModel):
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            scikit_model    = AdaBoostClassifier,
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )


###########################
#     SCIKIT SURVIVAL     #
###########################


class SKSurvModel(FromTheShelfModel):
    def __init__(self, sksurv_model, accessor_code, hyperparameters=None):
        """
        Args:
            - sksurv_model    : A model from SciKitLearn
            - accessor_code   : ScikitClassificationModel are supervised models: `accessor_code` is no longer optional.
            - hyperparameters : a dict of parameters that defines the model (e.g. neural architecture, etc.).
        """
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        self.m_model = sksurv_model(**self.hyperparameters)

    def fit(self, data_train, parameters=None):
        accessor = getattr(data_train, self.accessor_code)
        event    = accessor.event
        duration = accessor.duration

        x, y = get_x_y(data_frame=accessor.df, attr_labels=[event, duration], pos_label=1, survival=True)

        self.m_model = self.model.fit(x, y)

        return self

    def predict(self, x, parameters=None):
        return self.model.predict(x)

# COX #

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
            sksurv_model    = sklin.CoxPHSurvivalAnalysis,
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )

# ENSEMBLE METHODS #

class RandomSurvivalForest(SKSurvModel):
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            sksurv_model    = skens.RandomSurvivalForest,
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )

        # TODO: If time is continuous, make sure to DISCRETISE!
        #
        # discretiser = get_discretiser(model_name, pre_df_train)
        # df_train = discretiser(pre_df_train.copy())
        # df_val = discretiser(pre_df_val.copy())
        #
        # visitor.prefit(i, model_name, discretiser, df_train, df_val)


class XGBoost(SKSurvModel):
    """
    Loss functions include coxph (default) , squared , ipcwls (Only default works on our dataset)
    """
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            sksurv_model    = skens.GradientBoostingSurvivalAnalysis,
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
