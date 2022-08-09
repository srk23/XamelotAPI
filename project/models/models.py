class Model:
    def __init__(self):
        pass

    def train(self, sdm_train, parameters):
        """
        args:
            - sdm_train  : a SurvivalDataManager to easily distinguish covariates, events and durations.
            - parameters : model related parameters for training.
        """
        pass

    def __call__(self, df):
        """
        Args:
            - df : a DataFrame containing a set of covariates $\mathbf{x}$.
        """
        pass
