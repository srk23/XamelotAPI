# Provide a standard design regarding models.

class Model:
    """
    This Class introduces a framework to unify various models and implementations.
    It is inspired from the SciKit API's design (cf. https://arxiv.org/abs/1309.0238).
    In particular,
        - Consistency                  : All objects share a consistent interface.
        - Inspection                   : Access to constructor and model parameters.
        - Non-proliferation of classes : Not checked. 
                                         For convenience, datasets are expected to be DataFrames;
                                         possibly enriched with Accessors.
        - Composition                  : A priori not checked?
        - Sensible defaults            : TODO - Provide by default a baseline for parameterisation.

    """

    def __init__(self, accessor_code, hyperparameters=None):
        """
        Args:
            - accessor_code: a code to relate to an Accessor 
                             (which aims at enhancing the use of pandas DataFrames).
        """
        self.m_accessor_code = accessor_code
        self.m_hyperparameters = hyperparameters if hyperparameters is not None else dict()

    @property
    def accessor_code(self):
        return self.m_accessor_code

    @property
    def hyperparameters(self):
        return self.m_hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def fit(self, x_train, y_train, parameters=None):
        """
        Following SciKit's design.

        Args:
            - x_train    : a DataFrame that contains all a set of unlabeled data points.
            - y_train    : a DataFrame that contains the corresponding labels for each point of 'x_train'.
                           The shape of this DataFrame can vary depending on the problem of interest.
            - parameters : model related parameters for training.

        Returns: the trained model.
        """
        _ = (x_train, y_train, parameters)
        return self

    def predict(self, x):
        """
        Following SciKit's design.

        Args:
            - x : a DataFrame that represents a set of data points.

        Returns: the predictions for each data point.
        """
        pass


class FromTheShelfModel(Model):
    """
    Wrap any existing model implementation into the present framework.
    """

    def __init__(self, accessor_code, hyperparameters=None):
        super().__init__(accessor_code, hyperparameters)
        self.m_model = None

    @property
    def model(self):
        return self.m_model
