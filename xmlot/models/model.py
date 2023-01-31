# Provide a standard design regarding models.


##############
#   MODELS   #
##############


class Model:
    """
    This Class introduces a framework to unify various models and implementations.
    It is loosely inspired from the SciKit API's design (cf. https://arxiv.org/abs/1309.0238).
    In particular,
        - Consistency                  : All objects share a consistent interface.
        - Inspection                   : Access to constructor and model parameters.
        - Non-proliferation of classes : Not checked. 
                                         For convenience, datasets are expected to be DataFrames;
                                         possibly enriched with Accessors.
        - Composition                  : A priori not checked?
        - Sensible defaults            : TODO - Provide by default a baseline for parameterisation.

    """

    def __init__(self, accessor_code=None, hyperparameters=None):
        """
        Args:
            - accessor_code  : an optional accessor code to handle (training) data
            - hyperparameters: a dict of parameters that defines the model (e.g. neural architecture, k for k-NN, etc.).
        """
        self.m_accessor_code   = accessor_code
        self.m_hyperparameters = hyperparameters if hyperparameters is not None else dict()

    @property
    def accessor_code(self):
        return self.m_accessor_code

    @accessor_code.setter
    def accessor_code(self, new_code):
        self.m_accessor_code = new_code

    @property
    def hyperparameters(self):
        return self.m_hyperparameters

    def fit(self, data_train, parameters=None):
        """
        Depending on the type of model (e.g. supervised, unsupervised, etc.) or the type of problem
        (e.g. classification, survival analysis, etc.), the way data is used can vary. Therefore, we provide the whole
        dataset as input and let the model takes what it needs for training.

        This allows to compare models that trains differently as lons as their predictions can be compared with regard
        to the same metric.

        Args:
            - data_train : a DataFrame that contains the data to train from.
            - parameters : model related parameters for training.

        Returns: the trained model.
        """
        _ = (data_train, parameters)
        return self

    def predict(self, x):
        """
        Naturally, any target column must not be provided to the model at this stage.

        Args:
            - x          : a DataFrame that represents a set of data points.
            - parameters : model related parameters for prediction.

        Returns: predictions for each data point.
        Their nature dependss on the type of model.
        """
        pass


class FromTheShelfModel(Model):
    """
    Wrap any existing model implementation into the present framework.
    """

    def __init__(self, accessor_code=None, hyperparameters=None):
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        self.m_model = None

    @property
    def model(self):
        return self.m_model
