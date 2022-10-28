# Wrap various classifiers from the shelf so that they match the Model design.

from sklearn.naive_bayes    import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble       import RandomForestClassifier, AdaBoostClassifier

from xmlot.models.model import FromTheShelfModel


##############################
#     SCIKIT CLASSIFIERS     #
##############################


class ScikitClassificationModel(FromTheShelfModel):
    def __init__(self, accessor_code, scikit_model, hyperparameters=None):
        super().__init__(accessor_code, hyperparameters)
        self.m_model = scikit_model(**self.hyperparameters)

    def fit(self, x_train, y_train, parameters=None):
        self.m_model.fit(x_train, y_train.values.ravel())
        return self

    def predict(self, x):
        return self.model.predict(x)


class NaiveBayes(ScikitClassificationModel):
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            accessor_code,
            scikit_model=GaussianNB,
            hyperparameters=hyperparameters
        )


class NeuralNet(ScikitClassificationModel):
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            accessor_code,
            scikit_model=MLPClassifier,
            hyperparameters=hyperparameters
        )


class RandomForest(ScikitClassificationModel):
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            accessor_code,
            scikit_model=RandomForestClassifier,
            hyperparameters=hyperparameters
        )


class AdaBoost(ScikitClassificationModel):
    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            accessor_code,
            scikit_model=AdaBoostClassifier,
            hyperparameters=hyperparameters
        )
