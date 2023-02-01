# Provides a suite of metrics to evaluate performances of classification models.
import sklearn.metrics as sk

from xmlot.metrics.metric import Metric

class ClassificationMetric(Metric):
    def __init__(self, accessor_code):
        super().__init__(accessor_code=accessor_code)

    def _build_prediction_true_values_(self, model, df_test):
        """
        A prediction is:
            - either an array of shape (n_samples)
            - or an array of shape (n_samples, n_classes)

        Ideally predicts the occurrence of an event
        (in opposition of being alive or being censored).
        """

        accessor = getattr(df_test, self.accessor_code)

        y_pred = model.predict_proba(accessor.features)
        y_true = accessor.targets.to_numpy().ravel()

        return y_pred, y_true


###############
#   METRICS   #
###############

class Auroc(ClassificationMetric):
    def __init__(self, accessor_code, multi_class='ovo', average="macro"):
        super().__init__(accessor_code=accessor_code)
        self.m_multi_class = multi_class
        self.m_average     = average

    def __call__(self, model, df_test):
        y_pred, y_true = self._build_prediction_true_values_(model, df_test)
        return sk.roc_auc_score(
            y_true, y_pred, multi_class=self.m_multi_class, average=self.m_average
        )


class Auprc(ClassificationMetric):
    def __init__(self, accessor_code):
        super().__init__(accessor_code=accessor_code)

    def __call__(self, model, df_test):
        y_pred, y_true = self._build_prediction_true_values_(model, df_test)
        return sk.average_precision_score(y_true, y_pred)

class Precision(ClassificationMetric):
    def __init__(self, accessor_code, threshold=.5):
        super().__init__(accessor_code=accessor_code)
        self.m_threshold = threshold

    def __call__(self, model, df_test):
        y_pred, y_true = self._build_prediction_true_values_(model, df_test)
        y_pred = list(map(lambda p: 1 if p > self.m_threshold else 0, y_pred))

        return sk.precision_score(y_true, y_pred, zero_division=0)


class Recall(ClassificationMetric):
    def __init__(self, accessor_code, threshold=.5):
        super().__init__(accessor_code=accessor_code)
        self.m_threshold = threshold

    def __call__(self, model, df_test):
        y_pred, y_true = self._build_prediction_true_values_(model, df_test)
        y_pred = list(map(lambda p: 1 if p > self.m_threshold else 0, y_pred))

        return sk.recall_score(y_true, y_pred, zero_division=0)


class F1Score(ClassificationMetric):
    def __init__(self, accessor_code, threshold=.5):
        super().__init__(accessor_code=accessor_code)
        self.m_threshold = threshold

    def __call__(self, model, df_test):
        y_pred, y_true = self._build_prediction_true_values_(model, df_test)
        y_pred = list(map(lambda p: 1 if p > self.m_threshold else 0, y_pred))

        return sk.f1_score(y_true, y_pred, zero_division=0)
