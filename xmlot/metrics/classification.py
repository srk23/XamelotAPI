# Provides a suite of metrics to evaluate performances of classification models.

import numpy as np

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


#################################
#   NON PROBABILISTIC METRICS   #
#################################


def _from_proba_to_prediction_(y_pred, threshold):
    if len(np.shape(y_pred)) == 1:
        return list(map(lambda p: 1 if p > threshold else 0, y_pred))
    else:
        return list(map(lambda p: np.argmax(p), y_pred))


class Precision(ClassificationMetric):
    def __init__(self, accessor_code, threshold=.5, average='binary'):
        super().__init__(accessor_code=accessor_code)
        self.m_threshold = threshold
        self.m_average   = average

    def __call__(self, model, df_test):
        y_pred, y_true = self._build_prediction_true_values_(model, df_test)
        y_pred = _from_proba_to_prediction_(y_pred, self.m_threshold)

        return sk.precision_score(y_true, y_pred, average=self.m_average, zero_division=0)


class Recall(ClassificationMetric):
    def __init__(self, accessor_code, threshold=.5, average='binary'):
        super().__init__(accessor_code=accessor_code)
        self.m_threshold = threshold
        self.m_average   = average

    def __call__(self, model, df_test):
        y_pred, y_true = self._build_prediction_true_values_(model, df_test)
        y_pred = _from_proba_to_prediction_(y_pred, self.m_threshold)

        return sk.recall_score(y_true, y_pred, average=self.m_average, zero_division=0)


class F1Score(ClassificationMetric):
    def __init__(self, accessor_code, threshold=.5, average='binary'):
        super().__init__(accessor_code=accessor_code)
        self.m_threshold = threshold
        self.m_average   = average

    def __call__(self, model, df_test):
        y_pred, y_true = self._build_prediction_true_values_(model, df_test)
        y_pred = _from_proba_to_prediction_(y_pred, self.m_threshold)

        return sk.f1_score(y_true, y_pred, average=self.m_average, zero_division=0)


class Specificity(ClassificationMetric):
    def __init__(self, accessor_code, threshold=.5, average='binary'):
        super().__init__(accessor_code=accessor_code)
        self.m_threshold = threshold
        self.m_average   = average

    def __call__(self, model, df_test):
        y_pred, y_true = self._build_prediction_true_values_(model, df_test)
        y_pred = _from_proba_to_prediction_(y_pred, self.m_threshold)

        def _score_(_tp_, _fp_, _fn_, _tn_):
            if _tn_ + _fp_ == 0:
                return 0
            else:
                return _tn_ / (_tn_ + _fp_)

        m = sk.confusion_matrix(y_true, y_pred)
        n_class = len(m)

        if n_class == 2:  # Binary
            tn, fp, fn, tp = m.ravel()
            return _score_(tp, fp, fn, tn)
        else:             # Multi
            tps = list()
            tns = list()
            fps = list()
            fns = list()

            for i in range(n_class):
                tps.append(m[i, i])
                fps.append(np.sum(m[i, :]) - tps[-1])
                fns.append(np.sum(m[:, i]) - tps[-1])
                tns.append(np.sum(m) - tps[-1] - fps[-1] - fns[-1])

            if self.m_average == "macro":
                return np.mean(list(map(lambda x: _score_(*x), zip(tps, fps, fns, tns))))
            if self.m_average == "micro":
                return _score_(np.sum(tps), np.sum(fps), np.sum(fns), np.sum(tns))
            else:
                raise ValueError(
                    "The option 'average' must be set to either 'micro' or 'macro' in the multiclass case."
                )

class NegativePredictiveValue(ClassificationMetric):
    def __init__(self, accessor_code, threshold=.5, average='binary'):
        super().__init__(accessor_code=accessor_code)
        self.m_threshold = threshold
        self.m_average   = average

    def __call__(self, model, df_test):
        y_pred, y_true = self._build_prediction_true_values_(model, df_test)
        y_pred = _from_proba_to_prediction_(y_pred, self.m_threshold)

        def _score_(_tp_, _fp_, _fn_, _tn_):
            if _tn_ + _fn_ == 0:
                return 0
            else:
                return _tn_ / (_tn_ + _fn_)

        m = sk.confusion_matrix(y_true, y_pred)
        n_class = len(m)

        if n_class == 2:  # Binary
            tn, fp, fn, tp = m.ravel()
            return _score_(tp, fp, fn, tn)
        else:             # Multiclass
            tps = list()
            tns = list()
            fps = list()
            fns = list()

            for i in range(n_class):
                tps.append(m[i, i])
                fps.append(np.sum(m[i, :]) - tps[-1])
                fns.append(np.sum(m[:, i]) - tps[-1])
                tns.append(np.sum(m) - tps[-1] - fps[-1] - fns[-1])

            if self.m_average == "macro":
                return np.mean(list(map(lambda x: _score_(*x), zip(tps, fps, fns, tns))))
            if self.m_average == "micro":
                return _score_(np.sum(tps), np.sum(fps), np.sum(fns), np.sum(tns))
            else:
                raise ValueError(
                    "The option 'average' must be set to either 'micro' or 'macro' in the multiclass case."
                )
