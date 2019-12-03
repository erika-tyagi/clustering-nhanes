import pandas as pd
import sklearn.metrics as metrics
import math

class Evaluator:
    """
    A class that wraps a model's prediction results and can be used to generate
    various metrics based on a threshold.
    """
    def __init__(self, y_predict, y_actual):
        self.df = pd.DataFrame(data={ 'predict': y_predict, 'actual': y_actual })


class ClassifierEvaluator(Evaluator):
    def __init__(self, y_predict, y_actual, threshold_percentage):
        Evaluator.__init__(self, y_predict, y_actual)
        self.df = self.df.sort_values('predict', ascending=False)
        self.threshold_percentage = threshold_percentage


    def prevalence(self):
        """Returns the ratio of actually true outcomes to total outcomes."""
        count_true = len(self.df[self.df.actual == 1])
        return count_true / len(self.df)


    def accuracy(self):
        return metrics.accuracy_score(self.df.actual.values,
                                      self._threshold_labels())


    def precision(self):
        return metrics.precision_score(self.df.actual.values,
                                       self._threshold_labels())


    def recall(self):
        return metrics.recall_score(self.df.actual.values,
                                    self._threshold_labels())


    def f1(self):
        return metrics.f1_score(self.df.actual.values,
                                self._threshold_labels())


    def auc(self):
        return metrics.roc_auc_score(self.df.actual.values,
                                     self._threshold_labels())


    def metric_names():
        """
        Returns the names of all the metrics in the order that the metrics
        are returned from all_metrics().
        """
        return [
            'accuracy',
            'precision',
            'recall',
            'f1',
            'auc'
        ]


    def all_metrics(self):
        return [
            self.accuracy(),
            self.precision(),
            self.recall(),
            self.f1(),
            self.auc()
        ]


    def _threshold_labels(self):
        # Stolen from Rayid Ghani
        # DataFrame must be sorted by predicted score!!!!
        cutoff_index = int(len(self.df) * (self.threshold_percentage / 100.0))
        return [1 if i < cutoff_index else 0 for i in range(len(self.df))]


class RegressionEvaluator(Evaluator):
    def mean_squared_error(self):
        return metrics.mean_squared_error(self.df.actual.values,
                                          self.df.predict.values)


    def root_mean_squared_error(self):
        return math.sqrt(metrics.mean_squared_error(self.df.actual.values,
                                                    self.df.predict.values))


    def explained_variance(self):
        return metrics.explained_variance_score(self.df.actual.values,
                                                self.df.predict.values)


    def r2(self):
        return metrics.r2_score(self.df.actual.values,
                                self.df.predict.values)


    def metric_names():
        """
        Returns the names of all the metrics in the order that the metrics
        are returned from all_metrics().
        """
        return [
            'mean_squared_error',
            'root_mean_squared_error',
            'explained_variance',
            'r2'
        ]


    def all_metrics(self):
        return [
            self.mean_squared_error(),
            self.root_mean_squared_error(),
            self.explained_variance(),
            self.r2()
        ]
