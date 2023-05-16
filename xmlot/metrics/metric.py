# A default Metric class, similar to the Model class

class Metric:
    def __init__(self, accessor_code):
        self.m_accessor_code = accessor_code

    @property
    def accessor_code(self):
        return self.m_accessor_code

    @accessor_code.setter
    def accessor_code(self, accessor_code):
        self.m_accessor_code = accessor_code

    def __call__(self, model, df_test, seed=None):
        """
        Note: All additional parameters are set as attributes:
        it ensures a consistent interface while used by functions such as xmlot.benchmark.

        In terms of design, two choices were possible:
            - either: Metric(parameters)(model, df) with Metric as a class;
            - or : lambda model, df: metric(model, df, parameters=parameters)

        The first option seems more convenient.
        """
        pass

class DummyMetric(Metric):
    """
    A dummy (constant) metric for debugging puposes.
    """
    def __init__(self):
        super().__init__(None)

    def __call__(self, model, df_test, seed=None):
        return 0
