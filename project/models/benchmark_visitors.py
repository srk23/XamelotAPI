import time
import numpy as np

class DefaultBenchmarkVisitor:
    def __init__(self):
        pass

    def start(self):
        pass

    def new_fold(self, i, k):
        pass

    def split(self):
        pass

    def standardisation(self):
        pass

    def binnisation(self):
        pass

    def initialisation(self):
        pass

    def training(self):
        pass

    def testing(self):
        pass

    def end_fold(self, performance):
        pass

    def end(self):
        pass


class EvaluationBenchmarkVisitor(DefaultBenchmarkVisitor):
    def __init__(self):
        super().__init__()
        self.m_init_times   = list()
        self.m_train_times  = list()
        self.m_test_times   = list()
        self.m_performances = list()
        self.m_time         = None

    def start(self):
        pass

    def new_fold(self, i, k):
        self.m_time = time.time()

    def training(self):
        self.m_init_times.append(time.time() - self.m_time)
        self.m_time = time.time()

    def testing(self):
        self.m_train_times.append(time.time() - self.m_time)
        self.m_time = time.time()

    def end_fold(self, performance):
        self.m_test_times.append(time.time() - self.m_time)
        self.m_time = time.time()
        self.m_performances.append(performance)

    def end(self):
        print("Average training time  : {0}s ; (std: {1})".format(np.mean(self.m_train_times),
                                                                  np.std(self.m_train_times)))
        print(
            "Average execution time : {0}s ; (std: {1})".format(np.mean(self.m_test_times), np.std(self.m_test_times)))
        print("Average performance    : {0}  ; (std: {1})".format(np.mean(self.m_performances),
                                                                  np.std(self.m_performances)))
class TalkativeBenchmarkVisitor(DefaultBenchmarkVisitor):
    def __init__(self):
        super().__init__()

    def start(self):
        pass

    def new_fold(self, i, k):
        if i > 0:
            print()
        print("Fold {0} / {1}".format(i+1, k))

    def split(self):
        print("    > Splitting data")

    def standardisation(self):
        print("    > Standardising data")

    def binnisation(self):
        print("    > Binning data")

    def initialisation(self):
        print("    > Model initialisation")

    def training(self):
        print("    > Training model")

    def testing(self):
        print("    > Testing model")

    def end_fold(self, performance):
        pass

    def end(self):
        print()

class AggregateBenchmarkVisitor(DefaultBenchmarkVisitor):
    def __init__(self, list_of_evaluation_visitors):
        super().__init__()
        self.m_visitors = list_of_evaluation_visitors

    def start(self):
        for vis in self.m_visitors:
            vis.start()

    def new_fold(self, i, k):
        for vis in self.m_visitors:
            vis.new_fold(i, k)

    def split(self):
        for vis in self.m_visitors:
            vis.split()

    def standardisation(self):
        for vis in self.m_visitors:
            vis.standardisation()

    def binnisation(self):
        for vis in self.m_visitors:
            vis.binnisation()

    def initialisation(self):
        for vis in self.m_visitors:
            vis.initialisation()

    def training(self):
        for vis in self.m_visitors:
            vis.training()

    def testing(self):
        for vis in self.m_visitors:
            vis.testing()

    def end_fold(self, performance):
        for vis in self.m_visitors:
            vis.end_fold(performance)

    def end(self):
        for vis in self.m_visitors:
            vis.end()
