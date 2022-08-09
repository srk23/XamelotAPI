import time

import numpy as np

from copy import deepcopy

from project.data.survival_data_manager import split_sdm, concat_sdms
from project.data.standardise import Standardiser, get_standardisation
from project.models.metrics import  concordance
class DefaultEvaluationVisitor:
    def __init__(self):
        pass

    def new_fold(self, k):
        pass

    def start(self):
        pass

    def initialisation(self):
        pass

    def training(self):
        pass

    def testing(self, performance):
        pass

    def end(self):
        pass


class EvaluationVisitor:
    def __init__(self):
        self.m_init_times   = list()
        self.m_train_times  = list()
        self.m_test_times   = list()
        self.m_performances = list()
        self.m_time         = None

    def start(self):
        pass

    def new_fold(self):
        self.m_time = time.time()

    def initialisation(self):
        self.m_init_times.append(time.time() - self.m_time)
        self.m_time = time.time()

    def training(self):
        self.m_train_times.append(time.time() - self.m_time)
        self.m_time = time.time()

    def testing(self, performance):
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


def k_fold_evaluation(sdm, k, class_of_model, hyperparameters=dict(), parameters=dict(), metric=concordance,
                      visitor=EvaluationVisitor()):
    visitor.start()
    splits = split_sdm(sdm, [1 / k] * k)

    model = None
    for i in range(k):
        visitor.new_fold()

        # SPLITTING DATA
        sdm_test = splits[i]
        sdm_train = concat_sdms([splits[j] for j in range(k) if j != i])

        # DATA STANDARDISATION
        standardiser = Standardiser(sdm_train, **get_standardisation(sdm_train))
        sdm_train = standardiser(sdm_train)
        sdm_test = standardiser(sdm_test)

        # MODEL INITIALISATION
        model = class_of_model(hyperparameters)
        visitor.initialisation()

        # TRAINING
        model.train(sdm_train, deepcopy(parameters))  # We need deepcopy for torch's callbacks
        visitor.training()

        # TESTING
        performance = metric(model, sdm_test)
        visitor.testing(performance)

    visitor.end()
    return model
