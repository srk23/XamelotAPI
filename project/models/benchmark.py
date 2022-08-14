from copy import deepcopy

from project.data.survival_data_manager import split_sdm, concat_sdms
from project.data.standardise import Standardiser, get_standardisation
from project.data.bin         import EquidistantBinner
from project.models.metrics import  concordance
from project.models.benchmark_visitors import DefaultBenchmarkVisitor

def k_fold_evaluation(
        sdm,
        k,
        class_of_model,
        hyperparameters=None,
        parameters=None,
        # metric=concordance,
        standardisation=True,
        binnisation=False,
        visitor=DefaultBenchmarkVisitor()
):
    hyperparameters = hyperparameters if hyperparameters is not None else dict()
    parameters      = parameters      if parameters      is not None else dict()

    visitor.start()
    splits = split_sdm(sdm, [1 / k] * k)

    model = None
    for i in range(k):
        visitor.new_fold(i, k)

        # SPLITTING DATA
        visitor.split()
        sdm_test = splits[i]
        sdm_train = concat_sdms([splits[j] for j in range(k) if j != i])

        # DATA STANDARDISATION
        if standardisation:
            visitor.standardisation()
            standardiser = Standardiser(sdm_train, **get_standardisation(sdm_train))
            sdm_train = standardiser(sdm_train)
            sdm_test = standardiser(sdm_test)

        # DURATION BINNING
        if binnisation:
            visitor.binnisation()
            binner    = EquidistantBinner(sdm_train.durations, num_bins=22)
            sdm_train = binner(sdm_train)
            sdm_test  = binner(sdm_test)

        # MODEL INITIALISATION
        visitor.initialisation()
        model = class_of_model(hyperparameters)

        # TRAINING
        visitor.training()
        model.train(sdm_train, deepcopy(parameters))  # We need deepcopy for torch's callbacks

        # TESTING
        visitor.testing()
        performance = model.score(sdm_test)  # metric(model, sdm_test)

        visitor.end_fold(performance)

    visitor.end()
    return model
