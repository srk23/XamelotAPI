# Benchmarking tools to compare various models on a fair ground.
# Models can come from various sources as long as they follow the same design (cf. Model in xmlot.models.model)

import time

from   copy   import deepcopy
import numpy  as     np
import pandas as     pd

from xmlot.data.discretise  import DefaultDiscretiser
from xmlot.data.split       import split_dataset
from xmlot.misc.misc        import set_seed


######################
#      VISITORS      #
######################
# Visitors allow to enrich functions (here 'embed_data') with optional behaviours.

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
            vis.separate()

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


#####################
#     BENCHMARK     #
#####################


def benchmark(
        models,
        metric,
        df,
        accessor_code,
        get_standardiser,
        get_discretiser  = lambda _model_name_, _df_: DefaultDiscretiser(),
        k                = 5,
        test_frac        = .2,
        seed             = None,
        visitor          = DefaultBenchmarkVisitor()
):
    """
    Args:
        - models           : a dictionary that stores the models to compare;
                             each entry is a key-value pair with the key being the model name
                             and the value a dictionary:
                             {
                                 "model"      : instance of untrained model
                                 "parameters" : training parameters (dictionary)
                             }
        - metric           : the metric used to compare models
        - df               : the DataFrame on which will be performed the benchmark
        - accessor_code    : a code to ease the access targets and features (via an Accessor)
        - get_standardiser : a function that returns a Standardiser given a DataFrame
        - get_discretiser  : a function that returns a Discretiser given a DataFrame
        - k                : the number of folds for k-fold cross validation
        - test_frac        : the proportion of data used for test
        - seed             : set a seed for random numbers generation
        - visitor          : TODO complete integration

    Returns: a Python dictionary that contains for each model the following entries:
        - validation_scores : the list of the validation scores for each fold
        - instances         : the list of the corresponding instances
        - test_score        : the score on test data reached by the best instance
    """
    visitor.start()

    # Set seed if required
    if seed is not None:
        set_seed(seed)
        random_states_split = np.random.randint(1000, size=2)
    else:
        random_states_split = (None, None)

    # Initialise output
    output = {model_name: {
        "validation_scores": list(),
        "instances": list()
    } for model_name in models.keys()}

    # We split the data, reserving some for testing (taken from the most recent entries)
    # Split is stratified
    accessor_df = getattr(df, accessor_code)
    pre_df_test, df_ = split_dataset(
        df,
        [test_frac, 1 - test_frac],
        main_target=accessor_df.stratification_target
    )

    # We split the remaining part in k folds.
    splits = split_dataset(
        df_, [1 / k] * k,
        main_target=accessor_df.stratification_target,
        random_states=random_states_split
    )

    for i in range(k):
        # Then we arrange data as train / validation sets
        pre_df_train = pd.concat([splits[j] for j in range(k) if j != i])
        pre_df_val   = splits[i]

        # We standardise each dataset based on the information we know from the training set.
        standardiser = get_standardiser(pre_df_train)
        pre_df_train = standardiser(pre_df_train)
        pre_df_val   = standardiser(pre_df_val)

        # Let's train each model regarding those datasets.
        for model_name in models.keys():
            # Depending on the model, we may need to discretise some target columns (eg. time).
            discretiser = get_discretiser(model_name, pre_df_train)
            df_train    = discretiser(pre_df_train.copy())
            df_val      = discretiser(pre_df_val.copy())

            # We take a copy of the untrained model: we train it and return the corresponding validation score.
            model = deepcopy(models[model_name]["model"])

            # We perform a deepcopy of the parameters as well
            # It is for example needed for torch's callbacks
            parameters = deepcopy(models[model_name]["parameters"])
            # We add validation data for the models which rely on it (cf. PyCox)
            parameters["val_data"] = df_val

            # Train
            accessor_train = getattr(df_train, accessor_code)
            model = model.fit(
                accessor_train.features,
                accessor_train.targets,
                parameters
            )
            # ... and validate
            validation_score = metric(model, df_val)

            # We store the results obtained with our model for that fold.
            output[model_name]["validation_scores"].append(validation_score)
            output[model_name]["instances"].append(model)

    # Now, we test the best instance of each model on our test data.
    for model_name in models.keys():
        # We get back the best instance.
        best_i = np.argmax(output[model_name]["validation_scores"])
        best_model = output[model_name]["instances"][best_i]
        df_train = pd.concat([splits[j] for j in range(k) if j != best_i])

        # We standardise based on the used training dataset.
        standardiser = get_standardiser(df_train)
        df_test = standardiser(pre_df_test.copy())

        # We discretise time.
        discretiser = get_discretiser(model_name, df_train)
        df_test = discretiser(df_test)

        # We store the result.
        output[model_name]["test_score"] = metric(best_model, df_test)
        output[model_name]["df_test"]           = df_test
        output[model_name]["best_discretiser"]  = discretiser
        output[model_name]["best_standardiser"] = standardiser
    return output
