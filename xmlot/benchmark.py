# Benchmarking tools to compare various models on a fair ground.
# Models can come from various sources as long as they follow the same design (cf. Model in xmlot.models.model)


from   copy   import deepcopy
import numpy  as     np
import pandas as     pd

from xmlot.data.split       import split_dataset
from xmlot.misc.misc        import SeedGenerator, identity

######################
#      VISITORS      #
######################
# Visitors allow to enrich functions (here 'embed_data') with optional behaviours.

class DefaultBenchmarkVisitor:
    def __init__(self):
        pass

    def split(self, splits, pre_df_test):
        pass

    def fold(self, i, k_fold, scaler, df_train, df_val):
        pass

    def fit(self, i, model_name, model, parameters):
        pass

    def test(self, results):
        pass

    def bootstrap(self, i, k_bootstrap, sampled_df_test):
        pass

    def predict(self, i, model_name, best_model, df_train, scaler, sampled_df_test):
        pass

    def end(self, results):
        pass


class AggregateBenchmarkVisitor(DefaultBenchmarkVisitor):
    def __init__(self, list_of_evaluation_visitors):
        super().__init__()
        self.m_visitors = list_of_evaluation_visitors

    def split(self, splits, pre_df_test):
        for vis in self.m_visitors:
            vis.split(splits, pre_df_test)

    def fold(self, i, k_fold, scaler, df_train, df_val):
        for vis in self.m_visitors:
            vis.fold(i, k_fold, scaler, df_train, df_val)

    def fit(self, i, model_name, model, parameters):
        for vis in self.m_visitors:
            vis.fit(i, model_name, model, parameters)

    def test(self, results):
        for vis in self.m_visitors:
            vis.test(results)

    def bootstrap(self, i, k_bootstrap, sampled_df_test):
        for vis in self.m_visitors:
            vis.bootstrap(i, k_bootstrap, sampled_df_test)

    def predict(self, i, model_name, best_model, df_train, scaler, sampled_df_test):
        for vis in self.m_visitors:
            vis.predict(i, model_name, best_model, df_train, scaler, sampled_df_test)

    def end(self, results):
        for vis in self.m_visitors:
            vis.end(results)


class TalkativeBenchmarkVisitor(DefaultBenchmarkVisitor):
    def __init__(self):
        super().__init__()

    def split(self, splits, pre_df_test):
        print("\ntraining data: {0}, validation data: {1}, test data: {2}\n".format(
            len(splits[0]) * (len(splits) - 1),
            len(splits[0]),
            len(pre_df_test)
        ))
        print("\n--- TRAINING ---\n")

    def fold(self, i, k_fold, scaler, df_train, df_val):
        print("> Fold {0}/{1}".format(i+1, k_fold))

    def fit(self, i, model_name, model, parameters):
        print("\t> {0}".format(model_name))

    def test(self, results):
        print("\nValidation scores:")
        for model in results.keys():
            scores = results[model]["validation_scores"]
            print("> {0}:\n\t> Best : {1}\n\t> Mean : {2}\n\t> Std  : {3}".format(
                model,
                np.max(scores),
                np.mean(scores),
                np.std(scores))
            )

        print("\n--- TESTING ---\n")

    def bootstrap(self, i, k_bootstrap, sampled_df_test):
        print("> Bootstrap {0}/{1}".format(i+1, k_bootstrap))

    def predict(self, i, model_name, best_model, df_train, scaler, sampled_df_test):
        print("\t> {0}".format(model_name))

    def end(self, results):
        print("\nTest scores:")
        for model in results.keys():
            scores = results[model]["test_scores"]
            print("> {0}:\n\t> Best : {1}\n\t> Mean : {2}\n\t> Std  : {3}".format(
                model,
                np.max(scores),
                np.mean(scores),
                np.std(scores))
            )


class DataframeBenchmarkVisitor(DefaultBenchmarkVisitor):
    def __init__(self):
        super().__init__()
        self.m_train = None
        self.m_val   = None
        self.m_test  = dict()

        self.m_train_scaler = None
        self.m_test_scaler  = dict()
        self.m_best_models  = dict()

    @property
    def train_data(self):
        return self.m_train

    @property
    def val_data(self):
        return self.m_val

    @property
    def test_data(self):
        return self.m_test

    @property
    def train_scaler(self):
        return self.m_train_scaler

    @property
    def test_scaler(self):
        return self.m_test_scaler

    @property
    def best_models(self):
        return self.m_best_models

    def fold(self, i, k_fold, scaler, df_train, df_val):
        if i == 0:
            self.m_train_scaler = [scaler]
            self.m_train = [df_train]
            self.m_val   = [df_val]
        else:
            self.m_train_scaler.append(scaler)
            self.m_train.append(df_train)
            self.m_val.append(df_val)

    def predict(self, i, model_name, best_model, df_train, scaler, sampled_df_test):
        if i == 0:
            self.m_test_scaler             = [scaler]
            self.m_test[model_name]        = [sampled_df_test]
            self.m_best_models[model_name] = [best_model]
        else:
            self.m_test_scaler.append(scaler)
            self.m_test[model_name].append(sampled_df_test)
            self.m_best_models[model_name].append(best_model)


#####################
#     BENCHMARK     #
#####################


def benchmark(
        models,
        metric,
        df,
        stratification_target,
        get_scaler,
        preprocessing      = identity,
        k_fold             = 5,
        test_frac          = .2,
        k_bootstrap        = 10,
        bootstrap_frac     = .5,
        seed               = None,
        visitor            = DefaultBenchmarkVisitor()
):
    """
    Args:
        - models                : a dictionary that stores the models to compare;
                                  each entry is a key-value pair with the key being the model name
                                  and the value a dictionary:
                                  {
                                      "model"      : instance of untrained model
                                      "parameters" : training parameters (dictionary)
                                  }
        - metric                : the metric used to compare models
        - df                    : the DataFrame on which will be performed the benchmark
        - stratification_target : specify the target used for stratification (during split)
        - get_scaler            : a function that returns a Scaler given a DataFrame
        - k_fold                : the number of folds for k-fold cross validation
        - test_frac             : the proportion of data used for test
        - seed                  : set a seed for random numbers generation
        - visitor               : inherits from DefaultBenchmarkVisitor, giving access to optional features

    Returns: a Python dictionary that contains for each model the following entries:
        - validation_scores : the list of the validation scores for each fold
        - instances         : the list of the corresponding instances
        - test_score        : the score on test data reached by the best instance
    """

    # Set seed if required;
    seed_generator = SeedGenerator(seed)

    # Initialise results
    results = {model_name: {
        "instances"        : list(),
        "validation_scores": list(),
        "test_scores"      : list()
    } for model_name in models.keys()}

    # We split the data, reserving some for testing (taken from the most recent entries)
    # Split is stratified
    df_, pre_df_test = split_dataset(
        df,
        [1 - test_frac, test_frac],
        main_target=stratification_target,
        seed=seed_generator()
    )

    df_ = preprocessing(df_)

    # We split the remaining part in k folds.
    splits = split_dataset(
        df_,
        [1 / k_fold] * k_fold,
        main_target=stratification_target,
        seed=seed_generator()
    )

    visitor.split(splits=splits, pre_df_test=pre_df_test)

    for i in range(k_fold):
        # Then we arrange data as train / validation sets
        pre_df_train = pd.concat([splits[j] for j in range(k_fold) if j != i])
        pre_df_val   = splits[i]

        # We re-scale each dataset based on the information we know from the training set.
        scaler = get_scaler(pre_df_train)

        df_train = scaler(pre_df_train)
        df_val   = scaler(pre_df_val)

        visitor.fold(i, k_fold, scaler, df_train, df_val)

        # Let's train each model regarding those datasets.
        # Fitting process is seeded to the fold (two same models should lead to the same results)
        parameters_seed = seed_generator()
        for model_name in models.keys():
            # We take a copy of the untrained model: we train it and return the corresponding validation score.
            model = deepcopy(models[model_name]["model"])

            # We perform a deepcopy of the parameters as well (it is for example needed for torch's callbacks)
            parameters = deepcopy(models[model_name]["parameters"])

            # We add validation data for the models which rely on it (cf. PyCox)
            parameters["val_data"] = df_val
            parameters["seed"]     = parameters_seed

            # Final model specific preprocessing
            try:
                df_train_ = parameters["preprocessing"](df_train)
            except KeyError:
                df_train_ = df_train

            # Train
            model = model.fit(
                df_train_,
                parameters
            )

            # ... and validate
            validation_score = metric(model, df_val)

            # We store the results respective to the current model for that fold.
            visitor.fit(i, model_name, model, parameters)
            results[model_name]["validation_scores"].append(validation_score)
            results[model_name]["instances"].append(model)

    visitor.test(results)

    # Now, we test the best instance of each model on our test data.
    # We repeat `k_bootstrap` times the same test.
    for i in range(k_bootstrap):
        # We select a random sample of the test data.
        sampled_df_test, _ = split_dataset(
            pre_df_test.copy(),
            [bootstrap_frac, 1 - bootstrap_frac],
            main_target=stratification_target,
            seed=seed_generator()
        )

        visitor.bootstrap(i, k_bootstrap, sampled_df_test)

        # We compute the performance score over that sample for each model.
        testing_seed = seed_generator()
        for model_name in models.keys():
            # We get back the best instance.
            best_i     = np.argmax(results[model_name]["validation_scores"])
            best_model = results[model_name]["instances"][best_i]
            df_train   = pd.concat([splits[j] for j in range(k_fold) if j != best_i])

            # We re-scale based on the used training dataset.
            scaler         = get_scaler(df_train)
            scaled_df_test = scaler(sampled_df_test)

            visitor.predict(i, model_name, best_model, df_train, scaler, scaled_df_test)

            # We store the result.
            results[model_name]["test_scores"].append(
                metric(
                    best_model,
                    scaled_df_test,
                    seed=testing_seed  # #### #
                )
            )
    visitor.end(results)
    return results
