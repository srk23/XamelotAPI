import numpy  as np
import pandas as pd

from copy import copy


from project.data.split   import split_dataset

def k_folds(df, duration, event, k, model, training_parameters):
    splits = split_dataset(df, [1 / k] * k, event)

    performances = list()
    for i in range(k):
        testing  = splits[i]
        training = pd.concat([splits[j] for j in range(k) if i != j])

        # MODEL "INITIALIZATION"
        model_ = copy(model)

        # TRAINING
        model_.fit(training, duration, event, training_parameters)
    
        # EVALUATION
        performance = model_.concordance(testing, duration, event)
        performances.append(performance)
    average_performance = np.mean(performances)

    return average_performance
