# Wrap various classifiers from PyCox.
# Everything follows Model's design.
#
# More details on: https://github.com/havakv/pycox

import pycox.models as pycox
import torch
import torchtuples  as tt

from xmlot.models.model import FromTheShelfModel


class PyCoxModel(FromTheShelfModel):
    def __init__(self, accessor_code, hyperparameters=None):
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        self.m_net  = None
        self.m_log  = None

    @property
    def net(self):
        return self.m_net

    @property
    def log(self):
        return self.m_log

    def _df_to_xy_(self, df):
        """
        Extract features and targets from a DataFrame into the intended PyCox fromat.
        """
        accessor = getattr(df, self.accessor_code)
        x = accessor.features.to_numpy()
        y = (
            accessor.durations.values,
            accessor.events.values
        )
        return x, y

# DEEPSURV #

class DeepSurv(PyCoxModel):
    """
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/cox-ph.ipynb

    Args:
        - in_features : dimension of a feature vector as input.
        - num_nodes   : sizes for intermediate layers.
        - batch_norm  : boolean enabling batch normalisation
        - dropout     : drop out rate.
        - output_bias : if set to False, no additive bias will be learnt.
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        in_features  = hyperparameters["in_features"]
        num_nodes    = hyperparameters["num_nodes"]    # [32, 32]
        out_features = 1
        batch_norm   = hyperparameters["batch_norm"]   # True
        dropout      = hyperparameters["dropout"]      # 0.1
        output_bias  = hyperparameters["output_bias"]  # False

        self.m_net = tt.practical.MLPVanilla(
            in_features,
            num_nodes,
            out_features,
            batch_norm,
            dropout,
            output_bias=output_bias)

        self.m_model = pycox.CoxPH(self.m_net, tt.optim.Adam)

    def fit(self, data_train, parameters=None):
        x, y, = self._df_to_xy_(data_train)

        # Compute learning rate
        if parameters['lr']:
            self.model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.model.lr_finder(
                x,
                y,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.model.optimizer.set_lr(lr)

        # Train
        self.m_log = self.model.fit(
            x,
            y,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=self._df_to_xy_(parameters["val_data"]),
            val_batch_size=parameters['batch_size']
        )

        _ = self.model.compute_baseline_hazards()

        return self

    def predict(self, x, parameters=None):
        input_tensor = torch.tensor(x.values).cuda()
        output = self.net(input_tensor).cpu().detach().numpy()
        output = output.reshape((output.shape[0],))
        return output


# DEEPHIT #

class DeepHit(PyCoxModel):
    """
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit.ipynb (single risk)
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit_competing_risks.ipynb (competing risks)

    Args:
        - in_features  : dimension of a feature vector as input.
        - num_nodes    : sizes for intermediate layers.
        - out_features : matches the size of the grid on which time has been discretised
        - batch_norm   : boolean enabling batch normalisation
        - dropout      : drop out rate.
        - alpha        :
        - sigma        :
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(accessor_code=accessor_code)

        in_features = hyperparameters["in_features"]
        num_nodes = hyperparameters["num_nodes"]  # [32, 32]
        out_features = hyperparameters["out_features"]
        batch_norm = hyperparameters["batch_norm"]  # True
        dropout = hyperparameters["dropout"]  # 0.1

        self.m_net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

        self.m_model = pycox.DeepHitSingle(
            self.m_net,
            tt.optim.Adam,
            alpha=hyperparameters["alpha"],  # 0.2,
            sigma=hyperparameters["sigma"],  # 0.1,
        )

    def fit(self, data_train, parameters=None):

        # TODO: If time is continuous, make sure to DISCRETISE!
        #
        # discretiser = get_discretiser(model_name, pre_df_train)
        # df_train = discretiser(pre_df_train.copy())
        # df_val = discretiser(pre_df_val.copy())
        #
        # visitor.prefit(i, model_name, discretiser, df_train, df_val)

        x, y, = self._df_to_xy_(data_train)

        # Compute learning rate
        if parameters['lr']:
            self.m_model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.m_model.lr_finder(
                x,
                y,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.m_model.optimizer.set_lr(lr)

        # Train
        self.m_log = self.m_model.fit(
            x,
            y,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=self._df_to_xy_(parameters["val_data"]),
            val_batch_size=parameters['batch_size']
        )

        return self

    def predict(self, x, parameters=None):
        pass  # TODO
