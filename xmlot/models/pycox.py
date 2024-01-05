# Wrap various classifiers from PyCox.
# Everything follows Model's design.
#
# More details on: https://github.com/havakv/pycox

import pandas as pd
import numpy  as np

import pycox.models as pycox
import torch
import torchtuples  as tt

from xmlot.models.model import FromTheShelfModel

def _adapt_input_(x):
    # Adapt input
    if type(x) == pd.DataFrame:
        return torch.tensor(x.values, dtype=torch.float32)
    elif type(x) == np.ndarray:
        return torch.tensor(x, dtype=torch.float32)
    else:
        return x

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

class DeepHitSingle(PyCoxModel):
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


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                 out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input_):
        out = self.shared_net(input_)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out

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

        in_features      = hyperparameters["in_features"]       # x_train.shape[1]
        num_nodes_shared = hyperparameters["num_nodes_shared"]  # [64, 64]
        num_nodes_indiv  = hyperparameters["num_nodes_indiv"]   # [32]
        num_risks        = hyperparameters["num_risks"]         # y_train[1].max()
        out_features     = hyperparameters["out_features"]      # len(labtrans.cuts)
        batch_norm       = hyperparameters["batch_norm"]        # True
        dropout          = hyperparameters["dropout"]           # 0.1
        seed             = hyperparameters["seed"]

        if seed is not None:
            torch.manual_seed(seed)
        self.m_net = CauseSpecificNet(
            in_features,
            num_nodes_shared,
            num_nodes_indiv,
            num_risks,
            out_features,
            batch_norm,
            dropout
        )

        self.m_model = pycox.DeepHit(
            self.m_net,
            tt.optim.AdamWR(
                lr=0.01,
                decoupled_weight_decay=0.01,
                cycle_eta_multiplier=0.8
            ),
            alpha=hyperparameters["alpha"],  # 0.2,
            sigma=hyperparameters["sigma"],  # 0.1,
            duration_index=hyperparameters["cuts"]
        )

    def fit(self, data_train, parameters=None):

        if parameters['seed'] is not None:
            torch.manual_seed(parameters['seed'])

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
        val_data = self._df_to_xy_(parameters["val_data"])

        self.m_log = self.m_model.fit(
            x,
            y,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=val_data,
            val_batch_size=parameters['batch_size']
        )

        return self

    def predict_CIF(self, x, parameters=None):
        _ = parameters
        x_ = _adapt_input_(x)
        return self.model.predict_cif(x_)

    def predict_surv(self, x, parameters=None):
        _ = parameters
        x_ = _adapt_input_(x)
        return self.model.predict_surv_df(x_)

    def predict_pmf(self, x, parameters=None):
        _ = parameters
        x_ = _adapt_input_(x)
        return self.model.predict_pmf(x_)

    def predict(self, x, parameters=None):
        return self.predict_CIF(x, parameters=None)
