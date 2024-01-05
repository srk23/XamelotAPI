import numpy as np
import pandas as pd
from xmlot.misc.misc import SeedGenerator
import torch
from torch import nn
from torch.nn.functional import one_hot
from xmlot.models.model import Model
from xmlot.data.discretise import BalancedDiscretiser
from xmlot.data.split import split_dataset

import matplotlib.pyplot as plt

def _adapt_input_(x):
    # Adapt input
    if type(x) == pd.DataFrame:
        return torch.tensor(x.values, dtype=torch.float32)
    elif type(x) == np.ndarray:
        return torch.tensor(x, dtype=torch.float32)
    else:
        return x

class DefaultFitVisitor:
    def __init__(self):
        pass

    def track_loss(self, loss):
        pass

    def track_val_loss(self, n_iter, loss):
        pass

    def end(self):
        pass

class LossTrackerVisitor(DefaultFitVisitor):
    def __init__(self):
        super().__init__()
        self.m_losses     = []
        self.m_val_losses = []
        self.m_x_val      = []

    def track_loss(self, loss):
        self.m_losses.append(loss.detach())

    def track_val_loss(self, n_iter, loss):
        self.m_x_val.append(n_iter)
        self.m_val_losses.append(loss.detach())

    def end(self):
        plt.figure(figsize=(7, 5))
        plt.plot(self.m_losses, label="training loss")
        plt.plot(self.m_x_val, self.m_val_losses, label="validation loss")
        plt.legend()
        plt.show()

        print("{0} -> {1}".format(self.m_val_losses[0], self.m_val_losses[-1]))

class NeuralModel(Model):
    def __init__(self, accessor_code=None, hyperparameters=None):
        super().__init__(
            accessor_code=accessor_code,
            hyperparameters=hyperparameters
        )

        self.m_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Computation relies on {0}.".format(self.m_device))

        # Load hyperparameters
        self.m_hyperparameters = hyperparameters if hyperparameters is not None else dict()
        self.m_input_size      = self.m_hyperparameters["input_size"]
        self.m_output_size     = self.m_hyperparameters["output_size"]
        self.m_dropout         = self.m_hyperparameters["dropout"] \
            if "dropout" in self.m_hyperparameters.keys() \
            else 0

        # Seed for initialisation
        if "seed" in self.m_hyperparameters.keys():
            seed = self.m_hyperparameters["seed"]
            seed_generator = SeedGenerator(seed)
            torch.manual_seed(seed_generator())
            torch.cuda.manual_seed(seed_generator())

        # Architecture
        self.m_net = nn.Sequential(
            nn.Linear(self.m_input_size, self.m_output_size),
            nn.ReLU()
        ).to(self.m_device)

    @property
    def parameters(self):
        return self.m_net.parameters()

    def _get_y(self, targets):
        n_points = len(targets)
        return one_hot(
            torch.tensor(targets.values, dtype=torch.long), num_classes=self.m_output_size
        ) \
            .reshape((n_points, self.m_output_size)) \
            .type(torch.float32) \
            .to(self.m_device)

    def _get_loss_function(self, data_train):
        _, _ = self, data_train
        return nn.CrossEntropyLoss()

    def predict(self, x, train=False):
        x_ = _adapt_input_(x)
        if train:
            self.m_net.train()
        else:
            self.m_net.eval()
        return self.m_net(x_.to(self.m_device))

    def fit(self, data_train, parameters=None):
        # Load parameters
        parameters = parameters if parameters is not None else dict()
        size_batch = parameters["size_batch"]                     \
            if "size_batch" in parameters.keys()                  \
            else min(200, len(data_train))
        n_epochs = parameters["n_epochs"]                         \
            if "n_epochs" in parameters.keys()                    \
            else 1
        optimizer = parameters["optimizer"](self.parameters)      \
            if "optimizer" in parameters.keys()                   \
            else None
        learning_rate_decays = parameters["learning_rate_decays"] \
            if "learning_rate_decays" in parameters.keys()        \
            else []
        data_val = parameters["val_data"]                         \
            if "val_data"  in parameters.keys()                   \
            else None
        early_stopping = parameters["early_stopping"]             \
            if "early_stopping" in parameters.keys()              \
            else True
        patience = parameters["patience"]                         \
            if "patience"  in parameters.keys()                   \
            else 10
        min_delta = parameters["min_delta"]                       \
            if "min_delta" in parameters.keys()                   \
            else 0.
        val_size_batch = parameters["val_size_batch"]             \
            if "val_size_batch" in parameters.keys()              \
            else None
        visitor        = parameters["visitor"]                    \
            if "visitor" in parameters.keys()                     \
            else DefaultFitVisitor()
        seed = parameters["seed"]                                 \
            if "seed" in parameters.keys()                        \
            else None

        def _lr_(_epoch_, decays=()):
            if _epoch_ < len(decays):
                return decays[_epoch_]
            else:
                return 1

        seed_generator = SeedGenerator(seed)
        if seed is not None:
            torch.manual_seed(seed_generator())
            torch.cuda.manual_seed(seed_generator())

        # Loss
        loss_function = self._get_loss_function(data_train)

        # Initialisation
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda e: _lr_(e, decays=learning_rate_decays))

        batch_fracs = [size_batch / len(data_train) for _ in range(int(len(data_train) / size_batch))]

        # Validation (for future early stopping)
        iter_since_best = None
        best_val_loss   = None

        for epoch in range(n_epochs):
            # Split data in batches
            batches = split_dataset(data_train, batch_fracs, seed=seed_generator())
            for n_batch, batch in enumerate(batches):
                # Get data
                acc = getattr(batch, self.accessor_code)
                x = acc.features
                y = self._get_y(acc.targets)

                # Compute gradients (w.r.t. our model)
                optimizer.zero_grad()
                y_ = self.predict(x, train=True)

                loss = loss_function(y_, y)
                visitor.track_loss(loss)

                loss.backward()

                # Update model's parameters
                optimizer.step()

            # Validation and early stopping
            if data_val is not None:
                if val_size_batch is not None:
                    val_loss = 0

                    batch_fracs = [val_size_batch / len(data_val) for _ in range(int(len(data_val) / val_size_batch))]
                    batches = split_dataset(data_val, batch_fracs, seed=seed_generator())
                    for batch in batches:
                        val_acc = getattr(batch, self.accessor_code)
                        val_x = val_acc.features
                        val_y = self._get_y(val_acc.targets)

                        val_y_ = self.predict(val_x, train=False).detach()
                        val_loss += loss_function(val_y_, val_y)
                else:
                    val_acc = getattr(data_val, self.accessor_code)
                    val_x = val_acc.features
                    val_y = self._get_y(val_acc.targets)

                    val_y_ = self.predict(val_x, train=False).detach()
                    val_loss = loss_function(val_y_, val_y)

                visitor.track_val_loss(len(batches) * (epoch + 1), val_loss)

                if early_stopping:
                    if best_val_loss is None or val_loss - best_val_loss > min_delta:
                        iter_since_best  = 0
                        best_val_loss    = val_loss
                    else:
                        iter_since_best += 1

                    if iter_since_best > patience:
                        break

            # Update learning rate
            scheduler.step()

        visitor.end()
        return self

class NeuralNet(NeuralModel):
    def __init__(self, accessor_code=None, hyperparameters=None):
        super().__init__(
            accessor_code=accessor_code,
            hyperparameters=hyperparameters
        )

        # Architecture
        hidden_layer_sizes = hyperparameters["hidden_layer_sizes"] \
            if "hidden_layer_sizes" in hyperparameters.keys()      \
            else [100]
        modules = [nn.Linear(self.m_input_size, hidden_layer_sizes[0]), nn.ReLU(), nn.Dropout(p=self.m_dropout)]
        for i in range(len(hidden_layer_sizes) - 1):
            modules.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=self.m_dropout))
        modules.append(nn.Linear(hidden_layer_sizes[-1], self.m_output_size))
        modules.append(nn.Softmax(dim=1))
        self.m_net = nn.Sequential(*modules).to(self.m_device)

    def _get_loss_function(self, data_train):
        value_counts = getattr(data_train, self.m_accessor_code).targets.value_counts()
        weights = torch.zeros(self.m_output_size)
        for k, v in value_counts.items():
            k_ = int(k[0])
            weights[k_] = 1/v
        weights /= weights.sum()

        return nn.CrossEntropyLoss(weight=weights)

    def predict_proba(self, x, train=False):
        y = self.predict(x, train)
        if np.shape(y)[1] == 2:
            return y[:, 1]
        return y


class DeepHit(NeuralModel):
    def __init__(self, accessor_code=None, hyperparameters=None):
        super().__init__(
            accessor_code=accessor_code,
            hyperparameters=hyperparameters
        )

        self.m_causes = range(self.m_output_size)

        shared_hidden_layer_sizes = hyperparameters["shared_hidden_layer_sizes"] \
            if "shared_hidden_layer_sizes" in hyperparameters.keys() \
            else [100]
        cs_hidden_layer_sizes = hyperparameters["cs_hidden_layer_sizes"] \
            if "cs_hidden_layer_sizes" in hyperparameters.keys() \
            else [100]
        self.m_censored = hyperparameters["censored"] \
            if "censored" in hyperparameters.keys() \
            else -1
        self.m_alpha = hyperparameters["alpha"] \
            if "alpha" in hyperparameters.keys() \
            else torch.ones(len(list(self.m_causes)))
        self.m_sigma = hyperparameters["sigma"] \
            if "sigma" in hyperparameters.keys() \
            else 1

        # Discretiser (placeholder)
        self.m_discretiser = None

        # Architecture
        modules = [nn.Linear(self.m_input_size, shared_hidden_layer_sizes[0]), nn.ReLU(), nn.Dropout(p=self.m_dropout)]
        for i in range(len(shared_hidden_layer_sizes) - 1):
            modules.append(nn.Linear(shared_hidden_layer_sizes[i], shared_hidden_layer_sizes[i + 1]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=self.m_dropout))
        self.m_shared_net = nn.Sequential(*modules).to(self.m_device)

        self.m_cause_specific_nets = list()
        for _ in self.m_causes:
            modules = [
                nn.Linear(shared_hidden_layer_sizes[-1] + self.m_input_size, cs_hidden_layer_sizes[0]),
                nn.ReLU(),
                nn.Dropout(p=self.m_dropout)
            ]
            for i in range(len(cs_hidden_layer_sizes) - 1):
                modules.append(nn.Linear(cs_hidden_layer_sizes[i], cs_hidden_layer_sizes[i + 1]))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(p=self.m_dropout))
            modules.append(nn.Linear(cs_hidden_layer_sizes[-1], self.m_output_size))
            modules.append(nn.Softmax(dim=1))
            self.m_cause_specific_nets.append(nn.Sequential(*modules).to(self.m_device))

    @property
    def parameters(self):
        parameters = [p for p in self.m_shared_net.parameters()]
        for cause in self.m_causes:
            parameters += [p for p in self.m_cause_specific_nets[cause].parameters()]
        return parameters

    @property
    def discretiser(self):
        return self.m_discretiser

    def _get_y(self, targets):
        targets = self.discretiser(targets).values
        k = targets[:, 0]
        s = targets[:, 1]
        return torch.tensor(k, dtype=torch.long), torch.tensor(s, dtype=torch.long)

    def _get_loss_function(self, data_train):
        def _L_(y_, y):

            k, s = y
            F = torch.cumsum(y_, dim=2)

            L1 = 0
            for i in range(len(s)):
                if k[i] != self.m_censored:
                    L1 -= torch.log(y_[k[i], i, s[i]])
                else:
                    L1 -= torch.log(1 - torch.sum(F[:, i, s[i]]))
            L2 = 0
            for i in range(len(s)):
                F_ki_i_si = F[k[i], i, s[i]] * torch.ones(len(s) - (i + 1))
                F_ki_j_si = F[k[i], i + 1:, s[i]]
                mask = torch.tensor([s[i] < sj for sj in s[i + 1:]])

                L2 += torch.sum(mask * torch.exp((F_ki_j_si - F_ki_i_si) / self.m_sigma) * self.m_alpha[k[i]])
            return L1 + L2

        return _L_

    def predict(self, x, train=False):
        x_ = _adapt_input_(x)
        z = self.m_shared_net(x_)

        # Residual connection
        z_ = torch.cat((z, x_), dim=1)

        return torch.stack(tuple(net(z_) for net in self.m_cause_specific_nets))

    def predict_CIF(self, x, train=False):
        y_ = self.predict(x, train)
        return torch.cumsum(y_, dim=2)

    def fit(self, data_train, parameters=None):
        self.m_discretiser = BalancedDiscretiser(
            data_train,
            size_grid=self.m_output_size,
            accessor_code=self.m_accessor_code
        )
        return super().fit(data_train, parameters)
