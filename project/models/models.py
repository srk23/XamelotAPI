import torch
import torchtuples  as tt

# from lifelines           import CoxPHFitter
from   lifelines.utils import concordance_index
import lifelines

# from pycox.models        import CoxPH, DeepHitSingle
import pycox.models as pycox
from sksurv.datasets     import get_x_y

# from sksurv.linear_model import CoxPHSurvivalAnalysis
import sksurv.linear_model as sklin
# from sksurv.ensemble     import GradientBoostingSurvivalAnalysis
# from sksurv.ensemble     import RandomSurvivalForest as SksurvRSF
import sksurv.ensemble     as skens


#####################################
#      MISCELLANEOUS FUNCTIONS      #
#####################################


def _get_xy_(sdm):
    x = sdm.covariates.to_numpy()
    y = (sdm.durations.values, sdm.events.values)
    return x, y


###################################
#      GENERIC MODEL CLASSES      #
###################################


class Model:
    def __init__(self):
        pass

    def train(self, sdm_train, sdm_val, parameters):
        """
        Args:
            - sdm_train  : a SurvivalDataManager to easily distinguish covariates, events and durations.
            - sdm_val    : a SurvivalDataManager to easily distinguish covariates, events and durations.
            - parameters : model related parameters for training.
        """
        pass

    def __call__(self, covariates):
        pass

class FromTheShelfModel(Model):
    def __init__(self):
        super().__init__()
        self.m_model = None

    @property
    def model(self):
        return self.m_model

class DeepLearningModel(Model):
    def __init__(self):
        super().__init__()
        self.m_net = None
        self.m_log = None

    @property
    def net(self):
        return self.m_net

    @property
    def log(self):
        return self.m_log


###################################
#      FROM THE SHELF MODELS      #
###################################


# Cox #

class LifelinesCoxModel(FromTheShelfModel):
    def __init__(self, hyperparameters):
        super().__init__()
        self.m_model = lifelines.CoxPHFitter(**hyperparameters)

    def train(self, sdm_train, sdm_val, parameters):
        self.m_model = self.model.fit(
            sdm_train.df,
            sdm_train.duration_name,
            sdm_train.event_name,
            **parameters
        )
        return self.score(sdm_val)

    def __call__(self, covariates):
        # negative partial (no baseline) hazard function: -dot(betas, X)
        output = -self.model.predict_partial_hazard(covariates).to_numpy()
        return output

    def score(self, sdm):
        return concordance_index(
            sdm.durations,
            -self.model.predict_partial_hazard(sdm.covariates).to_numpy(),
            sdm.events
        )

class ScikitCoxModel(FromTheShelfModel):
    def __init__(self, hyperparameters):
        super().__init__()
        self.m_model = sklin.CoxPHSurvivalAnalysis()
        self.m_model.set_params(alpha=hyperparameters['penaliser'])

    def train(self, sdm_train, sdm_val, parameters=None):
        x, y = get_x_y(sdm_train.df, [sdm_train.event_name, sdm_train.duration_name], pos_label=1, survival=True)

        self.model.fit(x, y)

        return self.score(sdm_val)

    def __call__(self, covariates):
        # negative partial (no baseline) hazard function: -dot(betas, X)
        return -self.model.predict(covariates)

    def score(self, sdm):
        x, _ = get_x_y(sdm.df, [sdm.event_name, sdm.duration_name], pos_label=1, survival=True)

        return concordance_index(
            sdm.durations,
            -self.model.predict(x),
            sdm.events
        )

class PycoxCoxModel(FromTheShelfModel, DeepLearningModel):
    def __init__(self, hyperparameters):
        super().__init__()
        self.m_net = torch.nn.Linear(
            in_features=hyperparameters["in_features"],
            out_features=1,
            bias=hyperparameters["bias"]
        )

        self.m_model = pycox.CoxPH(self.m_net, tt.optim.Adam)

    def train(self, sdm_train, sdm_val, parameters):
        x_train, y_train = _get_xy_(sdm_train)
        x_val, y_val     = _get_xy_(sdm_val)

        # Compute learning rate
        if parameters['lr']:
            self.model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.model.lr_finder(
                x_train,
                y_train,
                batch_size=parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.model.optimizer.set_lr(lr)

        # Train
        self.m_log = self.model.fit(
            x_train,
            y_train,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=(x_val, y_val),
            val_batch_size=parameters['batch_size']
        )

        _ = self.model.compute_baseline_hazards()

        return self.score(sdm_val)

    def __call__(self, covariates):
        # negative partial (no baseline) hazard function: -dot(betas, X)
        input_tensor = torch.tensor(covariates.values).cuda()
        output = -self.net(input_tensor).cpu().detach().numpy()
        output = output.reshape((output.shape[0],))
        return output

    def score(self, sdm):
        return concordance_index(
            sdm.durations,
            -self(sdm.covariates),
            sdm.events
        )

# Random Survival Forests #

class RandomSurvivalForest(FromTheShelfModel):
    def __init__(self, hyperparameters):
        super().__init__()
        self.m_model = skens.RandomSurvivalForest()
        self.m_model.set_params(**hyperparameters)

    def train(self, sdm_train, sdm_val, parameters=None):
        x, y = get_x_y(sdm_train.df, [sdm_train.event_name, sdm_train.duration_name], pos_label=1, survival=True)
        self.model.fit(x, y)

        return self.score(sdm_val)

    def __call__(self, covariates):
        pass

    def score(self, sdm):
        x, y = get_x_y(sdm.df, [sdm.event_name, sdm.duration_name], pos_label=1, survival=True)
        return self.model.score(x, y)

# XGBoost #

class XGBoost(FromTheShelfModel):
    def __init__(self, hyperparameters):
        super().__init__()
        # Loss functions include coxph (default) , squared , ipcwls (Only default works on our dataset)
        self.m_model = skens.GradientBoostingSurvivalAnalysis(**hyperparameters)

    def train(self, sdm_train, sdm_val, parameters=None):
        x, y = get_x_y(sdm_train.df, [sdm_train.event_name, sdm_train.duration_name], pos_label=1, survival=True)
        self.model.fit(x, y)

        return self.score(sdm_val)

    def __call__(self, covariates):
        pass

    def score(self, sdm):
        x, y = get_x_y(sdm.df, [sdm.event_name, sdm.duration_name], pos_label=1, survival=True)
        return self.model.score(x, y)

# DeepSurv #

class DeepSurv(FromTheShelfModel, DeepLearningModel):
    def __init__(self, hyperparameters):
        super().__init__()
        in_features = hyperparameters["in_features"]
        num_nodes = hyperparameters["num_nodes"]  # [32, 32]
        out_features = 1
        batch_norm = hyperparameters["batch_norm"]  # True
        dropout = hyperparameters["dropout"]  # 0.1
        output_bias = hyperparameters["output_bias"]  # False

        self.m_net = tt.practical.MLPVanilla(
            in_features,
            num_nodes,
            out_features,
            batch_norm,
            dropout,
            output_bias=output_bias)

        self.m_model = pycox.CoxPH(self.m_net, tt.optim.Adam)

    def train(self, sdm_train, sdm_val, parameters):
        x_train  , y_train = _get_xy_(sdm_train)
        x_val    , y_val   = _get_xy_(sdm_val)

        # Compute learning rate
        if parameters['lr']:
            self.model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.model.lr_finder(
                x_train,
                y_train,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.model.optimizer.set_lr(lr)

        # Train
        self.m_log = self.model.fit(
            x_train,
            y_train,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=(x_val, y_val),
            val_batch_size=parameters['batch_size']
        )

        _ = self.model.compute_baseline_hazards()

        return self.score(sdm_val)

    def __call__(self, covariates):
        input_tensor = torch.tensor(covariates.values).cuda()
        output = self.net(input_tensor).cpu().detach().numpy()
        output = output.reshape((output.shape[0],))
        return output

    def score(self, sdm):
        return concordance_index(
            sdm.durations,
            -self(sdm.covariates),
            sdm.events
        )

# DeepHit #

class DeepHit(FromTheShelfModel, DeepLearningModel):
    def __init__(self, hyperparameters):
        super().__init__()

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

    def train(self, sdm_train, sdm_val, parameters):
        x_train, y_train = _get_xy_(sdm_train)
        x_val, y_val = _get_xy_(sdm_val)

        # Compute learning rate
        if parameters['lr']:
            self.m_model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.m_model.lr_finder(
                x_train,
                y_train,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.m_model.optimizer.set_lr(lr)

        # Train
        self.m_log = self.m_model.fit(
            x_train,
            y_train,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=(x_val, y_val),
            val_batch_size=parameters['batch_size']
        )

        return self.score(sdm_val)

    def __call__(self, df):
        pass

    def score(self, sdm):
        from pycox.evaluation import EvalSurv

        interpolation = 10
        surv = self.model.interpolate(interpolation).predict_surv_df(sdm.covariates.to_numpy())
        ev = EvalSurv(surv, sdm.durations.to_numpy(), sdm.events.to_numpy(), censor_surv='km')
        return ev.concordance_td('antolini')
