from lifelines.utils  import concordance_index
from pycox.evaluation import EvalSurv

def concordance(model, sdm):
    return concordance_index(
            sdm.durations,
            model(sdm.covariates),
            sdm.events
        )

def concordance_td(model, sdm, interpolation=10):
    surv = model.m_model.interpolate(interpolation).predict_surv_df(sdm.covariates.to_numpy())
    ev = EvalSurv(surv, sdm.durations.to_numpy(), sdm.events.to_numpy(), censor_surv='km')
    return ev.concordance_td('antolini')
