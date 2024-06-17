from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.learning.jpt.jpt import JPT
import numpy as np


class MonteCarloEstimator:

    model: ProbabilisticModel

    def __init__(self, model: ProbabilisticModel):
        self.model = model

    def area_validation_metric(self, other: ProbabilisticModel):
        ...


def monte_carlo_estimation_area_validation_metric(sample_amount: int, first_model: ProbabilisticModel, senc_model: ProbabilisticModel):


    half_sample_amount = int(sample_amount / 2) if sample_amount > 0 else 1
    p_p_amount = monte_carlo_densty_event(half_sample_amount, first_model, senc_model)
    q_q_amount= monte_carlo_densty_event(half_sample_amount, senc_model, first_model)
    return (p_p_amount + q_q_amount)/sample_amount
def monte_carlo_densty_event(sample_amount: int, fist_model: ProbabilisticModel, senc_model: ProbabilisticModel):
    first_amount = 0
    sample_amount = fist_model.sample(sample_amount)
    for sample in sample_amount:
        own_likli = fist_model.likelihood(np.array([sample]))
        other_likli = senc_model.likelihood(np.array([sample]))
        if own_likli > other_likli:
            first_amount += 1

    return first_amount