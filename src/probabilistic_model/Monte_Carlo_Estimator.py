from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.learning.jpt.jpt import JPT
import numpy as np


class MonteCarloEstimator:

    model: ProbabilisticModel
    sample_size: int

    def __init__(self, model: ProbabilisticModel, sample_size: int=100):
        self.model = model
        self.sample_size = sample_size

    def area_validation_metric(self, other_model: ProbabilisticModel):
        p_p_amount, q_q_amount = self.monte_carlo_densty_events(other_model)
        return (p_p_amount + q_q_amount) / self.sample_size
    def monte_carlo_densty_events(self, other_model: ProbabilisticModel):
        half_sample_amount = int(self.sample_size / 2) if self.sample_size > 0 else 1
        own_amount = 0
        other_amount = 0
        own_samples = self.model.sample(half_sample_amount)
        other_samples = other_model.sample(half_sample_amount)
        for sample in own_samples:
            own_liklihood = self.model.likelihood(np.array([sample]))
            other_liklihood = other_model.likelihood(np.array([sample]))
            if own_liklihood > other_liklihood:
                own_amount += 1
        for sample in other_samples:
            own_liklihood = self.model.likelihood(np.array([sample]))
            other_liklihood = other_model.likelihood(np.array([sample]))
            if own_liklihood < other_liklihood:
                other_amount += 1

        return own_amount, other_amount