from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.learning.jpt.jpt import JPT
import numpy as np


class MonteCarloEstimator:

    model: ProbabilisticModel
    sample_size: int

    def __init__(self, model: ProbabilisticModel, sample_size: int=100):
        self.model = model
        self.sample_size = sample_size

    def area_validation_metric2(self, other: ProbabilisticModel) -> float:
        own_samples = self.model.sample(self.sample_size)
        other_samples = other.sample(self.sample_size)

        ll_own_samples_self = self.model.likelihood(own_samples)
        ll_other_samples_self = other.likelihood(own_samples)

        p_x_greater_q_x_own_samples = (ll_own_samples_self > ll_other_samples_self).sum()
        q_x_greater_p_x_own_samples = (ll_other_samples_self > ll_own_samples_self).sum()

        ll_own_samples_other = self.model.likelihood(other_samples)
        ll_other_samples_other = other.likelihood(other_samples)

        p_x_greater_q_x_other_samples = (ll_own_samples_other > ll_other_samples_other).sum()
        q_x_greater_p_x_other_samples = (ll_other_samples_other > ll_own_samples_other).sum()

        result = (p_x_greater_q_x_own_samples - q_x_greater_p_x_own_samples + q_x_greater_p_x_other_samples -
                  p_x_greater_q_x_other_samples)
        return result/self.sample_size


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