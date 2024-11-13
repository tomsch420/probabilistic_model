from collections import defaultdict

from random_events.variable import Continuous, Symbolic, Integer
from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares

from probabilistic_model.probabilistic_circuit.nx.distributions import SymbolicDistribution, IntegerDistribution, \
    UniformDistribution
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import ProductUnit, SumUnit
from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.learning.jpt.jpt import JPT
import numpy as np

from probabilistic_model.utils import MissingDict


class MonteCarloEstimator:

    model: ProbabilisticModel
    sample_size: int

    def __init__(self, model: ProbabilisticModel, sample_size: int=100):
        self.model = model
        self.sample_size = sample_size

    def set_sample_size(self, sample_size: int):
        self.sample_size = sample_size

    def area_validation_metric2(self, other: ProbabilisticModel) -> float:
        own_samples = self.model.sample(self.sample_size)
        other_samples = other.sample(self.sample_size)

        ll_own_samples_self = self.model.likelihood(own_samples)
        ll_other_samples_self = other.likelihood(own_samples)

        p_x_greater_q_x_own_samples = (ll_own_samples_self > ll_other_samples_self).sum()
        q_x_greater_p_x_own_samples = (ll_other_samples_self > ll_own_samples_self).sum()

        ll_own_samples_other = np.round(self.model.likelihood(other_samples), 5)
        ll_other_samples_other = np.round(other.likelihood(other_samples), 5)

        p_x_greater_q_x_other_samples = (ll_own_samples_other > ll_other_samples_other).sum()
        q_x_greater_p_x_other_samples = (ll_other_samples_other > ll_own_samples_other).sum()

        result = (p_x_greater_q_x_own_samples - q_x_greater_p_x_own_samples + q_x_greater_p_x_other_samples -
                  p_x_greater_q_x_other_samples)
        return result/self.sample_size

    def l1_metric(self, other: ProbabilisticModel) -> float:
        own_samples = self.model.sample(self.sample_size)
        other_samples = other.sample(self.sample_size)
        ll_own_samples_self = self.model.likelihood(own_samples)
        ll_other_samples_self = other.likelihood(own_samples)
        ll_own_samples_other = self.model.likelihood(other_samples)
        ll_other_samples_other = other.likelihood(other_samples)
        diff_on_self = np.abs(ll_own_samples_self - ll_other_samples_self)
        diff_on_other = np.abs(ll_other_samples_other - ll_own_samples_other)
        return np.mean(diff_on_other + diff_on_self)

    def l1_metric_but_with_uniform_measure(self, other: ProbabilisticModel):
        supp_of_self = self.model.support
        supp_of_other = other.support
        union_of_supps = supp_of_other | supp_of_self
        bounding_box = union_of_supps.bounding_box()

        uniform_model = ProductUnit()
        for variable, assignment in bounding_box.items():
            if isinstance(variable, Continuous):
                distribution = SumUnit()
                for assignment_ in assignment:
                    u = UniformDistribution(variable, assignment_)
                    distribution.add_subcircuit(u, 1/u.pdf_value())
                distribution.normalize()
            elif isinstance(variable, Symbolic):
                distribution = SymbolicDistribution(variable, MissingDict(float, {value: 1/len(assignment.simple_sets) for
                                                                                  value in assignment}))
            elif isinstance(variable, Integer):
                distribution = IntegerDistribution(variable, MissingDict(float, {value.lower: 1/len(assignment.simple_sets) for
                                                                                  value in assignment}))
            else:
                raise NotImplementedError
            uniform_model.add_subcircuit(distribution)
        uniform_model, _ = uniform_model.conditional(union_of_supps)
        import plotly.graph_objects as go
        samples = uniform_model.sample(self.sample_size)
        density_of_uniform_model = uniform_model.likelihood(samples[:1])[0]
        p_self = self.model.likelihood(samples)
        p_other = other.likelihood(samples)
        l1_metric = np.mean(np.abs(p_self - p_other)) / density_of_uniform_model
        return l1_metric


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

    def l1(self, other: ProbabilisticModel, tolerance = 10e-8):
        samples_p = self.model.sample(self.sample_size)
        l_p = self.model.likelihood(samples_p)
        l_q = other.likelihood(samples_p)
        diff = l_p - l_q
        p = (diff > tolerance).mean()
        print(p)

        samples_q = other.sample(self.sample_size)
        l_p = self.model.likelihood(samples_q)
        l_q = other.likelihood(samples_q)
        diff = l_p - l_q
        q = (diff > tolerance).mean()
        print(q)
        return 2 * (p - q)
