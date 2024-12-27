from collections import defaultdict

from random_events.variable import Continuous, Symbolic, Integer
from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares

from probabilistic_model.probabilistic_circuit.nx.distributions import SymbolicDistribution, IntegerDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import ProductUnit, SumUnit
from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.learning.jpt.jpt import JPT
import numpy as np

from probabilistic_model.utils import MissingDict


class MonteCarloEstimator:
    """
    This class has different Monte Carlo estimators for L1 Metric on Shallow Circuits.

    """

    model: ProbabilisticModel
    sample_size: int

    def __init__(self, model: ProbabilisticModel, sample_size: int=100):
        """
        :model: Probabilistic model the model on which the sample be calculated on
        :sample_size: Number of Monte Carlo samples
        """
        self.model = model
        self.sample_size = sample_size

    def set_sample_size(self, sample_size: int):
        self.sample_size = sample_size

    def l1_metric_but_with_uniform_measure(self, other: ProbabilisticModel):
        """
        This estimator uses the union of supports both models to reduce sample time.
        :other: the 2. Model need to estimate the L1 Metric.
        """
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
        samples = uniform_model.sample(self.sample_size)
        density_of_uniform_model = uniform_model.likelihood(samples[:1])[0]
        p_self = self.model.likelihood(samples)
        p_other = other.likelihood(samples)
        l1_metric = np.mean(np.abs(p_self - p_other)) / density_of_uniform_model
        return l1_metric


    def l1(self, other: ProbabilisticModel, tolerance = 10e-8):
        """
        Estimates the L1 metric with Monte Carlo estimator.
        :other: the 2. Model need to estimate the L1 Metric.
        :tolerance: Tolerance for how close to zero should be zero.
        """
        samples_p = self.model.sample(self.sample_size)
        l_p = self.model.likelihood(samples_p)
        l_q = other.likelihood(samples_p)
        diff = l_p - l_q
        p = (diff > tolerance).mean()
        samples_q = other.sample(self.sample_size)
        l_p = self.model.likelihood(samples_q)
        l_q = other.likelihood(samples_q)
        diff = l_p - l_q
        q = (diff > tolerance).mean()
        return 2 * (p - q)
