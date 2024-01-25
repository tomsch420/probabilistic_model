import math
from typing import Tuple, List, Optional, Dict, Any, Union

import numpy as np
import random_events.variables
from scipy.stats import gamma
from typing_extensions import Self

import portion
from random_events.events import Event, EncodedEvent, VariableMap
from random_events.variables import Continuous, Variable

from ..distribution import ContinuousDistribution, Unit, DeterministicSumUnit
from ...probabilistic_model import OrderType, CenterType, MomentType
from ...distributions.gaussian import (GaussianDistribution as PMGaussianDistribution,
                                       TruncatedGaussianDistribution as PMTruncatedGaussianDistribution)


class GaussianDistribution(PMGaussianDistribution, ContinuousDistribution):
    """
    Class for Gaussian distributions.
    """

    def __init__(self, variable: Continuous, mean: float, variance: float, parent=None):
        super().__init__(variable, mean, variance)
        ContinuousDistribution.__init__(self, variable, parent)

    def to_json(self) -> Dict[str, Any]:
        return {**ContinuousDistribution.to_json(self), "mean": self.mean, "variance": self.variance}

    @classmethod
    def from_json_with_variables_and_children(cls, data: Dict[str, Any], variables: List[Variable],
                                              children: List['Unit']) -> Self:
        variable = random_events.variables.Variable.from_json(data["variable"])
        return cls(variable, data["mean"], data["variance"])

    def conditional_from_simple_interval(self, interval: portion.Interval) \
            -> Tuple[Optional[Union[DeterministicSumUnit, Self]], float]:

        conditional, probability = super().conditional_from_simple_interval(interval)
        if conditional is None:
            return None, probability

        resulting_distribution = TruncatedGaussianDistribution(self.variable, interval, conditional.mean,
                                                               conditional.variance)
        return resulting_distribution, probability


class TruncatedGaussianDistribution(PMTruncatedGaussianDistribution, GaussianDistribution):
    """
    Class for Truncated Gaussian distributions.
    """

    def __init__(self, variable: Continuous, interval: portion.Interval, mean: float, variance: float, parent=None):
        super().__init__(variable, interval, mean, variance)
        GaussianDistribution.__init__(self, variable, mean, variance, parent)

    def conditional_from_simple_interval(self, interval: portion.Interval) \
            -> Tuple[Optional[Union[DeterministicSumUnit, Self]], float]:

        # calculate the probability of the interval
        probability = self._probability(EncodedEvent({self.variable: interval}))

        # if the probability is 0, return None
        if probability == 0:
            return None, 0

        # else, form the intersection of the interval and the domain
        intersection = self.interval & interval
        resulting_distribution = TruncatedGaussianDistribution(self.variable, interval=intersection,
                                                               mean= self.mean, variance=self.variance)
        return resulting_distribution, probability

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "interval": portion.to_data(self.interval), "mean": self.mean, "variance": self.variance}

    @classmethod
    def from_json_with_variables_and_children(cls, data: Dict[str, Any], variables: List[Variable],
                                              children: List['Unit']) -> Self:
        variable = random_events.variables.Variable.from_json(data["variable"])
        return cls(variable, portion.from_data(data["interval"]), data["mean"], data["variance"])
