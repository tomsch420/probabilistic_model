from __future__ import annotations

from typing import Tuple, Optional, Union, Type, List

import random_events.interval
from random_events.interval import SimpleInterval, Bound
from typing_extensions import Self

import torch
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.learning.torch.input_layer import ContinuousLayerWithFiniteSupport
from probabilistic_model.learning.torch.pc import AnnotatedLayer
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution as UniformUnit
from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.utils import simple_interval_to_open_tensor


class UniformLayer(ContinuousLayerWithFiniteSupport):
    """
    A layer that represents a uniform distribution over a single variable.
    """

    variable: Continuous
    """
    The index of the variable that this layer represents.
    """

    def merge_with(self, others: List[Self]):
        self.interval = torch.cat([self.interval] + [other.interval for other in others])

    def __init__(self, variable: Continuous, interval: torch.Tensor):
        """
        Initialize the uniform layer.
        """
        super().__init__(variable, interval)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the cumulative distribution function at x.
        :param x: The data
        :return: The cumulative distribution function at x with shape (#x, #number_of_nodes)
        """
        result = (x - self.lower) / (self.upper - self.lower)
        result = torch.clamp(result, 0, 1)
        return result

    @classmethod
    def original_class(cls) -> Tuple[Type, ...]:
        return UniformDistribution,

    def validate(self):
        assert self.lower.shape == self.upper.shape, "The shapes of the lower and upper bounds must match."

    @property
    def number_of_nodes(self) -> int:
        """
        The number of nodes in the layer.
        """
        return len(self.lower)

    def log_pdf_value(self) -> torch.Tensor:
        """
        Calculate the log-density of the uniform distribution.
        """
        return -torch.log(self.upper - self.lower)

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the uniform distribution.

        :param x: The input tensor of shape (n, 1).
        :return: The log-likelihood of the uniform distribution.
        """
        return torch.where(self.included_condition(x), self.log_pdf_value(), -torch.inf)

    def log_mode(self) -> Tuple[Event, float]:
        pass

    def sample(self, amount: int) -> torch.Tensor:
        pass

    def log_conditional_from_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, torch.Tensor]:
        probabilities = self.probability_of_simple_event(SimpleEvent({self.variable: interval})).log()
        intersections = [interval.intersection_with(SimpleInterval(lower.item(), upper.item(),
                                                                   Bound.OPEN, Bound.OPEN))
                         for lower, upper in self.interval]

        non_empty_intervals = [simple_interval_to_open_tensor(intersection) for intersection in intersections
                               if not intersection.is_empty()]
        if len(non_empty_intervals) == 0:
            return None, probabilities
        new_intervals = torch.stack(non_empty_intervals)
        return self.__class__(self.variable, new_intervals), probabilities

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[UniformUnit],
                                                         child_layers: List[AnnotatedLayer]) -> \
            AnnotatedLayer:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        intervals = torch.stack([simple_interval_to_open_tensor(node.interval) for node in nodes])
        result = cls(nodes[0].variable, intervals)
        return AnnotatedLayer(result, nodes, hash_remap)
