from __future__ import annotations

from abc import ABC
from typing import Optional

import random_events
import torch
from random_events.interval import Interval, SimpleInterval, singleton
from random_events.product_algebra import Event, SimpleEvent
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Continuous
from sortedcontainers import SortedSet
from typing_extensions import List, Tuple, Self

from .pc import InputLayer, AnnotatedLayer, SumLayer
from ...probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuitMixin
from ...utils import interval_as_array


class ContinuousLayer(InputLayer, ABC):
    """
    Abstract base class for continuous univariate input units.
    """

    variable: Continuous

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the cumulative distribution function at x.
        :param x: The data
        :return: The cumulative distribution function at x with shape (#x, #number_of_nodes)
        """
        raise NotImplementedError

    def probability_of_simple_event(self, event: SimpleEvent) -> torch.Tensor:
        interval: Interval = event[self.variable]
        points = torch.tensor(interval_as_array(interval))
        upper_bound_cdf = self.cdf(points[:, (1,)])
        lower_bound_cdf = self.cdf(points[:, (0,)])
        return (upper_bound_cdf - lower_bound_cdf).sum(dim=0).unsqueeze(-1)

    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:

        if event.is_empty():
            return None, -torch.inf

        # extract the interval of the event
        marginal_event = event.marginal(SortedSet(self.variables))
        assert len(marginal_event.simple_sets) == 1, "The event must be a simple event."
        interval = marginal_event.simple_sets[0][self.variable]

        if len(interval.simple_sets) == 1:
            return self.log_conditional_from_simple_interval(interval.simple_sets[0])
        else:
            return self.log_conditional_from_interval(interval)

    def log_conditional_from_singletons(self, singletons: List[SimpleInterval]) -> Tuple[DiracDeltaLayer, torch.Tensor]:
        """
        Calculate the conditional distribution given a list singleton events with p(event) > zero forall events.

        In this case, the conditional distribution is a Dirac delta distribution and the log-likelihood is chosen
        for the log-probability.

        :param singletons: The singleton events
        :return: The dirac delta layer and the log-likelihoods with shape (#singletons, 1).
        """
        values = torch.tensor([s.lower for s in singletons])
        log_likelihoods = self.log_likelihood(values.reshape(-1, 1))
        return DiracDeltaLayer(self.variable, values, log_likelihoods), log_likelihoods

    def log_conditional_from_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, torch.Tensor]:
        """
        Calculate the conditional distribution given a simple interval with p(interval) > 0.
        The interval could also be a singleton.

        :param interval: The simple interval
        :return: The conditional distribution and the log-probability of the interval.
        """
        # form the intersection per node
        intersection = [interval.intersection_with(node_interval.simple_sets[0]) for node_interval in
                        self.univariate_support_per_node]
        singletons = [simple_interval for simple_interval in intersection if simple_interval.is_singleton()]
        non_singletons = [simple_interval for simple_interval in intersection if not simple_interval.is_singleton()]


    def log_conditional_from_non_singleton_simple_interval(self, interval: SimpleInterval) -> Tuple[SumLayer, float]:
        """
        Calculate the conditional distribution given a non-singleton, simple interval with p(interval) > 0.
        :param interval: The simple interval.
        :return: The conditional distribution and the log-probability of the interval.
        """
        raise NotImplementedError

    def log_conditional_from_interval(self, interval) -> Tuple[Self, float]:
        """
        Calculate the conditional distribution given an interval with p(interval) > 0.
        :param interval: The simple interval
        :return: The conditional distribution and the log-probability of the interval.
        """
        raise NotImplementedError


class ContinuousLayerWithFiniteSupport(ContinuousLayer, ABC):
    """
    Abstract class for continuous univariate input units with finite support.
    """

    interval: torch.Tensor
    """
    The interval of the distribution as a tensor of shape (num_nodes, 2).
    The first column contains the lower bounds and the second column the upper bounds.
    The intervals are treated as open intervals (>/< comparator).
    """

    def __init__(self, variable, interval):
        super().__init__(variable)
        self.interval = interval

    @property
    def lower(self) -> torch.Tensor:
        return self.interval[:, 0]

    @property
    def upper(self) -> torch.Tensor:
        return self.interval[:, 1]

    @property
    def univariate_support_per_node(self) -> List[AbstractCompositeSet]:
        return [random_events.interval.open(lower, upper) for lower, upper in self.interval]

    def left_included_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check if x is included in the left bound of the intervals.
        :param x: The data
        :return: A boolean array of shape (#x, #nodes)
        """
        return self.lower < x

    def right_included_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
         Check if x is included in the right bound of the intervals.
         :param x: The data
         :return: A boolean array of shape (#x, #nodes)
         """
        return x < self.upper

    def included_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
         Check if x is included in the interval.
         :param x: The data
         :return: A boolean array of shape (#x, #nodes)
         """
        return self.left_included_condition(x) & self.right_included_condition(x)


class DiracDeltaLayer(ContinuousLayer):

    location: torch.Tensor
    """
    The locations of the Dirac delta distributions.
    """

    density_cap: torch.Tensor
    """
    The density caps of the Dirac delta distributions.
    This value will be used to replace infinity in likelihoods.
    """

    def __init__(self, variable: Continuous, location: torch.Tensor, density_cap: Optional[torch.Tensor] = None):
        super().__init__(variable)
        self.location = location
        self.density_cap = density_cap if density_cap is not None else torch.full_like(location, torch.inf)

    def validate(self):
        assert self.location.shape == self.density_cap.shape, "The shapes of the location and density cap must match."
        assert all(self.density_cap > 0), "The density cap must be positive."

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x == self.location, torch.log(self.density_cap), -torch.inf)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[ProbabilisticCircuitMixin],
                                                         child_layers: List[AnnotatedLayer]) -> \
            AnnotatedLayer:
        raise NotImplementedError

    @property
    def univariate_support_per_node(self) -> List[AbstractCompositeSet]:
        return [singleton(location) for location in self.location]

    def log_conditional_of_simple_event(self, event: SimpleEvent) -> Tuple[Optional[Self], torch.Tensor]:
        pass

    def log_mode(self) -> Tuple[Event, float]:
        pass

    def sample(self, amount: int) -> torch.Tensor:
        pass

