from __future__ import annotations

from typing_extensions import Tuple, Type, List, Self

from random_events.interval import SimpleInterval, Bound

import torch
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous

from ...distributions.uniform import UniformDistribution
from .input_layer import ContinuousLayerWithFiniteSupport
from .pc import AnnotatedLayer
from ..nx.distributions import UniformDistribution as UniformUnit
from .utils import simple_interval_to_open_tensor, create_sparse_tensor_indices_from_row_lengths


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

    def cdf_of_nodes(self, x: torch.Tensor) -> torch.Tensor:
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

    def log_likelihood_of_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the uniform distribution.

        :param x: The input tensor of shape (n, 1).
        :return: The log-likelihood of the uniform distribution.
        """
        return torch.where(self.included_condition(x), self.log_pdf_value(), -torch.inf)

    def log_mode_of_nodes(self) -> Tuple[List[Event], torch.Tensor]:
        return self.support_per_node, self.log_pdf_value()

    def sample_from_frequencies(self, frequencies: torch.Tensor) -> torch.Tensor:
        max_frequency = max(frequencies)

        # create indices for the sparse result
        indices = create_sparse_tensor_indices_from_row_lengths(frequencies)

        # sample from U(0,1)
        standard_uniform_samples = torch.rand((indices.shape[1]))

        # calculate range for each node
        range_per_sample = (self.upper - self.lower).repeat_interleave(frequencies)

        # calculate the right shift for each node
        right_shift_per_sample = self.lower.repeat_interleave(frequencies)

        # apply the transformation to the desired intervals
        samples = standard_uniform_samples * range_per_sample + right_shift_per_sample
        samples = samples.unsqueeze(-1)

        result = torch.sparse_coo_tensor(indices=indices, values=samples, is_coalesced=True,
                                         size=(self.number_of_nodes, max_frequency, 1))
        return result

    def log_conditional_from_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, torch.Tensor]:
        probabilities = self.probability_of_simple_event(SimpleEvent({self.variable: interval})).log()
        intersections = [interval.intersection_with(SimpleInterval(lower.item(), upper.item(),
                                                                   Bound.OPEN, Bound.OPEN))
                         for lower, upper in self.interval]

        non_empty_intervals = [simple_interval_to_open_tensor(intersection) for intersection in intersections
                               if not intersection.is_empty()]
        if len(non_empty_intervals) == 0:
            return self.impossible_condition_result
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

    def __deepcopy__(self):
        return self.__class__(self.variable, self.interval.clone())

    def moment_of_nodes(self, order: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """
        Calculate the moment of the uniform distribution.
        """

        order = order.item()
        center = center.item()
        pdf_value = torch.exp(self.log_pdf_value())
        lower_integral_value = (pdf_value * (self.lower - center) ** (order + 1)) / (order + 1)
        upper_integral_value = (pdf_value * (self.upper - center) ** (order + 1)) / (order + 1)
        return (upper_integral_value - lower_integral_value).unsqueeze(-1)