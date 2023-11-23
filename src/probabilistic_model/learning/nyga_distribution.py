import collections
import dataclasses
from typing import Optional, List, Deque, Any, Iterable, Union

import portion
from anytree import RenderTree
from random_events.variables import Continuous
from typing_extensions import Self

from ..probabilistic_circuit.distributions import ContinuousDistribution, UniformDistribution
from ..probabilistic_circuit.units import DeterministicSumUnit, Unit


@dataclasses.dataclass
class InductionStep:
    data: List[float]
    """
    The entire sorted data.
    """

    weights: List[float]
    """
    The weights of the samples in dataset.
    """

    begin_index: int
    """
    Included index of the first sample.
    """

    end_index: int
    """
    Excluded index of the last sample.
    """

    current_node: 'NygaDistribution'
    """
    The current node in the induction step.
    """

    previous_average_likelihood: float
    """
    The likelihood of the parent node
    """

    index_in_parent: int
    """
    The index of this induction step in the parent node.
    """

    @property
    def variable(self):
        return self.current_node.variable

    @property
    def min_samples_per_quantile(self):
        return self.current_node.min_samples_per_quantile

    def left_connecting_point(self) -> float:
        """
        Calculate the left connecting point.
        """
        return self.left_connecting_point_from_index(self.begin_index)

    def left_connecting_point_from_index(self, index):
        """
        Calculate the left connecting point given some begin index.
        """
        if index > 0:
            left_connecting_point = (self.data[index - 1] + self.data[index]) / 2
        else:
            left_connecting_point = self.data[index]
        return left_connecting_point

    def right_connecting_point(self) -> float:
        """
        Calculate the right connecting point.
        """
        return self.right_connecting_point_from_index(self.end_index)

    def right_connecting_point_from_index(self, index):
        """
        Calculate the right connecting point given some end index.
        """
        if index < len(self.data):
            right_connecting_point = (self.data[index] + self.data[index - 1]) / 2
        else:
            right_connecting_point = self.data[index - 1]
        return right_connecting_point

    def create_uniform_distribution(self):
        """
        Create a uniform distribution from this induction step.
        """
        return self.create_uniform_distribution_from_indices(self.begin_index, self.end_index)

    def create_uniform_distribution_from_indices(self, begin_index: int, end_index: int):
        if end_index == len(self.data):
            interval = portion.closed(self.left_connecting_point_from_index(begin_index),
                                      self.right_connecting_point_from_index(end_index))
        else:
            interval = portion.closedopen(self.left_connecting_point_from_index(begin_index),
                                          self.right_connecting_point_from_index(end_index))
        return UniformDistribution(self.variable, interval)

    def sum_weights_from_indices(self, begin_index: int, end_index: int):
        return sum(self.weights[begin_index:end_index])

    def sum_weights(self):
        return sum(self.weights[self.begin_index:self.end_index])

    def create_deterministic_uniform_mixture_from_split_index(self, split_index: int) -> 'NygaDistribution':

        # creat uniform distribution from the left including to the right excluding
        left_uniform_distribution = self.create_uniform_distribution_from_indices(self.begin_index, split_index)
        right_uniform_distribution = self.create_uniform_distribution_from_indices(split_index, self.end_index)

        weights_left = self.sum_weights_from_indices(self.begin_index, split_index)
        weights_right = self.sum_weights_from_indices(split_index, self.end_index)

        result = NygaDistribution(self.variable)
        result.weights = [weights_left, weights_right]
        result.children = [left_uniform_distribution, right_uniform_distribution]
        return result

    def compute_best_split(self):

        maximum_likelihood = 0
        best_split_index = None

        for split_index in range(self.min_samples_per_quantile, len(self.data) - self.min_samples_per_quantile + 1):
            distribution = self.create_deterministic_uniform_mixture_from_split_index(split_index)
            likelihoods = [distribution._pdf(value) for value in self.data[self.begin_index:self.end_index]]
            average_likelihood = sum(likelihoods) / len(self.data)
            if average_likelihood > maximum_likelihood:
                maximum_likelihood = average_likelihood
                best_split_index = split_index

        return maximum_likelihood, best_split_index

    def construct_left_induction_step(self, split_index: int) -> Self:
        self.current_node.weights += [self.sum_weights_from_indices(self.begin_index, split_index)]
        new_nyga_distribution = NygaDistribution(self.variable, parent=self.current_node)
        return InductionStep(self.data, self.weights, self.begin_index, split_index, new_nyga_distribution,
                             self.previous_average_likelihood, 0,)

    def construct_right_induction_step(self, split_index: int) -> Self:
        self.current_node.weights += [self.sum_weights_from_indices(split_index, self.end_index)]
        new_nyga_distribution = NygaDistribution(self.variable, parent=self.current_node)
        return InductionStep(self.data, self.weights, split_index, self.end_index, new_nyga_distribution,
                             self.previous_average_likelihood, 1,)


class NygaDistribution(DeterministicSumUnit, ContinuousDistribution):
    """
    A Nyga distribution is a deterministic mixture of uniform distributions.
    """

    min_likelihood_improvement: float = 1.1
    """
    The relative, minimal likelihood improvement needed to create a new quantile.
    """

    min_samples_per_quantile: int = 1
    """
    The minimal number of samples per quantile.
    """

    def __init__(self, variable: Continuous, min_likelihood_improvement: Optional[float] = None, parent: 'Unit' = None):
        ContinuousDistribution.__init__(self, variable, parent)
        DeterministicSumUnit.__init__(self, self.variables, [], parent)

        if min_likelihood_improvement is not None:
            self.min_likelihood_improvement = min_likelihood_improvement

    def _pdf(self, value: Union[float, int]) -> float:
        return sum([child._pdf(value) * weight for child, weight in zip(self.children, self.weights)])