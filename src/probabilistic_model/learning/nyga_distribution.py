import collections
import dataclasses
from typing import Optional, List, Deque

import portion
from random_events.variables import Continuous
from typing_extensions import Self

from ..probabilistic_circuit.distributions import ContinuousDistribution, UniformDistribution
from ..probabilistic_circuit.units import DeterministicSumUnit, Unit


@dataclasses.dataclass
class InductionParameters:
    begin_index: int
    """
    Included index of the first sample.
    """

    end_index: int
    """
    Excluded index of the last sample.
    """

    previous_average_likelihood: float
    """
    The likelihood of the parent node
    """

    current_node: DeterministicSumUnit
    """
    The parent node of this induction step
    """

    index_in_parent: int
    """
    The index of this induction step in the parent node.
    """

    def left_connecting_point(self, sorted_data: List[float]) -> float:
        """
        Calculate the left connecting point from a sorted list of data.
        """
        if self.begin_index > 0:
            left_connecting_point = (sorted_data[self.begin_index - 1] +
                                     sorted_data[self.begin_index]) / 2
        else:
            left_connecting_point = sorted_data[self.begin_index]
        return left_connecting_point

    def right_connecting_point(self, sorted_data: List[float]) -> float:
        """
        Calculate the right connecting point from a sorted list of data.
        """
        if self.end_index < len(sorted_data):
            right_connecting_point = (sorted_data[self.end_index] + sorted_data[self.end_index - 1]) / 2
        else:
            right_connecting_point = sorted_data[self.end_index - 1]
        return right_connecting_point

    def total_number_of_samples(self) -> float:
        """
        Calculate the total number of samples in this induction step.
        """
        return self.end_index - self.begin_index

    def get_from_sorted_data(self, sorted_data: List[float]):
        """
        Get the data for this induction step from the sorted data.
        """
        return sorted_data[self.begin_index: self.end_index]

    def construct_left_induction_parameters(self, split_index: int, new_maximum_likelihood: float) -> Self:
        """
        Construct the left induction step.

        :param split_index: The index of the split, which will be excluded for the left subset.
        """
        # construct the left induction step
        self.current_node.weights.append((split_index - self.begin_index) / self.total_number_of_samples())
        left_sum_unit = DeterministicSumUnit(self.current_node.variables, [], self.current_node)
        return InductionParameters(self.begin_index, split_index, new_maximum_likelihood, left_sum_unit, 0)

    def construct_right_induction_parameters(self, split_index: int, new_maximum_likelihood: float) -> Self:
        """
        Construct the right induction step.

        :param split_index: The index of the split, which will be included for the right subset.
        """
        # construct the right induction step
        self.current_node.weights.append((self.end_index - split_index) / self.total_number_of_samples())
        right_sum_unit = DeterministicSumUnit(self.current_node.variables, [], self.current_node)
        return InductionParameters(split_index, self.end_index, new_maximum_likelihood, right_sum_unit, 1)


class NygaDistribution(ContinuousDistribution, DeterministicSumUnit):
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
        super().__init__(variable, None)
        DeterministicSumUnit.__init__(self, self.variables, [], parent)

        if min_likelihood_improvement is not None:
            self.min_likelihood_improvement = min_likelihood_improvement

    def fit(self, data: List[float]) -> DeterministicSumUnit:
        """
        Fit the model to the data.
        :param data:
        :return:
        """
        # sort the data
        sorted_data = list(sorted(data))

        # initialize induction
        minimal_distribution = UniformDistribution(self.variable, portion.closed(sorted_data[0], sorted_data[-1]))
        minimal_average_likelihood = (
                    sum([minimal_distribution.likelihood([value]) for value in sorted_data]) / len(sorted_data))
        root_of_distribution = DeterministicSumUnit(self.variables, [])

        # create initial iteration
        parameter_queue: Deque[InductionParameters] = collections.deque()
        parameter_queue.append(InductionParameters(0, len(sorted_data), minimal_average_likelihood,
                                                   root_of_distribution, 0))

        while len(parameter_queue) > 0:

            # get element from queue
            induction_parameters = (parameter_queue.pop())

            # calculate the best possible split
            new_maximum_likelihood, best_sum_node, split_index = self.compute_most_likely_split(
                induction_parameters.get_from_sorted_data(sorted_data),
                induction_parameters.left_connecting_point(sorted_data),
                induction_parameters.right_connecting_point(sorted_data))

            # if no further splits could be made or the likelihood improvement is too small
            if (split_index is None or
                    new_maximum_likelihood <= induction_parameters.previous_average_likelihood *
                    self.min_likelihood_improvement):
                parent_children = list(induction_parameters.current_node.parent.children)
                parent_children[induction_parameters.index_in_parent] = best_sum_node
                induction_parameters.current_node.parent.children = tuple(parent_children)
                continue

            # append induction steps
            parameter_queue.append(induction_parameters.construct_left_induction_parameters(split_index,
                                                                                            new_maximum_likelihood))
            parameter_queue.append(induction_parameters.construct_right_induction_parameters(split_index,
                                                                                             new_maximum_likelihood))

        # ensure the right most border is closed
        for uniform_distribution in root_of_distribution.leaves:
            if uniform_distribution.interval.upper == sorted_data[-1]:
                uniform_distribution.interval = portion.closed(uniform_distribution.interval.lower,
                                                               uniform_distribution.interval.upper)

        return root_of_distribution

    def compute_most_likely_split(self, data: List[float], left_connecting_point: Optional[float],
                                  right_connecting_point: Optional[float]):
        """
        Compute the most likely split of the data.

        :param data: The data evaluate the splits on.
        :param left_connecting_point: The connecting point that this distribution has to connect to on the left.
        :param right_connecting_point: The connection point that this distribution has to connect to on the right.
        :return: A deterministic uniform mixture over the provided data.
        """

        # initialize the max and argmax
        maximum_likelihood = 0
        best_sum_node = None
        best_split_index = None

        # for every possible split position
        for index in range(self.min_samples_per_quantile, len(data) - self.min_samples_per_quantile + 1):

            # create the distribution for the split
            distribution = self._create_deterministic_uniform_mixture_from_datasets(data[:index], data[index:],
                                                                                    left_connecting_point,
                                                                                    right_connecting_point)

            # calculate the average likelihood of the distribution
            average_likelihood = sum([distribution.likelihood([value]) for value in data]) / len(data)

            # if the average likelihood is better
            if average_likelihood > maximum_likelihood:

                # update the best distribution
                maximum_likelihood = average_likelihood
                best_sum_node = distribution
                best_split_index = index

        # if no split was found, create a uniform distribution
        if best_sum_node is None:
            best_sum_node = UniformDistribution(self.variable, portion.closedopen(left_connecting_point,
                                                                                  right_connecting_point))
            maximum_likelihood = sum([best_sum_node.likelihood([value]) for value in data]) / len(data)

        return maximum_likelihood, best_sum_node, best_split_index

    def _create_deterministic_uniform_mixture_from_datasets(self, left_data: List[float], right_data: List[float],
                                                            left_connecting_point: float,
                                                            right_connecting_point: float) -> DeterministicSumUnit:
        """
        Create a deterministic uniform mixture from two datasets.
        The left dataset is included in the left uniform distribution up to the middle point between the last
        point in the left dataset and the first point in the right dataset.
        The right dataset is included in the right uniform distribution from the middle point.
        The weights of the mixture correspond to the relative size of the datasets.

        :param left_data: The data for the left distribution.
        :param right_data: The data for the right distribution.
        :param left_connecting_point: The connecting point that this distribution has to connect to on the left.
        :param right_connecting_point: The connection point that this distribution has to connect to on the right.
        :return: A deterministic uniform mixture of the two datasets.
        """
        middle_connecting_point = (left_data[-1] + right_data[0]) / 2

        # creat uniform distribution from the left including to the right excluding
        left_uniform_distribution = UniformDistribution(self.variable, portion.closedopen(left_connecting_point,
                                                                                          middle_connecting_point))
        right_uniform_distribution = UniformDistribution(self.variable, portion.closed(middle_connecting_point,
                                                                                       right_connecting_point))

        datapoints_total = len(left_data) + len(right_data)

        result = DeterministicSumUnit(self.variables,
                                      [len(left_data) / datapoints_total, len(right_data) / datapoints_total])
        result.children = [left_uniform_distribution, right_uniform_distribution]
        return result
