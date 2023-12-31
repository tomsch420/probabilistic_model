import collections
import dataclasses
from typing import Optional, List, Deque, Union, Tuple, Dict, Any

import plotly.graph_objects as go
import portion
from random_events.events import Event
from random_events.variables import Continuous, Variable
from typing_extensions import Self

from ..probabilistic_circuit.distribution import ContinuousDistribution, DiracDeltaDistribution
from ..probabilistic_circuit.distributions.uniform import UniformDistribution
from ..probabilistic_circuit.units import DeterministicSumUnit, Unit, SmoothSumUnit


@dataclasses.dataclass
class InductionStep:
    """
    Class for performing induction in the NygaDistributions.
    """

    data: List[float]
    """
    The entire sorted and unique data points.
    """

    total_number_of_samples: int
    """
    The total number of samples in the dataset.
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

    nyga_distribution: 'NygaDistribution'
    """
    The Nyga Distribution to mount the quantile distributions into and read the parameters from.
    """

    @property
    def variable(self):
        """
        The variable of the distribution.
        """
        return self.nyga_distribution.variable

    @property
    def min_samples_per_quantile(self):
        """
        The minimal number of samples per quantile.
        """
        return self.nyga_distribution.min_samples_per_quantile

    @property
    def min_likelihood_improvement(self):
        """
        The relative, minimal likelihood improvement needed to create a new quantile.
        """
        return self.nyga_distribution.min_likelihood_improvement

    def left_connecting_point(self) -> float:
        """
        Calculate the left connecting point.
        """
        return self.left_connecting_point_from_index(self.begin_index)

    def left_connecting_point_from_index(self, index) -> float:
        """
        Calculate the left connecting point given some beginning index.

        :param index: The index of the left datapoint.
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

    def right_connecting_point_from_index(self, index) -> float:
        """
        Calculate the right connecting point given some ending index.

        :param index: The index of the right datapoint.
        """
        if index < len(self.data):
            right_connecting_point = (self.data[index] + self.data[index - 1]) / 2
        else:
            right_connecting_point = self.data[index - 1]
        return right_connecting_point

    def create_uniform_distribution(self) -> UniformDistribution:
        """
        Create a uniform distribution from this induction step.
        """
        return self.create_uniform_distribution_from_indices(self.begin_index, self.end_index)

    def create_uniform_distribution_from_indices(self, begin_index: int, end_index: int) -> UniformDistribution:
        """
        Create a uniform distribution from the datapoint at `begin_index` to the datapoint at `end_index`.

        :param begin_index: The index of the first datapoint.
        :param end_index: The index of the last datapoint.
        """
        if end_index == len(self.data):
            interval = portion.closed(self.left_connecting_point_from_index(begin_index),
                                      self.right_connecting_point_from_index(end_index))
        else:
            interval = portion.closedopen(self.left_connecting_point_from_index(begin_index),
                                          self.right_connecting_point_from_index(end_index))
        return UniformDistribution(self.variable, interval)

    def sum_weights_from_indices(self, begin_index: int, end_index: int) -> float:
        """
        Sum the weights from `begin_index` to `end_index`.
        """
        return sum(self.weights[begin_index:end_index])

    def sum_weights(self):
        """
        Sum the weights of this induction step.
        """
        return self.sum_weights_from_indices(self.begin_index, self.end_index)

    def compute_best_split(self) -> Tuple[float, Optional[int]]:

        # initialize the maximum likelihood and the best split index
        maximum_likelihood = 0
        best_split_index = None

        # calculate the connecting points
        right_connecting_point = self.right_connecting_point()
        left_connecting_point = self.left_connecting_point()

        # calculate the number of samples that are involved in this partition
        total_number_of_samples = self.sum_weights() * self.total_number_of_samples

        # for every possible splitting index
        for split_index in range(self.begin_index + self.min_samples_per_quantile,
                                 self.end_index - self.min_samples_per_quantile + 1):

            # calculate the split value
            split_value = (self.data[split_index - 1] + self.data[split_index]) / 2

            # calculate left likelihood
            weight_sum_left = self.sum_weights_from_indices(self.begin_index, split_index)
            number_of_samples_left = self.total_number_of_samples * weight_sum_left
            average_likelihood_left = weight_sum_left / (split_value - left_connecting_point) * number_of_samples_left

            # calculate right sum of likelihoods
            weight_sum_right = self.sum_weights_from_indices(split_index, self.end_index)
            number_of_samples_right = self.total_number_of_samples * weight_sum_right
            average_likelihood_right = (
                        weight_sum_right / (right_connecting_point - split_value) * number_of_samples_right)

            # calculate average likelihood
            average_likelihood = (average_likelihood_left + average_likelihood_right) / total_number_of_samples

            # update the maximum likelihood and the best split index
            if average_likelihood > maximum_likelihood:
                maximum_likelihood = average_likelihood
                best_split_index = split_index

        return maximum_likelihood, best_split_index

    def construct_left_induction_step(self, split_index: int) -> Self:
        """
        Construct the left induction step.

        :param split_index: The index of the split.
        """
        return InductionStep(self.data, self.total_number_of_samples, self.weights, self.begin_index, split_index,
                             self.nyga_distribution)

    def construct_right_induction_step(self, split_index: int) -> Self:
        """
        Construct the right induction step.

        :param split_index: The index of the split.
        """
        return InductionStep(self.data, self.total_number_of_samples, self.weights, split_index, self.end_index,
                             self.nyga_distribution)

    def induce(self) -> List[Self]:
        """
        Perform one induction step.

        :return: The (possibly empty) list of new induction steps.
        """
        # calculate the likelihood without splitting
        average_likelihood_without_split = (
                    (self.sum_weights()) / (self.right_connecting_point() - self.left_connecting_point()))

        # calculate the best likelihood with splitting
        maximum_likelihood, best_split_index = self.compute_best_split()

        # if the improvement is good enough
        if maximum_likelihood > (1 + self.min_likelihood_improvement) * average_likelihood_without_split:

            # create the left and right induction steps
            left_induction_step = self.construct_left_induction_step(best_split_index)
            right_induction_step = self.construct_right_induction_step(best_split_index)
            return [left_induction_step, right_induction_step]

        # if the improvement is not good enough
        else:
            # mount a uniform distribution
            distribution = self.create_uniform_distribution()
            self.nyga_distribution.weights += [self.sum_weights()]
            distribution.parent = self.nyga_distribution
            return []


class NygaDistribution(DeterministicSumUnit, ContinuousDistribution):
    """
    A Nyga distribution is a deterministic mixture of uniform distributions.
    """

    min_likelihood_improvement: float
    """
    The relative, minimal likelihood improvement needed to create a new quantile.
    """

    min_samples_per_quantile: int
    """
    The minimal number of samples per quantile.
    """

    weights: List[float]

    def __init__(self, variable: Continuous, min_samples_per_quantile: Optional[int] = 1,
                 min_likelihood_improvement: Optional[float] = 1.1, parent: 'Unit' = None):
        DeterministicSumUnit.__init__(self, [variable], [], parent)
        ContinuousDistribution.__init__(self, variable, parent)
        self.min_samples_per_quantile = min_samples_per_quantile
        self.min_likelihood_improvement = min_likelihood_improvement

    def _pdf(self, value: Union[float, int]) -> float:
        return sum([child._pdf(value) * weight for child, weight in zip(self.children, self.weights)])

    def _cdf(self, value: Union[float, int]) -> float:
        return sum([child._cdf(value) * weight for child, weight in zip(self.children, self.weights)])

    def fit(self, data: List[float]):
        return self._fit(list(self.variable.encode_many(data)))

    def _fit(self, data: List[float]) -> Self:
        """
        Fit the distribution to the data.

        :param data: The data to fit the distribution to.

        :return: The fitted distribution.
        """

        # sort the data and calculate the weights
        sorted_data = sorted(set(data))

        if len(sorted_data) == 1:
            self.weights.append(1.)
            distribution = DiracDeltaDistribution(self.variable, sorted_data[0], parent=self)
            return self

        weights = [data.count(value) / len(data) for value in sorted_data]

        # construct the initial induction step
        initial_induction_step = InductionStep(data=sorted_data, total_number_of_samples=len(data), weights=weights,
                                               begin_index=0, end_index=len(sorted_data), nyga_distribution=self)
        # initialize the queue
        induction_steps: Deque[InductionStep] = collections.deque([initial_induction_step])

        # induce the distribution
        while len(induction_steps) > 0:
            induction_step = induction_steps.popleft()
            new_induction_steps = induction_step.induce()
            induction_steps.extend(new_induction_steps)

        return self

    def pdf_trace(self) -> go.Scatter:
        """
        Create a plotly trace for the probability density function.
        """
        domain_size = self.domain[self.variable].upper - self.domain[self.variable].lower
        x = [self.domain[self.variable].lower - domain_size * 0.05, self.domain[self.variable].lower, None]
        y = [0, 0, None]
        for uniform in self.leaves:
            lower_value = uniform.interval.lower
            upper_value = uniform.interval.upper
            pdf_value = uniform.pdf_value() * uniform.get_weight_if_possible()
            x += [lower_value, upper_value, None]
            y += [pdf_value, pdf_value, None]

        x.extend([self.domain[self.variable].upper, self.domain[self.variable].upper + domain_size * 0.05])
        y.extend([0, 0])
        return go.Scatter(x=x, y=y, mode='lines', name="Probability Density Function")

    def cdf_trace(self) -> go.Scatter:
        """
        Create a plotly trace for the cumulative distribution function.
        """
        domain_size = self.domain[self.variable].upper - self.domain[self.variable].lower
        x = [self.domain[self.variable].lower - domain_size * 0.05, self.domain[self.variable].lower, None]
        y = [0, 0, None]
        for uniform in self.leaves:
            lower_value = uniform.interval.lower
            upper_value = uniform.interval.upper
            x += [lower_value, upper_value, None]
            y += [self.cdf(lower_value), self.cdf(upper_value), None]

        x.extend([self.domain[self.variable].upper, self.domain[self.variable].upper + domain_size * 0.05])
        y.extend([1, 1])
        return go.Scatter(x=x, y=y, mode='lines', name="Cumulative Distribution Function")

    def plot(self) -> List[go.Scatter]:
        """
        Plot the distribution with PDF, CDF, Expectation and Mode.
        """
        traces = [self.pdf_trace(), self.cdf_trace()]

        mode, maximum_likelihood = self.mode()
        mode = mode[0][self.variable]

        expectation = self.expectation([self.variable])[self.variable]
        traces.append(go.Scatter(x=[mode.lower, mode.lower, mode.upper, mode.upper, ],
                                 y=[0, maximum_likelihood * 1.05, maximum_likelihood * 1.05, 0], mode='lines+markers',
                                 name="Mode", fill="toself"))
        traces.append(go.Scatter(x=[expectation, expectation], y=[0, maximum_likelihood * 1.05], mode='lines+markers',
                                 name="Expectation"))

        return traces

    def __eq__(self, other: Self):
        return (isinstance(other, NygaDistribution) and
                self.min_likelihood_improvement == other.min_likelihood_improvement and
                self.min_samples_per_quantile == other.min_samples_per_quantile and super().__eq__(other))

    def __copy__(self) -> Self:
        """
        Create a copy of the distribution.
        """
        result = NygaDistribution(self.variable, self.min_samples_per_quantile, self.min_likelihood_improvement)
        result.weights = self.weights.copy()
        result.children = self._copy_children()
        return result

    def _parameter_copy(self):
        result = NygaDistribution(variable=self.variable,
                                  min_samples_per_quantile=self.min_samples_per_quantile,
                                  min_likelihood_improvement=self.min_likelihood_improvement)
        result.weights = self.weights
        return result

    def to_json(self) -> Dict[str, Any]:
        """
        Create a json representation of the distribution.
        """
        result = super().to_json()
        result["min_samples_per_quantile"] = self.min_samples_per_quantile
        result["min_likelihood_improvement"] = self.min_likelihood_improvement
        return result

    @classmethod
    def from_json_with_variables_and_children(cls, data: Dict[str, Any],
                                              variables: List[Variable],
                                              children: List['Unit']) -> Self:
        result = cls(list(variables)[0], data["min_samples_per_quantile"], data["min_likelihood_improvement"])
        result.weights = data["weights"]
        result.children = children
        return result

    @classmethod
    def from_uniform_mixture(cls, mixture: SmoothSumUnit) -> Self:
        """
        Construct a Nyga Distribution from a mixture of uniform distributions.
        The mixture does not have to be deterministic.
        :param mixture: An arbitrary, univariate mixture of uniform distributions
        :return: A Nyga Distribution describing the same function.
        """
        variable: Continuous = mixture.variables[0]
        result = cls(variable)

        all_mixture_points = set()
        for leaf in mixture.leaves:
            leaf: UniformDistribution
            all_mixture_points.add(leaf.interval.lower)
            all_mixture_points.add(leaf.interval.upper)

        all_mixture_points = sorted(list(all_mixture_points))

        for index, (lower, upper) in enumerate(zip(all_mixture_points[:-1], all_mixture_points[1:])):
            if index == len(all_mixture_points) - 2:
                interval = portion.closed(lower, upper)
            else:
                interval = portion.closedopen(lower, upper)
            leaf = UniformDistribution(variable, interval, parent=result)
            weight = mixture.probability(Event({variable: interval}))
            result.weights.append(weight)

        return result
