import collections
import dataclasses
from functools import cached_property
from typing import Optional, List, Deque, Union, Tuple, Dict, Any

import numpy as np
import plotly.graph_objects as go
import portion
from random_events.events import Event
from random_events.variables import Continuous, Variable
from typing_extensions import Self

from ..probabilistic_circuit.distributions import ContinuousDistribution, DiracDeltaDistribution, UniformDistribution
from ..probabilistic_circuit.probabilistic_circuit import (DeterministicSumUnit, SmoothSumUnit, cache_inference_result)


@dataclasses.dataclass
class InductionStep:
    """
    Class for performing induction in the NygaDistributions.
    """

    data: np.array
    """
    The entire sorted and unique data points
    """

    cumulative_weights: np.array
    """
    The cumulative weights of the samples in the dataset.
    """

    cumulative_log_weights: np.array
    """
    The cumulative logarithmic weights of the samples in the dataset.
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

    @property
    def number_of_samples(self):
        """
        The number of samples in the induction step.
        """
        return self.end_index - self.begin_index

    @property
    def total_weights(self):
        """
        The total sum of weights of the samples in the induction step.
        """
        return self.cumulative_weights[self.end_index] - self.cumulative_weights[self.begin_index]

    @property
    def total_log_weights(self):
        """
        The total sum of logarithmic weights of the samples in the induction step.
        """
        return self.cumulative_log_weights[self.end_index] - self.cumulative_log_weights[self.begin_index]

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
        return self.cumulative_weights[end_index] - self.cumulative_weights[begin_index]

    def sum_weights(self):
        """
        Sum the weights of this induction step.
        """
        return self.sum_weights_from_indices(self.begin_index, self.end_index)

    def sum_log_weights_from_indices(self, begin_index: int, end_index: int) -> float:
        """
        Sum the logarithmic weights from `begin_index` to `end_index`.
        """
        return self.cumulative_log_weights[end_index] - self.cumulative_log_weights[begin_index]

    def sum_log_weights(self):
        """
        Sum the logarithmic weights of this induction step.
        """
        return self.sum_log_weights_from_indices(self.begin_index, self.end_index)

    def compute_best_split(self) -> Tuple[float, Optional[int]]:
        """
        Compute the best split of the data.

        The best split of the data is computed by evaluating the log likelihood of every possible split and memorizing
        the best one.

        :return: The maximum log likelihood and the best split index.
        """

        # initialize the maximum likelihood and the best split index
        maximum_log_likelihood = -float("inf")
        best_split_index = None

        # calculate the connecting points
        right_connecting_point = self.right_connecting_point()
        left_connecting_point = self.left_connecting_point()

        # for every possible splitting index
        for split_index in range(self.begin_index + self.min_samples_per_quantile,
                                 self.end_index - self.min_samples_per_quantile + 1):

            # calculate log likelihoods
            log_likelihood_left = self.log_likelihood_of_split_side(split_index, left_connecting_point)
            log_likelihood_right = self.log_likelihood_of_split_side(split_index, right_connecting_point)
            log_likelihood = (log_likelihood_left + log_likelihood_right)

            # update the maximum likelihood and the best split index
            if log_likelihood > maximum_log_likelihood:
                maximum_log_likelihood = log_likelihood
                best_split_index = split_index

        return maximum_log_likelihood, best_split_index

    def log_likelihood_without_split(self) -> float:
        """
        Calculate the log likelihood without splitting.

        :return: The log likelihood without splitting.
        """
        log_density = -np.log(self.right_connecting_point() - self.left_connecting_point())
        return self.sum_log_weights() + (self.number_of_samples * log_density)

    def log_likelihood_of_split_side(self, split_index: int, connecting_point: float) -> float:
        """
        Calculate the log likelihood of a split side.

        This method automatically determines if this is the left or right side of the split.

        :param split_index: The index of the split.
        :param connecting_point: The connecting point.

        :return: The log likelihood of the split.
        """

        # calculate the split value
        split_value = (self.data[split_index - 1] + self.data[split_index]) / 2

        # calculate the log density
        density = split_value - connecting_point
        is_left = density > 0
        log_density = np.log(np.abs(density))

        # calculate the log of the weight of this partition in the sum node
        log_weight_sum_of_split = np.log(self.sum_weights_from_indices(self.begin_index, split_index)) if is_left \
            else np.log(self.sum_weights_from_indices(split_index, self.end_index))

        # calculate the log of the sum of the weights of both partitions
        log_weight_sum = np.log(self.total_weights)

        # calculate the number of samples in this partition
        number_of_samples = split_index - self.begin_index if is_left else self.end_index - split_index

        # calculate the sum of the logarithmic weights of the samples in this partition
        sum_of_log_weights_of_samples = self.sum_log_weights_from_indices(self.begin_index, split_index) if is_left \
            else self.sum_log_weights_from_indices(split_index, self.end_index)

        # add the terms together
        log_likelihood = (number_of_samples * (log_weight_sum_of_split - log_weight_sum - log_density) +
                          sum_of_log_weights_of_samples)

        return log_likelihood

    def construct_left_induction_step(self, split_index: int) -> Self:
        """
        Construct the left induction step.

        :param split_index: The index of the split.
        """
        return InductionStep(self.data, self.cumulative_weights, self.cumulative_log_weights, self.begin_index,
                             split_index, self.nyga_distribution)

    def construct_right_induction_step(self, split_index: int) -> Self:
        """
        Construct the right induction step.

        :param split_index: The index of the split.
        """
        return InductionStep(self.data, self.cumulative_weights, self.cumulative_log_weights, split_index,
                             self.end_index, self.nyga_distribution)

    def improvement_is_good_enough(self, maximum_log_likelihood: float) -> bool:
        """
        Check if the improvement is good enough.
        :param maximum_log_likelihood: The improved maximum log likelihood.
        :return: Rather the improvement is good enough
        """
        log_likelihood_without_split = self.log_likelihood_without_split()
        return np.exp(maximum_log_likelihood - log_likelihood_without_split) > self.min_likelihood_improvement

    def induce(self) -> List[Self]:
        """
        Perform one induction step.

        :return: The (possibly empty) list of new induction steps.
        """

        # calculate the best likelihood with splitting
        maximum_log_likelihood, best_split_index = self.compute_best_split()

        # if the improvement is good enough
        if self.improvement_is_good_enough(maximum_log_likelihood):

            # create the left and right induction steps
            left_induction_step = self.construct_left_induction_step(best_split_index)
            right_induction_step = self.construct_right_induction_step(best_split_index)
            return [left_induction_step, right_induction_step]

        # if the improvement is not good enough
        else:
            # calculate the weight of the uniform distribution
            weight = self.total_weights / self.cumulative_weights[-1]

            # mount a uniform distribution
            distribution = self.create_uniform_distribution()
            self.nyga_distribution.add_subcircuit(distribution, weight)

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

    def __init__(self, variable: Continuous, min_samples_per_quantile: Optional[int] = 1,
                 min_likelihood_improvement: Optional[float] = 0.1):
        DeterministicSumUnit.__init__(self)
        ContinuousDistribution.__init__(self, variable)
        self.min_samples_per_quantile = min_samples_per_quantile
        self.min_likelihood_improvement = min_likelihood_improvement

    @property
    def variables(self) -> Tuple[Variable, ...]:
        if len(self.subcircuits) > 0:
            return DeterministicSumUnit.variables.fget(self)
        else:
            return self._variables

    @cache_inference_result
    def _pdf(self, value: Union[float, int]) -> float:
        return sum([weight * subcircuit._pdf(value) for weight, subcircuit in self.weighted_subcircuits])

    @cache_inference_result
    def _cdf(self, value: Union[float, int]) -> float:
        return sum([weight * subcircuit._cdf(value) for weight, subcircuit in self.weighted_subcircuits])

    def fit(self, data: np.array, weights: Optional[np.array] = None) -> Self:
        """
        Fit the distribution to the data.

        :param data: The data to fit the distribution to.
        :param weights: The optional weights of the data points.

        :return: The fitted distribution.
        """
        return self._fit(list(self.variable.encode_many(data)), weights)

    def _fit(self, data: List[float], weights: Optional[np.array] = None) -> Self:

        # make the data unique and sort it
        sorted_unique_data, counts = np.unique(data, return_counts=True)

        # if the data contains only one value
        if len(sorted_unique_data) == 1:
            # mount a dirac delta distribution and return
            distribution = DiracDeltaDistribution(self.variable, sorted_unique_data[0])
            self.probabilistic_circuit.add_edge(self, distribution, weight=1)
            return self

        # if the weights are not given
        if weights is None:
            weights = counts

        log_weights = np.log(weights)
        cumulative_log_weights = np.cumsum(log_weights)
        cumulative_log_weights = np.insert(cumulative_log_weights, 0, 0)

        cumulative_weights = np.cumsum(weights)
        cumulative_weights = np.insert(cumulative_weights, 0, 0)

        # construct the initial induction step
        initial_induction_step = InductionStep(data=sorted_unique_data,
                                               cumulative_weights=cumulative_weights,
                                               cumulative_log_weights=cumulative_log_weights,
                                               begin_index=0, end_index=len(sorted_unique_data),
                                               nyga_distribution=self)

        # initialize the queue
        induction_steps: Deque[InductionStep] = collections.deque([initial_induction_step])

        # induce the distribution
        while len(induction_steps) > 0:
            induction_step = induction_steps.popleft()
            new_induction_steps = induction_step.induce()
            induction_steps.extend(new_induction_steps)

        self.normalize()
        return self

    def empty_copy(self) -> Self:
        return NygaDistribution(self.variable, min_samples_per_quantile=self.min_samples_per_quantile,
                                min_likelihood_improvement=self.min_likelihood_improvement)

    def to_json(self) -> Dict[str, Any]:
        """
        Create a json representation of the distribution.
        """
        result = super().to_json()
        result["min_samples_per_quantile"] = self.min_samples_per_quantile
        result["min_likelihood_improvement"] = self.min_likelihood_improvement
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        smooth_sum_unit = DeterministicSumUnit()._from_json(data)
        result = cls([], min_samples_per_quantile=data["min_samples_per_quantile"],
                     min_likelihood_improvement=data["min_likelihood_improvement"])
        for weight, subcircuit in smooth_sum_unit.weighted_subcircuits:
            result.mount(subcircuit)
            result.probabilistic_circuit.add_edge(result, subcircuit, weight=weight)
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
            leaf = UniformDistribution(variable, interval)
            weight = mixture.probability(Event({variable: interval}))
            result.probabilistic_circuit.add_edge(result, leaf, weight=weight)

        return result

    def pdf_trace(self) -> go.Scatter:
        """
        Create a plotly trace for the probability density function.
        """
        domain = self.domain.events[0][self.variable]
        domain_size = domain.upper - domain.lower
        x = [domain.lower - domain_size * 0.05, domain.lower, None]
        y = [0, 0, None]
        for weight, subcircuit in self.weighted_subcircuits:
            uniform: UniformDistribution = subcircuit
            lower_value = uniform.interval.lower
            upper_value = uniform.interval.upper
            pdf_value = uniform.pdf_value() * weight
            x += [lower_value, upper_value, None]
            y += [pdf_value, pdf_value, None]

        x.extend([domain.upper, domain.upper + domain_size * 0.05])
        y.extend([0, 0])
        return go.Scatter(x=x, y=y, mode='lines', name="PDF")

    def cdf_trace(self) -> go.Scatter:
        """
        Create a plotly trace for the cumulative distribution function.
        """
        domain = self.domain.events[0][self.variable]
        domain_size = domain.upper - domain.lower
        x = [domain.lower - domain_size * 0.05, domain.lower, None]
        y = [0, 0, None]
        for subcircuit in sorted(self.subcircuits, key=lambda d: d.interval.lower):
            uniform: UniformDistribution = subcircuit
            lower_value = uniform.interval.lower
            upper_value = uniform.interval.upper
            x += [lower_value, upper_value]
            y += [self.cdf(lower_value), self.cdf(upper_value)]

        x.extend([domain.upper, domain.upper + domain_size * 0.05])
        y.extend([1, 1])
        return go.Scatter(x=x, y=y, mode='lines', name="CDF")

    def plot(self) -> List[go.Scatter]:
        """
        Plot the distribution with PDF, CDF, Expectation and Mode.
        """
        traces = [self.pdf_trace(), self.cdf_trace()]
        mode_trace, maximum_likelihood = self.mode_trace_1d()
        self.reset_result_of_current_query()

        expectation = self.expectation([self.variable])[self.variable]
        traces.append(mode_trace)
        self.reset_result_of_current_query()
        traces.append(go.Scatter(x=[expectation, expectation], y=[0, maximum_likelihood * 1.05], mode='lines+markers',
                                 name="Expectation"))
        return traces
