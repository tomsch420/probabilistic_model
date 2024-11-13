from __future__ import annotations

from abc import ABC

import numpy as np
import random_events.interval
from random_events.interval import SimpleInterval, Interval
from random_events.product_algebra import SimpleEvent, Event
from random_events.sigma_algebra import AbstractCompositeSet, AbstractSimpleSet
import random_events.set
from sortedcontainers import SortedSet
from typing_extensions import Tuple, Optional, Self, List

from probabilistic_model.distributions.distributions import (ContinuousDistribution as PMContinuousDistribution,
                                                             DiracDeltaDistribution as PMDiracDeltaDistribution,
                                                             SymbolicDistribution as PMSymbolicDistribution,
                                                             IntegerDistribution as PMIntegerDistribution,
                                                             DiscreteDistribution as PMDiscreteDistribution,
                                                             UnivariateDistribution as PMUnivariateDistribution)
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import (ProbabilisticCircuitMixin, cache_inference_result,
                                                                                SumUnit)
from probabilistic_model.distributions.uniform import UniformDistribution as PMUniformDistribution
from probabilistic_model.distributions.gaussian import (GaussianDistribution as PMGaussianDistribution,
                                                        TruncatedGaussianDistribution as PMTruncatedGaussianDistribution)
from probabilistic_model.utils import MissingDict


class UnivariateDistribution(PMUnivariateDistribution, ProbabilisticCircuitMixin, ABC):

    def is_deterministic(self) -> bool:
        return True

    @property
    def variables(self) -> SortedSet:
        return SortedSet([self.variable])

    def __hash__(self):
        return ProbabilisticCircuitMixin.__hash__(self)

    # @cache_inference_result
    def log_conditional_of_simple_event(self, event: SimpleEvent) -> Tuple[Optional[Self], float]:
        return super().log_conditional(event.as_composite_set())

    # @cache_inference_result
    def simplify(self) -> Self:
        return self.__copy__()

    def empty_copy(self) -> Self:
        return self.__copy__()

    def event_of_higher_density(self, other: Self, own_weight, other_weight) -> Tuple[Event, Event]:
        """
        Calculate the event where $own_weight * self.pdf > other_weight * other.pdf$

        :return: A tuple of two events, the first describes where the weighted density of this is higher than the
        weighted density of other and the vice versa event
        """
        raise NotImplementedError


class ContinuousDistribution(UnivariateDistribution, PMContinuousDistribution, ProbabilisticCircuitMixin, ABC):

    def log_conditional_from_interval(self, interval: Interval) -> Tuple[SumUnit, float]:
        result = SumUnit()
        total_probability = 0.

        for simple_interval in interval.simple_sets:
            current_conditional, current_log_probability = self.log_conditional_from_simple_interval(simple_interval)
            current_probability = np.exp(current_log_probability)
            result.add_subcircuit(current_conditional, current_probability)
            total_probability += current_probability

        result.normalize()

        return result, np.log(total_probability)

    def log_conditional_from_singleton(self, interval: SimpleInterval) -> Tuple[DiracDeltaDistribution, float]:
        conditional, probability = super().log_conditional_from_singleton(interval)
        return DiracDeltaDistribution(conditional.variable, conditional.location,
                                      conditional.density_cap), probability


class DiscreteDistribution(UnivariateDistribution, PMDiscreteDistribution, ProbabilisticCircuitMixin, ABC):

    def as_deterministic_sum(self) -> SumUnit:
        """
        Convert this distribution to a deterministic sum unit that encodes the same distribution.
        The result has as many children as the domain of the variable and each child encodes the value of the variable.

        :return: A deterministic sum unit that encodes the same distribution.
        """
        result = SumUnit()

        for event in self.univariate_support.simple_sets:
            event = SimpleEvent({self.variable: event}).as_composite_set()
            conditional, probability = self.conditional(event)
            result.add_subcircuit(conditional, probability)

        return result

    @classmethod
    def from_sum_unit(cls, sum_unit: SumUnit):
        """
        Create a discrete distribution from a sum unit.

        :param sum_unit: The sum unit to create the distribution from.
        :return: The discrete distribution.
        """
        assert len(sum_unit.variables) == 1, "Can only convert unidimensional sum units to discrete distributions."
        variable = sum_unit.variables[0]
        probabilities = MissingDict(float)

        for element in sum_unit.support.simple_sets[0][variable].simple_sets:
            probability = sum_unit.probability_of_simple_event(SimpleEvent({variable: element}))
            if isinstance(element, SimpleInterval):
                element = element.lower
            probabilities[int(element)] = probability
        return cls(variable, probabilities)

    def create_event_from_value_list(self, values: List[AbstractSimpleSet]) -> AbstractCompositeSet:
        raise NotImplementedError

class DiracDeltaDistribution(ContinuousDistribution, PMDiracDeltaDistribution):

    def event_of_higher_density(self, other: Self, own_weight, other_weight) -> Tuple[Event, Event]:
        if self.location == other.location:
            own_density = self.density_cap * own_weight
            other_density = self.density_cap * other_weight

            empty_interval = random_events.interval.open(0, 0)

            if own_density > other_density:
                return self.support, SimpleEvent({other.variable: empty_interval}).as_composite_set()
            elif own_density < other_density:
                return SimpleEvent({self.variable: empty_interval}).as_composite_set(), other.support

        return self.support, other.support


class UniformDistribution(ContinuousDistribution, PMUniformDistribution):

    def event_of_higher_density(self, other: Self, own_weight, other_weight) -> Tuple[Event, Event]:
        own_density = self.pdf_value() * own_weight
        other_density = self.pdf_value() * other_weight

        own_event = self.support - other.support
        other_event = other.support - self.support
        intersection = self.support & other.support

        if own_density > other_density:
            own_event |= intersection
        else:
            other_event |= intersection

        return own_event, other_event


class GaussianDistribution(ContinuousDistribution, PMGaussianDistribution):

    def log_conditional_from_non_singleton_simple_interval(self, interval: SimpleInterval) -> (
            Tuple)[TruncatedGaussianDistribution, float]:
        conditional, log_probability = (PMGaussianDistribution.
                                        log_conditional_from_non_singleton_simple_interval(self, interval))
        return TruncatedGaussianDistribution(conditional.variable, conditional.interval,
                                             conditional.location, conditional.scale), log_probability


class TruncatedGaussianDistribution(GaussianDistribution, ContinuousDistribution, PMTruncatedGaussianDistribution):
    ...


class IntegerDistribution(DiscreteDistribution, PMIntegerDistribution):

    def event_of_higher_density(self, other: Self, own_weight, other_weight) -> Tuple[Event, Event]:
        own_result = []
        other_result = []
        for value in self.variable.domain.simple_sets:
            p_own = self.probabilities[value.lower]
            p_other = other.probabilities[value.lower]
            if p_own > p_other:
                own_result.append(value)
            elif p_own < p_other:
                other_result.append(value)

        own_result = self.create_event_from_value_list(own_result)

        if own_result.is_empty():
            own_result = Event()
        else:
            own_result = SimpleEvent({self.variable: own_result}).as_composite_set()

        other_result = self.create_event_from_value_list(other_result)
        if other_result.is_empty():

            other_result = Event()
        else:
            other_result = SimpleEvent({other.variable: other_result}).as_composite_set()
        return own_result, other_result

    def create_event_from_value_list(self, values: List[AbstractSimpleSet]) -> AbstractCompositeSet:
        if len(values) == 0:
            return random_events.interval.open(0, 0)
        result = values[0].as_composite_set()
        for value in values[1:]:
            result |= value.as_composite_set()
        return result

class SymbolicDistribution(DiscreteDistribution, PMSymbolicDistribution):

    def event_of_higher_density(self, other: Self, own_weight, other_weight) -> Tuple[Event, Event]:

        own_result = []
        other_result = []
        for value in self.variable.domain.simple_sets:
            p_own = self.probabilities[int(value)]
            p_other = other.probabilities[int(value)]
            if p_own > p_other:
                own_result.append(value)
            elif p_own < p_other:
                other_result.append(value)

        own_result = self.create_event_from_value_list(own_result)
        if own_result.is_empty():
            own_result = Event()
        else:
            own_result = SimpleEvent({self.variable: own_result}).as_composite_set()

        other_result = self.create_event_from_value_list(other_result)
        if other_result.is_empty():
            other_result = Event()
        else:
            other_result = SimpleEvent({other.variable: other_result}).as_composite_set()

        return own_result, other_result

    def create_event_from_value_list(self, values: List[AbstractSimpleSet]) -> AbstractCompositeSet:
        if len(values) == 0:
            return random_events.set.Set()

        result = values[0].as_composite_set()
        for value in values[1:]:
            result |= value.as_composite_set()
        return result
