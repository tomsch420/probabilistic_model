from __future__ import annotations

import numpy as np
from random_events.interval import SimpleInterval, Interval
from random_events.product_algebra import SimpleEvent
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Variable
from typing_extensions import Optional

from probabilistic_model.utils import MissingDict
from ..probabilistic_circuit import SumUnit, LeafUnit
from ....distributions.distributions import (ContinuousDistribution as PMContinuousDistribution)


class UnivariateLeaf(LeafUnit):

    @property
    def variable(self) -> Variable:
        return self.distribution.variables[0]

    def log_conditional_of_simple_event_in_place(self, event: SimpleEvent):
        return self.univariate_log_conditional_of_simple_event_in_place(event[self.variable])

    def univariate_log_conditional_of_simple_event_in_place(self, event: AbstractCompositeSet):
        raise NotImplementedError


class UnivariateContinuousLeaf(UnivariateLeaf):
    distribution: Optional[PMContinuousDistribution]

    def univariate_log_conditional_of_simple_event_in_place(self, event: Interval):

        event = self.distribution.univariate_support & event

        if event.is_empty():
            self.result_of_current_query = -np.inf
            self.distribution = None
            return None

        # if it is a simple truncation
        if len(event.simple_sets) == 1:
            self.distribution, self.result_of_current_query = self.distribution.log_conditional_from_simple_interval(
                event.simple_sets[0])
            return self

        total_probability = 0.

        # calculate the conditional distribution as sum unit
        result = SumUnit(self.probabilistic_circuit)

        for simple_interval in event.simple_sets:
            current_conditional, current_log_probability = self.distribution.log_conditional_from_simple_interval(
                simple_interval)
            current_probability = np.exp(current_log_probability)

            if current_probability == 0:
                continue

            current_conditional = self.__class__(current_conditional, self.probabilistic_circuit)
            result.add_subcircuit(current_conditional, current_probability, mount=False)
            total_probability += current_probability

        # if the event is impossible
        if total_probability == 0:
            self.result_of_current_query = -np.inf
            self.distribution = None
            return None

        # reroute the parent to the new sum unit
        self.connect_incoming_edges_to(result)

        # remove this node
        self.probabilistic_circuit.remove_node(self)

        # update result
        result.normalize()
        result.result_of_current_query = np.log(total_probability)

        return result


class DiscreteDistribution(UnivariateLeaf):

    def as_deterministic_sum(self) -> SumUnit:
        """
        Convert this distribution to a deterministic sum unit that encodes the same distribution.
        The result has as many children as the domain of the variable and each child encodes the value of the variable.

        :return: A deterministic sum unit that encodes the same distribution.
        """
        result = SumUnit()

        for event in self.variable.domain.simple_sets:
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
