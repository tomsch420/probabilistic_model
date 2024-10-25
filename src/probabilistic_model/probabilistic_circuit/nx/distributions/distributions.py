from __future__ import annotations

import networkx as nx
import numpy as np
from random_events.interval import SimpleInterval, Interval
from random_events.product_algebra import SimpleEvent
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Variable, Integer
from typing_extensions import Optional

from ....utils import MissingDict
from ..probabilistic_circuit import SumUnit, LeafUnit, ProbabilisticCircuit
from ....distributions.distributions import (ContinuousDistribution, DiscreteDistribution, IntegerDistribution,
                                             SymbolicDistribution)


class UnivariateLeaf(LeafUnit):

    @property
    def variable(self) -> Variable:
        return self.distribution.variables[0]


class UnivariateContinuousLeaf(UnivariateLeaf):
    distribution: Optional[ContinuousDistribution]

    def log_conditional_of_simple_event_in_place(self, event: SimpleEvent):
        return self.univariate_log_conditional_of_simple_event_in_place(event[self.variable])

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


class UnivariateDiscreteLeaf(UnivariateLeaf):

    distribution: Optional[DiscreteDistribution]

    def as_deterministic_sum(self) -> SumUnit:
        """
        Convert this distribution to a deterministic sum unit that encodes the same distribution in-place.
        The result has as many children as the probability dictionary of this distribution.
        Each child encodes the value of the variable.

        :return: The deterministic sum unit that encodes the same distribution.
        """
        result = SumUnit(self.probabilistic_circuit)

        for element, probability in self.distribution.probabilities.items():
            result.add_subcircuit(UnivariateDiscreteLeaf(self.distribution.__class__(self.variable,
                                                                                     MissingDict(float, {element: 1.})),
                                                         self.probabilistic_circuit), probability, mount=False)
        self.connect_incoming_edges_to(result)
        self.probabilistic_circuit.remove_node(self)
        return result

    @classmethod
    def from_mixture(cls, mixture: ProbabilisticCircuit):
        """
        Create a discrete distribution from a univariate mixture.

        :param mixture: The mixture to create the distribution from.
        :return: The discrete distribution.
        """
        assert len(mixture.variables) == 1, "Can only convert univariate sum units to discrete distributions."
        variable = mixture.variables[0]
        probabilities = MissingDict(float)

        for element in mixture.support.simple_sets[0][variable].simple_sets:
            probability = mixture.probability_of_simple_event(
                SimpleEvent({variable: element}))
            if isinstance(element, SimpleInterval):
                element = element.lower
            probabilities[int(element)] = probability

        distribution_class = IntegerDistribution if isinstance(variable, Integer) else SymbolicDistribution
        distribution = distribution_class(variable, probabilities)
        return cls(distribution)
