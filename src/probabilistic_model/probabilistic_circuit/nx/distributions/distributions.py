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

    @staticmethod
    def batch_log_conditional_of_simple_event_in_place(leaves, event: SimpleEvent):
        # Gather intervals for each leaf from the event
        intervals = [event[leaf.variable] for leaf in leaves]
        distributions = [leaf.distribution for leaf in leaves]
        # Try vectorizing if all distributions are same type and all intervals are singletons
        is_singleton = all(interval.is_singleton() for interval in intervals)
        if is_singleton:
            # Batch all values
            xs = np.array([interval.lower for interval in intervals]).reshape(-1, 1)
            # Assume all distributions are same class; check
            if len(set(type(d) for d in distributions)) == 1:
                log_liks = distributions[0].log_likelihood(xs)
                for leaf, ll in zip(leaves, log_liks):
                    leaf.result_of_current_query = ll
                return log_liks
        # Fallback: loop for more complex cases
        log_probs = []
        for leaf, interval in zip(leaves, intervals):
            # If the interval is composite, break into simple intervals
            simple_intervals = getattr(interval, "simple_sets", [interval])
            total_prob = 0.0
            for si in simple_intervals:
                _, ll = leaf.distribution.log_conditional_from_simple_interval(si)
                total_prob += np.exp(ll)
            # log-sum-exp trick for numerics
            ll = np.log(total_prob) if total_prob > 0 else -np.inf
            leaf.result_of_current_query = ll
            log_probs.append(ll)
        return np.array(log_probs)

    def log_conditional_of_simple_event_in_place(self, event: SimpleEvent):
        return self.univariate_log_conditional_of_simple_event_in_place(event[self.variable])

    def univariate_log_conditional_of_simple_event_in_place(self, event: Interval):
        """
        Condition this distribution on a simple event in-place but use sum units to create conditions on composite
        intervals.
        :param event: The simple event to condition on.
        """
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
            result.add_subcircuit(current_conditional, np.log(current_probability), mount=False)
            total_probability += current_probability

        # if the event is impossible
        if total_probability == 0:
            self.result_of_current_query = -np.inf
            self.distribution = None
            self.probabilistic_circuit.remove_node(result)
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
                                                         self.probabilistic_circuit), np.log(probability), mount=False)
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
            probabilities[hash(element)] = probability

        distribution_class = IntegerDistribution if isinstance(variable, Integer) else SymbolicDistribution
        distribution = distribution_class(variable, probabilities)
        return cls(distribution)
