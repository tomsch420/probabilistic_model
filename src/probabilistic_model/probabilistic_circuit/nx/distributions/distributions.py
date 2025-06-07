from __future__ import annotations

import networkx as nx
import numpy as np
from random_events.interval import SimpleInterval, Interval
from random_events.product_algebra import SimpleEvent
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Variable, Integer
from scipy.stats import norm
from typing_extensions import Optional

from ....distributions import GaussianDistribution
from ....utils import MissingDict
from ..probabilistic_circuit import SumUnit, LeafUnit, ProbabilisticCircuit
from ....distributions.distributions import (ContinuousDistribution, DiscreteDistribution, IntegerDistribution,
                                             SymbolicDistribution)


class UnivariateLeaf(LeafUnit):

    @property
    def variable(self) -> Variable:
        return self.distribution.variables[0]

def batch_log_likelihood_singletons(distributions, intervals):
    """
    distributions: list of GaussianDistribution, all same type, one per leaf
    intervals: list of SimpleInterval, all .is_singleton() is True
    """
    # All Gaussians must have the same .location and .scale for perfect vectorization,
    # but if not, we handle it elementwise in numpy
    locations = np.array([d.location for d in distributions])
    scales = np.array([d.scale for d in distributions])
    xs = np.array([interval.lower for interval in intervals])

    # If all location/scale are the same, vectorize
    if np.all(locations == locations[0]) and np.all(scales == scales[0]):
        logpdfs = norm.logpdf(xs, loc=locations[0], scale=scales[0])
    else:
        logpdfs = norm.logpdf(xs, loc=locations, scale=scales)
    return logpdfs


def batch_log_probability_intervals(distributions, intervals):
    """
    distributions: list of GaussianDistribution, all same type, one per leaf
    intervals: list of SimpleInterval, .is_singleton() may be False
    """
    locations = np.array([d.location for d in distributions])
    scales = np.array([d.scale for d in distributions])
    lowers = np.array([interval.lower for interval in intervals])
    uppers = np.array([interval.upper for interval in intervals])

    # CDF at upper and lower
    if np.all(locations == locations[0]) and np.all(scales == scales[0]):
        cdf_upper = norm.cdf(uppers, loc=locations[0], scale=scales[0])
        cdf_lower = norm.cdf(lowers, loc=locations[0], scale=scales[0])
    else:
        cdf_upper = norm.cdf(uppers, loc=locations, scale=scales)
        cdf_lower = norm.cdf(lowers, loc=locations, scale=scales)

    probs = cdf_upper - cdf_lower
    # Avoid log(0) for near-zero probs
    probs = np.clip(probs, 1e-300, 1.0)
    logps = np.log(probs)
    return logps


class UnivariateContinuousLeaf(UnivariateLeaf):
    distribution: Optional[ContinuousDistribution]

    @staticmethod
    def batch_log_conditional_of_simple_event_in_place(leaves, event: SimpleEvent):
        intervals = [event[leaf.variable] for leaf in leaves]
        distributions = [leaf.distribution for leaf in leaves]

        # If all are Gaussians...
        if all(type(d) is GaussianDistribution for d in distributions):
            if all(interval.is_singleton() for interval in intervals):
                logps = batch_log_likelihood_singletons(distributions, intervals)
            elif all(hasattr(interval, "lower") and hasattr(interval, "upper") for interval in intervals):
                logps = batch_log_probability_intervals(distributions, intervals)
            else:
                # fallback for composites
                logps = []
                for leaf, interval in zip(leaves, intervals):
                    simple_intervals = getattr(interval, "simple_sets", [interval])
                    total_prob = 0.0
                    for si in simple_intervals:
                        _, ll = leaf.distribution.log_conditional_from_simple_interval(si)
                        total_prob += np.exp(ll)
                    ll = np.log(total_prob) if total_prob > 0 else -np.inf
                    logps.append(ll)
                logps = np.array(logps)
            for leaf, lp in zip(leaves, logps):
                leaf.result_of_current_query = lp
            return logps
        else:
            # fallback: mixed or non-Gaussian
            logps = []
            for leaf, interval in zip(leaves, intervals):
                simple_intervals = getattr(interval, "simple_sets", [interval])
                total_prob = 0.0
                for si in simple_intervals:
                    _, ll = leaf.distribution.log_conditional_from_simple_interval(si)
                    total_prob += np.exp(ll)
                ll = np.log(total_prob) if total_prob > 0 else -np.inf
                leaf.result_of_current_query = ll
                logps.append(ll)
            return np.array(logps)

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
