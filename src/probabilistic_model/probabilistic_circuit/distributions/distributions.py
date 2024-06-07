from __future__ import annotations

from random_events.product_algebra import Event, SimpleEvent
from typing_extensions import Union, Tuple, Optional, Self, Iterable

from ...distributions.distributions import (ContinuousDistribution as PMContinuousDistribution,
                                            DiracDeltaDistribution as PMDiracDeltaDistribution,
                                            SymbolicDistribution as PMSymbolicDistribution,
                                            IntegerDistribution as PMIntegerDistribution,
                                            DiscreteDistribution as PMDiscreteDistribution,
                                            UnivariateDistribution as PMUnivariateDistribution)
from ..probabilistic_circuit import (DeterministicSumUnit, ProbabilisticCircuitMixin, cache_inference_result)
from ...distributions.uniform import UniformDistribution as PMUniformDistribution
from ...distributions.gaussian import (GaussianDistribution as PMGaussianDistribution,
                                       TruncatedGaussianDistribution as PMTruncatedGaussianDistribution)


class UnivariateDistribution(PMUnivariateDistribution, ProbabilisticCircuitMixin):

    def is_deterministic(self) -> bool:
        return True

    def __hash__(self):
        return ProbabilisticCircuitMixin.__hash__(self)

    @cache_inference_result
    def log_conditional_of_simple_event(self, event: SimpleEvent) -> Tuple[Optional[Self], float]:
        return super().log_conditional(event.as_composite_set())

    @cache_inference_result
    def simplify(self) -> Self:
        return self.__copy__()

    def empty_copy(self) -> Self:
        return self.__copy__()


class ContinuousDistribution(UnivariateDistribution, PMContinuousDistribution, ProbabilisticCircuitMixin):

    def conditional_from_complex_interval(self, interval: portion.Interval) -> \
            Tuple[Optional[DeterministicSumUnit], float]:

        # list for resulting distributions with their probabilities
        resulting_distributions = []
        resulting_probabilities = []

        # for every simple interval in the complex interval
        for simple_interval in interval:

            # if the interval is a singleton
            if simple_interval.lower == simple_interval.upper:
                conditional, probability = self.conditional_from_singleton(simple_interval)
            else:
                # get the conditional and the probability
                conditional, probability = self.conditional_from_simple_interval(simple_interval)

            # update lists
            resulting_probabilities.append(probability)
            resulting_distributions.append(conditional)

        # calculate the total probability
        total_probability = sum(resulting_probabilities)

        # normalize the probabilities
        normalized_probabilities = [probability / total_probability for probability in resulting_probabilities]

        # create and add the deterministic mixture as result
        conditional = DeterministicSumUnit()

        # for every distribution and its normalized probability
        for probability, distribution in zip(normalized_probabilities, resulting_distributions):
            conditional.mount(distribution)
            conditional.probabilistic_circuit.add_edge(conditional, distribution, weight=probability)

        return conditional, total_probability

    def conditional_from_singleton(self, singleton: portion.Interval) -> \
            Tuple['DiracDeltaDistribution', float]:
        conditional, probability = super().conditional_from_singleton(singleton)
        return DiracDeltaDistribution(conditional.variable, conditional.location, conditional.density_cap), probability

    @cache_inference_result
    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        return PMContinuousDistribution.marginal(self, variables)


class DiscreteDistribution(UnivariateDistribution, PMDiscreteDistribution, ProbabilisticCircuitMixin):

    def as_deterministic_sum(self) -> DeterministicSumUnit:
        """
        Convert this distribution to a deterministic sum unit that encodes the same distribution.
        The result has as many children as the domain of the variable and each child encodes the value of the variable.

        :return: A deterministic sum unit that encodes the same distribution.
        """
        result = DeterministicSumUnit()

        for event in self.variable.domain:
            event = Event({self.variable: event})
            conditional, probability = self.conditional(event)
            result.add_subcircuit(conditional, probability)

        return result


class DiracDeltaDistribution(ContinuousDistribution, PMDiracDeltaDistribution):
    ...


class UniformDistribution(ContinuousDistribution, PMUniformDistribution):
    ...


class GaussianDistribution(ContinuousDistribution, PMGaussianDistribution):

    def conditional_from_simple_interval(self, interval: portion.Interval) -> (
            Tuple)[Optional[TruncatedGaussianDistribution], float]:

        conditional, probability = super().conditional_from_simple_interval(interval)

        if probability == 0:
            return None, 0

        return TruncatedGaussianDistribution(conditional.variable, conditional.interval,
                                             conditional.mean, conditional.scale), probability


class TruncatedGaussianDistribution(GaussianDistribution, ContinuousDistribution, PMTruncatedGaussianDistribution):
    ...


class IntegerDistribution(DiscreteDistribution, PMIntegerDistribution):
    ...


class SymbolicDistribution(DiscreteDistribution, PMSymbolicDistribution):
    ...
