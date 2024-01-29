from typing import Iterable

from random_events.events import EncodedEvent, Event
from random_events.variables import Variable
from typing_extensions import Union, Tuple, Optional, Self

import portion

from ...distributions.distributions import (ContinuousDistribution as PMContinuousDistribution,
                                            DiracDeltaDistribution as PMDiracDeltaDistribution,
                                            SymbolicDistribution as PMSymbolicDistribution,
                                            IntegerDistribution as PMIntegerDistribution,
                                            DiscreteDistribution as PMDiscreteDistribution,
                                            UnivariateDistribution as PMUnivariateDistribution)
from ..probabilistic_circuit import (DeterministicSumUnit, ProbabilisticCircuitMixin,
                                     DirectedWeightedEdge, cache_inference_result)
from ...distributions.uniform import UniformDistribution as PMUniformDistribution
from ...distributions.gaussian import (GaussianDistribution as PMGaussianDistribution,
                                       TruncatedGaussianDistribution as PMTruncatedGaussianDistribution)


class UnivariateDistribution(PMUnivariateDistribution, ProbabilisticCircuitMixin):

    @property
    def variables(self) -> Tuple[Variable]:
        return self._variables


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
        self.probabilistic_circuit.add_node(conditional)

        # for every distribution and its normalized probability
        for distribution, probability in zip(resulting_distributions, normalized_probabilities):

            # create an edge from the mixture to the distribution
            edge = DirectedWeightedEdge(conditional, distribution, probability)
            conditional.probabilistic_circuit.add_edge(edge)

        return conditional, total_probability

    def conditional_from_singleton(self, singleton: portion.Interval) -> \
            Tuple[Optional['DiracDeltaDistribution'], float]:
        conditional, probability = super().conditional_from_singleton(singleton)
        return DiracDeltaDistribution(conditional.variable, conditional.location, conditional.density_cap), probability

    @cache_inference_result
    def _conditional(self, event: EncodedEvent) -> \
            Tuple[Optional[Union['ContinuousDistribution', 'DiracDeltaDistribution', DeterministicSumUnit]], float]:

        # get the conditional from the superclass
        conditional, probability = super()._conditional(event)

        # if the conditional is None
        if conditional is None:

            # remove self from the circuit
            self.probabilistic_circuit.remove_node(self)
            return None, 0

        # add the conditional node
        self.probabilistic_circuit.add_node(conditional)

        # get the edges
        new_edges = [edge.__copy__() for edge in self.incoming_edges()]
        for edge in new_edges:
            edge.target = conditional

        self.probabilistic_circuit.remove_node(self)
        self.probabilistic_circuit.add_edges_from(new_edges)
        return conditional, probability

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        return ProbabilisticCircuitMixin.marginal(self, variables)


class DiscreteDistribution(UnivariateDistribution, PMDiscreteDistribution, ProbabilisticCircuitMixin):
    ...


class DiracDeltaDistribution(ContinuousDistribution, PMDiracDeltaDistribution):
    ...


class UniformDistribution(ContinuousDistribution, PMUniformDistribution):
    ...


class GaussianDistribution(ContinuousDistribution, PMGaussianDistribution):
    ...


class TruncatedGaussianDistribution(ContinuousDistribution, PMTruncatedGaussianDistribution):
    ...


class IntegerDistribution(DiscreteDistribution, PMIntegerDistribution):
    ...


class SymbolicDistribution(DiscreteDistribution, PMSymbolicDistribution):
    ...
