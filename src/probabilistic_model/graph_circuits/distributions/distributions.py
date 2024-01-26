from typing import Iterable

from random_events.events import EncodedEvent, Event
from random_events.variables import Variable
from typing_extensions import Union, Tuple, Optional, Self

import portion

from ...distributions.distributions import (ContinuousDistribution as PMContinuousDistribution,
                                            DiracDeltaDistribution as PMDiracDeltaDistribution)
from ..probabilistic_circuit import (DeterministicSumUnit, ProbabilisticCircuitMixin,
                                     DirectedWeightedEdge, cache_inference_result)
from ...distributions.uniform import UniformDistribution as PMUniformDistribution
from ...distributions.gaussian import (GaussianDistribution as PMGaussianDistribution,
                                       TruncatedGaussianDistribution as PMTruncatedGaussianDistribution)


class ContinuousDistribution(PMContinuousDistribution, ProbabilisticCircuitMixin):

    @property
    def variables(self) -> Tuple[Variable]:
        return self._variables

    def conditional_from_complex_interval(self, interval: portion.Interval) -> \
            Tuple[Optional[DeterministicSumUnit], float]:

        resulting_distributions = []
        resulting_probabilities = []

        for simple_interval in interval:

            conditional, probability = self.conditional_from_simple_interval(simple_interval)

            resulting_probabilities.append(probability)
            resulting_distributions.append(conditional)

        total_probability = sum(resulting_probabilities)

        normalized_probabilities = [probability / total_probability for probability in resulting_probabilities]

        conditional = DeterministicSumUnit()
        self.probabilistic_circuit.add_node(conditional)

        for distribution, probability in zip(resulting_distributions, normalized_probabilities):
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
        conditional, probability = super()._conditional(event)

        if conditional is None:
            self.probabilistic_circuit.remove_node(self)
            return None, 0

        self.probabilistic_circuit.add_node(conditional)

        new_edges = [edge.__copy__() for edge in self.incoming_edges()]
        for edge in new_edges:
            edge.target = conditional

        self.probabilistic_circuit.remove_node(self)
        self.probabilistic_circuit.add_edges_from(new_edges)
        return conditional, probability

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        return ProbabilisticCircuitMixin.marginal(self, variables)


class DiracDeltaDistribution(ContinuousDistribution, PMDiracDeltaDistribution):
    ...


class UniformDistribution(ContinuousDistribution, PMUniformDistribution):
    ...


class GaussianDistribution(ContinuousDistribution, PMGaussianDistribution):
    ...


class TruncatedGaussianDistribution(ContinuousDistribution, PMTruncatedGaussianDistribution):
    ...

