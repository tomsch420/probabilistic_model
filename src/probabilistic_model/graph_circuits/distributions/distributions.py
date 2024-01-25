from typing import Tuple, Optional

from random_events.events import EncodedEvent
from typing_extensions import Self

import portion

from ...distributions.distributions import (ContinuousDistribution as PMContinuousDistribution,
                                            DiracDeltaDistribution as PMDiracDeltaDistribution)
from ..probabilistic_circuit import DeterministicSumUnit, ProbabilisticCircuitMixin, LeafComponent, DirectedWeightedEdge


class ContinuousDistribution(PMContinuousDistribution, ProbabilisticCircuitMixin):

    def conditional_from_complex_interval(self, interval: portion.Interval) -> Tuple[DeterministicSumUnit, float]:

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
            node = LeafComponent(distribution)
            edge = DirectedWeightedEdge(conditional, node, probability)
            conditional.probabilistic_circuit.add_edge(edge)

        return conditional, total_probability

    def conditional_from_singleton(self, singleton: portion.Interval) -> \
            Tuple[Optional['DiracDeltaDistribution'], float]:
        conditional, probability = super().conditional_from_singleton(singleton)
        return DiracDeltaDistribution(conditional.variable, conditional.location, conditional.density_cap), probability

    def _conditional(self, event: EncodedEvent) -> Tuple[Optional['ContinuousDistribution'], float]:
        conditional, probability = super()._conditional(event)

        if conditional is None:
            self.probabilistic_circuit.remove_node(self)
            return None, 0

        self.probabilistic_circuit.remove_node(self)
        self.probabilistic_circuit.add_node(conditional)

        return conditional, probability


class DiracDeltaDistribution(ContinuousDistribution, PMDiracDeltaDistribution):
    ...
