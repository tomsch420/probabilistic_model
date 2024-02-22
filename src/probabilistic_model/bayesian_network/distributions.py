import numpy as np
from random_events.events import Event, EncodedEvent
from typing_extensions import Tuple, Dict, Iterable, List, Type, Union, Optional

from .bayesian_network import BayesianNetworkMixin
from ..probabilistic_model import ProbabilisticModel
from random_events.variables import Discrete
from ..distributions.distributions import DiscreteDistribution

from ..probabilistic_circuit.probabilistic_circuit import (ProbabilisticCircuit, DeterministicSumUnit,
                                                           ProbabilisticCircuitMixin)
from ..probabilistic_circuit.probabilistic_circuit import DecomposableProductUnit
from ..probabilistic_circuit.distributions import (SymbolicDistribution as PCSymbolicDistribution,
                                                   IntegerDistribution as PCIntegerDistribution,
                                                   DiscreteDistribution as PCDiscreteDistribution)
from ..distributions.distributions import SymbolicDistribution, IntegerDistribution
from ..utils import type_converter


class RootDistribution(BayesianNetworkMixin, PCDiscreteDistribution):

    forward_message: Optional[PCDiscreteDistribution]

    def forward_pass(self, event: EncodedEvent):
        self.forward_message, self.forward_probability = self._conditional(event)

    def joint_distribution_with_parent(self) -> DeterministicSumUnit:
        result = DeterministicSumUnit()

        for event in self.variable.domain:

            event = Event({self.variable: event})
            conditional, probability = self.forward_message.conditional(event)
            result.add_subcircuit(conditional, probability)

        return result


    def __repr__(self):
        return f"P({self.variable.name})"


class ConditionalProbabilityTable(BayesianNetworkMixin):

    variables: Tuple[Discrete, ...]
    conditional_probability_distributions: Dict[Tuple, DiscreteDistribution] = dict()

    def __init__(self, variable: Discrete):
        ProbabilisticModel.__init__(self, [variable])

    @property
    def variable(self) -> Discrete:
        return self.variables[0]

    def likelihood(self, event: Iterable) -> float:
        return self._likelihood([variable.encode(value) for variable, value
                                 in zip(self.parent_and_node_variables, event)])

    def _likelihood(self, event: Iterable) -> float:
        parent_event = tuple(event[:1])
        node_event = tuple(event[1:])
        return self.conditional_probability_distributions[parent_event]._likelihood(node_event)

    def forward_pass(self, event: EncodedEvent):

        # if the parent distribution is None, the forward message is None since it is an impossible event
        if self.parent.forward_message is None:
            self.forward_message = None
            self.forward_probability = 0
            return

        # initialize the weights
        weights = np.zeros(len(self.variable.domain))

        # initialize the forward probability
        forward_probability = 0

        # for every parent state
        for parent_state in event[self.parent.variable]:

            # wrap the parent state
            parent_state = (parent_state,)

            # calculate the probability of said state
            parent_state_probability = self.parent.forward_message.likelihood(parent_state)

            # construct the conditional distribution
            conditional, current_probability = (self.conditional_probability_distributions[parent_state]
                                                ._conditional(event))

            # if the conditional is None, skip
            if conditional is None:
                continue

            # update weights and forward probability
            weights += conditional.weights
            forward_probability += parent_state_probability * current_probability

        # if weights sum to zero, the forward message is None
        if weights.sum() == 0:
            self.forward_message = None
            self.forward_probability = 0
        else:
            # create the new forward message
            self.forward_message = DiscreteDistribution(self.variable, (weights/weights.sum()).tolist())

        self.forward_probability = forward_probability

    def __repr__(self):
        return f"P({self.variable.name}|{self.parent.variable.name})"

    def to_tabulate(self) -> List[List[str]]:
        table = [[self.parent.variable.name, self.variable.name, repr(self)]]
        for parent_event, distribution in self.conditional_probability_distributions.items():
            for event, probability in zip(self.variable.domain, distribution.weights):
                table.append([str(parent_event[0]), str(event), str(probability)])
        return table

    def joint_distribution_with_parent(self) -> ProbabilisticModel:
        result = DeterministicSumUnit()

        for parent_event, distribution in self.conditional_probability_distributions.items():

            event = Event({self.parent.variable: parent_event})
            product_unit = DecomposableProductUnit()

            parent_distribution, _ = self.parent.forward_message.conditional(event)
            product_unit.add_subcircuit(parent_distribution)
            product_unit.add_subcircuit(distribution.__copy__())

            weight = self.parent.forward_message.likelihood(parent_event)

            result.add_subcircuit(product_unit, weight)

        return result
