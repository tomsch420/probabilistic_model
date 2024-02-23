import numpy as np
from matplotlib import pyplot as plt
from random_events.events import Event, EncodedEvent, VariableMap
from typing_extensions import Tuple, Dict, Iterable, List, Type, Union, Optional, Self

from .bayesian_network import BayesianNetworkMixin
from ..probabilistic_model import ProbabilisticModel
from random_events.variables import Discrete, Variable

from ..probabilistic_circuit.probabilistic_circuit import (ProbabilisticCircuit, DeterministicSumUnit,
                                                           ProbabilisticCircuitMixin, SmoothSumUnit)
from ..probabilistic_circuit.probabilistic_circuit import DecomposableProductUnit
from ..probabilistic_circuit.distributions import (SymbolicDistribution as PCSymbolicDistribution,
                                                   IntegerDistribution as PCIntegerDistribution,
                                                   DiscreteDistribution as PCDiscreteDistribution)
from ..distributions.multinomial import MultinomialDistribution


class DiscreteDistribution(BayesianNetworkMixin, PCDiscreteDistribution):
    """
    Intermediate Distribution for Bayesian Network root nodes.
    """

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


class SymbolicDistribution(DiscreteDistribution, PCSymbolicDistribution):
    ...


class IntegerDistribution(DiscreteDistribution, PCIntegerDistribution):
    ...


class ConditionalProbabilityTable(BayesianNetworkMixin):

    variables: Tuple[Discrete, ...]
    conditional_probability_distributions: Dict[Tuple, PCDiscreteDistribution]

    def __init__(self, variable: Discrete):
        ProbabilisticModel.__init__(self, [variable])
        self.conditional_probability_distributions = dict()

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
        variables = ", ".join([variable.name for variable in self.variables])
        return f"P({variables}|{self.parent.variable.name})"

    def to_tabulate(self) -> List[List[str]]:
        table = [[self.parent.variable.name, self.variable.name, repr(self)]]
        for parent_event, distribution in self.conditional_probability_distributions.items():
            for event, probability in zip(self.variable.domain, distribution.weights):
                table.append([str(parent_event[0]), str(event), str(probability)])
        return table

    def joint_distribution_with_parent(self) -> DeterministicSumUnit:

        # initialize result
        result = DeterministicSumUnit()

        # a map from the state of this nodes variable to the distribution
        distribution_nodes = dict()
        distribution_template = self.conditional_probability_distributions[(self.variable.domain[0],)].__copy__()
        distribution_template.weights = [1/len(self.variable.domain) for _ in self.variable.domain]
        for value in self.variable.domain:
            event = Event({self.variable: value})
            distribution_nodes[value], _ = distribution_template.conditional(event)

        # for every parent event and conditional distribution
        for parent_event, distribution in self.conditional_probability_distributions.items():

            # wrap the parent event
            parent_event = Event({self.parent.variable: parent_event})

            # encode the parent state as distribution
            parent_distribution, parent_probability = self.parent.forward_message.conditional(parent_event)

            for child_event, child_probability in zip(self.variable.domain, distribution.weights):

                # initialize the product unit
                product_unit = DecomposableProductUnit()

                # add the encoded parent distribution and a copy of this distribution to the product unit
                product_unit.add_subcircuit(parent_distribution)
                product_unit.add_subcircuit(distribution_nodes[child_event])

                result.add_subcircuit(product_unit, parent_probability * child_probability)

        return result

    def forward_message_as_sum_unit(self) -> DeterministicSumUnit:
        return self.forward_message.as_deterministic_sum()

    def interaction_term(self, node_latent_variable: Discrete, parent_latent_variable: Discrete) -> \
            ProbabilisticCircuit:
        interaction_term = self.joint_distribution_with_parent().probabilistic_circuit
        interaction_term.update_variables(VariableMap({self.variable: node_latent_variable,
                                                       self.parent.variable: parent_latent_variable}))
        return interaction_term

    def from_multinomial_distribution(self, distribution: MultinomialDistribution) -> Self:
        """
        Get the conditional probability table from a multinomial distribution.

        :param distribution: The multinomial distribution to get the data from
        :return:
        """
        assert len(distribution.variables) == 2
        assert self.variable in distribution.variables

        parent_variable = distribution.variables[0] \
            if distribution.variables[0] != self.variable else distribution.variables[1]

        for parent_event in parent_variable.domain:
            parent_event = Event({parent_variable: parent_event})
            conditional, _ = distribution.conditional(parent_event)
            marginal = conditional.marginal(self.variables).normalize()
            self.conditional_probability_distributions[parent_event[parent_variable]] = (
                DiscreteDistribution(self.variable, marginal.probabilities.tolist()))

        return self


class ConditionalProbabilisticCircuit(ConditionalProbabilityTable):

    conditional_probability_distributions: Dict[Tuple, ProbabilisticCircuit]
    forward_message: SmoothSumUnit

    def __init__(self, variables: Iterable[Variable]):
        ProbabilisticModel.__init__(self, variables)
        self.conditional_probability_distributions = dict()

    def forward_pass(self, event: EncodedEvent):
        forward_message, self.forward_probability = self.joint_distribution_with_parent()._conditional(event)
        self.forward_message = forward_message.marginal(self.variables)

    def joint_distribution_with_parent(self) -> DeterministicSumUnit:

        result = DeterministicSumUnit()

        for parent_event, distribution in self.conditional_probability_distributions.items():
            parent_event = Event({self.parent.variable: parent_event})
            parent_distribution, parent_probability = self.parent.forward_message.conditional(parent_event)

            if parent_probability == 0:
                continue

            product_unit = DecomposableProductUnit()
            product_unit.add_subcircuit(parent_distribution)
            product_unit.add_subcircuit(distribution.root)

            result.add_subcircuit(product_unit, parent_probability)

        return result

    def forward_message_as_sum_unit(self) -> SmoothSumUnit:
        return self.forward_message

    def interaction_term(self, node_latent_variable: Discrete, parent_latent_variable: Discrete) -> \
            ProbabilisticCircuit:

        assert node_latent_variable.domain == parent_latent_variable.domain

        result = DeterministicSumUnit()

        for index, weight in zip(node_latent_variable.domain, self.parent.forward_message.weights):
            probabilities = [0] * len(node_latent_variable.domain)
            probabilities[index] = 1
            product_unit = DecomposableProductUnit()
            product_unit.add_subcircuit(DiscreteDistribution(parent_latent_variable, probabilities))
            product_unit.add_subcircuit(DiscreteDistribution(node_latent_variable, probabilities))
            result.add_subcircuit(product_unit, weight)

        return result.probabilistic_circuit

    def from_unit(self, unit: ProbabilisticCircuitMixin) -> Self:
        """
        Get the conditional probability table from a probabilistic circuit by mounting all children as conditional
        probability distributions.
        :param unit: The probabilistic circuit to get the data from
        :return: The conditional probability distribution
        """
        for index, subcircuit in enumerate(unit.subcircuits):
            self.conditional_probability_distributions[(index, )] = subcircuit.__copy__().probabilistic_circuit
        return self
