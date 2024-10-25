import numpy as np
from random_events.product_algebra import SimpleEvent, VariableMap
from random_events.variable import Variable, Symbolic
from typing_extensions import Tuple, Dict, Iterable, List, Union, Self

from .bayesian_network import BayesianNetworkMixin
from ..distributions.multinomial import MultinomialDistribution
from ..distributions import SymbolicDistribution
from ..probabilistic_circuit.nx.distributions import UnivariateDiscreteLeaf
from ..probabilistic_circuit.nx.probabilistic_circuit import (ProbabilisticCircuit, Unit, SumUnit,
                                                                                ProductUnit)
from ..utils import MissingDict


class RootDistribution(BayesianNetworkMixin, SymbolicDistribution):
    """
    Distribution for Bayesian Network root nodes.
    """

    @property
    def variables(self) -> Tuple[Symbolic, ...]:
        return (self.variable,)

    def forward_pass(self, event: SimpleEvent):
        self.forward_message, self.forward_probability = self.log_conditional_of_composite_set(event[self.variable])
        self.forward_probability = np.exp(self.forward_probability)

    def joint_distribution_with_parent(self) -> SumUnit:
        result = SumUnit()

        for event in self.variable.domain.simple_sets:
            event = SimpleEvent({self.variable: event})
            conditional, probability = self.forward_message.log_conditional_of_composite_set(event[self.variable])
            result.add_subcircuit(UnivariateDiscreteLeaf(conditional), np.exp(probability))

        return result

    def __repr__(self):
        return f"P({self.variable.name})"


class ConditionalProbabilityTable(BayesianNetworkMixin):
    """
    Conditional probability distribution for Bayesian Network nodes given their parents.
    The parent in this case must be exactly one node.
    """

    variable: Symbolic
    conditional_probability_distributions: Dict[int, SymbolicDistribution]

    def __init__(self, variable: Symbolic):
        self.variable = variable
        self.conditional_probability_distributions = dict()

    @property
    def variables(self) -> Tuple[Symbolic, ...]:
        return (self.variable,)

    def forward_pass(self, event: SimpleEvent):

        # if the parent distribution is None, the forward message is None since it is an impossible event
        if self.parent.forward_message is None:
            self.forward_message = None
            self.forward_probability = 0
            return

        # initialize the weights
        probabilities = MissingDict(float)

        # initialize the forward probability
        forward_probability = 0

        # for every parent state
        for parent_state in event[self.parent.variable].simple_sets:

            # calculate the probability P(self.parent.variable = parent_state)
            parent_state_probability = self.parent.forward_message.probabilities[int(parent_state)]
            if parent_state_probability == 0:
                continue

            # construct the conditional distribution P(self.variable | self.parent.variable = parent_state)
            conditional, current_log_probability = (self.conditional_probability_distributions[int(parent_state)].
                                                    log_conditional_of_composite_set(event[self.variable]))

            # if the conditional is None, skip
            if conditional is None:
                continue

            # update probability and forward probability (perform sum-product)
            for state, probability in conditional.probabilities.items():
                probabilities[state] += parent_state_probability * probability
            forward_probability += parent_state_probability * np.exp(current_log_probability)

        # if weights sum to zero, the forward message is None
        if sum(probabilities.values()) == 0:
            self.forward_message = None
            self.forward_probability = 0
        else:
            # create the new forward message
            self.forward_message = SymbolicDistribution(self.variable, probabilities)
            self.forward_message.normalize()

        self.forward_probability = forward_probability

    def __repr__(self):
        return f"P({self.variable.name}|{self.parent.variable.name})"

    def to_tabulate(self) -> List[List[str]]:
        table = [[self.parent.variable.name, self.variable.name, repr(self)]]
        for parent_event, distribution in self.conditional_probability_distributions.items():
            for index, probability in distribution.probabilities.items():
                table.append([self.parent.variable.domain_type()(parent_event),
                              self.variable.domain_type()(index), str(probability)])
        return table

    def joint_distribution_with_parent(self) -> SumUnit:

        # initialize result
        result = SumUnit()

        # a map from the state of this nodes variable to the distribution
        distribution_nodes: Dict[int, UnivariateDiscreteLeaf] = dict()
        template_probabilities = MissingDict(float, {int(element): 1 / len(self.variable.domain.simple_sets) for element
                                                     in self.variable.domain.simple_sets})
        distribution_template = SymbolicDistribution(self.variable, template_probabilities)
        for value in self.variable.domain.simple_sets:
            distribution_node, _ = distribution_template.log_conditional_of_composite_set(value.as_composite_set())
            distribution_nodes[int(value)] = UnivariateDiscreteLeaf(distribution_node)

        # for every parent event and conditional distribution
        for parent_event, distribution in self.conditional_probability_distributions.items():

            # wrap the parent event
            parent_event = SimpleEvent({self.parent.variable: self.parent.variable.domain_type()(parent_event)})

            # encode the parent state as distribution
            parent_distribution, parent_log_probability = (
                self.parent.forward_message.log_conditional_of_composite_set(parent_event[self.parent.variable]))

            for child_event_index, child_probability in distribution.probabilities.items():
                # initialize the product unit
                product_unit = ProductUnit()

                # add the encoded parent distribution and a copy of this distribution to the product unit
                product_unit.add_subcircuit(UnivariateDiscreteLeaf(parent_distribution))
                product_unit.add_subcircuit(distribution_nodes[child_event_index])

                result.add_subcircuit(product_unit, np.exp(parent_log_probability) * child_probability)

        return result

    def forward_message_as_sum_unit(self) -> SumUnit:
        result = UnivariateDiscreteLeaf(self.forward_message)
        return result.as_deterministic_sum()

    def interaction_term(self, node_latent_variable: Symbolic, parent_latent_variable: Symbolic) -> \
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

        # set the parent variable to the variable that is not this one
        parent_variable = distribution.variables[0] \
            if distribution.variables[0] != self.variable else distribution.variables[1]

        for parent_simple_set in parent_variable.domain.simple_sets:
            parent_event = SimpleEvent({parent_variable: parent_simple_set})
            conditional, _ = distribution.conditional(parent_event.as_composite_set())
            marginal = conditional.marginal(self.variables)

            conditional_distribution = SymbolicDistribution(self.variable,
                                                              MissingDict(float, zip(range(
                                                                  len(self.variable.domain.simple_sets)),
                                                                  marginal.probabilities.tolist())))

            self.conditional_probability_distributions[int(parent_simple_set)] = conditional_distribution

        return self


class ConditionalProbabilisticCircuit(BayesianNetworkMixin):
    """
    Conditional probability distribution represented as Circuit for Bayesian Network nodes given their parents.
    """

    conditional_probability_distributions: Dict[int, ProbabilisticCircuit]
    forward_message: SumUnit
    _variables: Tuple[Variable, ...]

    def __init__(self, variables: Iterable[Variable]):
        super().__init__()
        self._variables = tuple(variables)
        self.conditional_probability_distributions = dict()

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return self._variables

    @property
    def parent(self) -> Union[RootDistribution, ConditionalProbabilityTable]:
        return super().parent

    def forward_pass(self, event: SimpleEvent):
        joint_distribution_with_parent = self.joint_distribution_with_parent()
        conditional, log_prob = joint_distribution_with_parent.probabilistic_circuit.log_conditional_of_simple_event_in_place(event)
        self.forward_probability = np.exp(log_prob)
        marginal = conditional.marginal(self.variables)
        self.forward_message = marginal.root

    def joint_distribution_with_parent(self) -> SumUnit:

        result = SumUnit()

        for parent_event, distribution in self.conditional_probability_distributions.items():
            parent_event = SimpleEvent({self.parent.variable: self.parent.variable.domain_type()(parent_event)})
            parent_distribution, parent_log_probability = self.parent.forward_message.log_conditional_of_composite_set(
                parent_event[self.parent.variable])

            if parent_distribution is None:
                continue

            product_unit = ProductUnit()
            product_unit.add_subcircuit(UnivariateDiscreteLeaf(parent_distribution))
            product_unit.add_subcircuit(distribution.root)

            result.add_subcircuit(product_unit, np.exp(parent_log_probability))

        return result

    def forward_message_as_sum_unit(self) -> SumUnit:
        return self.forward_message

    def interaction_term(self, node_latent_variable: Symbolic, parent_latent_variable: Symbolic) -> \
            ProbabilisticCircuit:

        assert node_latent_variable.domain == parent_latent_variable.domain

        result = SumUnit()

        for state, weight in self.parent.forward_message.probabilities.items():
            probabilities = MissingDict(float)
            probabilities[int(state)] = 1
            product_unit = ProductUnit()
            product_unit.add_subcircuit(UnivariateDiscreteLeaf(SymbolicDistribution(parent_latent_variable, probabilities)))
            product_unit.add_subcircuit(UnivariateDiscreteLeaf(SymbolicDistribution(node_latent_variable, probabilities)))
            result.add_subcircuit(product_unit, weight)

        return result.probabilistic_circuit

    def from_unit(self, unit: Unit) -> Self:
        """
        Get the conditional probability table from a probabilistic circuit by mounting all children as conditional
        probability distributions.
        :param unit: The probabilistic circuit to get the data from
        :return: The conditional probability distribution
        """
        for index, subcircuit in enumerate(unit.subcircuits):
            self.conditional_probability_distributions[index] = subcircuit.__copy__().probabilistic_circuit
        return self

    def __repr__(self):
        return f"P({', '.join([v.name for v in self.variables])} | {self.parent.variable.name})"