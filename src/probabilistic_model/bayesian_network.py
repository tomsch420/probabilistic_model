from __future__ import annotations

import itertools
from functools import cached_property
from typing import Tuple

from random_events.events import EncodedEvent, Event
from random_events.variables import Variable
from typing_extensions import Self, List, Tuple, Iterable, Optional, Dict

from .probabilistic_model import ProbabilisticModel
from .distributions.multinomial import MultinomialDistribution
import networkx as nx
import numpy as np

from .probabilistic_circuit.probabilistic_circuit import (ProbabilisticCircuit, DeterministicSumUnit,
                                                          DecomposableProductUnit)


class BayesianNetworkMixin:

    bayesian_network: BayesianNetwork

    forward_message: MultinomialDistribution
    """
    The marginal distribution (message) as calculated in the forward pass.
    """

    forward_probability: float
    """
    The probability of the forward message at each node.
    """

    @property
    def parents(self) -> List[Self]:
        return list(self.bayesian_network.predecessors(self))

    @property
    def is_root(self):
        return len(self.parents) == 0

    @property
    def variables(self) -> Tuple[Variable, ...]:
        raise NotImplementedError

    @property
    def parent_variables(self) -> Tuple[Variable, ...]:
        parent_variables = [variable for parent in self.parents for variable in parent.variables]
        return tuple(sorted(parent_variables))

    @property
    def parent_and_node_variables(self):
        return self.parent_variables + self.variables

    def __hash__(self):
        return id(self)

    def _likelihood(self, event: Iterable, parent_event: Iterable) -> float:
        raise NotImplementedError

    def as_probabilistic_circuit(self) -> DeterministicSumUnit:
        raise NotImplementedError


class ConditionalMultinomialDistribution(BayesianNetworkMixin, MultinomialDistribution):

    variables: Tuple[Variable, ...]

    _probabilities: np.ndarray
    """
    Private array of probabilities.
    """

    def __init__(self, variables: Iterable[Variable]):
        ProbabilisticModel.__init__(self, variables)
        BayesianNetworkMixin.__init__(self)

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return self._variables

    @property
    def probabilities(self):
        return self._probabilities

    @probabilities.setter
    def probabilities(self, probabilities: np.ndarray):
        """
        Set the probabilities of this distribution. The probabilities have to have the shape that is obtained by the
        concatenation of the parent variables shape and the own variables shape.
        """
        own_variables_shape = tuple(len(variable.domain) for variable in self.variables)
        parent_variables_shape = tuple(len(variable.domain) for variable in self.parent_variables)

        if parent_variables_shape + own_variables_shape != probabilities.shape:
            raise ValueError(
                f"""The probabilities have to have the shape that is obtained by the concatenation of the parent 
                variables shape and the own variables shape. 
                Parent Variables {self.parent_variables} \n 
                Own Variables {self.variables} \n
                Probability Shape {probabilities.shape}""")
        self._probabilities = probabilities

    def normalize(self):
        normalized_probabilities = self.probabilities / np.sum(self.probabilities, axis=-1).reshape(-1, 1)
        self.probabilities = normalized_probabilities

    def _likelihood(self, event: Iterable, parent_event: Optional[Iterable] = None) -> float:
        if parent_event is None:
            parent_event = tuple()
        return self.probabilities[tuple(parent_event) + tuple(event)].item()

    def calculate_forward_message(self, event: EncodedEvent):
        """
        Calculate the forward message for this node given the event and the forward probability of said event.
        :param event: The event to account for
        """

        forward_message = self.joint_distribution_with_parents()

        # calculate conditional probability
        forward_message, forward_probability = forward_message._conditional(event)

        # marginalize with respect to the node variables
        forward_message = forward_message.marginal(self.variables).normalize()

        # save forward message and probability
        self.forward_message = forward_message
        self.forward_probability = forward_probability

    def __hash__(self):
        return BayesianNetworkMixin.__hash__(self)

    def __repr__(self):
        node_variables_representation = ', '.join([repr(v) for v in self.variables])
        if len(self.parent_variables) == 0:
            return f"P({node_variables_representation})"
        else:
            return f"P( {node_variables_representation} | {', '.join([repr(v) for v in self.parent_variables])})"

    def as_probabilistic_circuit(self) -> DeterministicSumUnit:
        return MultinomialDistribution.as_probabilistic_circuit(self)

    def joint_distribution_with_parents(self) -> MultinomialDistribution:
        """
        Calculate the joint distribution of the node and its parents.
        :return: The joint distribution of the node and its parents.
        """

        if self.is_root:
            return MultinomialDistribution(self.variables, self.probabilities)

        # get the parent
        parent = self.parents[0]

        # multiply the parent forward message with the own probabilities
        probabilities = self.probabilities * parent.forward_message.probabilities.reshape(-1, 1)

        # create the new forward message
        result = MultinomialDistribution(self.parent_and_node_variables, None)
        result._variables = self.parent_and_node_variables
        result.probabilities = probabilities

        return result


class BayesianNetwork(ProbabilisticModel, nx.DiGraph):

    def __init__(self):
        ProbabilisticModel.__init__(self, None)
        nx.DiGraph.__init__(self)

    @cached_property
    def nodes(self) -> Iterable[ConditionalMultinomialDistribution]:
        return super().nodes

    @property
    def variables(self) -> Tuple[Variable, ...]:
        variables = [variable for node in self.nodes for variable in node.variables]
        return tuple(sorted(variables))

    @property
    def leaves(self) -> List[ConditionalMultinomialDistribution]:
        return [node for node in self.nodes if self.out_degree(node) == 0]

    def add_node(self, node: BayesianNetworkMixin, **attr):
        node.bayesian_network = self
        super().add_node(node, **attr)

    def add_nodes_from(self, nodes: Iterable[BayesianNetworkMixin], **attr):
        [self.add_node(node) for node in nodes]

    def _likelihood(self, event: Iterable) -> float:
        event = EncodedEvent(zip(self.variables, event))
        result = 1.
        for node in self.nodes:
            parent_event = [event[variable][0] for variable in node.parent_variables]
            node_event = [event[variable][0] for variable in node.variables]
            result *= node._likelihood(node_event, parent_event)
        return result

    def forward_pass(self, event: EncodedEvent):
        """
        Calculate all forward messages.
        """
        # calculate forward pass
        for node in self.nodes:
            node.calculate_forward_message(event)

    def _probability(self, event: EncodedEvent) -> float:
        self.forward_pass(event)
        result = 1.

        for node in self.nodes:
            result *= node.forward_probability
        return result

    def brute_force_joint_distribution(self) -> MultinomialDistribution:
        """
        Compute the joint distribution of the factor graphs variables by brute force.

        .. Warning::
            This method is only feasible for a small number of variables as it has exponential runtime.

        :return: A Multinomial distribution over all the variables.
        """
        worlds = list(itertools.product(*[variable.domain for variable in self.variables]))
        worlds = np.array(worlds)
        potentials = np.zeros(tuple(len(variable.domain) for variable in self.variables))

        for idx, world in enumerate(worlds):
            potentials[tuple(world)] = self._likelihood(world.tolist())

        return MultinomialDistribution(self.variables, potentials)

    @property
    def root(self) -> BayesianNetworkMixin:
        """
        The root of the circuit is the node with in-degree 0.
        This is the output node, that will perform the final computation.

        :return: The root of the circuit.
        """
        possible_roots = [node for node in self.nodes if self.in_degree(node) == 0]
        if len(possible_roots) > 1:
            raise ValueError(f"More than one root found. Possible roots are {possible_roots}")

        return possible_roots[0]

    def as_probabilistic_circuit(self) -> DeterministicSumUnit:
        """
        Convert the BayesianNetwork to a probabilistic circuit that expresses the same probability distribution.
        :return:
        """

        # this only works for bayesian trees
        assert nx.is_tree(self)

        # calculate forward pass
        self.forward_pass(self.preprocess_event(Event()))

        # initialize dict that maps from the basic network node to the component in the circuit
        pointers_to_sum_units: Dict[BayesianNetworkMixin, DeterministicSumUnit] = dict()

        # warm start the algorithm
        for leaf in self.leaves:

            # by creating the circuit for every leafs marginal distribution
            distribution = leaf.joint_distribution_with_parents().marginal(leaf.variables)
            pointers_to_sum_units[leaf] = distribution.as_probabilistic_circuit().simplify()

        # iterate over the edges in reversed bfs order
        edges = nx.bfs_edges(self, self.root)
        edges = reversed(list(edges))

        # for every edge
        for parent, child in edges:

            # if the parent is not in the dict
            if parent not in pointers_to_sum_units:
                # create the parent
                pointers_to_sum_units[parent] = (parent.joint_distribution_with_parents().marginal(parent.variables).
                                                 as_probabilistic_circuit().simplify())

            # calculate the interaction term between parent and child
            joint_distribution = child.joint_distribution_with_parents()
            joint_distribution._variables = (pointers_to_sum_units[parent].latent_variable,
                                             pointers_to_sum_units[child].latent_variable)

            # mount child into the parent using interaction term
            pointers_to_sum_units[parent].mount_with_interaction_terms(pointers_to_sum_units[child],
                                                                       joint_distribution)

        return pointers_to_sum_units[self.root]
