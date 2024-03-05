from __future__ import annotations

import itertools
from functools import cached_property

from matplotlib import pyplot as plt
from random_events.events import EncodedEvent, Event, VariableMap
from random_events.variables import Variable, Symbolic, Integer, Discrete
from typing_extensions import Self, List, Tuple, Iterable, Optional, Dict, TYPE_CHECKING

from probabilistic_model.probabilistic_circuit.distributions import (SymbolicDistribution, IntegerDistribution,
                                                                     DiscreteDistribution as PCDiscreteDistribution)
from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.distributions.multinomial import MultinomialDistribution
import networkx as nx
import numpy as np

from ..probabilistic_circuit.probabilistic_circuit import (ProbabilisticCircuit, DeterministicSumUnit,
                                                           DecomposableProductUnit, ProbabilisticCircuitMixin,
                                                           SmoothSumUnit)

if TYPE_CHECKING:
    from .distributions import DiscreteDistribution as RootDistribution, ConditionalProbabilisticCircuit


class BayesianNetworkMixin(ProbabilisticModel):
    """
    Mixin class for conditional probability distributions in tree shaped bayesian networks.
    """

    bayesian_network: BayesianNetwork

    forward_message: Optional[PCDiscreteDistribution]
    """
    The marginal distribution of this nodes variable (message) as calculated in the forward pass.
    """

    forward_probability: float
    """
    The probability of the forward message at each node.
    """

    @property
    def parent(self) -> Optional[Self]:
        """
        The parent node if it exists and None if this is a root.
        :return:
        """
        parents = list(self.bayesian_network.predecessors(self))
        if len(parents) > 1:
            raise ValueError("Bayesian Network is not a tree.")
        elif len(parents) == 1:
            return parents[0]
        else:
            return None

    @property
    def is_root(self) -> bool:
        """
        :return: Rather this is the root or not.
        """
        return self.parent is None

    @property
    def parent_and_node_variables(self) -> Tuple[Variable, ...]:
        """
        Get the parent variables together with this nodes variable.
        :return: A tuple containing first the parent variable and second this nodes variable.
        """
        if self.is_root:
            return self.variables
        else:
            return self.parent.variables + self.variables

    def __hash__(self):
        return id(self)

    def joint_distribution_with_parent(self) -> DeterministicSumUnit:
        """
        Calculate the joint distribution of the node and its parent.
        The joint distribution is formed w. r. t. the forward message of the parent.
        Hence, this can only be called after the forward pass has been performed.

        :return: The joint distribution of the node and its parents.
        """
        raise NotImplementedError

    def forward_pass(self, event: EncodedEvent):
        """
        Calculate the forward pass for this node given the event.
        This includes calculating the forward message and the forward probability of said event.
        :param event: The event to account for
        """
        raise NotImplementedError

    def forward_message_as_sum_unit(self) -> SmoothSumUnit:
        """
        Convert this leaf nodes forward message to a sum unit.
        This is used for the start of the conversion to a probabilistic circuit and only called for leaf nodes.

        :return: The forward message as sum unit.
        """
        raise NotImplementedError

    def interaction_term(self, node_latent_variable: Discrete, parent_latent_variable: Discrete) \
            -> ProbabilisticCircuit:
        """
        Generate the interaction term that is used for mounting into the parent circuit in the generation of a
        probabilistic circuit form the bayesian network.
        :return: The interaction term as probabilistic circuit.
        """
        raise NotImplementedError


class BayesianNetwork(ProbabilisticModel, nx.DiGraph):
    """
    Class for Bayesian Networks that are rooted, tree shaped and have univariate inner nodes.
    """

    def __init__(self):
        ProbabilisticModel.__init__(self, None)
        nx.DiGraph.__init__(self)

    @cached_property
    def nodes(self) -> Iterable[BayesianNetworkMixin]:
        return super().nodes

    @cached_property
    def edges(self) -> Iterable[Tuple[BayesianNetworkMixin, BayesianNetworkMixin]]:
        return super().edges

    @property
    def variables(self) -> Tuple[Variable, ...]:
        variables = [variable for node in self.nodes for variable in node.variables]
        return tuple(sorted(variables))

    @property
    def leaves(self) -> List[BayesianNetworkMixin]:
        return [node for node in self.nodes if self.out_degree(node) == 0]

    def add_node(self, node: BayesianNetworkMixin, **attr):
        node.bayesian_network = self
        super().add_node(node, **attr)

    def add_nodes_from(self, nodes: Iterable[BayesianNetworkMixin], **attr):
        [self.add_node(node) for node in nodes]

    def _likelihood(self, event: Iterable) -> float:
        event = VariableMap(zip(self.variables, event))
        result = 1.
        for node in self.nodes:
            node_event = [event[variable] for variable in node.parent_and_node_variables]
            result *= node._likelihood(node_event)
        return result

    def forward_pass(self, event: EncodedEvent):
        """
        Calculate all forward messages.
        """
        # calculate forward pass
        for node in nx.bfs_tree(self, self.root):
            node.forward_pass(event)

    def _probability(self, event: EncodedEvent) -> float:
        self.forward_pass(event)
        result = 1.

        for node in self.nodes:
            result *= node.forward_probability
        return result

    def brute_force_joint_distribution(self) -> MultinomialDistribution:
        """
        Compute the joint distribution of this bayes network variables by brute force.
        This only works if only discrete variables are present in the network.

        .. Warning::
            This method is only feasible for a small number of variables as it has exponential runtime.

        :return: A Multinomial distribution over all the variables.
        """
        assert all([isinstance(variable, Discrete) for variable in self.variables])

        worlds = list(itertools.product(*[variable.domain for variable in self.variables]))
        worlds = np.array(worlds)
        potentials = np.zeros(tuple(len(variable.domain) for variable in self.variables))

        for idx, world in enumerate(worlds):
            potentials[tuple(world)] = self._likelihood(world)

        return MultinomialDistribution(self.variables, potentials)

    @property
    def root(self) -> RootDistribution:
        """
        The root of the circuit is the node with in-degree 0.
        This is the output node, that will perform the final computation.

        :return: The root of the circuit.
        """
        possible_roots = [node for node in self.nodes if self.in_degree(node) == 0]
        if len(possible_roots) > 1:
            raise ValueError(f"More than one root found. Possible roots are {possible_roots}")

        return possible_roots[0]

    def as_probabilistic_circuit(self) -> ProbabilisticCircuit:
        """
        Convert the BayesianNetwork to a probabilistic circuit that expresses the same probability distribution.
        :return:
        """

        # this only works for bayesian trees
        assert nx.is_tree(self)

        # calculate forward pass
        self.forward_pass(self.preprocess_event(Event()))

        pointers_to_sum_units: Dict[BayesianNetworkMixin, SmoothSumUnit] = dict()

        for leaf in self.leaves:
            pointers_to_sum_units[leaf] = leaf.forward_message_as_sum_unit()

        # iterate over the edges in reversed bfs order
        edges = nx.bfs_edges(self, self.root)

        # for each edge in reverse bfs order
        for parent, child in reversed(list(edges)):

            # type hinting
            parent: BayesianNetworkMixin
            child: BayesianNetworkMixin

            # if the parent circuit does not yet exist
            if parent not in pointers_to_sum_units.keys():

                # create the parent circuit
                pointers_to_sum_units[parent] = parent.forward_message.as_deterministic_sum()

            # get parent and child circuits
            parent_sum_unit = pointers_to_sum_units[parent]
            child_sum_unit = pointers_to_sum_units[child]

            # calculate interaction term
            interaction_term = child.interaction_term(child_sum_unit.latent_variable,
                                                      parent_sum_unit.latent_variable)

            # mount child into parent
            parent_sum_unit.mount_with_interaction_terms(pointers_to_sum_units[child], interaction_term)

        return pointers_to_sum_units[self.root].probabilistic_circuit
