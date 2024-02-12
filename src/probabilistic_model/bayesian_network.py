from __future__ import annotations

from typing import Tuple

from random_events.variables import Variable
from typing_extensions import Self, List, Tuple, Iterable, Optional

from .probabilistic_model import ProbabilisticModel
from .distributions.multinomial import MultinomialDistribution
import networkx as nx
import numpy as np


class NodeMixin:

    bayesian_network: BayesianNetwork

    @property
    def parents(self) -> List[Self]:
        return list(self.bayesian_network.predecessors(self))

    @property
    def variables(self) -> Tuple[Variable, ...]:
        raise NotImplementedError

    @property
    def parent_variables(self) -> Tuple[Variable, ...]:
        parent_variables = [variable for parent in self.parents for variable in parent.variables]
        return tuple(sorted(parent_variables))

    def __hash__(self):
        return id(self)


class ConditionalMultinomialDistribution(MultinomialDistribution, NodeMixin):

    variables: Tuple[Variable, ...]

    _probabilities: np.ndarray
    """
    Private array of probabilities.
    """

    def __init__(self, variables: Iterable[Variable]):
        ProbabilisticModel.__init__(self, variables)
        NodeMixin.__init__(self)

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

    def __hash__(self):
        return NodeMixin.__hash__(self)


class BayesianNetwork(ProbabilisticModel, nx.DiGraph):

    def __init__(self):
        ProbabilisticModel.__init__(self, None)
        nx.DiGraph.__init__(self)

    @property
    def variables(self) -> Tuple[Variable, ...]:
        variables = [variable for node in self.nodes for variable in node.variables]
        return tuple(sorted(variables))

    def add_node(self, node: NodeMixin, **attr):
        node.bayesian_network = self
        super().add_node(node, **attr)

    def add_nodes_from(self, nodes: Iterable[NodeMixin], **attr):
        [self.add_node(node) for node in nodes]
