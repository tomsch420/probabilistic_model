from __future__ import annotations

from abc import abstractmethod
from typing import Union, Iterable

import numpy as np
import rustworkx as rx
from random_events.product_algebra import SimpleEvent, Event
from random_events.utils import SubclassJSONSerializer
from random_events.variable import Variable, Symbolic, Continuous
from sortedcontainers import SortedSet
from typing_extensions import List, Optional, Any, Self, Dict, Tuple

from ...distributions import UnivariateDistribution, DiscreteDistribution, \
    ContinuousDistribution
from ...probabilistic_model import ProbabilisticModel


class Unit(SubclassJSONSerializer):
    """
    Class for all units of a probabilistic circuit.

    This class should not be used by users directly.

    Use the class :class:`ProbabilisticCircuit` as interface to users.
    """

    index: Optional[int] = None
    """
    The index this node has in its circuit.
    """

    _probabilistic_circuit: ProbabilisticCircuit = None
    """
    The circuit this component is part of. 
    """

    result_of_current_query: Any = None
    """
    The result of the current query. 
    """

    def __init__(self, probabilistic_circuit: Optional[ProbabilisticCircuit] = None):
        if probabilistic_circuit is None:
            probabilistic_circuit = ProbabilisticCircuit()

        probabilistic_circuit.add_node(self)

    @property
    @abstractmethod
    def subcircuits(self) -> List[Unit]:
        """
        :return: The subcircuits of this unit.
        """
        raise NotImplementedError

    @property
    def parents(self) -> List[InnerUnit]:
        """
        :return: The parents of this unit.
        """
        return list(self._probabilistic_circuit.predecessors(self))

    @abstractmethod
    def support(self):
        """
        Calculate the support of this unit.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_leaf(self):
        """
        :return: If this unit is a leaf unit.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def variables(self) -> SortedSet:
        raise NotImplementedError

    @property
    def leaves(self) -> List[LeafUnit]:
        """
        :return: The leaves of the circuit that are descendants of this node.
        """
        raise NotImplementedError

    @property
    def probabilistic_circuit(self) -> ProbabilisticCircuit:
        return self._probabilistic_circuit

    def mount(self, other: Unit):
        """
        Mount another unit including its descendants. There will be no edge from `self` to `other`.

        :param other: The other unit to mount.
        """
        descendants = rx.descendants(other.probabilistic_circuit.graph, other.index)
        descendants = list(descendants.union([other.index]))
        subgraph = other.probabilistic_circuit.graph.subgraph(descendants, preserve_attrs=True)

        self.probabilistic_circuit.add_edges_from(subgraph.edges())

    def __hash__(self):
        return hash((self.index, id(self.probabilistic_circuit)))

class LeafUnit(Unit):
    """
    Class for Leaf units.
    """

    distribution: Optional[ProbabilisticModel]
    """
    The distribution contained in this leaf unit.
    """

    def __init__(self, distribution: ProbabilisticModel, probabilistic_circuit: Optional[ProbabilisticCircuit] = None):
        super().__init__(probabilistic_circuit)
        self.distribution = distribution


class InnerUnit(Unit):
    """
    Class for inner units
    """

    @abstractmethod
    def add_subcircuit(self, subcircuit: Unit, *args, mount: bool = True, **kwargs, ):
        """
        Add a subcircuit to the subcircuits of this unit.

        .. note::

            This method does not normalize the edges to the subcircuits.


        :param subcircuit: The subcircuit to add.

        :param mount: If the subcircuit should be mounted to the pc of this unit.
        """
        raise NotImplementedError


class SumUnit(InnerUnit):
    """
    Sum Units for Probabilistic Circuits
    """

    _latent_variable: Optional[Symbolic] = None
    """
    The latent variable of this unit.
    This has to be here due to the rvalue/lvalue problem in random events.
    """

    def add_subcircuit(self, subcircuit: Unit, log_weight: float = 0., *args, mount: bool = True, **kwargs, ):
        if mount:
            self.mount(subcircuit)
        self.probabilistic_circuit.add_edge(self, subcircuit, log_weight=log_weight)


class ProductUnit(InnerUnit):
    """
    Decomposable Product Units for Probabilistic Circuits
    """

    def add_subcircuit(self, subcircuit: Unit, *args, mount: bool = True, **kwargs, ):
        if mount:
            self.mount(subcircuit)
        self.probabilistic_circuit.add_edge(self, subcircuit)


class ProbabilisticCircuit(ProbabilisticModel, SubclassJSONSerializer):
    """
    Probabilistic Circuits as a directed, rooted, acyclic graph.
    """

    graph: rx.PyDAG[Unit]
    """
    The graph structure of the circuit.
    """

    def __init__(self):
        super().__init__()
        self.graph = rx.PyDAG()

    def add_node(self, unit: Unit):
        if unit.probabilistic_circuit != self:
            node_index = self.graph.add_node(unit)
            unit.index = node_index
            unit._probabilistic_circuit = self

    def add_edge(self, parent: Unit, child: Unit,  log_weight: Optional[float] = None):
        self.add_node(parent)
        self.add_node(child)
        self.graph.add_edge(parent.index, child.index, log_weight)

    def add_nodes_from(self, nodes: Iterable[Unit]):
        for unit in nodes:
            self.add_node(unit)

    def add_edges_from(self, edges: Iterable[Tuple]):
        for elem in edges:
            self.add_edge(*elem)

    def add_unweighted_edges_from(self, edges: Iterable[Tuple[Unit, Unit]]):
        for parent, child in edges:
            self.add_edge(parent, child)

    def add_weighted_edges_from(self, edges: Iterable[Tuple[Unit, Unit, float]]):
        for parent, child, log_weight in edges:
            self.add_edge(parent, child, log_weight)


    @property
    def nodes(self) -> List[Unit]:
        return self.graph.nodes()

    @property
    def root(self) -> Unit:
        """
        The root of the circuit is the node with in-degree 0.
        This is the output node, that will perform the final computation.

        :return: The root of the circuit.
        """
        possible_roots = [node for node in self.nodes if self.graph.in_degree(node.index) == 0]
        if len(possible_roots) > 1:
            raise ValueError(f"More than one root found. Possible roots are {possible_roots}")
        return possible_roots[0]

    @property
    def support(self) -> Event:
        pass

    def log_likelihood(self, events: np.array) -> np.array:
        pass

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        pass

    def log_mode(self) -> Tuple[Event, float]:
        pass

    def log_conditional(self, event: Event) -> Tuple[Optional[Union[ProbabilisticModel, Self]], float]:
        pass

    def log_conditional_of_point(self, point: Dict[Variable, Any]) -> Tuple[Optional[Self], float]:
        pass

    def sample(self, amount: int) -> np.array:
        pass

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        pass

    @property
    def variables(self) -> Tuple[Variable, ...]:
        pass


class UnivariateLeaf(LeafUnit):

    @property
    def variable(self) -> Variable:
        return self.distribution.variables[0]


class UnivariateContinuousLeaf(UnivariateLeaf):
    distribution: Optional[ContinuousDistribution]


class UnivariateDiscreteLeaf(UnivariateLeaf):
    distribution: Optional[DiscreteDistribution]


def leaf(distribution: UnivariateDistribution,
         probabilistic_circuit: Optional[ProbabilisticCircuit] = None) -> UnivariateLeaf:
    """
    Factory that creates the correct leaf from a distribution.

    :return: The leaf.
    """
    if isinstance(distribution.variable, Continuous):
        return UnivariateContinuousLeaf(distribution, probabilistic_circuit)
    else:
        return UnivariateDiscreteLeaf(distribution, probabilistic_circuit)
