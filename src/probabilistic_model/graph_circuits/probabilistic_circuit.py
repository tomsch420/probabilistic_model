import itertools
from typing import Tuple, Iterable

from random_events.events import EncodedEvent
from random_events.variables import Variable
from typing_extensions import List, Optional, Union, Any

from ..probabilistic_model import ProbabilisticModel, ProbabilisticModelWrapper
import networkx as nx


class ProbabilisticCircuitMixin:
    """
    Mixin class for all components of a probabilistic circuit.
    """

    probabilistic_circuit: Optional['ProbabilisticCircuit'] = None
    """
    The circuit this component is part of. 
    """

    id: Optional[int] = None
    """
    The id of this node in the circuit.
    """

    representation: str = None
    """
    The string representing this component.
    """

    result_of_current_query: Any = None
    """
    Cache of the result of the current query. If the circuit would be queried multiple times,
    this would be returned instead.
    """

    def __repr__(self):
        return self.representation

    def edges_to_sub_circuits(self) -> List[Union['Edge', 'DirectedWeightedEdge']]:
        """
        Return a list of targets to the children of this component.
        """
        return [self.probabilistic_circuit[source][target]["edge"]
                for source, target in self.probabilistic_circuit.out_edges(self)]

    @property
    def variables(self) -> Tuple[Variable]:
        variables = set([variable for distribution in self.leaf_nodes() for variable in distribution.variables])
        return tuple(sorted(variables))

    def leaf_nodes(self) -> List[ProbabilisticModel]:
        return [node for node in nx.descendants(self.probabilistic_circuit, self)
                if self.probabilistic_circuit.out_degree(node) == 0]

    def reset_result_of_current_query(self):
        """
        Reset the result of the current query recursively.
        """
        self.result_of_current_query = None

        for edge in self.edges_to_sub_circuits():
            edge.target.reset_result_of_current_query()


class Component(ProbabilisticCircuitMixin, ProbabilisticModel):
    """
    Class for non-leaf components in circuits.
    """

    def __init__(self):
        super().__init__(None)


class SmoothSumUnit(Component):
    representation = "+"

    def _likelihood(self, event: Iterable) -> float:

        # query cache
        if self.result_of_current_query is not None:
            return self.result_of_current_query

        result = 0.

        for edge in self.edges_to_sub_circuits():
            result += edge.weight * edge.target._likelihood(event)

        # update cache
        self.result_of_current_query = result

        return result

    def _probability(self, event: EncodedEvent) -> float:

        # query cache
        if self.result_of_current_query is not None:
            return self.result_of_current_query

        result = 0.

        for edge in self.edges_to_sub_circuits():
            result += edge.weight * edge.target._probability(event)

        # update cache
        self.result_of_current_query = result

        return result


class DeterministicSumUnit(SmoothSumUnit):
    """
    Deterministic Sum Units for Probabilistic Circuits
    """

    representation = "⊕"

    def merge_modes_if_one_dimensional(self, modes: List[EncodedEvent]) -> List[EncodedEvent]:
        """
        Merge the modes in `modes` to one mode if the model is one dimensional.

        :param modes: The modes to merge.
        :return: The (possibly) merged modes.
        """
        if len(self.variables) > 1:
            return modes

        # merge modes
        mode = modes[0]

        for mode_ in modes[1:]:
            mode = mode | mode_

        return [mode]

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:

        # query cache
        if self.result_of_current_query is not None:
            return self.result_of_current_query

        modes = []
        likelihoods = []

        # gather all modes from the children
        for edge in self.edges_to_sub_circuits():
            mode, likelihood = edge.target._mode()
            modes.append(mode)
            likelihoods.append(edge.weight * likelihood)

        # get the most likely result
        maximum_likelihood = max(likelihoods)

        result = []

        # gather all results that are maximum likely
        for mode, likelihood in zip(modes, likelihoods):
            if likelihood == maximum_likelihood:
                result.extend(mode)

        modes = self.merge_modes_if_one_dimensional(result)

        # update cache
        self.result_of_current_query = (modes, maximum_likelihood)

        return modes, maximum_likelihood


class DecomposableProductUnit(Component):
    """
    Decomposable Product Units for Probabilistic Circuits
    """

    representation = "⊗"

    def _likelihood(self, event: Iterable) -> float:

        # query cache
        if self.result_of_current_query is not None:
            return self.result_of_current_query

        variables = self.variables

        result = 1.

        for edge in self.edges_to_sub_circuits():
            subcircuit = edge.target
            subcircuit_variables = edge.target.variables
            partial_event = [event[variables.index(variable)] for variable in subcircuit_variables]

            result *= subcircuit._likelihood(partial_event)

        # update cache
        self.result_of_current_query = result

        return result

    def _probability(self, event: EncodedEvent) -> float:

        # query cache
        if self.result_of_current_query is not None:
            return self.result_of_current_query

        result = 1.

        for edge in self.edges_to_sub_circuits():

            subcircuit = edge.target
            subcircuit_variables = edge.target.variables

            subcircuit_event = EncodedEvent({variable: event[variable] for variable in subcircuit_variables})

            # construct partial event for child
            result *= subcircuit._probability(subcircuit_event)

        # update cache
        self.result_of_current_query = result

        return result

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:

        # query cache
        if self.result_of_current_query is not None:
            return self.result_of_current_query

        modes = []
        resulting_likelihood = 1.

        # gather all modes from the children
        for edge in self.edges_to_sub_circuits():
            subcircuit = edge.target
            mode, likelihood = subcircuit._mode()
            modes.append(mode)
            resulting_likelihood *= likelihood

        result = []

        # perform the cartesian product of all modes
        for mode_combination in itertools.product(*modes):

            # form the intersection of the modes inside one cartesian product mode
            mode = mode_combination[0]
            for mode_ in mode_combination[1:]:
                mode = mode | mode_

            result.append(mode)

        # update cache
        self.result_of_current_query = (result, resulting_likelihood)

        return result, resulting_likelihood



class Edge:
    """
    Class representing a directed edge in a probabilistic circuit.
    """

    source: ProbabilisticCircuitMixin
    """
    The source of the edge.
    """

    target: ProbabilisticCircuitMixin
    """
    The target of the edge.
    """

    def __init__(self, source: ProbabilisticCircuitMixin, target: ProbabilisticCircuitMixin):
        self.source = source
        self.target = target


class DirectedWeightedEdge(Edge):
    """
    Class representing a directed weighted edge in a probabilistic circuit.
    """

    weight: float
    """
    The weight of the edge.
    """

    source: SmoothSumUnit
    target: SmoothSumUnit

    def __init__(self, source: ProbabilisticCircuitMixin, target: ProbabilisticCircuitMixin, weight: float):
        super().__init__(source, target)
        self.weight = weight


class LeafComponent(ProbabilisticCircuitMixin, ProbabilisticModelWrapper):
    """
    Class for leaf components in circuits.
    """

    def __init__(self, model: ProbabilisticModel):
        super().__init__()
        self.model = model

    @property
    def representation(self):
        return str(self.model.representation)

    @property
    def variables(self):
        return self.model.variables


class ProbabilisticCircuit(ProbabilisticModel, nx.DiGraph):
    """
    Probabilistic Circuits as a directed, rooted, acyclic graph.
    """

    def __init__(self):
        super().__init__(None)
        nx.DiGraph.__init__(self)

    @property
    def variables(self) -> Tuple[Variable]:
        variables = set([variable for distribution in self.leaf_nodes() for variable in distribution.variables])
        return tuple(sorted(variables))

    def leaf_nodes(self) -> List[ProbabilisticModel]:
        return [node for node in self.nodes() if self.out_degree(node) == 0]

    def is_valid(self) -> bool:
        """
        Check if this graph is:

        - acyclic
        - connected

        :return: True if the graph is valid, False otherwise.
        """
        return nx.is_directed_acyclic_graph(self) and nx.is_weakly_connected(self)

    def add_node(self, component: ProbabilisticCircuitMixin, **attr):
        component.probabilistic_circuit = self
        component.id = max(node.id for node in self.nodes) + 1 if len(self.nodes) > 0 else 0
        super().add_node(component, **attr)

    def add_edge(self, edge: Edge, **kwargs):

        # check if edge from a sum unit is weighted.
        if isinstance(edge.source, SmoothSumUnit) and not isinstance(edge, DirectedWeightedEdge):
            raise ValueError(f"Sum units can only have weighted edges. Got {type(edge)} instead.")

        # check if edge from a product unit is unweighted
        if isinstance(edge.source, DecomposableProductUnit) and isinstance(edge, DirectedWeightedEdge):
            raise ValueError(f"Product units can only have un-weighted edges. Got {type(edge)} instead.")

        super().add_edge(edge.source, edge.target, edge=edge, **kwargs)

    def add_edges_from(self, edges: Iterable[Edge], **kwargs):
        for edge in edges:
            self.add_edge(edge, **kwargs)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for node in nodes_for_adding:
            self.add_node(node, **attr)

    @property
    def root(self) -> Union[LeafComponent, Component]:
        """
        The root of the circuit is the node with in-degree 0.
        This is the output node, that will perform the final computation.

        :return: The root of the circuit.
        """
        possible_roots = [node for node in self.nodes() if self.in_degree(node) == 0]
        if len(possible_roots) > 1:
            raise ValueError(f"More than one root found. Possible roots are {possible_roots}")

        return possible_roots[0]

    def _likelihood(self, event: Iterable) -> float:
        root = self.root
        result = self.root._likelihood(event)
        root.reset_result_of_current_query()
        return result

    def _probability(self, event: EncodedEvent) -> float:
        root = self.root
        result = self.root._probability(event)
        root.reset_result_of_current_query()
        return result

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:
        root = self.root
        result = self.root._mode()
        root.reset_result_of_current_query()
        return result