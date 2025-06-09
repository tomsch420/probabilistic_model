from __future__ import annotations

import copy
import itertools
import random
from abc import abstractmethod
from enum import IntEnum
from typing import Iterable

import numpy as np
import rustworkx as rx
from matplotlib import pyplot as plt
from random_events.interval import Interval, SimpleInterval
from random_events.product_algebra import SimpleEvent, Event
from random_events.set import Set
from random_events.utils import SubclassJSONSerializer
from random_events.variable import Variable, Symbolic, Continuous, Integer
from scipy.special import logsumexp
from sortedcontainers import SortedSet
from typing_extensions import List, Optional, Any, Self, Dict, Tuple

from ...distributions import UnivariateDistribution, DiscreteDistribution, ContinuousDistribution, IntegerDistribution, \
    SymbolicDistribution
from ...error import IntractableError
from ...probabilistic_model import ProbabilisticModel, OrderType, CenterType, MomentType
from ...utils import MissingDict


class Unit:
    """
    Class for all units of a probabilistic circuit.

    This class should not be used by users directly.

    Use the class :class:`ProbabilisticCircuit` as interface to users.
    """

    index: Optional[int] = None
    """
    The index this node has in its circuit.
    """

    _probabilistic_circuit: Optional[ProbabilisticCircuit] = None
    """
    The circuit this component is part of. 
    """

    result_of_current_query: Optional[Any] = None
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
        return list(self.probabilistic_circuit.graph.predecessors(self.index))

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

    def simplify(self):
        """
        Simplify the circuit by removing nodes and redirected edges that have no impact in-place.
        Essentially, this method transforms the circuit into an alternating order of sum and product units.

        :return: The simplified circuit.
        """
        raise NotImplementedError()

    @property
    def leaves(self) -> List[LeafUnit]:
        """
        :return: The leaves of the circuit that are descendants of this node.
        """
        raise NotImplementedError

    @abstractmethod
    def moment(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """
        Draw samples from the circuit.

        For sampling, a node gets requested a number of samples from all his parents.
        The parents write into the `result_of_current_query` attribute a tuple describing the beginning index of the
        sampling and how many samples are requested.
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
        edges = [(subgraph.get_node_data(p), subgraph.get_node_data(c), lw) for (p, c), lw in zip(subgraph.edge_list(), subgraph.edges())]
        self.probabilistic_circuit.add_edges_from(edges)

    def connect_incoming_edges_to(self, other: Unit):
        """
        Connect all incoming edges of this unit to another unit.

        :param other: The other unit to connect the incoming edges to.
        """
        incoming_edges = list(self.probabilistic_circuit.graph.in_edges(self.index))
        for parent_index, _, data in incoming_edges:
            parent = self.probabilistic_circuit.graph[parent_index]
            self.probabilistic_circuit.add_edge(parent, other, data)

    def __hash__(self):
        return hash((self.index, id(self.probabilistic_circuit)))

    def empty_copy(self) -> Self:
        result = self.__class__()
        return result


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

    @property
    def variables(self) -> Iterable[Variable]:
        return SortedSet(self.distribution.variables)

    @property
    def subcircuits(self) -> List[Unit]:
        return []

    @property
    def is_leaf(self):
        return True

    def log_likelihood(self, events: np.array):
        self.result_of_current_query = self.distribution.log_likelihood(events)

    def cdf(self, events: np.array):
        self.result_of_current_query = self.distribution.cdf(events)

    def probability_of_simple_event(self, event: SimpleEvent):
        self.result_of_current_query = self.distribution.probability_of_simple_event(event)

    def support(self):
        self.result_of_current_query = self.distribution.support  # .__deepcopy__()

    def simplify(self):
        if self.distribution is None:
            self.probabilistic_circuit.remove_node(self)

    def log_conditional_of_simple_event_in_place(self, event: SimpleEvent):
        self.distribution, self.result_of_current_query = self.distribution.log_conditional(event.as_composite_set())

    def empty_copy(self):
        result = self.__class__(self.distribution)
        result.index = self.index
        return result

    def moment(self, order, center, variable_to_index_map):
        result = np.zeros(len(variable_to_index_map))
        moment = self.distribution.moment(order, center)
        for variable in self.variables:
            result[variable_to_index_map[variable]] = moment[variable]
        self.result_of_current_query = result

    def sample(self, samples: np.array, variable_to_index_map: Dict[Variable, int]):
        """
        Sample from the distribution and write the samples into the samples array.
        :param samples: The array to write the samples into.
        :param variable_to_index_map: The map from variables to column indices in the samples array.
        """
        column_indices = [variable_to_index_map[variable] for variable in self.variables]
        for start_index, amount in self.result_of_current_query:
            samples[start_index:start_index + amount, column_indices] = self.distribution.sample(amount)

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        marginal = self.distribution.marginal(variables)
        if marginal is None:
            self.probabilistic_circuit.remove_node(self)
            return None
        else:
            self.distribution = marginal
            return self

    def log_mode(self):
        self.result_of_current_query = self.distribution.log_mode()

    def log_conditional_of_point_in_place(self, point: Dict[Variable, Any]):
        if any(variable for variable in self.variables if variable in point):
            self.distribution, self.result_of_current_query = self.distribution.log_conditional_of_point(point)
        else:
            self.result_of_current_query = 0.

    def __repr__(self):
        return f"leaf({repr(self.distribution)})"


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

    @property
    def subcircuits(self) -> List[Unit]:
        return self.probabilistic_circuit.graph.successors(self.index)

    @property
    def is_leaf(self):
        return False

    @property
    def leaves(self) -> List[LeafUnit]:
        return [node for node in nx.descendants(self.probabilistic_circuit, self) if node.is_leaf]

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def log_forward(self, *args, **kwargs):
        raise NotImplementedError

    def marginal(self, *args, **kwargs) -> Optional[Self]:
        if len(self.subcircuits) == 0:
            self.probabilistic_circuit.remove_node(self)
            return None
        return self

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls()


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

    @property
    def representation(self) -> str:
        return "+"

    @property
    def log_weighted_subcircuits(self) -> List[Tuple[float, Unit]]:
        """
        :return: The weighted subcircuits of this unit.
        """
        return [(lw, self.probabilistic_circuit.graph.get_node_data(u))
                for _, u, lw in self.probabilistic_circuit.graph.out_edges(self.index)]

    @property
    def variables(self) -> SortedSet:
        return self.subcircuits[0].variables

    @property
    def latent_variable(self) -> Symbolic:
        name = f"{hash(self)}.latent"
        subcircuit_enum = IntEnum(name,
                                  {str(hash(subcircuit)): index for index, subcircuit in enumerate(self.subcircuits)})
        result = Symbolic(name, Set.from_iterable(subcircuit_enum))
        self._latent_variable = result
        return result

    def forward(self, *args, **kwargs):
        self.result_of_current_query = np.sum(
            [np.exp(weight) * subcircuit.result_of_current_query for weight, subcircuit in
             self.log_weighted_subcircuits], axis=0)

    def log_forward(self, *args, **kwargs):
        result = [lw + s.result_of_current_query for lw, s in self.log_weighted_subcircuits]
        self.result_of_current_query = logsumexp(result, axis=0)

    moment = forward

    def support(self):
        support = self.subcircuits[0].result_of_current_query.__deepcopy__()
        for subcircuit in self.subcircuits[1:]:
            support |= subcircuit.result_of_current_query.__deepcopy__()
        self.result_of_current_query = support

    @property
    def log_weights(self) -> np.array:
        """
        :return: The log_weights of the subcircuits.
        """
        return np.array([weight for weight, _ in self.log_weighted_subcircuits])

    def sample(self, *args, **kwargs) -> np.array:
        weights, subcircuits = self.log_weights, self.subcircuits

        subcircuit_indices = list(range(len(subcircuits)))
        # for every sampling request
        for start_index, amount in self.result_of_current_query:

            # calculate the numbers of samples requested from the sub circuits
            counts = np.random.multinomial(amount, pvals=np.exp(weights))
            total = 0

            # shuffle the order to sample from the subcircuits to avoid bias
            random.shuffle(subcircuit_indices)

            # add the sampling requests to the subcircuits
            for index in subcircuit_indices:
                subcircuit = subcircuits[index]
                count = counts[index]
                subcircuit.result_of_current_query.append((start_index + total, count))
                total += count

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.log_weighted_subcircuits == other.log_weighted_subcircuits

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        result = cls()
        return result

    def __copy__(self):
        return self.empty_copy()

    def mount_with_interaction_terms(self, other: Self, interaction_model: ProbabilisticModel):
        """
        Create a distribution that factorizes as follows:

        .. math::
            p(self.latent\_variable) \cdot p(self.variables | self.latent\_variable) \cdot
            p(other.latent\_variable | self.latent\_variable) \cdot p(other.variables | other.latent\_variable)

        where `self.latent_variable` and `other.latent_variable` are the results of the latent variable interpretation
        of mixture models.

        :param other: The other distribution to mount at this distribution children level.
        :param interaction_model: The interaction probabilities between both latent variables
        """
        assert set(self.variables).intersection(set(other.variables)) == set()
        assert set(interaction_model.variables) == {self.latent_variable, other.latent_variable}

        # load latent variables
        own_latent_variable = self.latent_variable
        other_latent_variable = other.latent_variable

        # load subircuits
        own_subcircuits = self.subcircuits
        other_subcircuits = other.subcircuits

        for own_index, own_subcircuit in enumerate(own_subcircuits):

            # create denominator of weight
            own_index = own_latent_variable.domain.simple_sets[0].element.__class__(own_index)
            condition = SimpleEvent({own_latent_variable: own_index}).as_composite_set()
            p_condition = interaction_model.probability(condition)

            # skip iterations that are impossible
            if p_condition == 0:
                continue

            # create proxy nodes for mounting
            proxy_product_node = ProductUnit()
            proxy_sum_node = other.empty_copy()
            self.probabilistic_circuit.add_nodes_from([proxy_product_node, proxy_sum_node])

            # remove edge to old child and replace it by product proxy
            self.probabilistic_circuit.remove_edge(self, own_subcircuit)
            self.add_subcircuit(proxy_product_node, np.log(p_condition))

            # mount current child on the product proxy
            proxy_product_node.add_subcircuit(own_subcircuit)

            # mount the proxy for the children from other in the product proxy
            proxy_product_node.add_subcircuit(proxy_sum_node)

            for other_index, other_subcircuit in enumerate(other_subcircuits):

                # create numerator of weight
                other_index = other_latent_variable.domain.simple_sets[0].element.__class__(other_index)
                query = SimpleEvent({other_latent_variable: other_index}).as_composite_set() & condition
                p_query = interaction_model.probability(query)

                # skip iterations that are impossible
                if p_query == 0:
                    continue

                # calculate conditional probability
                weight = p_query / p_condition

                # create edge from proxy to subcircuit
                proxy_sum_node.add_subcircuit(other_subcircuit, log_weight=np.log(weight))
            proxy_sum_node.normalize()

    def mount_from_bayesian_network(self, other: Self):
        """
        Mount a distribution from tge `to_probabilistic_circuit` method in bayesian networks.
        The distribution is mounted as follows:


        :param other: The other distribution to mount at this distribution children level.
        :return:
        """
        assert set(self.variables).intersection(set(other.variables)) == set()
        assert len(self.subcircuits) == len(other.subcircuits)
        # mount the other subcircuit

        for (own_weight, own_subcircuit), other_subcircuit in zip(self.log_weighted_subcircuits, other.subcircuits):
            # create proxy nodes for mounting
            proxy_product_node = ProductUnit()
            self.probabilistic_circuit.add_node(proxy_product_node)

            # remove edge to old child and replace it by product proxy
            self.probabilistic_circuit.remove_edge(self, own_subcircuit)
            self.add_subcircuit(proxy_product_node, np.log(own_weight))
            proxy_product_node.add_subcircuit(own_subcircuit)
            proxy_product_node.add_subcircuit(other_subcircuit)

    def simplify(self):

        # if this has only one child
        if len(self.subcircuits) == 1:

            # redirect every incoming edge to the child
            incoming_edges = list(self.probabilistic_circuit.graph.in_edges(self.index))
            for parent, _, data in incoming_edges:
                self.probabilistic_circuit.graph.add_edge(parent, self.subcircuits[0].index, data)

            # remove this node
            self.probabilistic_circuit.remove_node(self)

            return

        # for every subcircuit
        for log_weight, subcircuit in self.log_weighted_subcircuits:

            # if the weight is 0, skip this subcircuit
            if log_weight == -np.inf:
                # remove the edge
                self.probabilistic_circuit.remove_edge(self, subcircuit)

            # if the simplified subcircuit is of the same type as this
            if type(subcircuit) is type(self):

                # type hinting
                subcircuit: Self

                # mount the children of that circuit directly
                for sub_weight, sub_subcircuit in subcircuit.log_weighted_subcircuits:
                    new_weight = sub_weight + log_weight

                    # add an edge to that subcircuit
                    self.add_subcircuit(sub_subcircuit, new_weight, mount=False)

                # remove the old node
                self.probabilistic_circuit.remove_node(subcircuit)

    def normalize(self):
        """
        Normalize the log_weights of the subcircuits such that they sum up to 1 inplace.
        """
        total_weight = logsumexp(self.log_weights)

        for log_weight, subcircuit in self.log_weighted_subcircuits:
            normalized_log_weight = log_weight - total_weight
            self.probabilistic_circuit.graph.add_edge(self.index, subcircuit.index, normalized_log_weight)

    def is_deterministic(self) -> bool:
        """
        :return: If this unit is deterministic or not.
        """
        # for every unique combination of subcircuits
        for subcircuit_a, subcircuit_b in itertools.combinations(self.subcircuits, 2):
            # check if they intersect
            if not subcircuit_a.result_of_current_query.intersection_with(
                    subcircuit_b.result_of_current_query).is_empty():
                return False

        # if none intersect, the subcircuit is deterministic
        return True

    def log_mode(self):
        log_maxima = [log_weight + subcircuit.result_of_current_query[1] for log_weight, subcircuit in
                      self.log_weighted_subcircuits]
        log_max = max(log_maxima)
        arg_log_maxima = [subcircuit.result_of_current_query[0] for lm, subcircuit in zip(log_maxima, self.subcircuits)
                          if lm == log_max]
        arg_log_max = arg_log_maxima[0]
        for event in arg_log_maxima[1:]:
            arg_log_max |= event
        self.result_of_current_query = (arg_log_max, log_max)

    def subcircuit_index_of_samples(self, samples: np.array) -> np.array:
        """
        :return: the index of the subcircuit where p(sample) > 0 and None if p(sample) = 0 for all subcircuits.
        """
        result = np.full(len(samples), np.nan)
        for index, subcircuit in enumerate(self.subcircuits):
            likelihood = subcircuit.log_likelihood(samples)
            result[likelihood > -np.inf] = index
        return result

    def __repr__(self):
        return "+"

class ProductUnit(InnerUnit):
    """
    Decomposable Product Units for Probabilistic Circuits
    """

    representation = "×"

    def add_subcircuit(self, subcircuit: Unit, *args, mount: bool = True, **kwargs, ):
        if mount:
            self.mount(subcircuit)
        self.probabilistic_circuit.add_edge(self, subcircuit)

    def forward(self, *args, **kwargs):
        self.result_of_current_query = np.prod([subcircuit.result_of_current_query for subcircuit in self.subcircuits])

    def log_forward(self, *args, **kwargs):
        self.result_of_current_query = np.sum([subcircuit.result_of_current_query for subcircuit in self.subcircuits],
                                              axis=0)

    moment = log_forward

    @property
    def variables(self) -> SortedSet:
        result = SortedSet()
        for subcircuit in self.subcircuits:
            result = result.union(subcircuit.variables)
        return result

    def support(self):
        support: Event = self.subcircuits[0].result_of_current_query
        support.fill_missing_variables(self.variables)

        for subcircuit in self.subcircuits[1:]:
            support &= subcircuit.result_of_current_query

        self.result_of_current_query = support

    def is_decomposable(self):
        for index, subcircuit in enumerate(self.subcircuits):
            variables = subcircuit.variables
            for subcircuit_ in self.subcircuits[index + 1:]:
                if len(set(subcircuit_.variables).intersection(set(variables))) > 0:
                    return False
        return True

    def log_mode(self):
        arg_log_max, log_max = self.subcircuits[0].result_of_current_query
        arg_log_max.fill_missing_variables(self.variables)
        for subcircuit in self.subcircuits[1:]:
            arg_log_max = arg_log_max.intersection_with(subcircuit.result_of_current_query[0])
            log_max += subcircuit.result_of_current_query[1]
        self.result_of_current_query = arg_log_max, log_max

    def __copy__(self):
        return self.empty_copy()

    def simplify(self):

        # if this has only one child
        if len(self.subcircuits) == 1:
            self.connect_incoming_edges_to(self.subcircuits[0])
            self.probabilistic_circuit.remove_node(self)
            return

        # for every subcircuit
        for subcircuit in self.subcircuits:

            # if the simplified subcircuit is of the same type as this
            if type(subcircuit) is type(self):

                # type hinting
                subcircuit: Self

                # mount the children of that circuit directly
                for sub_subcircuit in subcircuit.subcircuits:
                    subcircuit.add_subcircuit(sub_subcircuit, mount=False)

    def sample(self, *args, **kwargs):
        for start_index, amount in self.result_of_current_query:
            for subcircuit in self.subcircuits:
                subcircuit.result_of_current_query.append([start_index, amount])


class ProbabilisticCircuit(ProbabilisticModel):
    """
    Probabilistic Circuits as a directed, rooted, acyclic graph.
    """

    graph: rx.PyDAG[Unit]
    """
    The graph structure of the circuit.
    """

    def __init__(self):
        super().__init__()
        self.graph = rx.PyDAG(multigraph=False)

    def log_conditional_of_point(self, point: Dict[Variable, Any]) -> Tuple[Optional[Self], float]:
        pass

    def sample(self, amount: int) -> np.array:

        # initialize all results
        for node in self.nodes:
            node.result_of_current_query = []

        variable_to_index_map = self.variable_to_index_map

        # initialize the sample arguments
        self.root.result_of_current_query.append((0, amount))

        # initialize the samples
        samples = np.full((amount, len(variable_to_index_map)), np.nan)

        # forward through the circuit to sample
        [node.sample(samples, variable_to_index_map) for layer in self.layers for node in layer]

        return samples



    def validate(self):
        assert rx.is_directed_acyclic_graph(self.graph)
        for unit in self.nodes:
            if unit.is_leaf:
                continue
            elif isinstance(unit, SumUnit):
                assert np.isclose(logsumexp(unit.log_weights), 0.)
            elif isinstance(unit, ProductUnit):
                edge_data = [self.graph.get_edge_data(unit.index, ss.index) for ss in unit.subcircuits]
                for d in edge_data:
                    assert d is None
            else:
                raise NotImplementedError

    def add_node(self, unit: Unit):
        """
        Add a node to the circuit if it is not already present.
        This overwrites the `index` and sets the `_probabilistic_circuit` fields.

        :param unit: The unit to add.
        """
        if unit.probabilistic_circuit != self:
            node_index = self.graph.add_node(unit)
            unit.index = node_index
            unit._probabilistic_circuit = self

    def add_edge(self, parent: Unit, child: Unit, log_weight: Optional[float] = None):
        """
        Add an edge to the circuit.
        Adds units that are not in the graph to the graph.

        :param parent: The parent unit.
        :param child: The child unit.
        :param log_weight: The weight of the edge.
        """
        self.add_node(parent)
        self.add_node(child)
        self.graph.add_edge(parent.index, child.index, log_weight)

    def add_nodes_from(self, nodes: Iterable[Unit]):
        """
        Add many nodes to the circuit.
        See `func:add_node` for more details.
        :param nodes:
        :return:
        """
        for unit in nodes:
            self.add_node(unit)

    def add_edges_from(self, edges: Iterable[Tuple]):
        for elem in edges:
            self.add_edge(*elem)

    def remove_edge(self, parent: Unit, child: Unit):
        self.graph.remove_edge(parent.index, child.index)

    def remove_node(self, node: Unit):
        self.graph.remove_node(node.index)
        node.index = None
        node._probabilistic_circuit = None

    def remove_nodes_from(self, nodes: Iterable[Unit]):
        for unit in nodes:
            self.remove_node(unit)

    @property
    def nodes(self) -> List[Unit]:
        return self.graph.nodes()

    @property
    def edges(self) -> List[Tuple[Unit, Unit, Optional[float]]]:
        return [(self.graph.get_node_data(p), self.graph.get_node_data(c), lw) for (p, c), lw
                in zip(self.graph.edge_list(), self.graph.edges())]

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
    def leaves(self) -> List[Unit]:
        indices = self.graph.filter_nodes(lambda node: self.graph.out_degree(node.index) == 0)
        return [self.graph.nodes()[index] for index in indices]

    @property
    def variables(self) -> SortedSet:
        return self.root.variables

    @property
    def layers(self) -> List[List[Unit]]:
        return rx.layers(self.graph, [self.root.index])

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        variable_to_index_map = self.variable_to_index_map
        [node.moment(order, center, variable_to_index_map) for layer in reversed(self.layers) for node in layer]
        return MomentType({variable: moment for variable, moment in
                           zip(variable_to_index_map.keys(), self.root.result_of_current_query)})

    def log_likelihood(self, events: np.array) -> np.array:
        variable_to_index_map = self.variable_to_index_map
        for layer in reversed(self.layers):
            for unit in layer:  # open all the procesess
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.log_likelihood(events[:, [variable_to_index_map[variable] for variable in unit.variables]])
                else:
                    unit: InnerUnit
                    unit.log_forward()  # Synch trheads 1
        return self.root.result_of_current_query

    def cdf(self, events: np.array) -> np.array:
        variable_to_index_map = self.variable_to_index_map
        for layer in reversed(self.layers):
            for unit in layer:
                unit: LeafUnit
                if unit.is_leaf:
                    unit.cdf(events[:, [variable_to_index_map[variable] for variable in unit.variables]])
                else:
                    unit: InnerUnit
                    unit.forward()
        return self.root.result_of_current_query

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.probability_of_simple_event(event)
                else:
                    unit: InnerUnit
                    unit.forward()
        return self.root.result_of_current_query

    def log_mode(self, check_determinism: bool = True) -> Tuple[Event, float]:
        if check_determinism:
            if not self.is_deterministic():
                raise IntractableError("The circuit is not deterministic.")
        [unit.log_mode() for layer in reversed(self.layers) for unit in layer]
        return self.root.result_of_current_query

    def remove_unreachable_nodes(self, root: Unit):
        """
        Remove all nodes that are not reachable from the root.
        """
        reachable_nodes = rx.descendants(self.graph, root.index)
        reachable_nodes = set([self.graph.get_node_data(index) for index in reachable_nodes])
        unreachable_nodes = set(self.graph.nodes()) - (reachable_nodes | {root})
        self.remove_nodes_from(unreachable_nodes)

    def log_conditional_of_simple_event_in_place(self, simple_event: SimpleEvent) -> Tuple[Optional[Self], float]:
        """
        Construct the conditional circuit from a simple event.

        :param simple_event: The simple event to condition on.
        :return: The conditional circuit and the log-probability of the event
        """
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.log_conditional_of_simple_event_in_place(simple_event)
                else:
                    unit: InnerUnit
                    unit.log_forward()


        root = self.root
        [self.remove_node(node) for node in self.nodes if node.result_of_current_query == -np.inf]

        if root not in self.nodes:
            return None, -np.inf

        # clean the circuit up
        self.remove_unreachable_nodes(root)
        self.simplify()
        self.normalize()

        return self, root.result_of_current_query

    @property
    def variable_to_index_map(self) -> Dict[Variable, int]:
        return {variable: index for index, variable in enumerate(self.variables)}

    def log_conditional_in_place(self, event: Event) -> Tuple[Optional[Self], float]:
        """
        Efficiently compute the conditional for an Event, batching as much as possible.
        """
        # skip trivial case
        if event.is_empty():
            self.remove_nodes_from(list(self.nodes))
            return None, -np.inf


        # if the event is easy, don't create a proxy node
        elif len(event.simple_sets) == 1:
            result = self.log_conditional_of_simple_event_in_place(event.simple_sets[0])
            return result

        # create a conditional circuit for every simple event
        conditional_circuits = [self.__deepcopy__().log_conditional_of_simple_event_in_place(simple_event) for simple_event
                                in event.simple_sets]

        # clear this circuit
        self.remove_nodes_from(list(self.nodes))

        # filtered out impossible conditionals
        conditional_circuits = [(conditional, log_probability) for conditional, log_probability in conditional_circuits
                                if log_probability > -np.inf]

        # if all conditionals are impossible
        if len(conditional_circuits) == 0:
            return None, -np.inf

        # create a new sum unit
        result = SumUnit(self)

        # add the conditionals to the sum unit
        [result.add_subcircuit(conditional.root, log_probability) for conditional, log_probability in
         conditional_circuits]
        result.log_forward()
        result.normalize()
        return self, result.result_of_current_query

    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        result = self.__deepcopy__()
        return result.log_conditional_in_place(event)

    def simplify(self) -> Self:
        """
        Simplify the circuit inplace.
        """
        [node.simplify() for layer in reversed(self.layers) for node in layer]
        return self

    @property
    def support(self) -> Event:
        [node.support() for layer in reversed(self.layers) for node in layer]
        return self.root.result_of_current_query

    def is_decomposable(self) -> bool:
        """
        Check if the whole circuit is decomposed.

        A circuit is decomposed if all its product units are decomposed.

        :return: if the whole circuit is decomposed
        """
        return all([subcircuit.is_decomposable() for subcircuit in self.leaves if isinstance(subcircuit, ProductUnit)])

    def __eq__(self, other: Self):
        return id(self) == id(other)

    def empty_copy(self) -> Self:
        """
        Create a copy of this circuit without any nodes.
        Only the parameters should be copied.
        This is used whenever a new circuit has to be created during inference.

        :return: A copy of this circuit without any subcircuits that is not in this units graph.
        """
        return self.__class__()

    def __deepcopy__(self, *args, **kwargs):
        result = self.__class__()
        new_nodes_map = {n.index: n.empty_copy() for n in self.nodes}
        result.add_nodes_from(new_nodes_map.values())
        for (parent_index, child_index), log_weight in zip(self.graph.edge_list(), self.graph.edges()):
            parent = new_nodes_map[parent_index]
            child = new_nodes_map[child_index]
            result.add_edge(parent, child, log_weight)
        return result

    def normalize(self):
        """
        Normalize every sum node of this circuit in-place.
        """
        [node.normalize() for node in self.nodes if isinstance(node, SumUnit)]

    def is_isomorphic(self, other: Self):
        node_matcher = lambda x, y: type(x) == type(y)
        return rx.digraph_is_isomorphic(self.graph, other.graph, node_matcher=node_matcher)

    def is_deterministic(self) -> bool:
        """
        :return: Whether, this circuit is deterministic or not.
        """

        # calculate the support
        support = self.support

        # check for determinism of every node
        return all(node.is_deterministic() for node in self.nodes if isinstance(node, SumUnit))

    def marginal_in_place(self, variables: Iterable[Variable]) -> Optional[Self]:
        result = [node.marginal(variables) for layer in reversed(self.layers) for node in layer][-1]
        if result is not None:
            self.remove_unreachable_nodes(result)
            self.simplify()
            return self
        else:
            return None

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        result = self.__deepcopy__()
        return result.marginal_in_place(variables)

    def __repr__(self):
        return f"{self.__class__.__name__} with {len(self.nodes)} nodes and {len(self.edges)} edges"

class UnivariateLeaf(LeafUnit):

    @property
    def variable(self) -> Variable:
        return self.distribution.variables[0]


class UnivariateContinuousLeaf(UnivariateLeaf):
    distribution: Optional[ContinuousDistribution]

    def log_conditional_of_simple_event_in_place(self, event: SimpleEvent):
        return self.univariate_log_conditional_of_simple_event_in_place(event[self.variable])

    def univariate_log_conditional_of_simple_event_in_place(self, event: Interval):
        """
        Condition this distribution on a simple event in-place but use sum units to create conditions on composite
        intervals.
        :param event: The simple event to condition on.
        """

        # if it is a simple truncation
        if len(event.simple_sets) == 1:
            self.distribution, self.result_of_current_query = self.distribution.log_conditional_from_simple_interval(
                event.simple_sets[0])
            return self

        total_probability = 0.

        # calculate the conditional distribution as sum unit
        result = SumUnit(self.probabilistic_circuit)

        for simple_interval in event.simple_sets:
            current_conditional, current_log_probability = self.distribution.log_conditional_from_simple_interval(
                simple_interval)
            current_probability = np.exp(current_log_probability)

            if current_probability == 0:
                continue

            current_conditional = self.__class__(current_conditional, self.probabilistic_circuit)
            result.add_subcircuit(current_conditional, np.log(current_probability), mount=False)
            total_probability += current_probability

        # if the event is impossible
        if total_probability == 0:
            self.result_of_current_query = -np.inf
            self.distribution = None
            self.probabilistic_circuit.remove_node(result)
            return None

        # reroute the parent to the new sum unit
        self.connect_incoming_edges_to(result)

        # remove this node
        self.probabilistic_circuit.remove_node(self)

        # update result
        result.normalize()
        result.result_of_current_query = np.log(total_probability)
        return result


class UnivariateDiscreteLeaf(UnivariateLeaf):
    distribution: Optional[DiscreteDistribution]

    def as_deterministic_sum(self) -> SumUnit:
        """
        Convert this distribution to a deterministic sum unit that encodes the same distribution in-place.
        The result has as many children as the probability dictionary of this distribution.
        Each child encodes the value of the variable.

        :return: The deterministic sum unit that encodes the same distribution.
        """
        result = SumUnit(self.probabilistic_circuit)

        for element, probability in self.distribution.probabilities.items():
            result.add_subcircuit(
                UnivariateDiscreteLeaf(self.distribution.__class__(self.variable, MissingDict(float, {element: 1.})),
                                       self.probabilistic_circuit), np.log(probability), mount=False)
        self.connect_incoming_edges_to(result)
        self.probabilistic_circuit.remove_node(self)
        return result

    @classmethod
    def from_mixture(cls, mixture: ProbabilisticCircuit):
        """
        Create a discrete distribution from a univariate mixture.

        :param mixture: The mixture to create the distribution from.
        :return: The discrete distribution.
        """
        assert len(mixture.variables) == 1, "Can only convert univariate sum units to discrete distributions."
        variable = mixture.variables[0]
        probabilities = MissingDict(float)

        for element in mixture.support.simple_sets[0][variable].simple_sets:
            probability = mixture.probability_of_simple_event(SimpleEvent({variable: element}))
            if isinstance(element, SimpleInterval):
                element = element.lower
            probabilities[hash(element)] = probability

        distribution_class = IntegerDistribution if isinstance(variable, Integer) else SymbolicDistribution
        distribution = distribution_class(variable, probabilities)
        return cls(distribution)


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
