from __future__ import annotations

import itertools
import math
import queue
import random
from abc import abstractmethod
from collections import deque, defaultdict
from enum import IntEnum

import networkx as nx
import networkx.drawing
import numpy as np
from random_events.interval import SimpleInterval, Interval
from scipy.special import logsumexp
import tqdm
from matplotlib import pyplot as plt
from random_events.product_algebra import VariableMap, SimpleEvent, Event
from random_events.set import Set
from random_events.utils import SubclassJSONSerializer
from random_events.variable import Variable, Symbolic, Continuous, Integer
from sortedcontainers import SortedSet
from typing_extensions import List, Optional, Any, Self, Dict, Tuple, Iterable, Callable

from ...distributions import UnivariateDistribution, IntegerDistribution, SymbolicDistribution, DiscreteDistribution, \
    ContinuousDistribution
from ...distributions.helper import make_dirac
from ...error import IntractableError
from ...interfaces.drawio.drawio import DrawIOInterface, circled_product, circled_sum
from ...probabilistic_model import ProbabilisticModel, OrderType, CenterType, MomentType
from ...utils import MissingDict


class Unit(SubclassJSONSerializer, DrawIOInterface):
    """
    Class for all units of a probabilistic circuit.

    This class should not be used by users directly.
    Use :class:`ProbabilisticCircuit` as interface to users.
    """

    probabilistic_circuit: ProbabilisticCircuit
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

        self.probabilistic_circuit = probabilistic_circuit
        self.probabilistic_circuit.add_node(self)

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
        return list(self.probabilistic_circuit.predecessors(self))

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

    def update_variables(self, new_variables: VariableMap):
        """
        Update the variables of this unit and its descendants.

        :param new_variables: A map that maps the variables that should be replaced to their new variable.
        """
        for leaf in self.leaves:
            for variable in leaf.variables:
                if variable in new_variables:
                    leaf.distribution.variable = new_variables[leaf.variable]

    def connect_incoming_edges_to(self, other: Unit):
        """
        Connect all incoming edges to this unit to another unit.

        :param other: The other unit to connect the incoming edges to.
        """
        incoming_edges = list(self.probabilistic_circuit.in_edges(self, data=True))
        for parent, _, data in incoming_edges:
            self.probabilistic_circuit.add_edge(parent, other, **data)

    def mount(self, other: Unit):
        """
        Mount another unit including its descendants. There will be no edge from `self` to `other`.

        :param other: The other unit to mount.
        """
        descendants = nx.descendants(other.probabilistic_circuit, other)
        descendants = descendants.union([other])
        subgraph = other.probabilistic_circuit.subgraph(descendants)
        self.probabilistic_circuit.add_edges_and_nodes_from_circuit(subgraph)

    def filter_variable_map_by_self(self, variable_map: VariableMap):
        """
        Filter a variable map by the variables of this unit.

        :param variable_map: The map to filter
        :return: The map filtered by the variables of this unit.
        """
        variables = self.variables
        return variable_map.__class__(
            {variable: value for variable, value in variable_map.items() if variable in variables})

    @property
    def impossible_condition_result(self) -> Tuple[Optional[Unit], float]:
        """
        :return: The result of an impossible truncated query.
        """
        return None, -np.inf

    def log_mode(self):
        raise NotImplementedError

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.subcircuits == other.subcircuits

    def __copy__(self):
        raise NotImplementedError()

    def empty_copy(self) -> Self:
        """
        Creat a copy of this circuit without any subcircuits. Only the parameters should be copied.
        This is used whenever a new circuit has to be created during inference.

        :return: A copy of this circuit without any subcircuits that is not in this units graph.
        """
        return self.__class__()

    def simplify(self):
        """
        Simplify the circuit by removing nodes and redirected edges that have no impact in-place.
        Essentially, this method transforms the circuit into an alternating order of sum and product units.

        :return: The simplified circuit.
        """
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        """
        Draw samples from the circuit.

        For sampling, a node gets requested a number of samples from all his parents.
        The parents write into the `result_of_current_query` attribute a tuple describing the beginning index of the
        sampling and how many samples are requested.
        """
        raise NotImplementedError

    def marginal(self, *args, **kwargs) -> Optional[Self]:
        """
        Remove nodes that are not part of the marginal distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def moment(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def drawio_style(self) -> Dict[str, Any]:
        return {"style": self.drawio_label, "width": 30, "height": 30, }


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

    def __repr__(self):
        return repr(self.distribution)

    @property
    def drawio_label(self):
        return "ellipse;whiteSpace=wrap;html=1;aspect=fixed;"

    @property
    def drawio_style(self) -> Dict[str, Any]:
        return {"style": self.drawio_label, "width": 30, "height": 30, "label": self.distribution.abbreviated_symbol}

    @property
    def variables(self) -> Iterable[Variable]:
        return SortedSet(self.distribution.variables)

    @property
    def subcircuits(self) -> List[Unit]:
        return []

    @property
    def is_leaf(self):
        return True

    @property
    def leaves(self) -> List[LeafUnit]:
        return []

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

    def log_truncated_of_simple_event_in_place(self, event: SimpleEvent):
        self.distribution, self.result_of_current_query = self.distribution.log_truncated(event.as_composite_set())

    def __copy__(self):
        return self.__class__(self.distribution.__copy__())

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

    def to_json(self):
        result = super().to_json()
        result["distribution"] = self.distribution.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        distribution = SubclassJSONSerializer.from_json(data["distribution"])
        return cls(distribution)

    def log_conditional_in_place(self, point: Dict[Variable, Any]):
        if any(variable for variable in self.variables if variable in point):
            self.distribution, self.result_of_current_query = self.distribution.log_conditional(point)
        else:
            self.result_of_current_query = 0.


class InnerUnit(Unit):
    """
    Class for inner units
    """

    @property
    def subcircuits(self) -> List[Unit]:
        return list(self.probabilistic_circuit.successors(self))

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
    _latent_variable: Optional[Symbolic] = None
    """
    The latent variable of this unit.
    This has to be here due to the rvalue/lvalue problem in random events.
    
    TODO remove this when RE is fixed
    """

    def __repr__(self):
        return "⊕"

    @property
    def representation(self) -> str:
        return "+"

    @property
    def drawio_label(self) -> str:
        return circled_sum

    @property
    def log_weighted_subcircuits(self) -> List[Tuple[float, Unit]]:
        """
        :return: The weighted subcircuits of this unit.
        """
        return [(self.probabilistic_circuit.edges[self, subcircuit]["log_weight"], subcircuit) for subcircuit in
                self.subcircuits]

    @property
    def variables(self) -> SortedSet:
        return self.subcircuits[0].variables

    @property
    def latent_variable(self) -> Symbolic:
        name = f"{hash(self)}.latent"
        subcircuit_enum = IntEnum(name, {str(hash(subcircuit)): index
                                         for index, subcircuit in enumerate(self.subcircuits)})
        result = Symbolic(name, Set.from_iterable(subcircuit_enum))
        self._latent_variable = result
        return result

    def add_subcircuit(self, subcircuit: Unit, log_weight: float, mount: bool = True):
        """
        Add a subcircuit to the subcircuits of this unit.

        .. note::

            This method does not normalize the edges to the subcircuits.


        :param subcircuit: The subcircuit to add.
        :param log_weight: The logarithmic weight of the subcircuit.
        :param mount: If the subcircuit should be mounted to the pc of this unit.

        """
        if mount:
            self.mount(subcircuit)
        self.probabilistic_circuit.add_edge(self, subcircuit, log_weight=log_weight)

    def forward(self, *args, **kwargs):
        self.result_of_current_query = np.sum(
            [np.exp(weight) * subcircuit.result_of_current_query for weight, subcircuit in self.log_weighted_subcircuits], axis=0)

    def log_forward(self, *args, **kwargs):

        result = [lw + s.result_of_current_query for lw, s in self.log_weighted_subcircuits]
        self.result_of_current_query = logsumexp(result, axis=0)
    moment = forward

    def log_forward_conditioning(self, *args, **kwargs):
        result = [lw + s.result_of_current_query for lw, s in self.log_weighted_subcircuits]

        # update weights according to bayes rule
        for new_weight, subcircuit in zip(result, self.subcircuits):
            self.probabilistic_circuit.add_edge(self, subcircuit, log_weight=new_weight)

        self.result_of_current_query = logsumexp(result, axis=0)

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

                # calculate truncated probability
                weight = p_query / p_condition

                # create edge from proxy to subcircuit
                proxy_sum_node.add_subcircuit(other_subcircuit, log_weight=np.log(weight))
            proxy_sum_node.normalize()

    def mount_from_bayesian_network(self, other: 'SumUnit'):
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
            incoming_edges = list(self.probabilistic_circuit.in_edges(self, data=True))
            for parent, _, data in incoming_edges:
                self.probabilistic_circuit.add_edge(parent, self.subcircuits[0], **data)

            # remove this node
            self.probabilistic_circuit.remove_node(self)

            return

        # for every subcircuit
        for weight, subcircuit in self.log_weighted_subcircuits:

            # if the weight is 0, skip this subcircuit
            if weight == -np.inf:
                # remove the edge
                self.probabilistic_circuit.remove_edge(self, subcircuit)

            # if the simplified subcircuit is of the same type as this
            if type(subcircuit) is type(self):

                # type hinting
                subcircuit: Self

                # mount the children of that circuit directly
                for sub_weight, sub_subcircuit in subcircuit.log_weighted_subcircuits:
                    new_weight = sub_weight + weight

                    # add an edge to that subcircuit
                    self.add_subcircuit(sub_subcircuit, new_weight, mount=False)

                # remove the old node
                self.probabilistic_circuit.remove_node(subcircuit)

    def normalize(self):
        """
        Normalize the log_weights of the subcircuits such that they sum up to 1 inplace.
        """
        total_weight = logsumexp(self.log_weights)
        for subcircuit in self.subcircuits:
            self.probabilistic_circuit.edges[self, subcircuit]["log_weight"] -= total_weight

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


class ProductUnit(InnerUnit):
    """
    Decomposable Product Units for Probabilistic Circuits
    """

    representation = "×"

    @property
    def drawio_label(self) -> str:
        return circled_product

    def __repr__(self):
        return "⊗"

    def forward(self, *args, **kwargs):
        self.result_of_current_query = math.prod(
            [subcircuit.result_of_current_query for subcircuit in self.subcircuits])

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

    def add_subcircuit(self, subcircuit: Unit, mount: bool = True):
        """
        Add a subcircuit to the subcircuits of this unit.

        :param subcircuit: The subcircuit to add.
        :param mount: If the subcircuit should be mounted to this units pc instance.
        """
        if mount:
            self.mount(subcircuit)
        self.probabilistic_circuit.add_edge(self, subcircuit)

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


class ProbabilisticCircuit(ProbabilisticModel, nx.DiGraph, SubclassJSONSerializer):
    """
    Probabilistic Circuits as a directed, rooted, acyclic graph.

    The nodes of the graph are the units of the circuit.
    The edges of the graph indicate how the units are connected.
    The outgoing edges of a sum unit contain the log-log_weights of the subcircuits.
    """

    _cached_nodes: List[Unit]
    """
    Cached list of all nodes
    """

    _cached_unweighted_edges: List[Tuple[Unit, Unit]]
    """
    Cached list of unweighted edges
    """

    _cached_weighted_edges: List[Tuple[Unit, Unit, float]]
    """
    Cached list of weighted edges
    """

    _cached_topological_ordered_nodes: List[Unit]
    """
    Cached list of topological sorted units
    """

    _cached_reversed_topological_ordered_nodes: List[Tuple[Unit, Unit]]
    """
    Cached list of reverse topological sorted units
    """

    def __init__(self):
        super().__init__(None)
        nx.DiGraph.__init__(self)

    def cache_structure(self):
        """
        Call once after building the circuit:
        - cache nodes
        - cache edge lists
        - cache topo order & reverse topo order
        """
        assert not any(
            hasattr(self, k) for k in ["_cached_nodes", "_cached_unw", "_cached_w", "_cached_topo", "_cached_rev"]), (
            "Cache was already set! You should have invalidated it before changing the graph."
        )

        # 1) Nodes
        self._cached_nodes = list(self.nodes)

        # 2) Edges
        all_edges = list(self.edges(data=True))
        self._cached_unweighted_edges = [(u, v) for u, v, attr in all_edges
                                         if "log_weight" not in attr]
        self._cached_weighted_edges   = [(u, v, attr["log_weight"]) for u, v, attr in all_edges
                                         if "log_weight" in attr]

        # 3) Build successor map & in-degree
        succ = {n: [] for n in self._cached_nodes}
        in_deg = {n: 0 for n in self._cached_nodes}
        for u, v in self._cached_unweighted_edges + [(u, v) for (u, v, _) in self._cached_weighted_edges]:
            succ[u].append(v)
            in_deg[v] += 1

        # 4) Single Kahn run
        queue = deque(n for n, d in in_deg.items() if d == 0)
        topo = []
        while queue:
            n = queue.popleft()
            topo.append(n)
            for m in succ[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)

        self._cached_topological_ordered_nodes = topo
        self._cached_reversed_topological_ordered_nodes  = list(reversed(topo))

    def _invalidate_cache(self):
        """Call this before any structure-changing operation."""
        for k in ["_cached_nodes", "_cached_unw", "_cached_w", "_cached_topo", "_cached_rev"]:
            if hasattr(self, k):
                delattr(self, k)

    @classmethod
    def from_other(cls, other: Self) -> Self:
        result = cls()
        result.add_edges_and_nodes_from_circuit(other)
        return result

    @property
    def variables(self) -> SortedSet:
        return self.root.variables

    @property
    def variable_to_index_map(self) -> Dict[Variable, int]:
        return {variable: index for index, variable in enumerate(self.variables)}

    @property
    def layers(self) -> List[List[Unit]]:
        return list(nx.bfs_layers(self, self.root))

    @property
    def leaves(self) -> List[LeafUnit]:
        return self.root.leaves

    def is_valid(self) -> bool:
        """
        Check if this graph is:

        - acyclic
        - connected

        :return: True if the graph is valid, False otherwise.
        """
        return nx.is_directed_acyclic_graph(self) and nx.is_weakly_connected(self)

    def add_node(self, node: Unit, **attr):
        self._invalidate_cache()
        # write self as the nodes' circuit
        node.probabilistic_circuit = self

        # call super
        super().add_node(node, **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for node in nodes_for_adding:
            self.add_node(node, **attr)

    @property
    def root(self) -> Unit:
        """
        The root of the circuit is the node with in-degree 0.
        This is the output node, that will perform the final computation.

        :return: The root of the circuit.
        """
        possible_roots = [node for node in self.nodes() if self.in_degree(node) == 0]
        if len(possible_roots) > 1:
            raise ValueError(f"More than one root found. Possible roots are {possible_roots}")

        return possible_roots[0]

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
        reachable_nodes = nx.descendants(self, root)
        unreachable_nodes = set(self.nodes) - (reachable_nodes | {root})
        self.remove_nodes_from(unreachable_nodes)

    def log_truncated_of_simple_event_in_place(self, simple_event: SimpleEvent) -> Tuple[Optional[Self], float]:
        """
        Construct the truncated circuit from a simple event.

        :param simple_event: The simple event to condition on.
        :return: The truncated circuit and the log-probability of the event
        """
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.log_truncated_of_simple_event_in_place(simple_event)
                else:
                    unit: InnerUnit
                    unit.log_forward()

        root = self.root
        [self.remove_node(node) for layer in reversed(self.layers) for node in layer if
         node.result_of_current_query == -np.inf]

        if root not in self.nodes:
            return None, -np.inf

        # clean the circuit up
        self.remove_unreachable_nodes(root)
        self.simplify()
        self.normalize()

        return self, root.result_of_current_query

    def log_truncated_in_place(self, event: Event) -> Tuple[Optional[Self], float]:
        """
        Efficiently compute the truncated for an Event, batching as much as possible.
        """
        # skip trivial case
        if event.is_empty():
            self.remove_nodes_from(list(self.nodes))
            return None, -np.inf


        # if the event is easy, don't create a proxy node
        elif len(event.simple_sets) == 1:
            result = self.log_truncated_of_simple_event_in_place(event.simple_sets[0])
            return result

        # create a conditional circuit for every simple event
        conditional_circuits = [self.__copy__().log_truncated_of_simple_event_in_place(simple_event) for simple_event
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

    def log_truncated(self, event: Event) -> Tuple[Optional[Self], float]:
        result = self.__copy__()
        return result.log_truncated_in_place(event)

    def marginal_in_place(self, variables: Iterable[Variable]) -> Optional[Self]:
        result = [node.marginal(variables) for layer in reversed(self.layers) for node in layer][-1]
        if result is not None:
            self.remove_unreachable_nodes(result)
            self.simplify()
            return self
        else:
            return None

    def log_conditional_in_place(self, point: Dict[Variable, Any]) -> Tuple[Optional[Self], float]:

        # do forward pass
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.log_conditional_in_place(point)
                elif isinstance(unit, SumUnit):
                    unit.log_forward_conditioning()
                elif isinstance(unit, ProductUnit):
                    unit.log_forward()
                else:
                    raise NotImplementedError()

        # clean the circuit up
        root = self.root
        [self.remove_node(node) for layer in reversed(self.layers) for node in layer if
         node.result_of_current_query == -np.inf]

        if root not in self.nodes:
            return None, -np.inf

        self.remove_unreachable_nodes(root)

        # simplify dirac parts
        remaining_variables = [v for v in self.variables if v not in point]


        self.marginal_in_place(remaining_variables)

        if len(remaining_variables) > 0:
            root = self.root

        # add dirac parts
        new_root = ProductUnit(self)

        if len(remaining_variables) > 0:
            new_root.add_subcircuit(root, False)

        for variable, value in point.items():
            new_root.add_subcircuit(leaf(make_dirac(variable, value), self))

        new_root.result_of_current_query = root.result_of_current_query

        self.simplify()
        self.normalize()

        return self, root.result_of_current_query

    def log_conditional(self, point: Dict[Variable, Any]) -> Tuple[Optional[Self], float]:
        result = self.__copy__()
        return result.log_conditional_in_place(point)

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        result = self.__copy__()
        return result.marginal_in_place(variables)

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

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        variable_to_index_map = self.variable_to_index_map
        [node.moment(order, center, variable_to_index_map) for layer in reversed(self.layers) for node in layer]
        return MomentType({variable: moment for variable, moment in
                           zip(variable_to_index_map.keys(), self.root.result_of_current_query)})

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

    def __eq__(self, other: 'ProbabilisticCircuit'):
        return self.root == other.root

    def empty_copy(self) -> Self:
        """
        Create a copy of this circuit without any nodes.
        Only the parameters should be copied.
        This is used whenever a new circuit has to be created during inference.

        :return: A copy of this circuit without any subcircuits that is not in this units graph.
        """
        return self.__class__()

    def __copy__(self, precomputed=None):
        """
        Fast copy: only use precomputed = (nodes, unw, weighted) if provided,
        else fallback to walking the graph.
        """
        if precomputed is None:
            nodes = list(self.nodes)
            unw = [(u, v) for u, v, attr in self.edges(data=True)
                   if "log_weight" not in attr]
            wgt = [(u, v, attr["log_weight"]) for u, v, attr in self.edges(data=True)
                   if "log_weight" in attr]
        else:
            nodes, unw, wgt = precomputed

        # Use fast dict comprehension for copying nodes
        result = self.empty_copy()
        new_map = {n: n.__copy__() for n in nodes}
        result.add_nodes_from(new_map.values())
        # Add edges directly from precomputed lists
        result.add_edges_from((new_map[u], new_map[v]) for u, v in unw)
        result.add_weighted_edges_from(
            ((new_map[u], new_map[v], lw) for u, v, lw in wgt),
            weight="log_weight"
        )
        return result

    def to_json(self) -> Dict[str, Any]:
        # get super result
        result = super().to_json()

        hash_to_node_map = dict()

        for node in self.nodes:
            node_json = node.to_json()
            hash_to_node_map[hash(node)] = node_json

        unweighted_edges = [(hash(source), hash(target)) for source, target in self.unweighted_edges]
        weighted_edges = [(hash(source), hash(target), weight) for source, target, weight in self.log_weighted_edges]
        result["hash_to_node_map"] = hash_to_node_map
        result["unweighted_edges"] = unweighted_edges
        result["log_weighted_edges"] = weighted_edges
        return result

    @classmethod
    def parameters_from_json(cls, data: Dict[str, Any]) -> Self:
        return cls()

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        result = cls.parameters_from_json(data)
        hash_remap: Dict[int, Unit] = dict()

        for hash_, node_data in data["hash_to_node_map"].items():
            node = Unit.from_json(node_data)
            hash_remap[int(hash_)] = node
            result.add_node(node)

        for source_hash, target_hash in data["unweighted_edges"]:
            result.add_edge(hash_remap[source_hash], hash_remap[target_hash])

        for source_hash, target_hash, weight in data["log_weighted_edges"]:
            result.add_edge(hash_remap[source_hash], hash_remap[target_hash], log_weight=weight)

        return result

    def update_variables(self, new_variables: VariableMap):
        """
        Update the variables of this unit and its descendants.

        :param new_variables: The new variables to set.
        """
        self.root.update_variables(new_variables)

    @property
    def log_weighted_edges(self):
        """
        :return: All log-weighted edges of the circuit.
        """
        weighted_edges = []

        for edge in self.edges:
            edge_ = self.edges[edge]

            if "log_weight" in edge_.keys():
                weight = edge_["log_weight"]
                weighted_edges.append((*edge, weight))

        return weighted_edges

    @property
    def unweighted_edges(self):
        """
        :return: All unweighted edges of the circuit.
        """
        unweighted_edges = []

        for edge in self.edges:
            edge_ = self.edges[edge]

            if "weight" not in edge_.keys():
                unweighted_edges.append(edge)

        return unweighted_edges

    def is_deterministic(self) -> bool:
        """
        :return: Whether, this circuit is deterministic or not.
        """

        # calculate the support
        support = self.support

        # check for determinism of every node
        return all(node.is_deterministic() for node in self.nodes if isinstance(node, SumUnit))

    def normalize(self):
        """
        Normalize every sum node of this circuit in-place.
        """
        [node.normalize() for node in self.nodes if isinstance(node, SumUnit)]

    def add_edges_and_nodes_from_circuit(self, other: Self):
        """
        Add all edges and nodes from another circuit to this circuit.

        :param other: The other circuit to add.
        """
        self.add_nodes_from(other.nodes)
        self.add_edges_from(other.unweighted_edges)
        self.add_weighted_edges_from(other.log_weighted_edges, weight="log_weight")

    def add_weighted_edges_from(
        self, ebunch_to_add, weight = "log_weight", **attr
    ):
        self._invalidate_cache()
        return super().add_weighted_edges_from(ebunch_to_add, weight=weight, **attr)

    def subgraph_of(self, node: Unit) -> Self:
        """
        Create a subgraph with a node as root.

        :param node: The root of the subgraph.
        :return: The subgraph.
        """
        nodes_to_keep = list(nx.descendants(self, node)) + [node]
        return nx.subgraph(self, nodes_to_keep)

    def fill_node_colors(self, node_colors: Dict[Unit, str]):
        """
        Fill the node colors for the structure plot.

        :param node_colors: The node colors to fill.
        """
        # fill the colors for the nodes
        if node_colors is None:
            node_colors = dict()
        for node in self.nodes:
            if node not in node_colors:
                node_colors[node] = "black"
        return node_colors

    def plot_structure(self, node_colors: Optional[Dict[Unit, str]] = None, node_size=550, variable_name_offset=0.2,
                       plot_inference=False,
                       inference_representation: Callable = lambda node: str(node.result_of_current_query),
                       inference_result_offset: float = -0.25):
        """
        Plot the structure of the circuit using matplotlib.

        :param node_colors: Optionally specified colors of the node.
        If nodes are not specified in the dictionary, they will be black.
        :param node_size: The size of the nodes
        :param variable_name_offset: The offset to the right of the variable names.
        :param plot_inference: If the results of the inference should be plotted.
        :param inference_representation: The representation of the inference results as a function from node to string.
        :param inference_result_offset: The vertical offset of the inference results.
        """

        # fill the colors for the nodes
        node_colors = self.fill_node_colors(node_colors)

        # get the positions of the nodes
        positions = networkx.drawing.bfs_layout(self, self.root)
        position_for_variable_name = {node: (x + variable_name_offset, y) for node, (x, y) in positions.items()}



        # draw the edges
        alpha_for_edges = [np.exp(self.get_edge_data(*edge)["log_weight"]) if self.get_edge_data(*edge) else 1. for edge in
                           self.edges]

        nx.draw_networkx_edges(self, positions, alpha=alpha_for_edges, node_size=node_size)
        edge_labels = {(s, t): round(np.exp(w), 2) for (s, t, w) in self.log_weighted_edges}
        nx.draw_networkx_edge_labels(self, positions, edge_labels, label_pos=0.25)

        # filter different types of nodes
        sum_nodes = [node for node in self.nodes if isinstance(node, SumUnit)]
        sum_node_colors = [node_colors[node] for node in sum_nodes]
        product_nodes = [node for node in self.nodes if isinstance(node, ProductUnit)]
        product_node_colors = [node_colors[node] for node in product_nodes]
        leaf_nodes = [node for node in self.nodes if isinstance(node, LeafUnit)]
        leaf_node_colors = [node_colors[node] for node in leaf_nodes]

        # draw sum nodes
        nx.draw_networkx_nodes(self, positions, nodelist=sum_nodes, node_color="#FFFFFF", node_shape="o",
                               edgecolors=sum_node_colors, node_size=node_size)
        nx.draw_networkx_nodes(self, positions, nodelist=sum_nodes, node_color=sum_node_colors, node_shape="+",
                               node_size=node_size * 0.5)

        # draw product nodes
        nx.draw_networkx_nodes(self, positions, nodelist=product_nodes, node_color="#FFFFFF", node_shape="o",
                               edgecolors=product_node_colors, node_size=node_size)
        nx.draw_networkx_nodes(self, positions, nodelist=product_nodes, node_color=product_node_colors, node_shape="x",
                               node_size=node_size * 6 / 11 * 0.5)

        # draw leaf nodes
        labels = {node: node.distribution.abbreviated_symbol for node in leaf_nodes}
        nx.draw_networkx_nodes(self, positions, nodelist=leaf_nodes, node_color="#FFFFFF", node_shape="o",
                               edgecolors=leaf_node_colors, node_size=node_size)

        for node, label in labels.items():
            nx.draw_networkx_labels(self, positions, {node: label}, font_size=16, font_color=node_colors[node],
                                    verticalalignment="center_baseline", horizontalalignment="center")
            nx.draw_networkx_labels(self, position_for_variable_name, {node: node.variables[0].name}, font_size=16,
                                    font_color=node_colors[node], )

        # Iterating over all the axes in the figure
        # and make the Spines Visibility as False
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)

        # adjust plot size
        xticks, xticklabels = plt.xticks()
        xmin = (3 * xticks[0] - xticks[1]) / 2.
        plt.xlim(xmin, max([x for x, _ in positions.values()]) + 1)

        if not plot_inference:
            return

        # plot the results of the queries
        positions_for_results = {node: (x, y + inference_result_offset) for node, (x, y) in positions.items()}
        inference_labels = {node: inference_representation(node) for node in self.nodes if
                            node.result_of_current_query is not None}
        nx.draw_networkx_labels(self, positions_for_results, inference_labels, font_size=8, font_color="black",
                                verticalalignment="center_baseline", horizontalalignment="center")

    def nodes_weights(self) -> dict:
        """
        :return: dict with keys as nodes and values as list of all the log_weights for the node.
        """
        node_weights = {hash(self.root): [1]}
        seen_nodes = set()
        seen_nodes.add(hash(self.root))

        to_visit_nodes = queue.Queue()

        to_visit_nodes.put(self.root)
        while not to_visit_nodes.empty():
            node = to_visit_nodes.get()
            succ_iter = self.successors(node)
            for succ in succ_iter:
                if self.has_edge(node, succ):
                    weight = self.get_edge_data(node, succ).get("weight", 1)
                    node_weights[hash(succ)] = [old * weight for old in node_weights[hash(node)]] + node_weights.get(
                        hash(succ), [])
                    if hash(succ) not in seen_nodes:
                        seen_nodes.add(hash(succ))
                        to_visit_nodes.put(succ)
        return node_weights

    def replace_discrete_distribution_with_deterministic_sum(self):
        """
        splits the distribution into sum unit with all the discrete possibilities as leaf.
        """
        old_leafs = self.leaves
        for leaf in old_leafs:

            if isinstance(leaf, UnivariateDiscreteLeaf):
                leaf: UnivariateDiscreteLeaf
                sum_leaf = leaf.as_deterministic_sum()
                old_predecessors = list(self.predecessors(leaf))
                for predecessor in old_predecessors:
                    weight = self.get_edge_data(predecessor, leaf).get("log_weight", -1)
                    if weight == -1:
                        predecessor.add_subcircuit(sum_leaf)
                    else:
                        predecessor.add_subcircuit(sum_leaf, log_weight=weight)
                    self.remove_edge(predecessor, leaf)
                self.remove_node(leaf)

    def translate(self, translation: Dict[Variable, float]):
        for leaf in self.leaves:
            if any(v.is_numeric for v in leaf.variables):
                leaf.distribution.translate(translation)

    def scale(self, scale: Dict[Variable, float]):
        for leaf in self.leaves:
            if any(v.is_numeric for v in leaf.variables):
                leaf.distribution.scale(scale)

class ShallowProbabilisticCircuit(ProbabilisticCircuit):
    """
    class for PC in shallow form, sum unit as root followed by product units which only have leafs as children.
    """

    @classmethod
    def from_probabilistic_circuit(cls, probabilistic_circuit: ProbabilisticCircuit):
        """
        Initialization function, to input a PC to create its shallow version.
        """
        result = cls()
        shallow_pc = probabilistic_circuit.__copy__()
        cls.shallowing(result, node=shallow_pc.root, presucc=None)
        result.add_nodes_from(shallow_pc.nodes)
        result.add_edges_from(shallow_pc.edges)
        result.add_weighted_edges_from(shallow_pc.log_weighted_edges)
        return result

    def shallowing(self, node: Unit, presucc: Unit | None):
        """
        This function transforms the PC into it shallow form, in place.
        This function uses recursion and need to be called on the root of the PC.
        :node: the Node in focus to be shallowed
        :presucc: the predecessor of the node of before shallowing.
        """
        probabilistic_circuit = node.probabilistic_circuit
        succ_list: List = list(probabilistic_circuit.successors(node))
        if not isinstance(node, SumUnit) and not isinstance(node, ProductUnit):
            sum_unit = SumUnit()
            product_unit = ProductUnit()
            probabilistic_circuit.add_node(sum_unit)
            probabilistic_circuit.add_node(product_unit)
            probabilistic_circuit.add_edge(product_unit, node)
            probabilistic_circuit.add_edge(sum_unit, product_unit, log_weight=0.)
            if presucc is not None:
                data = probabilistic_circuit.get_edge_data(presucc, node, {"weight": 1})
                probabilistic_circuit.add_edge(presucc, sum_unit, **data)
                probabilistic_circuit.remove_edge(presucc, node)
            return
        elif isinstance(node, SumUnit):
            for succ in succ_list:
                self.shallowing(succ, presucc=node)
            new_succ_list = list(probabilistic_circuit.successors(node))
            for sum_succ in new_succ_list:
                first_weight = probabilistic_circuit.get_edge_data(node, sum_succ, {"log_weight": 0}).get("log_weight", 0)
                for succ_succ in list(probabilistic_circuit.successors(sum_succ)):
                    second_weight = probabilistic_circuit.get_edge_data(sum_succ, succ_succ, {"log_weight": 0}).get(
                        "weight", 1)
                    probabilistic_circuit.add_edge(node, succ_succ, log_weight=first_weight + second_weight)
                probabilistic_circuit.remove_edge(node, sum_succ)
                if len(list(probabilistic_circuit.predecessors(sum_succ))) == 0:
                    self.remove_node_and_successor_structure(sum_succ)
            return
        elif isinstance(node, ProductUnit):
            for succ in succ_list:
                self.shallowing(succ, presucc=node)
            new_succ_list = list(probabilistic_circuit.successors(node))
            combination_li = list()
            sum_unit = SumUnit()
            probabilistic_circuit.add_node(sum_unit)
            if presucc is not None:
                data = probabilistic_circuit.get_edge_data(presucc, node, {"weight": 1})
                probabilistic_circuit.add_edge(presucc, sum_unit, **data)
                probabilistic_circuit.remove_edge(presucc, node)
                if len(list(probabilistic_circuit.predecessors(node))) == 0:
                    probabilistic_circuit.remove_node(node)
            elif presucc is None:
                # None only happen if this Instance is root
                probabilistic_circuit.remove_node(node)
            for sum_succ in new_succ_list:
                pro_li = []
                for pro_succ in list(probabilistic_circuit.successors(sum_succ)):
                    data = probabilistic_circuit.get_edge_data(sum_succ, pro_succ, {"weight": 1})
                    pro_li.append((pro_succ, data.get("weight", 1)))
                combination_li.append(pro_li)
            for combination in itertools.product(*combination_li):
                product_unit = ProductUnit()
                probabilistic_circuit.add_node(product_unit)
                total_weight = 1
                for pro_tuple in combination:
                    under_node, weight = pro_tuple[0], pro_tuple[1]
                    total_weight *= weight
                    under_succ_li = probabilistic_circuit.successors(under_node)
                    for under_succ in under_succ_li:
                        data = probabilistic_circuit.get_edge_data(under_node, under_succ, {"log_weight": 0.})
                        probabilistic_circuit.add_edge(product_unit, under_succ, **data)
                probabilistic_circuit.add_edge(sum_unit, product_unit, log_weight=total_weight)
            for sum_succ in new_succ_list:
                if len(list(probabilistic_circuit.predecessors(sum_succ))) == 0:
                    self.remove_node_and_successor_structure(sum_succ)
            return

        else:
            raise TypeError(f"{type(node)} is not supported")

    def events_of_higher_density_product(self, other: Self, own_pro_unit, other_pro_unit, tolerance: float = 10e-8):
        """
        Construct E_p of a product unit in a shallow context.
        :own_pro_unit: product unit which is part of E_p
        :other: other product unit which is part of E_p
        :tolerance: float as how close to zero is zero, because of imprecision.
        """
        # supp_own = own_pro_unit.support
        # supp_other = other_pro_unit.support
        # intersection: Event = (supp_own & supp_other)
        own_copy = self.subgraph_of(own_pro_unit)
        other_copy = other.subgraph_of(other_pro_unit)
        intersection = own_copy.support & other_copy.support

        if intersection.is_empty():
            return Event()

        center = np.array([assignment.simple_sets[0].center() for variable, assignment in
                           intersection.simple_sets[0].items()]).reshape(1, -1)

        likelihood_own = self.likelihood(center)
        likelihood_other = other.likelihood(center)
        diff = likelihood_own - likelihood_other
        if diff > tolerance:
            return intersection
        else:
            return Event()

    def events_of_higher_density_sum(self, other: Self, tolerance: float = 10e-8):
        """
        Construct E_p of a sum unit in a shallow context.
        :other: the other Root shallow PC node to create the E_p
        :tolerance: float as how close to zero is zero, because of imprecisions
        """
        progress_bar = tqdm.tqdm(total=len(self.root.subcircuits) * len(other.root.subcircuits))
        result = self.support - other.support
        for own_prod, other_prod in itertools.product(self.root.subcircuits, other.root.subcircuits):
            result |= self.events_of_higher_density_product(other, own_prod, other_prod, tolerance)
            progress_bar.update()
        return result

    def l1(self, other: Self, tolerance: float = 10e-8) -> float:
        """
        The L1 metric between shallow Circuits are calculated.
        It is important, that before the shallowing the PC replace_discrete_distribution_with_deterministic_sum called on.´
        :other: the other shallow PC which the L1 metric is calculated
        :tolerance: float as how close to zero is zero, because of imprecision for the Creation of E_p.
        """
        e = self.events_of_higher_density_sum(other, tolerance)
        p_e = self.probability(e)
        q_e = other.probability(e)

        return 2 * (p_e - q_e)

    def remove_node_and_successor_structure(self, node: Unit):
        """
        This is an assist function for pruning disconnected subgraphs from the PC.
        :node: the node that needs to be checked if to be pruned and its children.
        """
        probabilistic_circuit = node.probabilistic_circuit
        succ_list: List = list(probabilistic_circuit.successors(node))
        probabilistic_circuit.remove_node(node)
        for succ in succ_list:
            if len(list(probabilistic_circuit.predecessors(succ))) == 0:
                self.remove_node_and_successor_structure(succ)


class UnivariateLeaf(LeafUnit):

    @property
    def variable(self) -> Variable:
        return self.distribution.variables[0]


class UnivariateContinuousLeaf(UnivariateLeaf):
    distribution: Optional[ContinuousDistribution]

    def log_truncated_of_simple_event_in_place(self, event: SimpleEvent):
        return self.univariate_log_truncated_of_simple_event_in_place(event[self.variable])

    def univariate_log_truncated_of_simple_event_in_place(self, event: Interval):
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

        # calculate the truncated distribution as sum unit
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
            result.add_subcircuit(UnivariateDiscreteLeaf(self.distribution.__class__(self.variable,
                                                                                     MissingDict(float, {element: 1.})),
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
            probability = mixture.probability_of_simple_event(
                SimpleEvent({variable: element}))
            if isinstance(element, SimpleInterval):
                element = element.lower
            probabilities[hash(element)] = probability

        distribution_class = IntegerDistribution if isinstance(variable, Integer) else SymbolicDistribution
        distribution = distribution_class(variable, probabilities)
        return cls(distribution)


def leaf(distribution: UnivariateDistribution, probabilistic_circuit: Optional[ProbabilisticCircuit] = None) -> UnivariateLeaf:
    """
    Factory that creates the correct leaf from a distribution.

    :return: The leaf.
    """
    if isinstance(distribution.variable, Continuous):
        return UnivariateContinuousLeaf(distribution, probabilistic_circuit)
    else:
        return UnivariateDiscreteLeaf(distribution, probabilistic_circuit)