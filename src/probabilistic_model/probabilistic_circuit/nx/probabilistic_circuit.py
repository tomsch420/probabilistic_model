from __future__ import annotations

import itertools
import math
from abc import abstractmethod

import networkx as nx
import numpy as np
from random_events.product_algebra import VariableMap, SimpleEvent, Event
from random_events.set import SetElement
from random_events.utils import SubclassJSONSerializer
from random_events.variable import Variable, Symbolic
from sortedcontainers import SortedSet
from typing_extensions import List, Optional, Any, Self, Dict, Tuple, Iterable

from probabilistic_model.error import IntractableError
from probabilistic_model.probabilistic_model import ProbabilisticModel, OrderType, CenterType, MomentType


class Unit(SubclassJSONSerializer):
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
            if leaf.variable in new_variables:
                leaf.variable = new_variables[leaf.variable]

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
        :return: The result of an impossible conditional query.
        """
        return None, -np.inf

    def log_conditional_in_place(self, event: Event) -> Optional[Self]:
        """
        Construct the conditional circuit from an event.
        The event is not required to be a disjoint union of simple events.

        However, if it is not a disjoint union, the probability of the event is not correct,
        but the conditional distribution is.

        :param event: The event to condition on.
        :return: The root of the conditional circuit.
        """

        # skip trivial case
        if event.is_empty():
            self.probabilistic_circuit.remove_node(self)
            return None

        # if the event is easy, don't create a proxy node
        elif len(event.simple_sets) == 1:
            result = self.log_conditional_of_simple_event_in_place(event.simple_sets[0])
            return result

        # create a conditional circuit for every simple event
        conditional_circuits = [
            self.probabilistic_circuit.__copy__().root.log_conditional_of_simple_event_in_place(simple_event)
            for simple_event in event.simple_sets]

        # clear this circuit
        [self.probabilistic_circuit.remove_node(node) for node in self.probabilistic_circuit.nodes]

        # filtered out impossible conditionals
        conditional_circuits = [conditional for conditional in conditional_circuits if conditional is not None]

        # if all conditionals are impossible
        if len(conditional_circuits) == 0:
            return None

        # create a new sum unit
        result = SumUnit(self.probabilistic_circuit)

        # add the conditionals to the sum unit
        [result.add_subcircuit(conditional, np.exp(conditional.result_of_current_query)) for conditional in
         conditional_circuits]
        result.log_forward()
        result.normalize()
        return result

    @abstractmethod
    def log_conditional_of_simple_event_in_place(self, event: SimpleEvent) -> Self:
        """
        Calculate the conditional circuit from a simple event in-place.

        :param event: The simple event to condition on.
        """
        raise NotImplementedError

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

    def simplify(self) -> Self:
        """
        Simplify the circuit by removing nodes and redirected edges that have no impact in-place.
        Essentially, this method transforms the circuit into an alternating order of sum and product units.

        :return: The simplified circuit.
        """
        raise NotImplementedError()

    def reset_cached_properties(self):
        """
        Reset all cached properties in this circuit.
        """
        self.reset_result_of_current_query()
        for subcircuit in self.subcircuits:
            subcircuit.reset_cached_properties()

    def log_likelihood(self, *args, **kwargs):
        raise NotImplementedError

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


class LeafUnit(Unit):
    """
    Class for Leaf units.
    """

    distribution: ProbabilisticModel
    """
    The distribution contained in this leaf unit.
    """

    def __init__(self, distribution: ProbabilisticModel,
                 probabilistic_circuit: Optional[ProbabilisticCircuit] = None):
        super().__init__(probabilistic_circuit)
        self.distribution = distribution

    @property
    def variables(self) -> SortedSet:
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
        self.result_of_current_query = self.distribution.support

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
        return [node for node in nx.descendants(self.probabilistic_circuit, self) if
                node.is_leaf]

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def log_forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def moment_forward(self):
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

    def __repr__(self):
        return "+"

    @property
    def weighted_subcircuits(self) -> List[Tuple[float, Unit]]:
        """
        :return: The weighted subcircuits of this unit.
        """
        return [(self.probabilistic_circuit.edges[self, subcircuit]["weight"], subcircuit) for subcircuit in
                self.subcircuits]

    @property
    def variables(self) -> SortedSet:
        return self.subcircuits[0].variables

    @property
    def latent_variable(self) -> Symbolic:
        name = f"{hash(self)}.latent"
        enum_elements = {"EMPTY_SET": -1}
        enum_elements.update({str(hash(subcircuit)): index for index, subcircuit in enumerate(self.subcircuits)})
        domain = SetElement(name, enum_elements)
        return Symbolic(name, domain)

    def add_subcircuit(self, subcircuit: Unit, weight: float, mount: bool = True):
        """
        Add a subcircuit to the subcircuits of this unit.

        .. note::

            This method does not normalize the edges to the subcircuits.


        :param subcircuit: The subcircuit to add.
        :param weight: The weight of the subcircuit.
        :param mount: If the subcircuit should be mounted to the pc of this unit.

        """
        if mount:
            self.mount(subcircuit)
        self.probabilistic_circuit.add_edge(self, subcircuit, weight=weight)

    def forward(self, *args, **kwargs):
        self.result_of_current_query = np.sum([weight * subcircuit.result_of_current_query
                                               for weight, subcircuit in self.weighted_subcircuits], axis=0)

    def log_forward(self, *args, **kwargs):
        self.result_of_current_query = np.log(np.sum([weight * np.exp(subcircuit.result_of_current_query)
                                                      for weight, subcircuit in self.weighted_subcircuits], axis=0))

    moment_forward = forward

    def support(self):
        support = self.subcircuits[0].result_of_current_query
        for subcircuit in self.subcircuits[1:]:
            support |= subcircuit.result_of_current_query
        self.result_of_current_query = support

    @property
    def weights(self) -> np.array:
        """
        :return: The weights of the subcircuits.
        """
        return np.array([weight for weight, _ in self.weighted_subcircuits])

    def log_conditional_of_simple_event_in_place(self, event: SimpleEvent) -> Optional[Self]:
        if len(self.subcircuits) == 0:
            self.probabilistic_circuit.remove_node(self)
            return None
        self.log_forward()
        self.normalize()
        return self

    def sample(self) -> np.array:
        weights, subcircuits = self.weights, self.subcircuits

        # for every sampling request
        for start_index, amount in self.result_of_current_query:

            # calculate the numbers of samples requested from the sub circuits
            counts = np.random.multinomial(amount, pvals=weights)
            total = 0

            # add the sampling requests to the subcircuits
            for count, subcircuit in zip(counts, subcircuits):
                if subcircuit.result_of_current_query is None:
                    subcircuit.result_of_current_query = []
                subcircuit.result_of_current_query.append((start_index + total, count))
                total += count

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.weighted_subcircuits == other.weighted_subcircuits

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        result = cls()
        return result

    def __copy__(self):
        return self.empty_copy()

    def mount_with_interaction_terms(self, other: 'SumUnit', interaction_model: ProbabilisticModel):
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

        own_latent_variable = self.latent_variable
        other_latent_variable = other.latent_variable
        own_subcircuits = self.subcircuits
        other_subcircuits = other.subcircuits

        for own_index, own_subcircuit in zip(own_latent_variable.domain.simple_sets, own_subcircuits):

            # create denominator of weight
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
            self.add_subcircuit(proxy_product_node, p_condition)

            # mount current child on the product proxy
            proxy_product_node.add_subcircuit(own_subcircuit)

            # mount the proxy for the children from other in the product proxy
            proxy_product_node.add_subcircuit(proxy_sum_node)

            for other_index, other_subcircuit in zip(other_latent_variable.domain, other_subcircuits):

                # create numerator of weight
                query = SimpleEvent({other_latent_variable: other_index}).as_composite_set() & condition
                p_query = interaction_model.probability(query)

                # skip iterations that are impossible
                if p_query == 0:
                    continue

                # calculate conditional probability
                weight = p_query / p_condition

                # create edge from proxy to subcircuit
                proxy_sum_node.add_subcircuit(other_subcircuit, weight=weight)

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

        for (own_weight, own_subcircuit), other_subcircuit in zip(self.weighted_subcircuits, other.subcircuits):
            # create proxy nodes for mounting
            proxy_product_node = ProductUnit()
            self.probabilistic_circuit.add_node(proxy_product_node)

            # remove edge to old child and replace it by product proxy
            self.probabilistic_circuit.remove_edge(self, own_subcircuit)
            self.add_subcircuit(proxy_product_node, own_weight)
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
        for weight, subcircuit in self.weighted_subcircuits:

            # if the weight is 0, skip this subcircuit
            if weight == 0:
                # remove the edge
                self.probabilistic_circuit.remove_edge(self, subcircuit)

            # if the simplified subcircuit is of the same type as this
            if type(subcircuit) is type(self):

                # type hinting
                subcircuit: Self

                # mount the children of that circuit directly
                for sub_weight, sub_subcircuit in subcircuit.weighted_subcircuits:
                    new_weight = sub_weight * weight

                    # add an edge to that subcircuit
                    self.add_subcircuit(sub_subcircuit, new_weight, mount=False)

                    # remove the old node
                    self.probabilistic_circuit.remove_node(subcircuit)

    def normalize(self):
        """
        Normalize the weights of the subcircuits such that they sum up to 1 inplace.
        """
        total_weight = sum([weight for weight, _ in self.weighted_subcircuits])
        for subcircuit in self.subcircuits:
            self.probabilistic_circuit.edges[self, subcircuit]["weight"] /= total_weight

    def is_deterministic(self) -> bool:
        """
        :return: If this unit is deterministic or not.
        """
        # for every unique combination of subcircuits
        for subcircuit_a, subcircuit_b in itertools.combinations(self.subcircuits, 2):
            # check if they intersect
            if not subcircuit_a.result_of_current_query.intersection_with(subcircuit_b.result_of_current_query).is_empty():
                return False

        # if none intersect, the subcircuit is deterministic
        return True

    def log_mode(self):
        log_maxima = [np.log(weight) + subcircuit.result_of_current_query[1] for weight, subcircuit in
                      self.weighted_subcircuits]
        log_max = max(log_maxima)
        arg_log_maxima = [subcircuit.result_of_current_query[0] for lm, subcircuit in zip(log_maxima, self.subcircuits)
                       if lm == log_max]
        arg_log_max = arg_log_maxima[0]
        for event in  arg_log_maxima[1:]:
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

    def forward(self, *args, **kwargs):
        self.result_of_current_query = math.prod([subcircuit.result_of_current_query(*args, **kwargs)
                                                  for subcircuit in self.subcircuits])

    def log_forward(self, *args, **kwargs):
        self.result_of_current_query = np.sum([subcircuit.result_of_current_query(*args, **kwargs)
                                               for subcircuit in self.subcircuits], axis=0)

    moment_forward = log_forward

    def __repr__(self):
        return "∗"

    @property
    def variables(self) -> SortedSet:
        result = SortedSet()
        for subcircuit in self.subcircuits:
            result = result.union(subcircuit.variables)
        return result

    def support(self):
        support = self.subcircuits[0].result_of_current_query
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

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        result = 1.
        for subcircuit in self.subcircuits:
            result *= subcircuit.probability_of_simple_event(event)
            if result == 0:
                return 0.
        return result

    def is_decomposable(self):
        for index, subcircuit in enumerate(self.subcircuits):
            variables = subcircuit.variables
            for subcircuit_ in self.subcircuits[index + 1:]:
                if len(set(subcircuit_.variables).intersection(set(variables))) > 0:
                    return False
        return True

    def log_mode(self) -> Tuple[Event, float]:
        ...

    def log_conditional_of_simple_event(self, event: SimpleEvent,
                                        probabilistic_circuit: ProbabilisticCircuit) -> Tuple[Optional[Self], float]:
        # initialize probability
        log_probability = 0.

        # create a new node with new circuit attached to it
        resulting_node = self.empty_copy()
        probabilistic_circuit.add_node(resulting_node)

        for subcircuit in self.subcircuits:

            # get conditional child and probability in pre-order
            conditional_subcircuit, conditional_log_probability = (
                subcircuit.log_conditional_of_simple_event(event, probabilistic_circuit=probabilistic_circuit))

            # if any is 0, the whole probability is 0
            if conditional_subcircuit is None:
                probabilistic_circuit.remove_node(resulting_node)
                return self.impossible_condition_result

            # update probability and children
            resulting_node.add_subcircuit(conditional_subcircuit)
            log_probability += conditional_log_probability

        return resulting_node, log_probability

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


class ProbabilisticCircuit(ProbabilisticModel, nx.DiGraph, SubclassJSONSerializer):
    """
    Probabilistic Circuits as a directed, rooted, acyclic graph.
    """

    def __init__(self):
        super().__init__(None)
        nx.DiGraph.__init__(self)

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
            for unit in layer:
                if unit.is_leaf:
                    unit.log_likelihood(events[:, [variable_to_index_map[variable] for variable in unit.variables]])
                else:
                    unit: InnerUnit
                    unit.log_forward()
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

    def log_mode(self) -> Tuple[Event, float]:
        [unit.log_mode() for layer in reversed(self.layers) for unit in layer]
        return self.root.result_of_current_query

    def remove_unreachable_nodes(self, root: Unit):
        """
        Remove all nodes that are not reachable from the root.
        """
        reachable_nodes = nx.descendants(self, root)
        unreachable_nodes = set(self.nodes) - (reachable_nodes | {root})
        self.remove_nodes_from(unreachable_nodes)

    def log_conditional_in_place(self, event: Event) -> Tuple[Optional[Self], float]:
        new_root = None
        for layer in reversed(self.layers):
            for unit in layer:
                new_root = unit.log_conditional_in_place(event)

        if new_root is None:
            return None, -np.inf

        # clean up unreachable nodes
        self.remove_unreachable_nodes(new_root)

        return self, new_root.result_of_current_query

    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        result = self.__copy__()
        return result.log_conditional_in_place(event)

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        result = None
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    result = unit.marginal(variables)
                else:
                    unit: InnerUnit
                    result = unit.marginal()
        if result is not None:
            self.remove_unreachable_nodes(result)
        return self

    def sample(self, amount: int) -> np.array:
        variable_to_index_map = self.variable_to_index_map

        # initialize the sample arguments
        self.root.result_of_current_query = [(0, amount)]

        # initialize the samples
        samples = np.full((amount, len(variable_to_index_map)), np.nan)

        # forward through the circuit to sample
        for layer in self.layers:
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.sample(samples, variable_to_index_map)
                else:
                    unit: InnerUnit
                    unit.sample()
        return samples

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        variable_to_index_map = self.variable_to_index_map
        for layer in reversed(self.layers):
            for unit in layer:
                if unit.is_leaf:
                    unit: LeafUnit
                    unit.moment(order, center, variable_to_index_map)
                else:
                    unit: InnerUnit
                    unit.forward()
        return MomentType({variable: moment for variable, moment in zip(variable_to_index_map.keys(),
                                                                        self.root.result_of_current_query)})

    def simplify(self) -> Self:
        """
        Simplify the circuit inplace.
        """
        bfs_layers = list(nx.bfs_layers(self, self.root))
        for layer in reversed(bfs_layers):
            for node in layer:
                node.simplify()
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
        return all([subcircuit.is_decomposable() for subcircuit in self.leaves if
                    isinstance(subcircuit, ProductUnit)])

    def __eq__(self, other: 'ProbabilisticCircuit'):
        return self.root == other.root

    def __copy__(self):
        result = self.__class__()
        new_node_map = {node: node.__copy__() for node in self.nodes}
        result.add_nodes_from(new_node_map.values())
        new_unweighted_edges = [(new_node_map[source], new_node_map[target]) for source, target in
                                self.unweighted_edges]
        new_weighted_edges = [(new_node_map[source], new_node_map[target], weight)
                              for source, target, weight in self.weighted_edges]
        result.add_edges_from(new_unweighted_edges)
        result.add_weighted_edges_from(new_weighted_edges)
        return result

    def to_json(self) -> Dict[str, Any]:
        # get super result
        result = super().to_json()

        hash_to_node_map = dict()

        for node in self.nodes:
            node_json = node.to_json()
            hash_to_node_map[hash(node)] = node_json

        unweighted_edges = [(hash(source), hash(target)) for source, target
                            in self.unweighted_edges]
        weighted_edges = [(hash(source), hash(target), weight)
                          for source, target, weight in self.weighted_edges]
        result["hash_to_node_map"] = hash_to_node_map
        result["unweighted_edges"] = unweighted_edges
        result["weighted_edges"] = weighted_edges
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        result = ProbabilisticCircuit()
        hash_remap: Dict[int, Unit] = dict()

        for hash_, node_data in data["hash_to_node_map"].items():
            node = Unit.from_json(node_data)
            hash_remap[int(hash_)] = node
            result.add_node(node)

        for source_hash, target_hash in data["unweighted_edges"]:
            result.add_edge(hash_remap[source_hash], hash_remap[target_hash])

        for source_hash, target_hash, weight in data["weighted_edges"]:
            result.add_edge(hash_remap[source_hash], hash_remap[target_hash], weight=weight)

        return result

    def update_variables(self, new_variables: VariableMap):
        """
        Update the variables of this unit and its descendants.

        :param new_variables: The new variables to set.
        """
        self.root.update_variables(new_variables)

    @property
    def weighted_edges(self):
        """
        :return: All weighted edges of the circuit.
        """
        weighted_edges = []

        for edge in self.edges:
            edge_ = self.edges[edge]

            if "weight" in edge_.keys():
                weight = edge_["weight"]
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

    def add_edges_and_nodes_from_circuit(self, other: Self):
        """
        Add all edges and nodes from another circuit to this circuit.

        :param other: The other circuit to add.
        """
        self.add_nodes_from(other.nodes)
        self.add_edges_from(other.unweighted_edges)
        self.add_weighted_edges_from(other.weighted_edges)

    def plot_structure(self):
        """
        Plot the structure of the circuit.
        # TODO make it more fancy
        """

        # create the subgraph with this node as root
        subgraph = nx.subgraph(self, list(nx.descendants(self, self.root)) + [self])

        # do a layer-wise BFS
        layers = list(nx.bfs_layers(subgraph, [self]))

        # calculate the positions of the nodes
        maximum_layer_width = max([len(layer) for layer in layers])
        positions = {}
        for depth, layer in enumerate(layers):
            number_of_nodes = len(layer)
            positions_in_layer = np.linspace(0, maximum_layer_width, number_of_nodes, endpoint=False)
            positions_in_layer += (maximum_layer_width - len(layer)) / (2 * len(layer))
            for position, node in zip(positions_in_layer, layer):
                positions[node] = (depth, position)

        # draw the edges
        alpha_for_edges = [subgraph.get_edge_data(*edge)["weight"] if subgraph.get_edge_data(*edge) else 1.
                           for edge in subgraph.edges]
        nx.draw_networkx_edges(subgraph, positions, alpha=alpha_for_edges)

        # draw the nodes and labels
        nx.draw_networkx_nodes(subgraph, positions)
        labels = {node: repr(node) for node in subgraph.nodes}
        nx.draw_networkx_labels(subgraph, positions, labels)
        #  plt.show()
