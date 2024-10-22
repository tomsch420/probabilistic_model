from __future__ import annotations

import itertools
import math
from abc import abstractmethod, ABC
from functools import cached_property

import networkx as nx
import numpy as np
from random_events.product_algebra import VariableMap, SimpleEvent, Event
from random_events.set import SetElement
from random_events.variable import Variable, Symbolic
from sortedcontainers import SortedSet

from typing_extensions import List, Optional, Any, Self, Dict, Tuple, Iterable, TYPE_CHECKING

from probabilistic_model.error import IntractableError
from probabilistic_model.probabilistic_model import ProbabilisticModel, OrderType, CenterType, MomentType
from random_events.utils import SubclassJSONSerializer


if TYPE_CHECKING:
    from probabilistic_model.probabilistic_circuit.nx.distributions import UnivariateDistribution


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
    @cached_property
    def support(self) -> Event:
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
    def leaves(self) -> List[UnivariateDistribution]:
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

        # gather all weighted and non-weighted edges from the subgraph
        weighted_edges = []
        normal_edges = []


        # TODO: i think i have better methods for this by now
        for edge in subgraph.edges:
            edge_ = subgraph.edges[edge]

            if "weight" in edge_.keys():
                weight = edge_["weight"]
                weighted_edges.append((*edge, weight))
            else:
                normal_edges.append(edge)

        self.probabilistic_circuit.add_nodes_from(subgraph.nodes())
        self.probabilistic_circuit.add_edges_from(normal_edges)
        self.probabilistic_circuit.add_weighted_edges_from(weighted_edges)

    @abstractmethod
    def is_deterministic(self) -> bool:
        """
        Calculate if this unit is deterministic or not.
        """
        raise NotImplementedError

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

    def log_conditional(self, event: Event) -> Tuple[Optional[Unit], float]:
        """
        Construct the conditional circuit from an event.
        The event is not required to be disjoint.
        If it is not disjoint, the probability of the event is not correct, but the conditional distribution is.

        :param event: The event to condition on.
        :return: The conditional circuit and log(p(event))
        """

        # skip trivial case
        if event.is_empty():
            return None, -np.inf

        # if the event is easy, don't create a proxy node
        elif len(event.simple_sets) == 1:
            result = self.log_conditional_of_simple_event_in_place(event.simple_sets[0])
            if result is None:
                return self.impossible_condition_result
            return result, result.result_of_current_query

        # construct the proxy node
        result = SumUnit(self.probabilistic_circuit)
        total_probability = 0

        for simple_event in event.simple_sets:

            # reset cache
            self.reset_result_of_current_query()

            # add the conditional distribution of the simple event in this circuit
            conditional, log_probability = self.log_conditional_of_simple_event(simple_event,
                                                                                probabilistic_circuit)

            # skip if impossible
            if log_probability == -np.inf:
                continue

            probability = np.exp(log_probability)

            total_probability += probability
            result.add_subcircuit(conditional, probability)

        if total_probability == 0:
            return self.impossible_condition_result

        result.normalize()

        return result, np.log(total_probability)

    @abstractmethod
    def log_conditional_of_simple_event_in_place(self, event: SimpleEvent) -> Self:
        """
        Calculate the conditional circuit from a simple event in-place.

        :param event: The simple event to condition on.
        """
        raise NotImplementedError

    def reset_result_of_current_query(self):
        """
        Reset the result of the current query recursively.
        If a subcircuit has a retested result, it will not recurse in that subcircuit.
        """
        self.result_of_current_query = None
        for subcircuit in self.subcircuits:
            if subcircuit.result_of_current_query is not None:
                subcircuit.reset_result_of_current_query()

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

    def plot_structure(self):
        """
        Plot the structure of the circuit.
        # TODO make it more fancy
        """

        # create the subgraph with this node as root
        subgraph = nx.subgraph(self.probabilistic_circuit, list(nx.descendants(self.probabilistic_circuit, self)) +
                               [self])

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

    def reset_cached_properties(self):
        """
        Reset all cached properties in this circuit.
        """
        self.reset_result_of_current_query()
        for subcircuit in self.subcircuits:
            subcircuit.reset_cached_properties()


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

    def is_deterministic(self) -> bool:
        return True

    def log_likelihood(self, events: np.array):
        self.result_of_current_query = self.distribution.log_likelihood(events)

    def cdf(self, events: np.array):
        self.result_of_current_query = self.distribution.cdf(events)

    def probability_of_simple_event(self, event: SimpleEvent):
        self.result_of_current_query = self.distribution.probability_of_simple_event(event)

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
        return sum([weight * subcircuit.result_of_current_query(*args, **kwargs)
                    for weight, subcircuit in self.weighted_subcircuits])

    @cached_property
    def support(self) -> Event:
        support = self.subcircuits[0].support
        for subcircuit in self.subcircuits[1:]:
            support |= subcircuit.support
        return support

    @property
    def weights(self) -> np.array:
        """
        :return: The weights of the subcircuits.
        """
        return np.array([weight for weight, _ in self.weighted_subcircuits])

    def log_likelihood(self, events: np.array) -> np.array:
        result = np.zeros(len(events))
        for weight, subcircuit in self.weighted_subcircuits:
            subcircuit_likelihood = np.exp(subcircuit.log_likelihood(events))
            result += weight * subcircuit_likelihood
        self.result_of_current_query = np.log(result)

     
    def cdf(self, events: np.array) -> np.array:
        result = np.zeros(len(events))
        for weight, subcircuit in self.weighted_subcircuits:
            subcircuit_cdf = subcircuit.cdf(events)
            result += weight * subcircuit_cdf
        return result

     
    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        return sum([weight * subcircuit.probability_of_simple_event(event) for weight, subcircuit in
                    self.weighted_subcircuits])

     
    def log_conditional_of_simple_event(self, event: SimpleEvent,
                                        probabilistic_circuit: ProbabilisticCircuit) -> Tuple[Optional[Self], float]:
        # initialize result
        result = self.empty_copy()
        probabilistic_circuit.add_node(result)

        # for every weighted subcircuit
        for weight, subcircuit in self.weighted_subcircuits:

            # condition the subcircuit
            conditional, subcircuit_log_probability = subcircuit.log_conditional_of_simple_event(event, probabilistic_circuit=probabilistic_circuit)

            # skip impossible subcircuits
            if conditional is None:
                continue

            subcircuit_probability = np.exp(subcircuit_log_probability)
            # add subcircuit
            result.add_subcircuit(conditional, weight * subcircuit_probability)

        # check if the result is valid
        total_probability = sum(result.weights)

        if total_probability == 0:
            return self.impossible_condition_result

        # normalize probabilities
        result.normalize()

        return result, np.log(total_probability)

     
    def sample(self, amount: int) -> np.array:
        """
        Sample from the sum node using the latent variable interpretation.
        """
        weights, subcircuits = self.weights, self.subcircuits
        counts = np.random.multinomial(amount, pvals=weights)

        # sample from the children
        result = self.subcircuits[0].sample(counts[0].item())
        for amount, subcircuit in zip(counts[1:], self.subcircuits[1:]):
            result = np.concatenate((result, subcircuit.sample(amount)))
        return result

     
    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        # create a map for orders and centers
        order_of_self = self.filter_variable_map_by_self(order)
        center_of_self = self.filter_variable_map_by_self(center)

        # initialize result
        result = VariableMap({variable: 0 for variable in order_of_self})

        # for every weighted child
        for weight, subcircuit in self.weighted_subcircuits:

            # calculate the moment of the child
            sub_circuit_moment = subcircuit.moment(order_of_self, center_of_self)

            # add up the linear combination of the child moments
            for variable, moment in sub_circuit_moment.items():
                result[variable] += weight * moment

        return result

     
    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        # TODO check with multiple parents
        # if this node has no variables that are required in the marginal, remove it.
        if set(self.variables).intersection(set(variables)) == set():
            return None

        result = self.empty_copy()

        # propagate to sub-circuits
        for weight, subcircuit in self.weighted_subcircuits:
            result.add_subcircuit(subcircuit.marginal(variables), weight)
        return result

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.weighted_subcircuits == other.weighted_subcircuits

    def to_json(self):
        return {**super().to_json(), "weighted_subcircuits": [(weight, subcircuit.to_json()) for weight, subcircuit in
                                                              self.weighted_subcircuits]}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        result = cls()
        for weight, subcircuit_data in data["weighted_subcircuits"]:
            subcircuit = Unit.from_json(subcircuit_data)
            result.mount(subcircuit)
            result.probabilistic_circuit.add_edge(result, subcircuit, weight=weight)
        return result

    def __copy__(self):
        result = self.empty_copy()
        for weight, subcircuit in self.weighted_subcircuits:
            copied_subcircuit = subcircuit.__copy__()
            result.mount(copied_subcircuit)
            result.probabilistic_circuit.add_edge(result, copied_subcircuit, weight=weight)
        return result

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
        # for every unique combination of subcircuits
        for subcircuit_a, subcircuit_b in itertools.combinations(self.subcircuits, 2):
            # check if they intersect
            if not subcircuit_a.support.intersection_with(subcircuit_b.support).is_empty():
                return False

        # if none intersect, the subcircuit is deterministic
        return True

     
    def log_mode(self) -> Tuple[Event, float]:

        if not self.is_deterministic():
            raise IntractableError("The mode of a non-deterministic sum unit cannot be calculated efficiently.")

        modes = []
        log_likelihoods = []

        # gather all modes from the children
        for weight, subcircuit in self.weighted_subcircuits:
            mode, log_likelihood = subcircuit.log_mode()
            modes.append(mode)
            log_likelihoods.append(np.log(weight) + log_likelihood)

        # get the most likely result
        maximum_log_likelihood = max(log_likelihoods)
        result = Event()

        # gather all results that are maximum likely
        for mode, likelihood in zip(modes, log_likelihoods):
            if likelihood == maximum_log_likelihood:
                result |= mode

        return result, maximum_log_likelihood

    def subcircuit_index_of_samples(self, samples: np.array) -> np.array:
        """
        :return: the index of the subcircuit where p(sample) > 0 and None if p(sample) = 0 for all subcircuits.
        """
        result = np.full(len(samples), np.nan)
        for index, subcircuit in enumerate(self.subcircuits):
            likelihood = subcircuit.log_likelihood(samples)
            result[likelihood > -np.inf] = index
        return result


class ProductUnit(Unit):
    """
    Decomposable Product Units for Probabilistic Circuits
    """

    def forward(self, *args, **kwargs):
        return math.prod([subcircuit.result_of_current_query(*args, **kwargs)
                             for subcircuit in self.subcircuits])

    def __repr__(self):
        return "∗"

    @property
    def variables(self) -> SortedSet:
        result = SortedSet()
        for subcircuit in self.subcircuits:
            result = result.union(subcircuit.variables)
        return result

    @cached_property
    def support(self) -> Event:
        support = self.subcircuits[0].support
        for subcircuit in self.subcircuits[1:]:
            support &= subcircuit.support
        return support

    def add_subcircuit(self, subcircuit: Unit, mount: bool = True):
        """
        Add a subcircuit to the subcircuits of this unit.

        :param subcircuit: The subcircuit to add.
        :param mount: If the subcircuit should be mounted to this units pc instance.
        """
        if mount:
            self.mount(subcircuit)
        self.probabilistic_circuit.add_edge(self, subcircuit)

    def log_likelihood(self, events: np.array) -> np.array:
        result = np.zeros(len(events))
        for subcircuit in self.subcircuits:
            result += subcircuit.result_of_current_query
        self.result_of_current_query = result
        return result

     
    def cdf(self, events: np.array) -> np.array:
        variables = self.variables
        result = np.zeros(len(events))
        for subcircuit in self.subcircuits:
            subcircuit_variables = subcircuit.variables
            variable_indices_in_events = np.array([variables.index(variable) for variable in subcircuit_variables])
            result += subcircuit.cdf(events[:, variable_indices_in_events])
        return result

     
    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        result = 1.
        for subcircuit in self.subcircuits:
            result *= subcircuit.probability_of_simple_event(event)
            if result == 0:
                return 0.
        return result

    def is_deterministic(self) -> bool:
        return True

    def is_decomposable(self):
        for index, subcircuit in enumerate(self.subcircuits):
            variables = subcircuit.variables
            for subcircuit_ in self.subcircuits[index+1:]:
                if len(set(subcircuit_.variables).intersection(set(variables))) > 0:
                    return False
        return True

     
    def log_mode(self) -> Tuple[Event, float]:

        # initialize mode and log likelihood
        mode, log_likelihood = self.subcircuits[0].log_mode()

        # gather all modes from the children
        for subcircuit in self.subcircuits[1:]:
            subcircuit_mode, subcircuit_log_likelihood = subcircuit.log_mode()
            mode &= subcircuit_mode
            log_likelihood += subcircuit_log_likelihood

        return mode, log_likelihood

     
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

     
    def sample(self, amount: int) -> np.array:

        # load on variables
        variables = self.variables

        # list for the samples content in the same order as self.variables
        rearranged_samples = np.full((amount, len(variables)), np.nan)

        # for every subcircuit
        for subcircuit in self.subcircuits:

            # calculate the indices of the variables
            variable_indices_in_events = np.array([variables.index(variable) for variable in subcircuit.variables])

            # sample from the subcircuit
            sample_subset = subcircuit.sample(amount)
            np.random.shuffle(sample_subset)
            rearranged_samples[:, variable_indices_in_events] = sample_subset

        return rearranged_samples

     
    def moment(self, order: OrderType, center: CenterType) -> MomentType:

        # initialize result
        result = VariableMap()

        for subcircuit in self.subcircuits:
            # calculate the moment of the child
            child_moment = subcircuit.moment(order, center)

            result = VariableMap({**result, **child_moment})

        return result

     
    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:

        # if this node has no variables that are required in the marginal, remove it.
        if set(self.variables).intersection(set(variables)) == set():
            return None

        result = self.empty_copy()

        # propagate to sub-circuits
        for subcircuit in self.subcircuits:
            marginal = subcircuit.marginal(variables)

            if marginal is None:
                continue

            result.add_subcircuit(marginal)

        return result

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "subcircuits": [subcircuit.to_json() for subcircuit in self.subcircuits]}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        result = cls()
        for subcircuit_data in data["subcircuits"]:
            subcircuit = Unit.from_json(subcircuit_data)
            result.add_subcircuit(subcircuit)
        return result

    def __copy__(self):
        result = self.__class__()
        for subcircuit in self.subcircuits:
            copied_subcircuit = subcircuit.__copy__()
            result.add_subcircuit(copied_subcircuit)
        return result

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
    def variable_to_index_map(self):
        return {variable: index for index, variable in enumerate(self.variables)}

    @property
    def layers(self) -> List[List[Unit]]:
        return list(nx.bfs_layers(self, self.root))

    @property
    def leaves(self) -> List[UnivariateDistribution]:
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
              unit.log_likelihood(events)
        return self.root.result_of_current_query

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        return self.root.probability_of_simple_event(event)

    def log_mode(self) -> Tuple[Event, float]:
        return self.root.log_mode()

    def remove_unreachable_nodes(self, root: Unit):
        """
        Remove all nodes that are not reachable from the root.
        """
        reachable_nodes = nx.descendants(self, root)
        unreachable_nodes = set(self.nodes)  - (reachable_nodes | {root})
        self.remove_nodes_from(unreachable_nodes)

    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        conditional, log_prob = self.root.log_conditional(event)

        if conditional is None:
            return None, -np.inf

        # clean up unreachable nodes
        self.remove_unreachable_nodes(conditional)

        return conditional.probabilistic_circuit, log_prob

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        root = self.root
        result = self.root.marginal(variables)
        if result is None:
            return None
        root.reset_result_of_current_query()
        return result.probabilistic_circuit

    def sample(self, amount: int) -> np.array:
        return self.root.sample(amount)

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        return self.root.moment(order, center)

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
        return self.root.support

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
        new_unweighted_edges = [(new_node_map[source], new_node_map[target]) for source, target in self.unweighted_edges]
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
            node_json = node.empty_copy().to_json()
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

        # gather all weighted and non-weighted edges from the subgraph
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
        # gather all weighted and non-weighted edges from the subgraph
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
        return all(node.is_deterministic() for node in self.nodes if isinstance(node, SumUnit))

    def cdf(self, events: np.array) -> np.array:
        return self.root.cdf(events)

    def add_edges_and_nodes_from_circuit(self, other: Self):
        """
        Add all edges and nodes from another circuit to this circuit.

        :param other: The other circuit to add.
        """
        self.add_nodes_from(other.nodes)
        self.add_edges_from(other.unweighted_edges)
        self.add_weighted_edges_from(other.weighted_edges)

    def plot_structure(self):
        return self.root.plot_structure()