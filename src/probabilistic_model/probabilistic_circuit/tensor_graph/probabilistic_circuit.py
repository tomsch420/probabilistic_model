from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, Any, List, Tuple, Self, Dict

import numpy as np
import rustworkx as rx
from random_events.interval import Interval, SimpleInterval
from random_events.product_algebra import Event, SimpleEvent
from random_events.sigma_algebra import AbstractSimpleSet
from random_events.variable import Variable, Continuous
from scipy.special import logsumexp
from sortedcontainers import SortedSet


@dataclass
class Layer:
    index: Optional[int] = field(init=False, default=None)
    probabilistic_circuit: Optional[ProbabilisticCircuit] = field(kw_only=True, default=None, repr=False)
    result_of_current_query: Optional[Any] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.probabilistic_circuit:
            self.probabilistic_circuit.add_layer(self, check_exists=False)

    @property
    def number_of_nodes(self) -> int:
        raise NotImplementedError()

    @property
    def variables(self) -> SortedSet[Variable]:
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        raise NotImplementedError()

    def log_conditional_of_simple_event_in_place(self, simple_event: SimpleEvent):
        raise NotImplementedError()


@dataclass
class InnerLayer(Layer):

    @property
    def child_layers(self) -> List[Layer]:
        return self.probabilistic_circuit.graph.successors(self.index)

    def add_child_layer(self, layer: Layer, data: np.array):
        self.probabilistic_circuit.add_edge(self, layer, data)

    @property
    def variables(self) -> SortedSet[Variable]:
        return SortedSet().union(*[cl.variables for cl in self.child_layers])

    def __deepcopy__(self, memo):
        return self.__class__()

@dataclass
class SumLayer(InnerLayer):

    def add_child_layer(self, layer: Layer, data: np.array):
        """
        :param layer:
        :param data: The logarithmic weights with shape (#self.nodes, #layer.nodes)
        :return:
        """
        super().add_child_layer(layer, data)

    @property
    def log_weights(self) -> List[np.array]:
        """
        The logarithmic weights associated with this layer.
        The weights have shape (#child_layers, #nodes, #child_layer_nodes)

        :return: The logarithmic weights associated with this layer.
        """
        return [self.probabilistic_circuit.graph.get_edge_data(self.index, cl.index) for cl in self.child_layers]

    @property
    def log_weighted_child_layers(self):
        yield from zip(self.log_weights, self.child_layers)

    @property
    def concatenated_log_weights(self):
        return np.concatenate(self.log_weights, axis=1)

    @property
    def log_normalization_constants(self) -> np.array:
        return logsumexp(self.concatenated_log_weights, axis=1)

    @property
    def number_of_nodes(self) -> int:
        return self.log_weights[0].shape[0]

    def log_conditional_of_simple_event_in_place(self, simple_event: SimpleEvent):
        result = np.full((self.number_of_nodes, len(self.child_layers)), np.nan)

        for index, (log_weights, layer) in enumerate(self.log_weighted_child_layers):

            current_result = np.copy(log_weights)
            reshaped_child_layer_result = layer.result_of_current_query.reshape(-1, 1).T
            assert reshaped_child_layer_result.shape == (1, layer.number_of_nodes)
            current_result += reshaped_child_layer_result
            assert current_result.shape == (self.number_of_nodes, layer.number_of_nodes)
            log_probs = logsumexp(current_result, axis=1)
            assert log_probs.shape == (self.number_of_nodes,)
            result[:, index] = log_probs

        # normalize that
        result = logsumexp(result, axis=1)
        result -= self.log_normalization_constants
        self.result_of_current_query = result
        return result

class ProductLayer(InnerLayer):

    @property
    def edges(self) -> List[np.array]:
        """
        The edges associated with this layer.
        The weights have shape (#child_layers, #nodes, #child_layer_nodes)

        :return: The edges associated with this layer.
        """
        return [self.probabilistic_circuit.graph.get_edge_data(self.index, cl.index) for cl in self.child_layers]

    def add_child_layer(self, layer: Layer, data: np.array):
        """
        Add a child layer to the product layer.

        :param layer: The layer to add.
        :param data: The edges of the product layer.
        """
        super().add_child_layer(layer, data)


@dataclass
class InputLayer(Layer):
    ...


@dataclass
class UnivariateInputLayer(InputLayer):
    variable: Variable

    @property
    def variables(self) -> SortedSet[Variable]:
        return SortedSet([self.variable])

    def log_conditional_of_simple_event_in_place(self, simple_event: SimpleEvent):
        return self.univariate_log_conditional_of_simple_event_in_place(simple_event[self.variable])

    def univariate_log_conditional_of_simple_event_in_place(self, simple_set: AbstractSimpleSet):
        raise NotImplementedError()


@dataclass
class ContinuousInputLayer(UnivariateInputLayer):
    variable: Continuous

    def univariate_log_conditional_of_simple_event_in_place(self, interval: Interval):
        """
        Condition this distribution on a simple event in-place but use sum units to create conditions on composite
        intervals.
        :param interval: The simple interval to condition on.
        """

        # if it is a simple truncation
        if len(interval.simple_sets) == 1:
            self.univariate_log_conditional_of_simple_interval_in_place(interval.simple_sets[0])
            return self
        raise NotImplementedError()


    def univariate_log_conditional_of_simple_interval_in_place(self, simple_interval: SimpleInterval):
        raise NotImplementedError()


@dataclass
class DiracDeltaLayer(ContinuousInputLayer):
    location: np.array
    density_cap: np.array
    tolerance: Optional[np.array] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.tolerance is None:
            self.tolerance = np.full_like(self.location, 1e-6)

    def univariate_log_conditional_of_simple_interval_in_place(self, simple_interval: SimpleInterval):

        # make condition for the inclusion of left and right boundary
        contained_left = self.location > (simple_interval.lower - self.tolerance)
        contained_right = self.location < (simple_interval.upper + self.tolerance)
        condition_both = (contained_left & contained_right).nonzero()[0]
        impossible_condition = (~(contained_left & contained_right)).nonzero()[0]

        # create the result
        self.result_of_current_query = np.full_like(self.location, -np.inf)
        self.result_of_current_query[condition_both] = 0.

        # invalidate the impossible nodes in this layer
        self.location[impossible_condition] = np.nan

    def __deepcopy__(self, memo):
        return DiracDeltaLayer(self.variable, np.copy(self.location), np.copy(self.density_cap), np.copy(self.tolerance))

    @property
    def number_of_nodes(self) -> int:
        return len(self.location)

@dataclass
class ProbabilisticCircuit:
    graph: rx.PyDAG[Layer] = field(default_factory=rx.PyDAG[Layer])

    @property
    def root(self) -> Layer:
        """
        The root of the circuit is the layer with in-degree 0.
        This is the output layer, that will perform the final computation.

        :return: The root of the circuit.
        """
        possible_roots = [layer for layer in self.layers if self.graph.in_degree(layer.index) == 0]
        if len(possible_roots) > 1:
            raise ValueError(f"More than one root found. Possible roots are {possible_roots}")
        return possible_roots[0]

    @property
    def layers(self) -> List[Layer]:
        return self.graph.nodes()

    def bottom_up_layers(self) -> List[Layer]:
        return [self.graph.get_node_data(i) for i in reversed(rx.topological_sort(self.graph))]

    def add_layer(self, layer: Layer, check_exists=True):
        if not check_exists or id(layer.probabilistic_circuit) != id(self):
            layer.index = self.graph.add_node(layer)
            layer.probabilistic_circuit = self

    def add_edge(self, parent: Layer, child: Layer, data: Optional[np.ndarray] = None):
        self.add_layer(parent)
        self.add_layer(child)
        self.graph.add_edge(parent.index, child.index, data)

    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        result = copy.deepcopy(self)
        return result.log_conditional_in_place(event)

    def log_conditional_in_place(self, event: Event) -> Tuple[Optional[Self], float]:
        simple_sets = event.simple_sets
        if len(simple_sets) == 0:
            self.graph.remove_nodes_from(self.graph.node_indices())
            return None, -np.inf

        if len(simple_sets) == 1:
            result = self.log_conditional_of_simple_event_in_place(simple_sets[0])
            return result

        raise NotImplementedError

    def log_conditional_of_simple_event_in_place(self, simple_event: SimpleEvent) -> Tuple[Optional[Self], float]:
        layers = self.bottom_up_layers()
        for layer in layers:
            layer.log_conditional_of_simple_event_in_place(simple_event)


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

        return self, layer.result_of_current_query


    def __deepcopy__(self, memo):
        result = self.__class__()
        layer_remap: Dict[int, Layer] = dict()

        for layer in self.layers:
            layer_remap[layer.index] = copy.deepcopy(layer)
            result.add_layer(layer_remap[layer.index])

        for p, c in self.graph.edge_list():
            data = self.graph.get_edge_data(p, c)
            result.add_edge(layer_remap[p], layer_remap[c], np.copy(data))
        return result