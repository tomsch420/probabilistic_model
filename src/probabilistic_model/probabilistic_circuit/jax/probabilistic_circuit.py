from __future__ import annotations

import collections

from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Tuple, Self, List

from .inner_layer import Layer, NXConverterLayer
from ..nx.probabilistic_circuit import ProbabilisticCircuit as NXProbabilisticCircuit
import jax
import tqdm
import networkx as nx


class ProbabilisticCircuit:
    """
    A probabilistic circuit as wrapper for a layered probabilistic model.
    """

    variables: SortedSet
    """
    The variables of the circuit.
    """

    root: Layer
    """
    The root layer of the circuit.
    """

    def __init__(self, variables: SortedSet, root: Layer):
        self.variables = variables
        self.root = root

    def log_likelihood(self, x: jax.Array) -> jax.Array:
        return self.root.log_likelihood_of_nodes(x)[:, 0]

    @classmethod
    def from_nx(cls, pc: NXProbabilisticCircuit, progress_bar: bool = False) -> ProbabilisticCircuit:
        """
        Convert a probabilistic circuit to a layered circuit.
        The result expresses the same distribution as `pc`.

        :param pc: The probabilistic circuit.
        :param progress_bar: Whether to show a progress bar.
        :return: The layered circuit.
        """

        # calculate the depth of each node
        node_to_depth_map = {node: len(path) for node, path in nx.single_source_shortest_path(pc, pc.root).items()}

        # group nodes by depth
        layer_to_nodes_map = {depth: [node for node, n_depth in node_to_depth_map.items() if depth == n_depth] for depth
                              in set(node_to_depth_map.values())}
        reversed_layers_to_nodes_map = dict(reversed(layer_to_nodes_map.items()))

        # create layers from nodes
        child_layers: List[NXConverterLayer] = []
        for layer_index, nodes in (tqdm.tqdm(reversed_layers_to_nodes_map.items(), desc="Creating Layers") if progress_bar
                                                     else reversed_layers_to_nodes_map.items()):

            child_layers = Layer.create_layers_from_nodes(nodes, child_layers, progress_bar)
        root = child_layers[0].layer

        return cls(pc.variables, root)

    @property
    def trainable_parameters(self):
        parameters = []
        for layer in self.root.all_layers():
            parameters.extend(layer.trainable_parameters)
        return parameters