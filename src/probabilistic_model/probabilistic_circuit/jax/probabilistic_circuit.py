from __future__ import annotations

import collections
from typing import Dict, Any

import numpy as np
import optax
from random_events.product_algebra import SimpleEvent
from random_events.utils import SubclassJSONSerializer
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Tuple, Self, List, Optional

from .inner_layer import Layer, NXConverterLayer
from ..nx.probabilistic_circuit import ProbabilisticCircuit as NXProbabilisticCircuit
import jax
import tqdm
import networkx as nx
import jax.numpy as jnp
import equinox as eqx


class ProbabilisticCircuit(SubclassJSONSerializer):
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

    def to_nx(self, progress_bar: bool = True) -> NXProbabilisticCircuit:
        """
        Convert the probabilistic circuit to a networkx graph.

        :param progress_bar: Whether to show a progress bar.
        :return: The networkx graph.
        """
        if progress_bar:
            number_of_edges = self.root.number_of_components
            progress_bar = tqdm.tqdm(total=number_of_edges, desc="Converting to nx")
        else:
            progress_bar = None
        result = NXProbabilisticCircuit()
        self.root.to_nx(self.variables, result, progress_bar)
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["variables"] = [variable.to_json() for variable in self.variables]
        result["root"] = self.root.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variables = SortedSet(Variable.from_json(variable) for variable in data["variables"])
        root = Layer.from_json(data["root"])
        return cls(variables, root)

    def fit_generative(self, data: jax.Array, epochs: int = 100,
                       optimizer: Optional[optax.GradientTransformation] = None) -> None:
        """
        Fit the circuit to the data using generative training with the negative average log-likelihood as loss.

        :param data: The data.
        :param epochs: The number of epochs.
        :param optimizer: The optimizer to use.
        If `None`, the Adam optimizer with a learning rate of 1e-3 is used.
        """

        @eqx.filter_jit
        def loss(p, x):
            ll = p.log_likelihood_of_nodes(x)
            return -jnp.mean(ll)

        if optimizer is None:
            optimizer = optax.adam(1e-3)

        opt_state = optimizer.init(eqx.filter(self.root, eqx.is_inexact_array))

        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")

        for epoch in progress_bar:
            loss_value, grads = eqx.filter_value_and_grad(loss)(self.root, data)

            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(self.root, eqx.is_inexact_array)
            )
            self.root = eqx.apply_updates(self.root, updates)
            progress_bar.set_postfix_str(f"Loss: {loss_value}")

