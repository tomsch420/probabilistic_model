from typing import List

from jax import numpy as jnp
from typing_extensions import Type, Tuple

from .inner_layer import NXConverterLayer
from .input_layer import ContinuousLayerWithFiniteSupport
from ..nx.distributions import UniformDistribution
from .utils import simple_interval_to_open_array
import tqdm


class UniformLayer(ContinuousLayerWithFiniteSupport):
    """
    A layer that represents uniform distributions over a single variable.
    """

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        return UniformDistribution,

    def validate(self):
        assert self.lower.shape == self.upper.shape, "The shapes of the lower and upper bounds must match."

    @property
    def number_of_nodes(self) -> int:
        return len(self.lower)

    def log_pdf_value(self) -> jnp.array:
        """
        Calculate the log-density of the uniform distribution.
        """
        return -jnp.log(self.upper - self.lower)

    def log_likelihood_of_nodes(self, x: jnp.array) -> jnp.array:
        return jnp.where(self.included_condition(x), self.log_pdf_value(), -jnp.inf)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[UniformDistribution],
                                                         child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> \
            NXConverterLayer:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}

        variable = nodes[0].variable

        intervals = jnp.vstack([simple_interval_to_open_array(node.interval) for node in
                                 (tqdm.tqdm(nodes, desc=f"Creating uniform layer for variable {variable.name}")
                                  if progress_bar else nodes)])

        result = cls(nodes[0].probabilistic_circuit.variables.index(variable), intervals)
        return NXConverterLayer(result, nodes, hash_remap)


    def __deepcopy__(self):
        return self.__class__(self.variables[0].item(), self.interval.copy())
