from typing import List, Dict, Any

import jax
from jax import numpy as jnp
from jax.experimental.sparse import BCOO
from typing_extensions import Type, Tuple, Self

from .inner_layer import NXConverterLayer
from .input_layer import ContinuousLayerWithFiniteSupport
from ..nx.distributions import UniformDistribution
from .utils import simple_interval_to_open_array, create_bcoo_indices_from_row_lengths
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

    def log_likelihood_of_nodes_single(self, x: jnp.array) -> jnp.array:
        return jnp.where(self.included_condition(x), self.log_pdf_value(), -jnp.inf)

    def log_likelihood_of_nodes(self, x: jnp.array) -> jnp.array:
        return jax.vmap(self.log_likelihood_of_nodes_single)(x)

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

    def sample_from_frequencies(self, frequencies: jax.Array, key: jax.random.PRNGKey) -> BCOO:
        max_frequency = jnp.max(frequencies)

        # create indices for the sparse result
        indices = create_bcoo_indices_from_row_lengths(frequencies)

        # sample from U(0,1)
        standard_uniform_samples = jax.random.uniform(key, shape=(indices.shape[0], 1))

        # calculate range for each node
        range_per_sample = (self.upper - self.lower).repeat(frequencies).reshape(-1, 1)

        # calculate the right shift for each node
        right_shift_per_sample = self.lower.repeat(frequencies).reshape(-1, 1)

        # apply the transformation to the desired intervals
        samples = standard_uniform_samples * range_per_sample + right_shift_per_sample

        result = BCOO((samples, indices), shape=(self.number_of_nodes, max_frequency, 1), indices_sorted=True,
                      unique_indices=True)
        return result

    def cdf_of_nodes_single(self, x: jnp.array) -> jnp.array:
        return jnp.clip((x - self.lower) / (self.upper - self.lower), 0, 1)

    def moment_of_nodes(self, order: jax.Array, center: jax.Array):
        """
        Calculate the moment of the uniform distribution.
        """
        order = order[self.variables[0]]
        center = center[self.variables[0]]
        pdf_value = jnp.exp(self.log_pdf_value())
        lower_integral_value = (pdf_value * (self.lower - center) ** (order + 1)) / (order + 1)
        upper_integral_value = (pdf_value * (self.upper - center) ** (order + 1)) / (order + 1)
        return (upper_integral_value - lower_integral_value).reshape(-1, 1)

    def __deepcopy__(self):
        return self.__class__(self.variables[0].item(), self.interval.copy())

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(data["variable"], jnp.array(data["interval"]))