from typing import List, Dict, Any, Optional

import jax
import numpy as np
import random_events
from jax import numpy as jnp
from jax.experimental.sparse import BCOO
from random_events.interval import SimpleInterval
from random_events.variable import Variable
from sortedcontainers import SortedSet
from sqlalchemy.sql.functions import random
from typing_extensions import Type, Tuple, Self

from .inner_layer import NXConverterLayer
from .input_layer import ContinuousLayerWithFiniteSupport
from ..nx.distributions import UniformDistribution
from .utils import simple_interval_to_open_array, create_bcoo_indices_from_row_lengths
import tqdm

from ..nx.probabilistic_circuit import ProbabilisticCircuitMixin, ProbabilisticCircuit as NXProbabilisticCircuit


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

    def sample_from_frequencies(self, frequencies: np.array, result: np.array, start_index = 0):
        # sample from U(0,1)
        standard_uniform_samples = np.random.uniform(size=(sum(frequencies), 1))

        # calculate range for each node
        range_per_sample = (self.upper - self.lower).repeat(frequencies).reshape(-1, 1)

        # calculate the right shift for each node
        right_shift_per_sample = self.lower.repeat(frequencies).reshape(-1, 1)

        # apply the transformation to the desired intervals
        samples = standard_uniform_samples * range_per_sample + right_shift_per_sample

        result[start_index:start_index + len(samples), self.variables] = samples

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

    def log_conditional_from_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, jax.Array]:
        probabilities = jnp.log(self.probability_of_simple_interval(interval))
        open_interval_array = simple_interval_to_open_array(interval)
        new_lowers = jnp.maximum(self.lower, open_interval_array[0])
        new_uppers = jnp.minimum(self.upper, open_interval_array[1])
        valid_intervals = new_lowers < new_uppers
        new_intervals = jnp.stack([new_lowers[valid_intervals], new_uppers[valid_intervals]]).T
        return self.__class__(self.variable, new_intervals), probabilities

    def merge_with(self, others: List[Self]) -> Self:
        return self.__class__(self.variable, jnp.vstack([self.interval] + [other.interval for other in others]))

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(data["variable"], jnp.array(data["interval"]))

    def to_nx(self, variables: SortedSet[Variable], progress_bar: Optional[tqdm.tqdm] = None) -> List[
        ProbabilisticCircuitMixin]:
        variable = variables[self.variable]

        if progress_bar:
            progress_bar.set_postfix_str(f"Creating Uniform distributions for variable {variable.name}")

        nx_pc = NXProbabilisticCircuit()
        nodes = [UniformDistribution(variable=variable,
                                     interval=random_events.interval.SimpleInterval(lower.item(), upper.item(),
                                                                                    random_events.interval.Bound.OPEN,
                                                                                    random_events.interval.Bound.OPEN))
                 for lower, upper in self.interval]

        if progress_bar:
            progress_bar.update(self.number_of_nodes)

        nx_pc.add_nodes_from(nodes)
        return nodes
