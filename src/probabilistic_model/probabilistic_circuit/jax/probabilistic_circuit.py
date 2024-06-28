from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import List, Iterable

import jax
import networkx as nx
from random_events.interval import SimpleInterval
from random_events.variable import Continuous, Variable
from typing_extensions import Self, Type, Dict, Any
from typing import NamedTuple

from probabilistic_model.probabilistic_circuit.probabilistic_circuit import (SumUnit as PCSumUnit,
                                                                             ProbabilisticCircuit as PMProbabilisticCircuit,
                                                                             ProductUnit as PCProductUnit,
                                                                             ProbabilisticCircuitMixin)
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution as PCUniformDistribution
import jax.numpy as jnp
from jax import Array, nn
import flax
from jax.tree_util import register_pytree_node_class
from random_events.utils import recursive_subclasses

from probabilistic_model.probabilistic_model import MomentType


def inverse_class_of(clazz: Type[ProbabilisticCircuitMixin]) -> Type[ModuleMixin]:
    for subclass in recursive_subclasses(ModuleMixin):
        if issubclass(clazz, subclass.origin_class()):
            return subclass
    raise TypeError(f"Could not find class for {clazz}")


StateDictType = Dict[int, Any]


class ModuleMixin:
    """
    Mixin for JAX modules that are capable of being converted to the original probabilistic circuit module.
    JAX modules are limited in functionality, as only the log_likelihood method is supported and automatically
    differentiable.
    """

    probabilistic_circuit: ProbabilisticCircuit

    @staticmethod
    @abstractmethod
    def origin_class() -> Type[ProbabilisticCircuitMixin]:
        """
        The original class of the module.

        :return: The original class of the module.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_unit(cls, unit: ProbabilisticCircuitMixin, probabilistic_circuit: ProbabilisticCircuit) -> Self:
        """
        Create a new instance of this class from a unit.

        :param unit: The unit to read the parameters from.
        :param probabilistic_circuit: The probabilistic circuit where the unit should be added.
        :return: The jax version of the unit.
        """
        raise NotImplementedError

    def log_likelihood(self, x: Array):
        """
        Calculate p(x)
        :param x:
        :return:
        """
        raise NotImplementedError

    def average_negative_log_likelihood(self, x: Array):
        log_likelihood = self.log_likelihood(x)
        return jnp.mean(log_likelihood)

    def __hash__(self):
        return id(self)

    @abstractmethod
    def get_parameters(self) -> Dict:
        ...

    @abstractmethod
    def set_parameters(self, parameters):
        ...


@register_pytree_node_class
class UniformDistribution(ModuleMixin, PCUniformDistribution):

    def get_parameters(self) -> Dict:
        return {}

    def set_parameters(self, parameters):
        ...

    def tree_flatten(self):
        return tuple(), self

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return aux_data

    def __init__(self, variable: Continuous, interval: SimpleInterval, probabilistic_circuit: ProbabilisticCircuit):
        super().__init__(variable, interval)
        self.probabilistic_circuit = probabilistic_circuit

    @staticmethod
    def origin_class() -> Type[PCUniformDistribution]:
        return PCUniformDistribution

    @classmethod
    def from_unit(cls, unit: PCUniformDistribution, probabilistic_circuit: ProbabilisticCircuit) -> Self:
        return cls(unit.variable, unit.interval, probabilistic_circuit)

    def log_pdf_value(self) -> Array:
        return -jnp.log(self.upper - self.lower)

    def log_likelihood_without_bounds_check(self, x: Array) -> Array:
        return jnp.full((x.shape[:-1]), self.log_pdf_value())

    def log_likelihood(self, x: Array) -> Array:
        result = jnp.full(x.shape[:-1], -jnp.inf)
        include_condition = self.included_condition(x)
        filtered_x = x[include_condition].reshape(-1, 1)
        likelihoods = self.log_likelihood_without_bounds_check(filtered_x)
        result = result.at[include_condition[:, 0]].set(likelihoods)
        return result

    def expand_batch_dim(self, size = 1000):
        pass

@register_pytree_node_class
class SumUnit(PCSumUnit, ModuleMixin):

    def get_parameters(self) -> Dict:
        return {hash(self): self.weights}

    def set_parameters(self, parameters):
        self.log_weights = parameters[hash(self)]

    def expand_batch_dim(self, size=1000):
        self.log_weights = self.log_weights.repeat(size, axis=0)

    log_weights: Array

    def __init__(self, initial_weights: Array, probabilistic_circuit: ProbabilisticCircuit):
        super().__init__()
        self.log_weights = jnp.log(initial_weights)
        self.probabilistic_circuit = probabilistic_circuit

    def tree_flatten(self):
        children = self.log_weights
        aux_data = self
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        self = aux_data
        self.log_weights = children
        return self

    @property
    def weights(self) -> Array:
        print(self.log_weights)
        return nn.softmax(self.log_weights, axis=-1)

    @cached_property
    def subcircuits(self) -> List[ModuleMixin]:
        return super().subcircuits

    @staticmethod
    def origin_class() -> Type[PCSumUnit]:
        return PCSumUnit

    def log_likelihood(self, x: Array) -> Array:
        result = jnp.zeros(x.shape[:-1])
        for weight, subcircuit in zip(self.weights, self.subcircuits):
            subcircuit_likelihood = jnp.exp(subcircuit.log_likelihood(x))
            result += weight * subcircuit_likelihood
        return jnp.log(result)

    def __call__(self, x):
        return self.log_likelihood(x)

    @classmethod
    def from_unit(cls, unit: PCSumUnit, probabilistic_circuit: ProbabilisticCircuit) -> Self:
        weights = jnp.array(unit.weights)
        result = cls(weights, probabilistic_circuit)
        return result

    def __hash__(self):
        return id(self)


class ProductUnit(ModuleMixin, PCProductUnit):

    @staticmethod
    def origin_class() -> Type[PCProductUnit]:
        return PCProductUnit


@register_pytree_node_class
class ProbabilisticCircuit(PMProbabilisticCircuit):

    def tree_flatten(self):
        children, aux_data = zip(*(node.tree_flatten() for node in self.nodes))
        aux_data = aux_data, self
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        aux_data, self = aux_data
        for node, data in zip(aux_data, children):
            node.tree_unflatten(node, data)
        return self

    @cached_property
    def root(self) -> ModuleMixin:
        return super().root

    @classmethod
    def from_probabilistic_circuit(cls, probabilistic_circuit: PMProbabilisticCircuit) -> Self:

        node_remap: Dict = dict()

        result = cls()
        for node in probabilistic_circuit.nodes:
            jax_node = inverse_class_of(type(node)).from_unit(unit=node, probabilistic_circuit=result)
            nx.DiGraph.add_node(result, jax_node)
            node_remap[node] = jax_node

        for edge in probabilistic_circuit.edges:
            result.add_edge(node_remap[edge[0]], node_remap[edge[1]])
        return result

    def log_likelihood(self, events: Array) -> Array:
        return self.root.log_likelihood(events)

    def get_parameters(self) -> Dict:
        result = {}
        for node in self.nodes:
            result.update(node.get_parameters())
        return result

    def set_parameters(self, parameters: StateDictType):
        for node in self.nodes:
            node.set_parameters(parameters)

    def expand_batch_dim(self, size=1000):
        for node in self.nodes:
            node.expand_batch_dim(size)

