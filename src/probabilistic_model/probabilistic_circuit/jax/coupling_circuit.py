from abc import abstractmethod

import jax
from jax.tree_util import tree_flatten, tree_unflatten

import equinox as eqx

from .probabilistic_circuit import Layer


class Conditioner:

    @abstractmethod
    def generate_parameters(self, x) -> jax.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_length(self):
        """
        :return: The length number of parameters that the model outputs.
        """
        raise NotImplementedError


class CouplingCircuit(eqx.Module):

    conditioner: Conditioner
    circuit: Layer

    def __init__(self, conditioner: Conditioner, circuit: Layer):
        self.conditioner = conditioner
        self.circuit = circuit

    def partition_circuit(self):
        return eqx.partition(self.circuit, eqx.is_inexact_array)

    def validate(self):
        self.circuit.validate()
        params, _ = self.partition_circuit()
        flattened_params = tree_flatten(params)[0]
        number_of_parameters = sum([len(p) for p in flattened_params])
        assert number_of_parameters == self.conditioner.output_length

    def conditional_log_likelihood(self, x):
        tree_def, static = self.partition_circuit()
        flat_model, treedef_model = jax.tree_util.tree_flatten(tree_def)
        params = self.conditioner.generate_parameters(x)
        params = tree_unflatten(treedef_model, [params[0]])
        circuit = eqx.combine(params, static)
        return circuit.log_likelihood_of_nodes(x)
