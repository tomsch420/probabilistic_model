from abc import abstractmethod

import jax
from jax.tree_util import tree_flatten, tree_unflatten

import equinox as eqx

from .probabilistic_circuit import Layer
import jax.numpy as jnp

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
    """
    A probabilistic circuit that uses a function to generate parameters for a circuit.
    """

    conditioner: Conditioner
    """
    The conditioner that generates the parameters for the circuit.
    """

    circuit: Layer
    """
    The circuit to generate the parameters for.
    """

    conditioner_columns: jax.Array
    """
    The columns in a matrix that the conditioner takes as input for producing circuit parameters.
    """

    circuit_columns: jax.Array
    """
    The columns in a matrix that the circuit takes as input for calculating likelihoods.
    """

    def __init__(self, conditioner: Conditioner, conditioner_columns: jax.Array,
                 circuit: Layer, circuit_columns):
        self.conditioner = conditioner
        self.conditioner_columns = conditioner_columns
        self.circuit = circuit
        self.circuit_columns = circuit_columns

    def partition_circuit(self):
        """
        Partition the circuit into the parameters and the static structure.
        :return:
        """
        return eqx.partition(self.circuit, eqx.is_inexact_array)

    def create_circuit_from_parameters(self, params) -> Layer:
        """
        Generate a circuit with the structure from self.circuit and the parameters from params.
        :param params: The parameters to be used in the circuit.
        :return: The circuit
        """

        # get the parameters and circuit definition
        tree_def, static = self.partition_circuit()

        # flatten the parameters
        flat_model, flat_tree_def = jax.tree_util.tree_flatten(tree_def)

        # update the parameters
        params = tree_unflatten(flat_tree_def, params)

        # assemble the parameterized model
        circuit = eqx.combine(params, static)
        return circuit

    def _conditional_log_likelihood_single(self, x, conditioner, create_circuit_from_parameters):
        """
        Calculate the conditional log likelihood of a single data point.

        :param x: The datapoint
        :param conditioner: The conditioner to use
        :param create_circuit_from_parameters: The function to create a circuit from parameters
        :return:
        """
        params = conditioner.generate_parameters(x[self.conditioner_columns]).reshape(1, -1)
        circuit = create_circuit_from_parameters(params)
        return circuit.log_likelihood_of_nodes(x[self.circuit_columns])

    def vectorized_conditional_log_likelihood_single(self, x):
        """
        Calculate the conditional log likelihood of data points as batched.
        :param x: The data points
        :return: The conditional log likelihoods
        """
        return (jax.vmap(self._conditional_log_likelihood_single, in_axes=(0, None, None))
                (x, self.conditioner, self.create_circuit_from_parameters))

    def conditional_log_likelihood(self, x):
        return self.vectorized_conditional_log_likelihood_single(x)

    def validate(self):
        self.circuit.validate()
        params, _ = self.partition_circuit()
        flattened_params = tree_flatten(params)[0]
        number_of_parameters = sum([len(p) for p in flattened_params])
        assert number_of_parameters == self.conditioner.output_length
