from jax import numpy as jnp

from probabilistic_model.probabilistic_circuit.jax.input_layer import ContinuousLayerWithFiniteSupport


class UniformLayer(ContinuousLayerWithFiniteSupport):
    """
    A layer that represents a uniform distribution over a single variable.
    """

    def validate(self):
        assert self.lower.shape == self.upper.shape, "The shapes of the lower and upper bounds must match."

    @property
    def number_of_nodes(self) -> int:
        """
        The number of nodes in the layer.
        """
        return len(self.lower)

    def log_pdf_value(self) -> jnp.array:
        """
        Calculate the log-density of the uniform distribution.
        """
        return -jnp.log(self.upper - self.lower)

    def log_likelihood_of_nodes(self, x: jnp.array) -> jnp.array:
        """
        Calculate the log-likelihood of the uniform distribution.

        :param x: The input tensor of shape (n, 1).
        :return: The log-likelihood of the uniform distribution.
        """
        return jnp.where(self.included_condition(x), self.log_pdf_value(), -jnp.inf)

    def __deepcopy__(self):
        return self.__class__(self.variables[0].item(), self.interval.copy())
