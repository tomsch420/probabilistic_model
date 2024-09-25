import jax

from jax import numpy as jnp
import equinox as eqx
from random_events.variable import Continuous


class DiracDeltaLayer(eqx.Module):

    variable_index: int
    """
    The index of the variable of this layer.
    """

    location: jax.Array
    """
    The locations of the Dirac delta distributions.
    """

    density_cap: jax.Array
    """
    The density caps of the Dirac delta distributions.
    This value will be used to replace infinity in likelihoods.
    """

    def __init__(self, variable_index, location, density_cap):
        self.variable_index = variable_index
        self.location = location
        self.density_cap = density_cap

    @property
    def number_of_nodes(self):
        return len(self.location)

    @jax.jit
    def log_likelihood_of_nodes(self, x: jax.Array) -> jax.Array:
        return jnp.where(x == self.location, jnp.log(self.density_cap), -jnp.inf)