import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from random_events.interval import SimpleInterval, Bound


def copy_bcoo(x: BCOO) -> BCOO:
    return BCOO((x.data.copy(), x.indices.copy()), shape=x.shape, indices_sorted=x.indices_sorted,
                unique_indices=x.unique_indices)

def simple_interval_to_open_array(interval: SimpleInterval) -> jnp.array:
    lower = jnp.array(interval.lower)
    if interval.left == Bound.CLOSED:
        lower = jnp.nextafter(lower, lower - 1)
    upper = jnp.array(interval.upper)
    if interval.right == Bound.CLOSED:
        upper = jnp.nextafter(upper, upper + 1)
    return jnp.array([lower, upper])
