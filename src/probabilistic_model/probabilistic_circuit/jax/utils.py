import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from random_events.interval import SimpleInterval, Bound
import jax
from typing_extensions import Tuple


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


def create_sparse_array_indices_from_row_lengths(row_lengths: jax.Array) -> jax.Array:
    """
    Create the indices of a sparse tensor with the given row lengths.

    The shape of the indices is (2, sum(row_lengths)).
    The shape of the sparse tensor that the indices describe should be (len(row_lengths), max(row_lengths)).

    Example::

        >>> row_lengths = jnp.array([2, 3])
        >>> create_sparse_array_indices_from_row_lengths(row_lengths)
            [[0 0]
             [0 1]
             [1 0]
             [1 1]
             [1 2]]

    :param row_lengths: The row lengths.
    :return: The indices of the sparse tensor
    """

    # create row indices
    row_indices = jnp.repeat(jnp.arange(len(row_lengths)), row_lengths)

    # offset the row lengths by the one element
    offset_row_lengths = jnp.concatenate([jnp.array([0]), row_lengths[:-1]])

    # create a cumulative sum of the offset row lengths and offset it by the first row length
    cum_sum = jnp.repeat(offset_row_lengths, row_lengths)

    # arrange column indices
    summed_row_lengths = jnp.arange(row_lengths.sum())

    # create the column indices
    col_indices = summed_row_lengths - cum_sum

    return jnp.vstack([row_indices, col_indices]).T


def embed_sparse_array_in_nan_array(sparse_array: BCOO) -> jax.Array:
    result = jnp.full(sparse_array.shape, jnp.nan, dtype=jnp.float32)
    result = result.at[sparse_array.indices[:, 0], sparse_array.indices[:, 1]].set(sparse_array.data)
    return result

def in_bound_elements_from_sparse_slice(sparse_slice: BCOO) -> Tuple[jax.Array, jax.Array]:
    """
    Get the indices and data that are not out of bounds in a slice of a sparse array.

    :param sparse_slice: The slice from the sparse array
    :return: The valid indices and valid data
    """
    sparse_slice_indices = sparse_slice.indices[:, 0]
    in_bound_elements = sparse_slice_indices < sparse_slice.nse
    edge_indices = sparse_slice_indices[in_bound_elements]
    edge_data = sparse_slice.data[in_bound_elements]
    return edge_indices, edge_data