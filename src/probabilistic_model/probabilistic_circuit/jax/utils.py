import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from random_events.interval import SimpleInterval, Bound
import jax
from scipy.sparse import csr_matrix, csr_array, csc_array
from typing_extensions import Tuple

from probabilistic_model.utils import timeit_print


def copy_bcoo(x: BCOO) -> BCOO:
    return x.__class__((x.data.copy(), x.indices.copy()), shape=x.shape, indices_sorted=x.indices_sorted,
                unique_indices=x.unique_indices)

def copy_bcsr(x: BCSR) -> BCSR:
    return x.__class__((x.data.copy(), x.indices.copy(), x.indptr.copy()), shape=x.shape, indices_sorted=x.indices_sorted,
                unique_indices=x.unique_indices)

def simple_interval_to_open_array(interval: SimpleInterval) -> jnp.array:
    lower = jnp.array(interval.lower)
    if interval.left == Bound.CLOSED:
        lower = jnp.nextafter(lower, lower - 1)
    upper = jnp.array(interval.upper)
    if interval.right == Bound.CLOSED:
        upper = jnp.nextafter(upper, upper + 1)
    return jnp.array([lower, upper])


def create_bcoo_indices_from_row_lengths(row_lengths: np.array) -> np.array:
    """
    Create the indices of a BCOO array with the given row lengths.

    The shape of the indices is (2, sum(row_lengths)).
    The shape of the sparse tensor that the indices describe should be (len(row_lengths), max(row_lengths)).

    Example::

        >>> row_lengths = jnp.array([2, 3])
        >>> create_bcoo_indices_from_row_lengths(row_lengths)
            [[0 0]
             [0 1]
             [1 0]
             [1 1]
             [1 2]]

    :param row_lengths: The row lengths.
    :return: The indices of the sparse tensor
    """

    # create row indices
    row_indices = np.repeat(np.arange(len(row_lengths)), row_lengths)

    # offset the row lengths by the one element
    offset_row_lengths = np.concatenate([jnp.array([0]), row_lengths[:-1]])

    # create a cumulative sum of the offset row lengths and offset it by the first row length
    cum_sum = np.repeat(offset_row_lengths, row_lengths)

    # arrange column indices
    summed_row_lengths = np.arange(row_lengths.sum())

    # create the column indices
    col_indices = summed_row_lengths - cum_sum

    return np.vstack((row_indices, col_indices)).T

def create_bcoo_indices_from_row_lengths_np(row_lengths: np.array) -> np.array:
    """
    Create the indices of a BCOO array with the given row lengths.

    The shape of the indices is (2, sum(row_lengths)).
    The shape of the sparse tensor that the indices describe should be (len(row_lengths), max(row_lengths)).

    Example::

        >>> row_lengths = jnp.array([2, 3])
        >>> create_bcoo_indices_from_row_lengths(row_lengths)
            [[0 0]
             [0 1]
             [1 0]
             [1 1]
             [1 2]]

    :param row_lengths: The row lengths.
    :return: The indices of the sparse tensor
    """

    # create row indices
    row_indices = np.repeat(jnp.arange(len(row_lengths)), row_lengths)

    # offset the row lengths by the one element
    offset_row_lengths = np.concatenate([jnp.array([0]), row_lengths[:-1]])

    # create a cumulative sum of the offset row lengths and offset it by the first row length
    cum_sum = np.repeat(offset_row_lengths, row_lengths)

    # arrange column indices
    summed_row_lengths = np.arange(row_lengths.sum())

    # create the column indices
    col_indices = summed_row_lengths - cum_sum

    return np.vstack((row_indices, col_indices))


def create_bcsr_indices_from_row_lengths(row_lengths: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Create the column indices and indent pointer of bcsr array with the given row lengths.

    The shape of the sparse tensor that the indices describe should be (len(row_lengths), max(row_lengths)).

    Example::

        >>> row_lengths = jnp.array([2, 3])
        >>> create_bcsr_indices_from_row_lengths(row_lengths)
        (Array([0, 1, 0, 1, 2], dtype=int32), Array([0, 2, 5], dtype=int32))

    :param row_lengths: The row lengths.
    :return: The indices of the sparse tensor
    """

    # offset the row lengths by the one element
    offset_row_lengths = jnp.concatenate([jnp.array([0]), row_lengths[:-1]])

    # create a cumulative sum of the offset row lengths and offset it by the first row length
    cum_sum = jnp.repeat(offset_row_lengths, row_lengths)

    # arrange column indices
    summed_row_lengths = jnp.arange(row_lengths.sum())

    # create the column indices
    col_indices = summed_row_lengths - cum_sum

    indent_pointer = jnp.concatenate([jnp.array([0]), jnp.cumsum(row_lengths)])

    return col_indices, indent_pointer


def embed_sparse_array_in_nan_array(sparse_array: BCOO) -> jax.Array:
    result = jnp.full(sparse_array.shape, jnp.nan, dtype=jnp.float32)
    result = result.at[sparse_array.indices[:, 0], sparse_array.indices[:, 1]].set(sparse_array.data)
    return result


def sample_from_sparse_probabilities_csc(probabilities: csr_array, amount: np.array) -> csc_array:
    """
    Sample from a sparse array of probabilities.
    Each row in the sparse array encodes a categorical probability distribution.

    :param probabilities: The sparse array of probabilities.
    :param amount: The amount of samples to draw from each row.
    :return: The samples that are drawn for each state in the probabilities indicies.
    """
    all_samples = np.concatenate([np.random.multinomial(amount_.item(), pvals=probability_row.data) for amount_, probability_row in zip(amount, probabilities)], axis=0)
    result = csr_array((all_samples, probabilities.indices, probabilities.indptr),
                       shape=probabilities.shape).tocsc(copy=False)
    return result
