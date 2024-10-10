import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
from random_events.interval import SimpleInterval, Bound
import jax
from typing_extensions import Tuple


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

def sample_from_sparse_probabilities_bcsr(log_probabilities: BCSR, bcoo_indices: jax.Array, amount: jax.Array,
                                          key: jax.random.PRNGKey) -> BCOO:
    """
    Sample from a sparse array of probabilities.
    Each row in the sparse array encodes a categorical probability distribution.

    :param log_probabilities: The unnormalized sparse array of log-probabilities.
    :param amount:  The amount of samples to draw from each row.
    :param key: The random key.
    :return: The samples that are drawn for each state in the probabilities indicies.
    """
    all_samples = []

    # iterate through every row of the sparse array
    for row_index, (start, end) in enumerate(zip(log_probabilities.indptr[:-1], log_probabilities.indptr[1:])):
        # get the log-probabilities of the current row
        probability_row = log_probabilities.data[start:end]

        # sample from the log-probabilities
        samples = jax.random.categorical(key, probability_row, shape=(amount[row_index].item(), ))
        # count the frequencies of the samples
        frequencies = jnp.zeros((probability_row.shape[0],), dtype=jnp.int32)
        frequencies = frequencies.at[samples].add(1)
        all_samples.append(frequencies)

    return BCOO((jnp.concatenate(all_samples), bcoo_indices), shape=log_probabilities.shape,
                indices_sorted=True, unique_indices=True)


def sample_from_sparse_probabilities(log_probabilities: BCOO, amount: jax.Array, key: jax.random.PRNGKey) -> BCOO:
    """
    Sample from a sparse array of probabilities.
    Each row in the sparse array encodes a categorical probability distribution.

    :param log_probabilities: The unnormalized sparse array of log-probabilities.
    :param amount:  The amount of samples to draw from each row.
    :param key: The random key.
    :return: The samples that are drawn for each state in the probabilities indicies.
    """
    all_samples = []

    for probability_row, row_amount in zip(log_probabilities, amount):
        probability_row: BCOO
        probability_row = probability_row.sum_duplicates(remove_zeros=False)

        samples = jax.random.categorical(key, probability_row.data, shape=(row_amount.item(), ))
        frequencies = jnp.zeros((probability_row.data.shape[0],), dtype=jnp.int32)
        frequencies = frequencies.at[samples].add(1)
        all_samples.append(frequencies)

    return BCOO((jnp.concatenate(all_samples), log_probabilities.indices), shape=log_probabilities.shape,
                indices_sorted=True, unique_indices=True)


def extraction_mask(reference_indices: jax.Array, extraction_indices: jax.Array) -> jax.Array:
    """
    Create a mask that represents for every element in the reference indices whether it is in the extraction indices.

    :param reference_indices: The reference indices.
    :param extraction_indices: The extraction indices.
    :return: The mask.
    """
    return jnp.isin(reference_indices, extraction_indices)