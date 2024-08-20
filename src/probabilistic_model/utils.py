from __future__ import annotations

import time
import types
from collections import defaultdict
from functools import wraps

import numpy as np
import torch
import torch.sparse
from random_events.interval import SimpleInterval, Interval, Bound
from random_events.utils import recursive_subclasses
from torch import nextafter
from typing_extensions import Type, List


def simple_interval_as_array(interval: SimpleInterval) -> np.ndarray:
    """
    Convert a simple interval to a numpy array.
    :param interval:  The interval
    :return:  [lower, upper] as numpy array
    """
    return np.array([interval.lower, interval.upper])


def interval_as_array(interval: Interval) -> np.ndarray:
    """
    Convert an interval to a numpy array.
    The resulting array has shape (n, 2) where n is the number of simple intervals in the interval.
    The first column contains the lower bounds and the second column the upper bounds of the simple intervals.
    :param interval: The interval
    :return:  as numpy array
    """
    return np.array([simple_interval_as_array(simple_interval) for simple_interval in interval.simple_sets])


def type_converter(abstract_type: Type, package: types.ModuleType):
    """
    Convert a type to a different type from a target sub-package that inherits from this type.

    :param abstract_type: The type to convert
    :param package: The sub-package to search in for that type

    :return: The converted type
    """
    for subclass in recursive_subclasses(abstract_type):
        if subclass.__module__.startswith(package.__name__):
            return subclass

    raise ValueError("Could not find type {} in package {}".format(abstract_type, package))


class MissingDict(defaultdict):
    """
    A defaultdict that returns the default value when the key is missing and does **not** add the key to the dict.
    """

    def __missing__(self, key):
        return self.default_factory()


def timeit(func):
    """
    Decorator to measure the time a function takes to execute.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{func.__name__} took {total_time} s')
        return result

    return timeit_wrapper


def simple_interval_to_open_tensor(interval: SimpleInterval) -> torch.Tensor:
    """
    Convert a simple interval to a tensor where the first element is the lower bound as if it was open and the
    second is the upper bound as if it was open.

    :param interval: The interval to convert.
    :return: The tensor.
    """
    lower = torch.tensor(interval.lower)
    if interval.left == Bound.CLOSED:
        lower = nextafter(lower, lower - 1)
    upper = torch.tensor(interval.upper)
    if interval.right == Bound.CLOSED:
        upper = nextafter(upper, upper + 1)
    return torch.tensor([lower, upper])


def remove_rows_and_cols_where_all(tensor: torch.Tensor, value: float) -> torch.Tensor:
    """
    Remove rows and columns from a tensor where all elements are equal to a given value.

    :param tensor: The tensor to remove rows and columns from.
    :param value: The value to remove.
    :return: The tensor without the rows and columns.

    Example::


        >>> t = torch.tensor([[1, 0, 3], [0, 0, 0], [7, 0, 9]])
        >>> remove_rows_and_cols_where_all(t, 0)
        tensor([[1, 3],
                [7, 9]])
    """

    # get the rows and columns that are not entirely -inf
    valid_rows = (tensor != value).any(dim=1)
    valid_cols = (tensor != value).any(dim=0)

    # remove rows and cols that are entirely -inf
    valid_tensor = tensor[valid_rows][:, valid_cols]
    return valid_tensor


def sparse_remove_rows_and_cols_where_all(tensor: torch.Tensor, value: float) -> torch.Tensor:
    """
    Remove rows and columns from a sparse tensor where all elements are equal to a given value.
    :param tensor: The sparse tensor to remove rows and columns from.
    :param value: The value to remove.
    :return: The tensor without the unnecessary rows and columns.
    """
    # get indices of values where all elements are equal to a given value
    values = tensor.values()
    valid_elements = (values != value)

    # filter indices by valid elements
    valid_indices = tensor.indices().T[valid_elements]

    # shrink indices
    valid_indices = shrink_index_tensor(valid_indices)

    # construct result tensor
    result = torch.sparse_coo_tensor(valid_indices.T, values[valid_elements]).coalesce()
    return result


def shrink_index_tensor(index_tensor: torch.Tensor) -> torch.Tensor:
    """
    Shrink a 2D index tensor to only contain successive indices.
    The tensor has shape (#indices, 2).

    Example::

        >>> shrink_index_tensor(torch.tensor([[0, 3], [1, 0], [4, 1]]))
            tensor([[0, 2], [1, 0], [2, 1]])
    :param index_tensor: The index tensor to shrink.
    :return: The shrunken index tensor.
    """

    result = index_tensor.clone()

    for dim in range(index_tensor.shape[1]):
        unique_indices = torch.unique(index_tensor[:, dim], sorted=True)

        for new_index, unique_index in zip(range(len(unique_indices)), unique_indices):
            result[result[:, dim] == unique_index, dim] = new_index

    # map the old indices to the new indices
    return result

def sparse_dense_mul_inplace(sparse: torch.Tensor, dense: torch.Tensor):
    """
    Multiply a sparse tensor with a dense tensor element-wise in-place of the sparse tensor.
    :param sparse: The sparse tensor
    :param dense: The dense tensor
    :return: The result of the multiplication
    """
    indices = sparse._indices()

    # get values from relevant entries of dense matrix
    dense_values_at_sparse_indices = dense[indices[0, :], indices[1, :]]

    # multiply sparse values with dense values inplace
    sparse.values().mul_(dense_values_at_sparse_indices)


def add_sparse_edges_dense_child_tensor_inplace(edges: torch.Tensor, dense_child_tensor: torch.Tensor):
    """
    Add a dense tensor to a sparse tensor at the positions specified by the edge tensor.

    This method is used when a weighted sum of the child tensor is necessary.
    The edges specify how to weight the child tensor and the dense tensor is the child tensor.
    The result is stored in the sparse tensor.


    Example::

        >>> edges = torch.tensor([[0, 1], [1, 0], [1, 1]]).T
        >>> values = torch.tensor([2., 3., 4.])
        >>> sparse = torch.sparse_coo_tensor(edges, values, ).coalesce()
        >>> dense = torch.tensor([1., 2.]).reshape(-1, 1)
        >>> add_sparse_edges_dense_child_tensor_inplace(sparse, dense)
        >>> sprase
            tensor(indices=tensor([[0, 1, 1],
                           [1, 0, 1]]),
           values=tensor([4., 4., 6.]),
           size=(2, 2), nnz=3, layout=torch.sparse_coo)

    :param edges: The edge tensor of shape (#edges, n).
    :param dense_child_tensor: The dense tensor of shape (n, 1).
    :return: The result of the addition
    """
    # get indices of the sparse tensor
    indices = edges._indices()

    # get values from relevant entries of dense matrix
    dense_values_at_sparse_indices = dense_child_tensor[indices[1]].squeeze()

    # add sparse values with dense values inplace
    edges.values().add_(dense_values_at_sparse_indices)


def embed_sparse_tensors_in_new_sparse_tensor(sparse_tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Embed a list of sparse coo tensors into a new sparse co tensor containing all tensors without intersecting their
    indices.

    :param sparse_tensors: The list of sparse tensors to embed.
    :return: The new sparse tensor.

    Example::

        >>> t1 = torch.tensor([[1, 2], [3, 4]]).to_sparse_coo()
        >>> t2 = torch.tensor([[5, 6, 7], [8, 9, 10], [11, 12, 13]]).to_sparse_coo()
        >>> result = embed_sparse_tensors_in_new_sparse_tensor([t1, t2]).to_dense()
        tensor([[ 1,  2,  0,  0,  0],
                [ 3,  4,  0,  0,  0],
                [ 0,  0,  5,  6,  7],
                [ 0,  0,  8,  9, 10],
                [ 0,  0, 11, 12, 13]])

    """
    # calculate the shape of the new tensor
    new_shape = sum(torch.tensor(sparse_tensor.shape) for sparse_tensor in sparse_tensors)

    # initialize the new indices
    new_indices = sparse_tensors[0].indices()

    # shift the indices of the other tensors to not intersect with the already added tensors
    for sparse_tensor in sparse_tensors[1:]:
        current_indices = sparse_tensor.indices()
        max_of_new_indices = new_indices.max(1).values.reshape(-1, 1)
        current_indices += max_of_new_indices + 1
        new_indices = torch.cat([new_indices, current_indices], dim=1)

    # stack the new values
    new_values = torch.cat([sparse_tensor.values() for sparse_tensor in sparse_tensors], 0)

    # create the result
    return torch.sparse_coo_tensor(new_indices, new_values, torch.Size(new_shape), is_coalesced=True)


def embed_sparse_tensor_in_nan_tensor(sparse_tensor: torch.Tensor) -> torch.Tensor:
    result = torch.full(sparse_tensor.shape, torch.nan, dtype=torch.double)
    result[sparse_tensor.indices()[0], sparse_tensor.indices()[1]] = sparse_tensor.values()
    return result

def create_sparse_tensor_indices_from_row_lengths(row_lengths: torch.Tensor) -> torch.Tensor:
    """
    Create the indices of a sparse tensor with the given row lengths.

    The shape of the indices is (2, sum(row_lengths)).
    The shape of the sparse tensor that the indices describe should be (len(row_lengths), max(row_lengths)).

    Example::

        >>> row_lengths = torch.tensor([2, 3])
        >>> create_sparse_tensor_indices_from_row_lengths(row_lengths)
        tensor([[0, 0, 1, 1, 1],
                [0, 1, 0, 1, 2]])

    :param row_lengths: The row lengths.
    :return: The indices of the sparse tensor
    """

    # create row indices
    row_indices = torch.arange(len(row_lengths)).repeat_interleave(row_lengths)

    # offset the row lengths by the one element
    offset_row_lengths = torch.concatenate([torch.tensor([0]), row_lengths[:-1]])

    # create a cumulative sum of the offset row lengths and offset it by the first row length
    cum_sum = offset_row_lengths.repeat_interleave(row_lengths)

    # arrange column indices
    summed_row_lengths = torch.arange(row_lengths.sum().item())

    # create the column indices
    col_indices = summed_row_lengths - cum_sum

    return torch.stack([row_indices, col_indices])