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
from typing_extensions import Type


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
    # get indices of values where all elements are equal to a given value
    values = tensor.values()
    valid_elements = (values != value)
    valid_indices = tensor.indices()[valid_elements]
    print(values)
    print(valid_elements)
    result = torch.sparse_coo_tensor(valid_indices, values[valid_elements]).coalesce()
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

    for dim in range(2):
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