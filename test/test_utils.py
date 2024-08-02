import unittest

from torch.testing import assert_close

from probabilistic_model.distributions.distributions import SymbolicDistribution
import probabilistic_model.probabilistic_circuit
import probabilistic_model.probabilistic_circuit.distributions
from probabilistic_model.utils import type_converter
import torch
from probabilistic_model.utils import sparse_dense_mul_inplace, add_sparse_edges_dense_child_tensor_inplace, shrink_index_tensor


class TypeConversionTestCase(unittest.TestCase):

    def test_univariate_distribution_type_converter(self):
        result = type_converter(SymbolicDistribution, probabilistic_model.probabilistic_circuit)
        self.assertEqual(result, probabilistic_model.probabilistic_circuit.distributions.SymbolicDistribution)


class TorchUtilsTestCase(unittest.TestCase):

    def test_sparse_dense_mul_inplace(self):
        indices = torch.tensor([[0, 1], [1, 0]])
        values = torch.tensor([2., 3.])
        sparse = torch.sparse_coo_tensor(indices, values, ).coalesce()
        dense = torch.tensor([[1., 2.], [3, 4]])
        sparse_dense_mul_inplace(sparse, dense)
        assert_close(sparse.values(), torch.tensor([4., 9.]))

    def test_add_sparse_edges_dense_child_tensor_inplace(self):
        indices = torch.tensor([[0, 1], [1, 0], [1, 1]]).T
        values = torch.tensor([2., 3., 4.])
        sparse = torch.sparse_coo_tensor(indices, values, ).coalesce()
        dense = torch.tensor([1., 2.]).reshape(-1, 1)
        add_sparse_edges_dense_child_tensor_inplace(sparse, dense)
        assert_close(sparse.values(), torch.tensor([4., 4., 6.]))

    def test_shrink_index_tensor(self):
        shrank = shrink_index_tensor(torch.tensor([[0, 3], [1, 0], [4, 1]]))
        assert_close(torch.tensor([[0, 2], [1, 0], [2, 1]]), shrank)


if __name__ == '__main__':
    unittest.main()
