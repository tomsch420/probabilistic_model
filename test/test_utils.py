import unittest

from torch.testing import assert_close

from probabilistic_model.distributions.distributions import SymbolicDistribution
import probabilistic_model.probabilistic_circuit
import probabilistic_model.probabilistic_circuit.nx.distributions
from probabilistic_model.utils import type_converter
import torch
from probabilistic_model.probabilistic_circuit.torch.utils import (sparse_dense_mul_inplace, add_sparse_edges_dense_child_tensor_inplace,
                                       shrink_index_tensor, embed_sparse_tensors_in_new_sparse_tensor,
                                       create_sparse_tensor_indices_from_row_lengths)


class TypeConversionTestCase(unittest.TestCase):

    def test_univariate_distribution_type_converter(self):
        result = type_converter(SymbolicDistribution, probabilistic_model.probabilistic_circuit)
        self.assertEqual(result, probabilistic_model.probabilistic_circuit.nx.distributions.SymbolicDistribution)


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

    def test_shrink_index_tensor_1d(self):
        shrank = shrink_index_tensor(torch.tensor([[0], [1], [4]]))
        assert_close(torch.tensor([[0], [1], [2]]), shrank)

    def test_embed_sparse_tensors_in_new_sparse_tensor(self):
        t1 = torch.tensor([[1, 2], [3, 4]]).to_sparse_coo()
        t2 = torch.tensor([[5, 6, 7], [8, 9, 10], [11, 12, 13]]).to_sparse_coo()

        dense_result = torch.tensor([[1, 2, 0, 0, 0],
                                     [3, 4, 0, 0, 0],
                                     [0, 0, 5, 6, 7],
                                     [0, 0, 8, 9, 10],
                                     [0, 0, 11, 12, 13]])

        result = embed_sparse_tensors_in_new_sparse_tensor([t1, t2])
        assert_close(result.to_dense(), dense_result)

    def test_embed_sparse_tensors_in_new_sparse_tensor_1d(self):
        row_lengths = torch.tensor([2, 3])
        indices = create_sparse_tensor_indices_from_row_lengths(row_lengths)
        result = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 0, 1, 2]])
        assert_close(result, indices)




if __name__ == '__main__':
    unittest.main()
