import unittest

import probabilistic_model.probabilistic_circuit
import probabilistic_model.probabilistic_circuit.nx.distributions
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.utils import type_converter


class TypeConversionTestCase(unittest.TestCase):

    def test_univariate_distribution_type_converter(self):
        result = type_converter(SymbolicDistribution, probabilistic_model.probabilistic_circuit)
        self.assertEqual(result, probabilistic_model.probabilistic_circuit.nx.distributions.SymbolicDistribution)


if __name__ == '__main__':
    unittest.main()
