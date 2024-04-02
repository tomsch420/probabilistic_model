import unittest

from probabilistic_model.distributions.distributions import SymbolicDistribution
import probabilistic_model.probabilistic_circuit
import probabilistic_model.probabilistic_circuit.distributions
from probabilistic_model.utils import type_converter


class TypeConversionTestCase(unittest.TestCase):

    def test_univariate_distribution_type_converter(self):
        result = type_converter(SymbolicDistribution, probabilistic_model.probabilistic_circuit)
        self.assertEqual(result, probabilistic_model.probabilistic_circuit.distributions.SymbolicDistribution)


if __name__ == '__main__':
    unittest.main()
