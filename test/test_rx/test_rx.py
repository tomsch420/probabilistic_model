import unittest
from enum import IntEnum

import plotly.graph_objects as go
import random_events.interval
from matplotlib import pyplot as plt
from random_events.interval import closed, SimpleInterval, singleton
from random_events.variable import Continuous

from probabilistic_model.distributions import SymbolicDistribution, GaussianDistribution, DiracDeltaDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rustworkx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rustworkx.probabilistic_circuit import *
from probabilistic_model.utils import MissingDict
import rustworkx.visualization

class SymbolEnum(IntEnum):
    A = 0
    B = 1
    C = 2


class RXSmallCircuitTestCast(unittest.TestCase):
    """
    Integration test for all classes in probabilistic circuits.
    """

    x = Continuous("x")
    y = Continuous("y")

    model: ProbabilisticCircuit
    sum1: SumUnit

    @classmethod
    def setUp(cls):
        cls.model = ProbabilisticCircuit()
        sum1, sum2, sum3 = SumUnit(cls.model), SumUnit(cls.model), SumUnit(cls.model)
        sum4, sum5 = SumUnit(cls.model), SumUnit(cls.model)
        prod1, prod2 = ProductUnit(cls.model), ProductUnit(cls.model)

        sum1.add_subcircuit(prod1, np.log(0.5))
        sum1.add_subcircuit(prod2, np.log(0.5))
        prod1.add_subcircuit(sum2)
        prod1.add_subcircuit(sum4)
        prod2.add_subcircuit(sum3)
        prod2.add_subcircuit(sum5)

        d_x1 = leaf(DiracDeltaDistribution(cls.x, 0, 1), cls.model)
        d_x2 = leaf(DiracDeltaDistribution(cls.x, 1, 2), cls.model)
        d_y1 = leaf(DiracDeltaDistribution(cls.y, 2, 3), cls.model)
        d_y2 = leaf(DiracDeltaDistribution(cls.y, 3, 4), cls.model)

        sum2.add_subcircuit(d_x1, np.log(0.8))
        sum2.add_subcircuit(d_x2, np.log(0.2))
        sum3.add_subcircuit(d_x1, np.log(0.7))
        sum3.add_subcircuit(d_x2, np.log(0.3))

        sum4.add_subcircuit(d_y1, np.log(0.5))
        sum4.add_subcircuit(d_y2, np.log(0.5))
        sum5.add_subcircuit(d_y1, np.log(0.1))
        sum5.add_subcircuit(d_y2, np.log(0.9))
        cls.sum1 = sum1
        cls.model.normalize()

    def test_index_and_circuit_setting(self):
        model = ProbabilisticCircuit()
        s1 = SumUnit(model)
        d_x1 = leaf(DiracDeltaDistribution(self.x, 0, 1), model)
        self.assertEqual(s1.probabilistic_circuit, d_x1.probabilistic_circuit)
        self.assertEqual(s1.index, 0)
        self.assertEqual(d_x1.index, 1)
        s1.add_subcircuit(d_x1, mount=True)
        self.assertEqual(d_x1.index, 1)

    def test_created_structure(self):
        self.assertEqual(self.model.root, self.sum1)
        self.assertEqual(len(self.model.nodes), 11)
        self.assertEqual(len(self.model.graph.edges()), 14)
        self.assertEqual(len(self.model.leaves), 4)
    #
    # def test_sampling(self):
    #     samples = self.model.sample(100)
    #     unique = np.unique(samples, axis=0)
    #     self.assertGreater(len(unique), 95)

    def test_conditioning(self):
        event = SimpleEvent({self.x: closed(0, 0.25) | closed(0.5, 0.75)}).as_composite_set()
        # rustworkx.visualization.mpl_draw(self.model.__deepcopy__().graph)
        # plt.show()
        conditional, prob = self.model.conditional(event)

        conditional.validate()
        self.assertAlmostEqual(prob, 0.375)



if __name__ == '__main__':
    unittest.main()