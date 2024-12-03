import unittest

import numpy as np
from matplotlib import pyplot as plt
from random_events.interval import singleton, closed
from random_events.product_algebra import SimpleEvent
from random_events.set import SetElement
from random_events.variable import Continuous, Symbolic
from probabilistic_model.probabilistic_circuit.nx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit, LeafUnit
import json
import plotly.graph_objects as go


class Animal(SetElement):
    EMPTY_SET = -1
    CAT = 0
    DOG = 1
    FISH = 2

class FullyFactorizedTestCase(unittest.TestCase):

    x = Continuous("x")
    y = Continuous("y")
    model: ProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        mean = {cls.x: 1, cls.y : 0}
        variance = {cls.x: 2, cls.y: 1}
        cls.model = fully_factorized([cls.x, cls.y], mean, variance)

    def test_fully_factorized(self):
        self.assertEqual(len(self.model.nodes), 3)
        self.assertEqual(len(self.model.edges), 2)

    def test_conditioning_on_mode(self):
        event = SimpleEvent({self.x: closed(0, 2), self.y: closed(0, 2)}).as_composite_set().complement()
        model = self.model
        model, _ = model.conditional(event)
        mode, _ = model.mode()
        conditional, _ = model.conditional(mode)
        self.assertIsNotNone(conditional)


if __name__ == '__main__':
    unittest.main()
