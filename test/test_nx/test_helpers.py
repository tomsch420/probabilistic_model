import unittest

from random_events.set import SetElement
from random_events.variable import Continuous, Symbolic
from probabilistic_model.probabilistic_circuit.nx.helper import fully_factorized

class Animal(SetElement):
    EMPTY_SET = -1
    CAT = 0
    DOG = 1
    FISH = 2

class FullyFactorizedTestCase(unittest.TestCase):

    def test_fully_factorized(self):
        x = Continuous("x")
        a = Symbolic("a", Animal)
        mean = {x: 1}
        variance = {x: 2}
        model = fully_factorized([x, a], mean, variance)
        self.assertEqual(len(model.nodes), 3)
        self.assertEqual(len(model.edges), 2)


if __name__ == '__main__':
    unittest.main()
