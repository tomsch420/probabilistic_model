import itertools
import unittest

import numpy as np
from random_events.events import Event
from random_events.variables import Symbolic

from probabilistic_model.distributions.multinomial import MultinomialDistribution


class MultinomialConstructionTestCase(unittest.TestCase):
    x: Symbolic
    y: Symbolic
    z: Symbolic

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        cls.x = Symbolic("X", range(2))
        cls.y = Symbolic("Y", range(3))
        cls.z = Symbolic("Z", range(5))

    def test_creation_with_probabilities(self):
        distribution = MultinomialDistribution([self.x, self.y, self.z], np.random.rand(len(self.x.domain),
                                                                                        len(self.y.domain),
                                                                                        len(self.z.domain)))
        self.assertTrue(distribution)

    def test_creation_without_probabilities(self):
        distribution = MultinomialDistribution([self.x])
        self.assertTrue(np.allclose(1., distribution.probabilities))

    def test_creation_with_invalid_probabilities_shape(self):
        probabilities = np.array([[0.1, 0.1], [0.2, 0.2]])
        with self.assertRaises(ValueError):
            distribution = MultinomialDistribution([self.x, self.y], probabilities)

    def test_copy(self):
        distribution_1 = MultinomialDistribution([self.x, self.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]]))
        distribution_2 = distribution_1.__copy__()
        self.assertEqual(distribution_1, distribution_2)
        distribution_2.probabilities = np.zeros_like(distribution_2.probabilities)
        self.assertNotEqual(distribution_2, distribution_1)

    def test_to_tabulate(self):
        distribution = MultinomialDistribution([self.x, self.y, self.z], np.random.rand(len(self.x.domain),
                                                                                        len(self.y.domain),
                                                                                        len(self.z.domain)))
        table = distribution.to_tabulate()
        self.assertTrue(table)

    def test_to_str(self):
        distribution = MultinomialDistribution([self.x, self.y, self.z])
        self.assertTrue(str(distribution))


class MultinomialInferenceTestCase(unittest.TestCase):
    x: Symbolic
    y: Symbolic
    z: Symbolic
    random_distribution: MultinomialDistribution
    random_distribution_mass: float
    crafted_distribution: MultinomialDistribution
    crafted_distribution_mass: float

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        cls.x = Symbolic("X", range(2))
        cls.y = Symbolic("Y", range(3))
        cls.z = Symbolic("Z", range(5))
        cls.random_distribution = MultinomialDistribution([cls.x, cls.y, cls.z], np.random.rand(len(cls.x.domain),
                                                                                                len(cls.y.domain),
                                                                                                len(cls.z.domain)))
        cls.random_distribution_mass = cls.random_distribution.probabilities.sum()

        cls.crafted_distribution = MultinomialDistribution([cls.x, cls.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]]))
        cls.crafted_distribution_mass = cls.crafted_distribution.probabilities.sum()

    def test_normalize_random(self):
        distribution = self.random_distribution.normalize()
        self.assertNotAlmostEqual(self.random_distribution.probabilities.sum(),1.)
        self.assertAlmostEqual(distribution.probabilities.sum(), 1.)

    def test_normalize_crafted(self):
        distribution = self.random_distribution.normalize()
        self.assertNotAlmostEqual(self.random_distribution.probabilities.sum(), self.crafted_distribution_mass)
        self.assertAlmostEqual(distribution.probabilities.sum(), 1.)

    def test_random_marginal_with_normalize(self):
        marginal = self.random_distribution.marginal([self.x, self.y]).normalize()
        self.assertAlmostEqual(marginal.probabilities.sum(), 1)

    def test_crafted_marginal_with_normalize(self):
        marginal = self.crafted_distribution.marginal([self.x]).normalize()
        self.assertAlmostEqual(marginal.probabilities.sum(), 1)
        self.assertAlmostEqual(marginal.probabilities[0], 0.6 / self.crafted_distribution_mass)
        self.assertAlmostEqual(marginal.probabilities[1], 1.2 / self.crafted_distribution_mass)

    def test_random_mode(self):
        mode, probability = self.random_distribution.mode()
        mode = mode[0]
        self.assertEqual(probability, self.random_distribution.probabilities.max())
        self.assertEqual(mode["X"], (0,))
        self.assertEqual(mode["Y"], (0,))

    def test_crafted_mode(self):
        mode, probability = self.crafted_distribution.mode()
        mode = mode[0]
        self.assertEqual(probability, self.crafted_distribution.probabilities.max())
        self.assertEqual(mode["X"], (1,))
        self.assertEqual(mode["Y"], (0,))

    def test_multiple_modes(self):
        distribution = MultinomialDistribution([self.x, self.y], np.array([[0.1, 0.7, 0.3], [0.7, 0.4, 0.1]]), )
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 0.7)
        self.assertEqual(len(mode), 2)
        self.assertEqual(mode[0]["X"], (0,))
        self.assertEqual(mode[0]["Y"], (1,))
        self.assertEqual(mode[1]["X"], (1,))
        self.assertEqual(mode[1]["Y"], (0,))

    def test_crafted_probability(self):
        distribution = self.crafted_distribution.normalize()
        event = Event()
        self.assertAlmostEqual(distribution.probability(event), 1)

        event[self.x] = 0
        self.assertAlmostEqual(distribution.probability(event), 1 / 3)

        event[self.y] = (0, 1)
        self.assertAlmostEqual(distribution.probability(event), 0.3 / self.crafted_distribution_mass)

    def test_random_probability(self):
        distribution = self.random_distribution.normalize()
        event = Event()
        self.assertAlmostEqual(distribution.probability(event), 1)

        event[self.x] = 0
        self.assertLessEqual(distribution.probability(event), 1.)

        event[self.y] = (0, 1)
        self.assertLessEqual(distribution.probability(event), 1.)

    def test_crafted_conditional(self):
        event = Event({self.y: (0, 1)})
        conditional, probability = self.crafted_distribution.conditional(event)
        conditional = conditional.normalize()
        self.assertEqual(conditional.probability(event), 1)
        self.assertEqual(conditional.probability(Event()), 1.)
        self.assertEqual(conditional.probability(Event({self.y: 2})), 0.)

    def test_random_conditional(self):
        event = Event({self.y: (0, 1)})
        conditional, _ = self.random_distribution.conditional(event)
        conditional = conditional.normalize()
        self.assertAlmostEqual(conditional.probability(event), 1)
        self.assertAlmostEqual(conditional.probability(Event()), 1.)
        self.assertEqual(conditional.probability(Event({self.y: 2})), 0.)

    def test_as_probabilistic_circuit(self):
        distribution = self.random_distribution.normalize()
        circuit = distribution.as_probabilistic_circuit()

        for event in itertools.product(self.x.domain, self.y.domain, self.z.domain):
            event = Event(zip([self.x, self.y, self.z], event))
            self.assertAlmostEqual(distribution.probability(event),
                                   circuit.probability(event))

    def test_serialization(self):
        distribution = self.random_distribution
        serialized = distribution.to_json()
        deserialized = MultinomialDistribution.from_json(serialized)
        self.assertEqual(distribution, deserialized)


if __name__ == '__main__':
    unittest.main()