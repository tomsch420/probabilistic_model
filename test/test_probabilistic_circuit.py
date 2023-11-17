import unittest

import portion
from random_events.events import Event

from probabilistic_model.probabilistic_circuit.units import Unit, DeterministicSumUnit, DecomposableProductUnit, \
    SumUnit, ProductUnit
from probabilistic_model.probabilistic_circuit.distributions import (IntegerDistribution, UniformDistribution,
                                                                     SymbolicDistribution)
from random_events.variables import Integer, Symbolic, Continuous
from anytree import PreOrderIter, RenderTree


class ProbabilisticCircuitTestCase(unittest.TestCase):

    model: Unit

    @classmethod
    def setUpClass(cls):
        integer = Integer("integer", (1, 2, 4))
        integer_distribution_0 = IntegerDistribution(integer, [0.1, 0.2, 0.7])
        integer_distribution_1 = IntegerDistribution(integer, [0.5, 0.3, 0.2])

        symbol = Symbolic("symbol", ("a", "b", "c"))
        symbolic_distribution_0 = SymbolicDistribution(symbol, [0, 1, 0])
        symbolic_distribution_1 = SymbolicDistribution(symbol, [0.5, 0, 0.5])

        real = Continuous("real")
        real_distribution_0 = UniformDistribution(real, portion.closedopen(0, 1))
        real_distribution_1 = UniformDistribution(real, portion.closedopen(0.5, 2))

        product_0 = integer_distribution_0 * symbolic_distribution_0 * real_distribution_0
        product_1 = integer_distribution_1 * symbolic_distribution_1 * real_distribution_1
        model = product_0 + product_1
        model = DeterministicSumUnit.from_sum_unit(model)
        model.weights = [0.4, 0.6]
        cls.model = model

    def test_model_layout(self):
        self.assertEqual(len(self.model.variables), 3)
        self.assertEqual(len(self.model.children), 2)
        self.assertIsInstance(self.model.children[0], DecomposableProductUnit)
        self.assertIsInstance(self.model.children[1], DecomposableProductUnit)
        self.assertEqual(len(list(PreOrderIter(self.model))), 11)
        print(RenderTree(self.model))
        print(self.model)

    def test_likelihood(self):
        likelihood = self.model.likelihood([1, 0.5, "a",])
        self.assertEqual(likelihood, 0.6 * 0.5 * (1/1.5) * 0.5)

    def test_probability(self):
        self.assertEqual(self.model.probability(Event()), 1)
        event = Event({
            self.model.variables[0]: {1, 4},
            self.model.variables[1]: portion.closedopen(0, 1),
            self.model.variables[2]: ["a", "b"]})
        self.assertEqual(self.model.probability(event), 0.39)

    def test_mode(self):
        modes, likelihood = self.model.mode()
        self.assertEqual(likelihood, 0.4*0.7)
        self.assertEqual(modes, [Event({
            self.model.variables[0]: 4,
            self.model.variables[1]: portion.closedopen(0, 1),
            self.model.variables[2]: "b"})])

    def test_conditional_of_simple_product(self):
        symbol = Symbolic("symbol", ("a", "b", "c"))
        symbolic_distribution_0 = SymbolicDistribution(symbol, [0, 1, 0])
        integer = Integer("integer", (1, 2, 4))
        integer_distribution_0 = IntegerDistribution(integer, [0.1, 0.2, 0.7])

        product = integer_distribution_0 * symbolic_distribution_0
        conditional, probability = product.conditional(Event({integer: 1}))
        self.assertEqual(probability, 0.1)
        self.assertEqual(conditional.probability(Event()), 1)

    def test_conditional(self):
        event = Event({
            self.model.variables[0]: {1, 4},
            self.model.variables[1]: portion.closedopen(0, 1),
            self.model.variables[2]: ["a", "b"]})
        conditional, probability = self.model.conditional(event)
        self.assertEqual(probability, 0.39)
        self.assertEqual(conditional.probability(Event()), 1)
        self.assertEqual(conditional.probability(event), 1)

    def test_sample(self):
        samples = self.model.sample(100)
        self.assertTrue(all([self.model.likelihood(sample) > 0 for sample in samples]))

    def test_marginal(self):
        marginal = self.model.marginal([self.model.variables[0], self.model.variables[1]])
        self.assertEqual(marginal.probability(Event()), 1)
        self.assertEqual(marginal.variables, self.model.variables[:2])

    def test_expectation(self):
        expectation = self.model.expectation([self.model.variables[0], self.model.variables[1]])
        self.assertEqual(expectation["integer"], 2.46)
        self.assertEqual(expectation["real"], 0.95)

    def test_variance(self):
        variance = self.model.variance([self.model.variables[0], self.model.variables[1]])
        self.assertAlmostEqual(variance["integer"], 1.7284, delta=0.001)
        self.assertAlmostEqual(variance["real"], 0.2808, delta=0.001)

    def test_domain(self):
        domain = self.model.domain
        self.assertEqual(domain["integer"], (1, 2, 4))
        self.assertEqual(domain["real"], portion.closedopen(0, 2))
        self.assertEqual(domain["symbol"], ("a", "b", "c"))

    def test_determinism(self):
        self.assertTrue(self.model.is_deterministic())

    def test_non_determinism(self):
        variable = Continuous("real")
        distribution = UniformDistribution(variable, portion.closedopen(0, 1)) + UniformDistribution(variable,
                                                                                                     portion.closedopen(0.5, 2))
        self.assertFalse(distribution.is_deterministic())

    def test_smoothness(self):
        self.assertTrue(self.model.is_smooth())

    def test_non_smoothness(self):
        variable_0 = Continuous("real0")
        variable_1 = Continuous("real1")
        distribution = UniformDistribution(variable_0, portion.closedopen(0, 1)) + UniformDistribution(variable_1, portion.closedopen(0.5, 2))
        self.assertFalse(distribution.is_smooth())

    def test_decomposable(self):
        self.assertTrue(self.model.is_decomposable())

    def test_non_decomposable(self):
        variable = Continuous("real")
        distribution = UniformDistribution(variable,  portion.closedopen(0, 1)) * UniformDistribution(variable, portion.closedopen(0.5, 2))
        self.assertFalse(distribution.is_decomposable())

    def test_maximum_expressiveness_of_sum(self):
        real = Continuous("real")
        distribution_1 = UniformDistribution(real, portion.closedopen(0, 1))
        distribution_2 = UniformDistribution(real, portion.closedopen(1, 3))
        distribution = SumUnit([real], [0.7, 0.3])
        distribution.children = [distribution_1, distribution_2]

        distribution = distribution.maximize_expressiveness()
        self.assertTrue(distribution.is_decomposable())
        self.assertTrue(distribution.is_smooth())
        self.assertTrue(distribution.is_deterministic())
        self.assertIsInstance(distribution, DeterministicSumUnit)

    def test_maximum_expressiveness_of_product(self):
        real_1 = Continuous("real1")
        real_2 = Continuous("real2")
        distribution = ProductUnit([real_1, real_2])
        distribution_1 = UniformDistribution(real_1, portion.closedopen(0, 1), parent=distribution)
        distribution_2 = UniformDistribution(real_2, portion.closedopen(1, 3), parent=distribution)
        self.assertEqual(distribution.children, (distribution_1, distribution_2))
        self.assertIsInstance(distribution, ProductUnit)
        distribution = distribution.maximize_expressiveness()
        self.assertTrue(distribution.is_decomposable())
        self.assertTrue(distribution.is_smooth())
        self.assertTrue(distribution.is_deterministic())
        self.assertIsInstance(distribution, DecomposableProductUnit)

    def test_equality(self):
        self.assertEqual(self.model, self.model)
        model_2 = self.model.__copy__()
        self.assertEqual(self.model, model_2)
        real2 = Continuous("real2")
        model_2 *= UniformDistribution(real2, 0, 1)
        self.assertNotEqual(self.model, model_2)

    def test_to_json(self):
        json = self.model.to_json()
        model = Unit.from_json(json)
        self.assertEqual(self.model, model)

        event = Event({
            self.model.variables[0]: {1, 4},
            self.model.variables[1]: portion.closedopen(0, 1),
            self.model.variables[2]: ["a", "b"]})

        self.assertEqual(self.model.probability(event), model.probability(event))


if __name__ == '__main__':
    unittest.main()
