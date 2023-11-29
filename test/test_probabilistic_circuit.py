import unittest

import anytree
import portion
from random_events.events import Event

from probabilistic_model.probabilistic_circuit.units import Unit, DeterministicSumUnit, DecomposableProductUnit, \
    SumUnit, ProductUnit, SmoothSumUnit
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
        model_2 *= UniformDistribution(real2, portion.closedopen(0, 1))
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

class SimplifyTestCase (unittest.TestCase):

    model: Unit

    @classmethod
    def setUpClass(cls):
        integer = Integer("integer", (1, 2, 4))
        integer_distribution_0 = IntegerDistribution(integer, [0.5, 0.5, 0])
        integer_distribution_0_not = IntegerDistribution(integer, [0, 0, 1])

        symbol = Symbolic("symbol", ("a", "b", "c"))
        symbolic_distribution_0 = SymbolicDistribution(symbol, [0, 1, 0])
        symbolic_distribution_0_not = SymbolicDistribution(symbol, [0.5, 0, 0.5])

        real = Continuous("real")
        real_distribution_0 = UniformDistribution(real, portion.closedopen(0, 1))
        real_distribution_0_not = UniformDistribution(real, portion.closedopen(1.5, 2.5))

        integer = Integer("integer", (6, 10, 12))
        integer_distribution_1 = IntegerDistribution(integer, [0.5, 0.5, 0])
        integer_distribution_1_not = IntegerDistribution(integer, [0, 0, 1])

        product_0 = integer_distribution_0 * real_distribution_0_not
        sum_0 = integer_distribution_0 + integer_distribution_0_not
        product_1 = sum_0 * real_distribution_0
        product_2 = symbolic_distribution_0 * integer_distribution_1
        sum_1 = integer_distribution_1 + integer_distribution_1_not
        product_3 = symbolic_distribution_0_not * sum_1
        sum_3 = product_0 + product_1
        sum_4 = product_2 + product_3

        model = sum_3 * sum_4
        model.weights = [0.4, 0.6]
        #print(RenderTree(model))
        # print(model.is_decomposable())
        # print(model.is_smooth())
        #print(model.is_deterministic())
        cls.model = model.maximize_expressiveness()
        #print(RenderTree(cls.model))
        print(RenderTree(cls.model.simplify()))

    def test_simplify_complex_case(self):
        simplified_model = self.model.simplify()
        self.assertIsInstance(simplified_model.children[0], DeterministicSumUnit)
        self.assertIsInstance(simplified_model.children[1], DeterministicSumUnit)
        self.assertIsInstance(simplified_model.children[0].children[0], UniformDistribution)
        self.assertIsInstance(simplified_model.children[0].children[1], DecomposableProductUnit)
        self.assertIsInstance(simplified_model.children[0].children[1].children[0], DeterministicSumUnit)
        self.assertIsInstance(simplified_model.children[0].children[1].children[1], UniformDistribution)
        self.assertIsInstance(simplified_model.children[1].children[0], SymbolicDistribution)
        self.assertIsInstance(simplified_model.children[1].children[1], DecomposableProductUnit)
        self.assertIsInstance(simplified_model.children[1].children[1].children[0], SymbolicDistribution)
        self.assertIsInstance(simplified_model.children[1].children[1].children[1], DeterministicSumUnit)
        self.assertIsInstance(simplified_model.children[1].children[1].children[1].children[0], IntegerDistribution)
        self.assertIsInstance(simplified_model.children[1].children[1].children[1].children[1], IntegerDistribution)

    def test_simplify_only_smooth_sum_units(self):
        real = Continuous("real")

        model = SmoothSumUnit([real], [0.4, 0.6])
        child_1 = SmoothSumUnit([real], [0.25, 0.75], parent=model)
        child_1.children = [UniformDistribution(real, portion.closedopen(0, 1)),
                            UniformDistribution(real, portion.closedopen(1, 2))]

        child_2 = SmoothSumUnit([real], [1.], parent=model)
        child_2.children = [UniformDistribution(real, portion.closedopen(0, 1))]

        simplified = model.simplify()
        self.assertEqual(simplified.weights, [0.4*0.25, 0.4*0.75, 0.6])

    def test_simplify_smooth_and_deterministic_sum_units(self):
        real = Continuous("real")

        model = SmoothSumUnit([real], [0.4, 0.6])
        child_1 = DeterministicSumUnit([real], [0.25, 0.75], parent=model)
        child_1.children = [UniformDistribution(real, portion.closedopen(0, 1)),
                            UniformDistribution(real, portion.closedopen(1, 2))]

        child_2 = SmoothSumUnit([real], [1.], parent=model)
        child_2.children = [UniformDistribution(real, portion.closedopen(0, 1))]
        simplified = model.simplify()
        self.assertEqual(simplified.weights, [0.4, 0.6])
        self.assertIsInstance(simplified.children[0], DeterministicSumUnit)
        self.assertIsInstance(simplified.children[1], UniformDistribution)
        self.assertIsInstance(simplified, SmoothSumUnit)

    def test_simplify_one_child_only(self):
        real = Continuous("real")
        model = SmoothSumUnit([real], [1.])
        deterministic_sum_unit = DeterministicSumUnit([real], [1.], parent=model)
        deterministic_sum_unit.children = [UniformDistribution(real, portion.closedopen(0, 1))]
        simplified = model.simplify()
        self.assertEqual(simplified, UniformDistribution(real, portion.closedopen(0, 1)))

    def test_simplify_decomposable_products(self):
        real = Continuous("real")
        real_2 = Continuous("real2")
        real_3 = Continuous("real3")
        real_4 = Continuous("real4")

        distribution_1 = UniformDistribution(real, portion.closedopen(0, 1))
        distribution_2 = UniformDistribution(real_2, portion.closedopen(1, 2))
        distribution_3 = UniformDistribution(real_3, portion.closedopen(2, 3))
        distribution_4 = UniformDistribution(real_4, portion.closedopen(3, 4))

        model = distribution_1 * distribution_2 * distribution_3 * distribution_4
        simplified_model = model.simplify()
        print(RenderTree(simplified_model))
        self.assertIsInstance(simplified_model.children[0], UniformDistribution)
        self.assertIsInstance(simplified_model.children[1], UniformDistribution)
        self.assertIsInstance(simplified_model.children[2], UniformDistribution)
        self.assertIsInstance(simplified_model.children[3], UniformDistribution)
        self.assertIsInstance(simplified_model, DecomposableProductUnit)



if __name__ == '__main__':
    unittest.main()
