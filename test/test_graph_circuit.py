import unittest

import portion
from matplotlib import pyplot as plt
from random_events.events import Event
from random_events.variables import Integer, Symbolic, Continuous

from probabilistic_model.graph_circuits.probabilistic_circuit import *

from probabilistic_model.distributions.uniform import UniformDistribution


class MinimalGraphCircuitTestCase(unittest.TestCase):
    integer = Integer("integer", (1, 2, 4))
    symbol = Symbolic("symbol", ("a", "b", "c"))
    real = Continuous("x")
    real2 = Continuous("y")

    model: ProbabilisticCircuit

    def setUp(self):
        model = ProbabilisticCircuit()
        u1 = LeafComponent(UniformDistribution(self.real, portion.closed(0, 1)))
        u2 = LeafComponent(UniformDistribution(self.real, portion.closed(3, 5)))
        model.add_node(u1)
        model.add_node(u2)

        sum_unit_1 = DeterministicSumUnit()

        model.add_node(sum_unit_1)

        e1 = DirectedWeightedEdge(sum_unit_1, u1, 0.5)
        e2 = DirectedWeightedEdge(sum_unit_1, u2, 0.5)

        model.add_edge(e1)
        model.add_edge(e2)

        u3 = LeafComponent(UniformDistribution(self.real2, portion.closed(2, 2.25)))
        u4 = LeafComponent(UniformDistribution(self.real2, portion.closed(2, 5)))
        sum_unit_2 = SmoothSumUnit()

        model.add_nodes_from([u3, u4, sum_unit_2])

        e3 = DirectedWeightedEdge(sum_unit_2, u3, 0.7)
        e4 = DirectedWeightedEdge(sum_unit_2, u4, 0.3)

        model.add_edges_from([e3, e4])

        product_1 = DecomposableProductUnit()
        model.add_node(product_1)

        e5 = Edge(product_1, sum_unit_1)
        e6 = Edge(product_1, sum_unit_2)
        model.add_edges_from([e5, e6])

        self.model = model

    def show(self):
        nx.draw(self.model, with_labels=True)
        plt.show()

    def test_setup(self):
        node_ids = set()
        for node in self.model.nodes():
            self.assertIsNotNone(node.id)
            self.assertTrue(node.id not in node_ids)
            node_ids.add(node.id)
            self.assertIsNotNone(node.probabilistic_circuit)

    def test_variables(self):
        self.assertEqual(self.model.variables, (self.real, self.real2))

    def test_is_valid(self):
        self.assertTrue(self.model.is_valid())

    def test_root(self):
        self.assertEqual(self.model.root.id, 6)

    def test_variables_of_component(self):
        self.assertEqual(self.model.root.variables, (self.real, self.real2))
        self.assertEqual(list(self.model.nodes)[2].variables, (self.real,))
        self.assertEqual(list(self.model.nodes)[5].variables, (self.real2,))
        self.assertEqual(list(self.model.nodes)[6].variables, (self.real, self.real2))

    def test_likelihood(self):
        event = [1., 2.]
        result = self.model.likelihood(event)
        self.assertEqual(result, 0.5 * ((4 * 0.7) + (1/3 * 0.3)))

    def test_probability_everywhere(self):
        event = Event({self.real: portion.closed(0, 5),
                       self.real2: portion.closed(2, 5)})
        result = self.model.probability(event)
        self.assertEqual(result, 1.)

    def test_probability_nowhere(self):
        event = Event({self.real: portion.closed(0, 0.5),
                       self.real2: portion.closed(0, 1)})
        result = self.model.probability(event)
        self.assertEqual(result, 0.)

    def test_probability_somewhere(self):
        event = Event({self.real2: portion.closed(0, 3)})
        result = self.model.probability(event)
        self.assertAlmostEqual(result, 0.8)

    def test_caching_reset(self):
        event = Event({self.real: portion.closed(0, 5),
                       self.real2: portion.closed(2, 5)})
        _ = self.model.probability(event)

        for node in self.model.nodes():
            self.assertIsNone(node.result_of_current_query)

    def test_caching(self):
        event = Event({self.real: portion.closed(0, 5),
                       self.real2: portion.closed(2, 5)})
        _ = self.model.root.probability(event)

        for node in self.model.nodes():
            if not isinstance(node, LeafComponent):
                self.assertIsNotNone(node.result_of_current_query)

    def test_mode(self):
        mode, likelihood = list(self.model.nodes)[2].mode()
        self.assertEqual(likelihood, 0.5)
        self.assertEqual(mode, [Event({self.real: portion.closed(0, 1)})])

    def test_mode_raising(self):
        with self.assertRaises(NotImplementedError):
            _ = self.model.mode()

    def test_mode_with_product(self):
        non_deterministic_node = [node for node in self.model.nodes() if node.id == 5][0]

        for descendant in nx.descendants(self.model, non_deterministic_node):
            self.model.remove_node(descendant)

        self.model.remove_node(non_deterministic_node)
        new_node = LeafComponent(UniformDistribution(self.real2, portion.closed(2, 3)))

        new_edge = Edge(self.model.root, new_node)
        self.model.add_edge(new_edge)

        self.assertTrue(new_node in self.model.nodes())

        mode, likelihood = self.model.mode()
        self.assertEqual(likelihood, 0.5)
        self.assertEqual(mode, [Event({self.real: portion.closed(0, 1),
                                       self.real2: portion.closed(2, 3)})])


if __name__ == '__main__':
    unittest.main()
