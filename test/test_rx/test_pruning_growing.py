import unittest
import copy
import numpy as np

from probabilistic_model.distributions import GaussianDistribution, UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit, SumUnit, ProductUnit, leaf
from probabilistic_model.probabilistic_circuit.rx.flow_analyzer import CircuitFlowAnalyzer
from random_events.interval import SimpleInterval
from random_events.product_algebra import Continuous


class PruningGrowingTestCase(unittest.TestCase):
    """
    Test pruning and growing functionality of probabilistic circuits.
    """

    x = Continuous("x")
    y = Continuous("y")
    model: ProbabilisticCircuit

    def setUp(self):
        self.model = ProbabilisticCircuit()
        
        root = SumUnit(probabilistic_circuit=self.model)
        prod1 = ProductUnit(probabilistic_circuit=self.model)
        prod2 = ProductUnit(probabilistic_circuit=self.model)
        
        leaf_x1 = leaf(GaussianDistribution(self.x, 0.0, 1.0), self.model)
        leaf_y1 = leaf(GaussianDistribution(self.y, 0.0, 1.0), self.model)
        
        leaf_x2 = leaf(GaussianDistribution(self.x, 2.0, 1.0), self.model)
        leaf_y2 = leaf(GaussianDistribution(self.y, 2.0, 1.0), self.model)
        
        root.add_subcircuit(prod1, np.log(0.7))
        root.add_subcircuit(prod2, np.log(0.3))
        
        prod1.add_subcircuit(leaf_x1)
        prod1.add_subcircuit(leaf_y1)
        
        prod2.add_subcircuit(leaf_x2)
        prod2.add_subcircuit(leaf_y2)

    def test_prune_basic(self):
        data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100)
        
        original_nodes = len(list(self.model.nodes()))
        original_edges = len(list(self.model.edges()))
        
        model_copy = copy.deepcopy(self.model)
        analyzer = CircuitFlowAnalyzer(model_copy)
        pruned = analyzer.prune(data, pruning_percentage=0.3)
        
        self.assertLessEqual(len(list(pruned.nodes())), original_nodes)
        self.assertLessEqual(len(list(pruned.edges())), original_edges)
        self.assertTrue(pruned.is_valid())

    def test_prune_returns_self(self):
        data = np.random.normal(0, 1, size=(50, 2))
        model_copy = copy.deepcopy(self.model)
        analyzer = CircuitFlowAnalyzer(model_copy)
        result = analyzer.prune(data, pruning_percentage=0.2)
        self.assertIs(result, model_copy)

    def test_prune_maintains_validity(self):
        data = np.random.normal(0, 1, size=(50, 2))
        model_copy = copy.deepcopy(self.model)
        analyzer = CircuitFlowAnalyzer(model_copy)
        analyzer.prune(data, pruning_percentage=0.4)
        self.assertTrue(model_copy.is_valid())

    def test_grow_basic(self):
        original_nodes = len(list(self.model.nodes()))
        original_edges = len(list(self.model.edges()))
        
        grown = copy.deepcopy(self.model).grow(noise_variance=0.1)
        
        self.assertGreaterEqual(len(list(grown.nodes())), original_nodes)
        self.assertGreaterEqual(len(list(grown.edges())), original_edges)
        self.assertTrue(grown.is_valid())

    def test_grow_returns_self(self):
        result = self.model.grow(noise_variance=0.1)
        self.assertIs(result, self.model)

    def test_grow_maintains_validity(self):
        self.model.grow(noise_variance=0.2)
        self.assertTrue(self.model.is_valid())

    def test_prune_grow_chain(self):
        data = np.random.normal(0, 1, size=(50, 2))
        
        model_copy = copy.deepcopy(self.model)
        analyzer = CircuitFlowAnalyzer(model_copy)
        pruned = analyzer.prune(data, 0.3)
        result = pruned.grow(0.1)
        
        self.assertTrue(result.is_valid())
        self.assertIs(result.__class__, ProbabilisticCircuit)

    def test_prune_with_uniform_data(self):
        circuit = ProbabilisticCircuit()
        root = SumUnit(probabilistic_circuit=circuit)
        
        u1 = leaf(UniformDistribution(self.x, SimpleInterval(0, 1)), circuit)
        u2 = leaf(UniformDistribution(self.x, SimpleInterval(2, 3)), circuit)
        
        root.add_subcircuit(u1, np.log(0.6))
        root.add_subcircuit(u2, np.log(0.4))
        
        data = np.random.uniform(0, 1, size=(100, 1))
        
        original_edges = len(list(circuit.edges()))
        analyzer = CircuitFlowAnalyzer(circuit)
        analyzer.prune(data, pruning_percentage=0.3)
        
        self.assertLessEqual(len(list(circuit.edges())), original_edges)
        self.assertTrue(circuit.is_valid())

    def test_grow_with_zero_variance(self):
        original_nodes = len(list(self.model.nodes()))
        grown = copy.deepcopy(self.model).grow(noise_variance=0.0)
        
        self.assertGreater(len(list(grown.nodes())), original_nodes)
        self.assertTrue(grown.is_valid())

    def test_extreme_pruning(self):
        data = np.random.normal(0, 1, size=(50, 2))
        
        model_copy = copy.deepcopy(self.model)
        analyzer = CircuitFlowAnalyzer(model_copy)
        pruned = analyzer.prune(data, pruning_percentage=0.8)
        self.assertTrue(pruned.is_valid())

    def test_single_node_circuit(self):
        circuit = ProbabilisticCircuit()
        leaf(GaussianDistribution(self.x, 0.0, 1.0), circuit)
        
        grown = circuit.grow(noise_variance=0.1)
        self.assertGreater(len(list(grown.nodes())), 1)
        self.assertTrue(grown.is_valid())

    def test_pruning_growing_integration(self):
        circuit = ProbabilisticCircuit()
        root = SumUnit(probabilistic_circuit=circuit)
        
        prod1 = ProductUnit(probabilistic_circuit=circuit)
        prod2 = ProductUnit(probabilistic_circuit=circuit)
        prod3 = ProductUnit(probabilistic_circuit=circuit)
        
        leaf_x1 = leaf(GaussianDistribution(self.x, 0.0, 1.0), circuit)
        leaf_y1 = leaf(GaussianDistribution(self.y, 0.0, 1.0), circuit)
        leaf_x2 = leaf(GaussianDistribution(self.x, 3.0, 1.0), circuit)
        leaf_y2 = leaf(GaussianDistribution(self.y, 3.0, 1.0), circuit)
        leaf_x3 = leaf(GaussianDistribution(self.x, -3.0, 1.0), circuit)
        leaf_y3 = leaf(GaussianDistribution(self.y, 3.0, 1.0), circuit)
        
        root.add_subcircuit(prod1, np.log(0.5))
        root.add_subcircuit(prod2, np.log(0.3))
        root.add_subcircuit(prod3, np.log(0.2))
        
        prod1.add_subcircuit(leaf_x1)
        prod1.add_subcircuit(leaf_y1)
        prod2.add_subcircuit(leaf_x2)
        prod2.add_subcircuit(leaf_y2)
        prod3.add_subcircuit(leaf_x3)
        prod3.add_subcircuit(leaf_y3)
        
        data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100)
        
        original_ll = circuit.log_likelihood(data[:10])
        original_nodes = len(list(circuit.nodes()))
        
        analyzer = CircuitFlowAnalyzer(circuit)
        pruned_circuit = analyzer.prune(data, 0.5)
        result = pruned_circuit.grow(0.1)
        
        final_ll = result.log_likelihood(data[:10])
        final_nodes = len(list(result.nodes()))
        
        self.assertTrue(result.is_valid())
        self.assertEqual(len(original_ll), 10)
        self.assertEqual(len(final_ll), 10)
        
        samples = result.sample(20)
        self.assertEqual(len(samples), 20)
        self.assertEqual(samples.shape[1], 2)

    def test_performance_with_larger_circuit(self):
        circuit = ProbabilisticCircuit()
        root = SumUnit(probabilistic_circuit=circuit)
        
        for i in range(5):
            prod = ProductUnit(probabilistic_circuit=circuit)
            leaf_x = leaf(GaussianDistribution(self.x, i * 0.5, 1.0), circuit)
            leaf_y = leaf(GaussianDistribution(self.y, i * 0.5, 1.0), circuit)
            
            prod.add_subcircuit(leaf_x)
            prod.add_subcircuit(leaf_y)
            
            weight = 1.0 / 5  # Equal weights
            root.add_subcircuit(prod, np.log(weight))
        
        data = np.random.normal(0, 1, size=(100, 2))
        
        analyzer = CircuitFlowAnalyzer(circuit)
        pruned = analyzer.prune(data, 0.3)
        grown = pruned.grow(0.1)
        
        self.assertTrue(grown.is_valid())
        self.assertGreater(len(list(grown.nodes())), 5)

if __name__ == '__main__':
    unittest.main()
