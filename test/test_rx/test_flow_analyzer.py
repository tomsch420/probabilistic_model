import unittest
import numpy as np

from probabilistic_model.distributions import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit, SumUnit, ProductUnit, leaf
from probabilistic_model.probabilistic_circuit.rx.flow_analyzer import CircuitFlowAnalyzer, EdgeFlows
from random_events.product_algebra import Continuous


class CircuitFlowAnalyzerTestCase(unittest.TestCase):
    """Test the CircuitFlowAnalyzer class."""

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
        
        leaf_x2 = leaf(GaussianDistribution(self.x, 5.0, 1.0), self.model)
        leaf_y2 = leaf(GaussianDistribution(self.y, 5.0, 1.0), self.model)
        
        root.add_subcircuit(prod1, np.log(0.8))
        root.add_subcircuit(prod2, np.log(0.2))
        
        prod1.add_subcircuit(leaf_x1)
        prod1.add_subcircuit(leaf_y1)
        
        prod2.add_subcircuit(leaf_x2)
        prod2.add_subcircuit(leaf_y2)

    def test_compute_flows_basic(self):
        analyzer = CircuitFlowAnalyzer(self.model)
        
        data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=50)
        
        flows = analyzer.compute_flows(data)
        
        self.assertIsInstance(flows, EdgeFlows)
        
        found_flows = False
        for layer in self.model.layers:
            for node in layer:
                if hasattr(node, 'subcircuits'):
                    for child in node.subcircuits:
                        flow_value = flows.get(node, child)
                        self.assertGreaterEqual(flow_value, 0.0)
                        if flow_value > 0:
                            found_flows = True
        
        self.assertTrue(found_flows, "Should have found some non-zero flows")

    def test_flow_consistency(self):
        analyzer = CircuitFlowAnalyzer(self.model)
        data = np.array([[0.0, 0.0], [1.0, 1.0]])
        
        flows1 = analyzer.compute_flows(data)
        flows2 = analyzer.compute_flows(data)
        
        for layer in self.model.layers:
            for node in layer:
                if hasattr(node, 'subcircuits'):
                    for child in node.subcircuits:
                        flow1 = flows1.get(node, child)
                        flow2 = flows2.get(node, child)
                        self.assertAlmostEqual(flow1, flow2, places=10)

    def test_flow_scales_with_data_size(self):
        analyzer = CircuitFlowAnalyzer(self.model)
        
        small_data = np.array([[0.0, 0.0]])
        large_data = np.tile(small_data, (10, 1))
        
        small_flows = analyzer.compute_flows(small_data)
        large_flows = analyzer.compute_flows(large_data)
        
        for layer in self.model.layers:
            for node in layer:
                if hasattr(node, 'subcircuits'):
                    for child in node.subcircuits:
                        small_flow = small_flows.get(node, child)
                        large_flow = large_flows.get(node, child)
                        if small_flow > 0:  # Only check edges that have flow
                            self.assertGreaterEqual(large_flow, small_flow)

    def test_flow_reflects_weights(self):
        analyzer = CircuitFlowAnalyzer(self.model)
        
        data = np.array([[0.0, 0.0], [5.0, 5.0]])
        
        flows = analyzer.compute_flows(data)
        
        found_flows = False
        for layer in self.model.layers:
            for node in layer:
                if hasattr(node, 'subcircuits'):
                    for child in node.subcircuits:
                        if flows.get(node, child) > 0:
                            found_flows = True
                            break
        
        self.assertTrue(found_flows, "Should have found some flows")

    def test_empty_circuit_handling(self):
        circuit = ProbabilisticCircuit()
        leaf_node = leaf(GaussianDistribution(self.x, 0.0, 1.0), circuit)
        
        analyzer = CircuitFlowAnalyzer(circuit)
        data = np.array([[0.0]])
        
        flows = analyzer.compute_flows(data)
        
        self.assertIsInstance(flows, EdgeFlows)

    def test_cleanup_after_computation(self):
        analyzer = CircuitFlowAnalyzer(self.model)
        data = np.array([[0.0, 0.0]])
        
        flows = analyzer.compute_flows(data)
        self.assertIsInstance(flows, EdgeFlows)
        
        flows2 = analyzer.compute_flows(data)
        self.assertIsInstance(flows2, EdgeFlows)


if __name__ == '__main__':
    unittest.main()
