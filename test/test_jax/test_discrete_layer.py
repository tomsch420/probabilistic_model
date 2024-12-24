import unittest

from random_events.set import SetElement
from random_events.variable import Continuous, Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.distributions import SymbolicDistribution
from probabilistic_model.probabilistic_circuit.jax.discrete_layer import DiscreteLayer
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import GaussianLayer, GaussianDistribution
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import \
    ProbabilisticCircuit as NXProbabilisticCircuit, SumUnit
from probabilistic_model.probabilistic_circuit.nx.distributions import UnivariateContinuousLeaf, UnivariateDiscreteLeaf
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit
import jax.numpy as jnp

from probabilistic_model.utils import MissingDict


class Animal(SetElement):
    EMPTY_SET = -1
    CAT = 0
    DOG = 1
    FISH = 2

class DiscreteLayerTestCase(unittest.TestCase):

    model: DiscreteLayer
    x = Symbolic("x", Animal)

    @classmethod
    def setUpClass(cls):
        cls.model = DiscreteLayer(0, jnp.log(jnp.array([[0, 1, 2], [3, 4, 0]])))
        cls.model.validate()

    def test_normalization(self):
        result = self.model.log_normalization_constant
        correct = jnp.log(jnp.array([3., 7.]))
        self.assertTrue(jnp.allclose(result, correct, atol=1e-3))

    def test_log_likelihood(self):
        x = jnp.array([0.0])
        result = self.model.log_likelihood_of_nodes_single(x)
        correct = jnp.log(jnp.array([.0, 3/7]))
        self.assertTrue(jnp.allclose(result, correct, atol=1e-3))

        x2 = jnp.array([0., 1., 2]).reshape(-1, 1)
        result = self.model.log_likelihood_of_nodes(x2)
        self.assertEqual(result.shape, (3, 2))
        correct = jnp.log(jnp.array([[0., 3./7.], [1./3., 4./7.], [2./3., 0.]]))
        self.assertTrue(jnp.allclose(result, correct, atol=1e-3))


    def test_from_nx(self):

        p1 = MissingDict(float, {Animal.CAT: 0., Animal.DOG: 1, Animal.FISH: 2})
        d1 = UnivariateDiscreteLeaf(SymbolicDistribution(self.x, p1))

        p2 = MissingDict(float, {Animal.CAT: 3, Animal.DOG: 4, Animal.FISH: 0})
        d2 = UnivariateDiscreteLeaf(SymbolicDistribution(self.x, p2))
        s = SumUnit()
        s.add_subcircuit(d1, 0.5)
        s.add_subcircuit(d2, 0.5)
        nx_pc = s.probabilistic_circuit
        jax_pc = ProbabilisticCircuit.from_nx(nx_pc)
        discrete_layer = jax_pc.root.child_layers[0]
        self.assertIsInstance(discrete_layer, DiscreteLayer)
        self.assertEqual(discrete_layer.variable, 0)
        self.assertEqual(discrete_layer.log_probabilities.shape, (2, 3))
        self.assertTrue(jnp.allclose(discrete_layer.log_probabilities, self.model.log_probabilities))

    def test_to_nx(self):
        nx_circuit = self.model.to_nx(SortedSet([self.x]), NXProbabilisticCircuit())[0].probabilistic_circuit
        self.assertEqual(len(nx_circuit.nodes()), 2)
        self.assertEqual(len(nx_circuit.edges()), 0)
        for node in nx_circuit.nodes():
            self.assertIsInstance(node, UnivariateDiscreteLeaf)
            self.assertEqual(node.variable, self.x)
            distribution: SymbolicDistribution = node.distribution
            self.assertAlmostEqual(sum(distribution.probabilities.values()), 1., places=5)


if __name__ == '__main__':
    unittest.main()
