import unittest

import jax
from random_events.variable import Continuous

from probabilistic_model.learning.nyga_distribution import NygaDistribution
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import *
import plotly.graph_objects as go
import numpy as np
from jax import random, numpy as jnp


class TestJaxUnits(unittest.TestCase):

    x: Continuous = Continuous("x")
    y: Continuous = Continuous("y")
    np.random.seed(69)
    data: np.ndarray = np.random.multivariate_normal(np.array([0, 0]), np.array([[1, .5], [.5, 1]]), size=(1000))
    nyga_distribution: NygaDistribution

    @classmethod
    def setUp(cls) -> None:
        cls.nyga_distribution = NygaDistribution(cls.y, 50)
        cls.nyga_distribution.fit(cls.data[:, 1])

    def show(self):
        fig = go.Figure(self.nyga_distribution.plot(), self.nyga_distribution.plotly_layout())
        fig.show()

    def test_from_probabilistic_circuit(self):
        probabilistic_circuit = ProbabilisticCircuit.from_probabilistic_circuit(self.nyga_distribution.probabilistic_circuit)
        self.assertIsInstance(probabilistic_circuit, ProbabilisticCircuit)
        self.assertEqual(len(probabilistic_circuit.nodes), len(self.nyga_distribution.probabilistic_circuit.nodes))
        self.assertEqual(len(probabilistic_circuit.edges), len(self.nyga_distribution.probabilistic_circuit.edges))

    def test_likelihood(self):
        probabilistic_circuit = ProbabilisticCircuit.from_probabilistic_circuit(self.nyga_distribution.probabilistic_circuit)
        log_likelihood = probabilistic_circuit.log_likelihood(jnp.array(self.data[:, (1, )]))
        self.assertTrue(jnp.allclose(log_likelihood, self.nyga_distribution.log_likelihood(self.data[:, (1, )])))

    def test_tunable_parameters(self):
        pc = ProbabilisticCircuit.from_probabilistic_circuit(self.nyga_distribution.probabilistic_circuit)
        params = pc.tunable_parameters()
        ll = pc.log_likelihood_given_state(jnp.array(self.data[:, (1, )]), params)
        self.assertTrue(all(jnp.exp(ll) > 0))

    def test_gradient(self):
        pc = ProbabilisticCircuit.from_probabilistic_circuit(self.nyga_distribution.probabilistic_circuit)
        params = pc.tunable_parameters()

        data = jnp.array(self.data[:500, (1, )])

        model = pc.root
        a_nll = model.average_negative_log_likelihood(params, data)
        grad = jax.grad(model.average_negative_log_likelihood)(params, data)

        new_params = jax.tree.map(
            lambda param, g: param - g * 0.01, params, grad)
        better_a_nll = model.average_negative_log_likelihood(new_params, data)
        self.assertGreater(a_nll, better_a_nll)

    def test_linear_regression(self):
        ...
