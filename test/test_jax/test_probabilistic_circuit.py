import unittest

import flax.linen
import jax
import numpy as np
import plotly.graph_objects as go
from jax import random, numpy as jnp
from random_events.variable import Continuous

from probabilistic_model.learning.jax.probabilistic_circuit import ProbabilisticCircuit
from probabilistic_model.learning.nyga_distribution import NygaDistribution


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
        probabilistic_circuit = ProbabilisticCircuit.from_probabilistic_circuit(
            self.nyga_distribution.probabilistic_circuit)
        self.assertIsInstance(probabilistic_circuit, ProbabilisticCircuit)
        self.assertEqual(len(probabilistic_circuit.nodes), len(self.nyga_distribution.probabilistic_circuit.nodes))
        self.assertEqual(len(probabilistic_circuit.edges), len(self.nyga_distribution.probabilistic_circuit.edges))

    def test_likelihood(self):
        probabilistic_circuit = ProbabilisticCircuit.from_probabilistic_circuit(
            self.nyga_distribution.probabilistic_circuit)
        log_likelihood = probabilistic_circuit.log_likelihood(jnp.array(self.data[:, (1,)]))
        self.assertTrue(jnp.allclose(log_likelihood, self.nyga_distribution.log_likelihood(self.data[:, (1,)])))

    def test_coupling_circuit(self):
        features = jnp.array(self.data[:, (0,)])
        targets = jnp.array(self.data[:, (1,)])

        pc = ProbabilisticCircuit.from_probabilistic_circuit(self.nyga_distribution.probabilistic_circuit)

        coupling_model = flax.linen.Dense(pc.number_of_parameters)
        coupling_params = coupling_model.init(random.PRNGKey(0), features)

        def loss(params, x, y):
            pc_params = coupling_model.apply(params, x)
            pc.set_parameters(pc_params)
            a_nll = -pc.log_likelihood(y).mean()
            return a_nll

        LEARNING_RATE = 0.05

        @jax.jit
        def update(params, x: jnp.ndarray, y: jnp.ndarray):
            """Performs one SGD update step on params using the given data."""
            grad = jax.grad(loss)(params, x, y)

            new_params = jax.tree_map(
                lambda param, g: param - g * LEARNING_RATE, params, grad)

            return new_params

        loss_values = []

        # Fit regression
        for _ in range(2):
            coupling_params = update(coupling_params, features, targets)
            loss_values.append(loss(coupling_params, features, targets))
        #
        # import plotly.express as px
        # fig = px.line(y=loss_values)
        # fig.update_layout(title="Coupling Circuit Learning Curve", xaxis_title="Epochs",
        #                   yaxis_title="Average Negative Log Likelihood")
        # fig.show()
