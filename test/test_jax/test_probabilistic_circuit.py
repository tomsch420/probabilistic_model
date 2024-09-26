import unittest

import jax.numpy as jnp
import numpy as np
from random_events.variable import Continuous

from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.jax import SumLayer
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit
from probabilistic_model.probabilistic_circuit.nx.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import (SumUnit, ProductUnit,
                                                                                ProbabilisticCircuit as NXProbabilisticCircuit)

np.random.seed(69)
import pandas as pd


class SmallCircuitIntegrationTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")

    nx_model = SumUnit()
    jax_model: ProbabilisticCircuit
    nx_model: NXProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        sum1, sum2, sum3 = SumUnit(), SumUnit(), SumUnit()
        sum4, sum5 = SumUnit(), SumUnit()
        prod1, prod2 = ProductUnit(), ProductUnit()

        sum1.add_subcircuit(prod1, 0.5)
        sum1.add_subcircuit(prod2, 0.5)
        prod1.add_subcircuit(sum2)
        prod1.add_subcircuit(sum4)
        prod2.add_subcircuit(sum3)
        prod2.add_subcircuit(sum5)

        d_x1, d_x2 = DiracDeltaDistribution(cls.x, 0, 1), DiracDeltaDistribution(cls.x, 1, 2)
        d_y1, d_y2 = DiracDeltaDistribution(cls.y, 2, 3), DiracDeltaDistribution(cls.y, 3, 4)

        sum2.add_subcircuit(d_x1, 0.8)
        sum2.add_subcircuit(d_x2, 0.2)
        sum3.add_subcircuit(d_x1, 0.7)
        sum3.add_subcircuit(d_x2, 0.3)

        sum4.add_subcircuit(d_y1, 0.5)
        sum4.add_subcircuit(d_y2, 0.5)
        sum5.add_subcircuit(d_y1, 0.1)
        sum5.add_subcircuit(d_y2, 0.9)

        cls.nx_model = sum1.probabilistic_circuit
        cls.jax_model = ProbabilisticCircuit.from_nx(cls.nx_model)

    def test_creation(self):
        self.assertEqual(self.jax_model.variables, self.nx_model.variables)
        self.assertIsInstance(self.jax_model.root, SumLayer)
        self.assertEqual(self.jax_model.root.number_of_nodes, 1)
        self.assertEqual(len(self.jax_model.root.child_layers), 1)
        product_layer = self.jax_model.root.child_layers[0]
        self.assertEqual(product_layer.number_of_nodes, 2)
        self.assertEqual(len(product_layer.child_layers), 2)
        sum_layer1 = product_layer.child_layers[0]
        sum_layer2 = product_layer.child_layers[1]
        self.assertEqual(sum_layer1.number_of_nodes, 2)
        self.assertEqual(sum_layer2.number_of_nodes, 2)

    def test_ll(self):
        samples = self.nx_model.sample(1000)
        nx_ll = self.nx_model.log_likelihood(samples)
        jax_ll = self.jax_model.log_likelihood(samples)
        self.assertTrue(jnp.allclose(nx_ll, jax_ll))


class JPTIntegrationTestCase(unittest.TestCase):
    number_of_variables = 2
    number_of_samples = 10000

    jpt: NXProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        mean = np.full(cls.number_of_variables, 0)
        cov = np.random.uniform(0, 1, (cls.number_of_variables, cls.number_of_variables))
        cov = np.dot(cov, cov.T)
        samples = np.random.multivariate_normal(mean, cov, cls.number_of_samples)
        df = pd.DataFrame(samples, columns=[f"x_{i}" for i in range(cls.number_of_variables)])
        variables = infer_variables_from_dataframe(df, min_samples_per_quantile=100)
        jpt = JPT(variables, min_samples_leaf=0.1)
        jpt.fit(df)
        cls.jpt = jpt.probabilistic_circuit

    def test_from_jpt(self):
        model = ProbabilisticCircuit.from_nx(self.jpt, False)
        samples = jnp.array(self.jpt.sample(1000))
        jax_ll = model.log_likelihood(samples)
        self.assertTrue((jax_ll > -jnp.inf).all())


if __name__ == '__main__':
    unittest.main()
