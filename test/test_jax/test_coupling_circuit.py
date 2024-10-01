import unittest

import jax
import numpy as np

from probabilistic_model.probabilistic_circuit.jax import UniformLayer, SumLayer
from probabilistic_model.probabilistic_circuit.jax.coupling_circuit import Conditioner, CouplingCircuit

import equinox as eqx
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

class TrivialConditioner(Conditioner):

    def generate_parameters(self, x):
         return jnp.log(jnp.array([[0.2, 0.8]]).repeat(x.shape[0], 0))

    @property
    def output_length(self):
        return 2

class LessTrivialConditioner(Conditioner):

    linear = eqx.nn.Linear(1, 2, key=jax.random.PRNGKey(0))

    def generate_parameters(self, x):
        return self.linear(x)

    @property
    def output_length(self):
        return self.linear.out_features

class CouplingCircuitTestCase(unittest.TestCase):

    data = jnp.array(np.vstack((np.random.uniform(0, 1, (100, 1)),
                      np.random.uniform(2, 3, (200, 1)))))
    uniform_layer = UniformLayer(0, jnp.array([[-0.01, 1.01],
                                                       [1.99, 3.01]]))
    sum_layer = SumLayer([uniform_layer], [BCOO((jnp.array([0., 0.]),
                                                 jnp.array([[0, 0], [0, 1]])),
                                                shape=(1, 2))])

    cc: CouplingCircuit

    def setUp(self):
        self.cc = CouplingCircuit(TrivialConditioner(), jnp.array([0]), self.sum_layer, jnp.array([0]))
        self.cc.validate()

    def test_log_likelihood(self):
        ll = self.cc.conditional_log_likelihood(self.data)
        ll1 = ll[0, 0]
        ll2 = ll[-1, 0]
        self.assertAlmostEqual(ll1, jnp.log(jnp.array([0.2/1.02])), delta=10e-4)
        self.assertAlmostEqual(ll2, jnp.log(jnp.array([0.8/1.02])), delta=10e-4)

    def test_vmap(self):
        cc = CouplingCircuit(LessTrivialConditioner(), jnp.array([0]), self.sum_layer, jnp.array([0]))
        r = cc.conditional_log_likelihood(self.data)

        p1 = jnp.exp(cc.conditioner.generate_parameters(self.data[0]))
        p1 /= jnp.sum(p1)
        p2 = jnp.exp(cc.conditioner.generate_parameters(self.data[1]))
        p2 /= jnp.sum(p2)

        r1 = r[0, 0]
        r2 = r[1, 0]

        # check that if the probability of the first uniform is higher than also,
        # the probability of the sample is higher
        self.assertEqual(p1[0] > p2[0], r1[0] > r2[0])

if __name__ == '__main__':
    unittest.main()
