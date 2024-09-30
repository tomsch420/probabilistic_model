import unittest
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

class CouplingCircuitTestCase(unittest.TestCase):

    data = np.vstack((np.random.uniform(0, 1, (100, 1)),
                      np.random.uniform(2, 3, (200, 1))))
    uniform_layer = UniformLayer(0, jnp.array([[-0.01, 1.01],
                                                       [1.99, 3.01]]))
    sum_layer = SumLayer([uniform_layer], [BCOO((jnp.array([0., 0.]),
                                                 jnp.array([[0, 0], [0, 1]])),
                                                shape=(1, 2))])

    cc = CouplingCircuit(TrivialConditioner(), sum_layer)
    cc.validate()

    def test_log_likelihood(self):
        x = jnp.array(self.data)
        ll = self.cc.conditional_log_likelihood(x)

if __name__ == '__main__':
    unittest.main()
