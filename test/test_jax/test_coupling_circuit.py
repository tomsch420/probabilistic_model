import unittest

import jax
import numpy as np

from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.jax import UniformLayer, SumLayer
from probabilistic_model.probabilistic_circuit.jax.coupling_circuit import Conditioner, CouplingCircuit, \
    LinearConditioner

import equinox as eqx
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import pandas as pd
import optax
import plotly.graph_objects as go
import tqdm


from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit


class TrivialConditioner(Conditioner):

    def generate_parameters(self, x):
         return jnp.log(jnp.array([0.2, 0.8]).repeat(x.shape[0], 0))

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
        r1 = r[0]
        r2 = r[1]

        # check that if the probability of the first uniform is higher than also,
        # the probability of the sample is higher
        self.assertEqual(p1[0] > p2[0], r1 > r2)

    def test_parameter_count(self):
        self.assertEqual(self.cc.circuit.number_of_trainable_parameters, 2)

    def test_create_circuit(self):
        cc = CouplingCircuit(LinearConditioner(1, 2), jnp.array([0]), self.sum_layer, jnp.array([0]))
        params = cc.conditioner.generate_parameters(jnp.array([0.1]))
        cc.create_circuit_from_parameters(params)

class CouplingCircuit4DTestCase(unittest.TestCase):

    number_of_variables = 4
    number_of_samples = 1000
    cc: CouplingCircuit
    data: jax.Array
    jpt: ProbabilisticCircuit
    non_marginalized_jpt: ProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        mean = np.full(cls.number_of_variables, 0)
        cov = np.random.uniform(0, 1, (cls.number_of_variables, cls.number_of_variables))
        cov = np.dot(cov, cov.T)
        samples = np.random.multivariate_normal(mean, cov, cls.number_of_samples)
        df = pd.DataFrame(samples, columns=[f"x_{i}" for i in range(cls.number_of_variables)])
        variables = infer_variables_from_dataframe(df, min_samples_per_quantile=100)
        jpt = JPT(variables, min_samples_leaf=0.2)
        jpt.fit(df)
        cls.non_marginalized_jpt = jpt.probabilistic_circuit
        cls.jpt = jpt.probabilistic_circuit.marginal(variables[cls.number_of_variables // 2:])
        circuit = ProbabilisticCircuit.from_nx(cls.jpt, False)
        conditioner = LinearConditioner(cls.number_of_variables // 2, circuit.root.number_of_trainable_parameters)
        cls.cc = CouplingCircuit(conditioner, jnp.array(list(range(cls.number_of_variables // 2))),
                             circuit.root, jnp.array(list(range(cls.number_of_variables // 2, cls.number_of_variables))))
        cls.cc.validate()
        cls.data = jnp.array(df)

    def test_learning(self):

        def loss(model, x):
            cll = model.conditional_log_likelihood(x)
            return -jnp.mean(cll)

        def loss2(p, s, x):
            model = eqx.combine(p, s)
            ll = model.conditional_log_likelihood(x)
            return -jnp.mean(ll)

        nll = loss(self.cc, self.data)

        value, grads = eqx.filter_value_and_grad(loss)(self.cc, self.data)



if __name__ == '__main__':
    unittest.main()
