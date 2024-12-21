import random
import unittest

from jax import tree_flatten
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from random_events.variable import Continuous
import pydot
from probabilistic_model.learning.region_graph.region_graph import *
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit as JPC
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import GaussianLayer
import numpy as np
import equinox as eqx
import optax
import tqdm
import plotly.graph_objects as go

np.random.seed(420)
random.seed(420)

class RandomRegionGraphTestCase(unittest.TestCase):

    variables = SortedSet([Continuous(str(i)) for i in range(4)])

    region_graph = RegionGraph(variables, partitions=2, depth=1, repetitions=2)
    region_graph = region_graph.create_random_region_graph()

    def test_region_graph(self):
        self.assertEqual(len(self.region_graph.nodes()), 19)

    def test_as_jpc(self):
        model = self.region_graph.as_probabilistic_circuit(input_units=10, sum_units=5)
        nx_model = model.to_nx()
        nx_model.plot_structure()
        # plt.show()
        self.assertEqual(len(list(node for node in nx_model.nodes() if isinstance(node, SumUnit))), 21)



class RandomRegionGraphLearningTestCase(unittest.TestCase):

    variables = SortedSet([Continuous(str(i)) for i in range(4)])
    region_graph = RegionGraph(variables, partitions=2, depth=1, repetitions=2)
    region_graph = region_graph.create_random_region_graph()

    def test_learning(self):
        data = np.random.uniform(0, 1, (10000, len(self.variables)))
        model = self.region_graph.as_probabilistic_circuit(input_units=5, sum_units=5)

        root = model.root

        @eqx.filter_jit
        def loss(model, x):
            ll = model.log_likelihood_of_nodes(x)
            return -jnp.mean(ll)

        optim = optax.adamw(0.01)
        opt_state = optim.init(eqx.filter(root, eqx.is_inexact_array))

        for _ in tqdm.trange(50):
            loss_value, grads = eqx.filter_value_and_grad(loss)(root, data)
            grads_of_sum_layer = eqx.filter(tree_flatten(grads), eqx.is_inexact_array)[0][0]
            self.assertTrue(jnp.all(jnp.isfinite(grads_of_sum_layer)))

            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(root, eqx.is_inexact_array)
            )
            root = eqx.apply_updates(root, updates)




if __name__ == '__main__':
    unittest.main()
