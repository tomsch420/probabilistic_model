import random
import unittest

from jax import tree_flatten
from random_events.set import SetElement

from probabilistic_model.learning.region_graph.region_graph import *
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit as JPC
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import GaussianLayer
import numpy as np
import equinox as eqx
import optax
import tqdm
import plotly.graph_objects as go




class Target(SetElement):
    EMPTY_SET = -1
    A = 0
    B = 1


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
        data = jnp.array(data)
        model = self.region_graph.as_probabilistic_circuit(input_units=5, sum_units=5)
        model.fit(data, epochs=50, optimizer=optax.adamw(0.01))


class ClassificationRegionGraphTestCase(unittest.TestCase):

    features = SortedSet([Continuous(str(i)) for i in range(4)])
    target = Symbolic("target", Target)
    region_graph = RegionGraph(features, partitions=2, depth=1, repetitions=2, classes=2)
    region_graph = region_graph.create_random_region_graph()

    def test_classification(self):
        data = np.random.uniform(0, 1, (100, len(self.features)))
        data = jnp.array(data)
        labels = np.random.randint(2, size = len(data))
        labels = jnp.array(labels)
        model = self.region_graph.as_probabilistic_circuit(input_units=5, sum_units=5)
        self.assertIsInstance(model, ClassificationCircuit)
        model.fit(data, labels=labels, epochs=50, optimizer=optax.adamw(0.01))


if __name__ == '__main__':
    unittest.main()
