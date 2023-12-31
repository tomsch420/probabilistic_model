import json
import math
import tempfile
import unittest
from datetime import datetime

import numpy as np
import pandas as pd
import random_events
import sklearn.datasets
from anytree import RenderTree
from jpt import infer_from_dataframe as old_infer_from_dataframe
from jpt.learning.impurity import Impurity
from jpt.trees import JPT as OldJPT, Leaf
from random_events.variables import Variable
from random_events.events import Event

from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import (ScaledContinuous, infer_variables_from_dataframe, Integer,
                                                        Symbolic, Continuous)
from probabilistic_model.learning.nyga_distribution import NygaDistribution
from probabilistic_model.probabilistic_circuit.distribution import IntegerDistribution, SymbolicDistribution, \
    UnivariateDiscreteDistribution, UnivariateDiscreteSumUnit
from probabilistic_model.probabilistic_circuit.exporter.dotexporter import GraphVizExporter
from probabilistic_model.probabilistic_circuit.units import DecomposableProductUnit, Unit
import plotly.graph_objects as go


class VariableTestCase(unittest.TestCase):
    variable: ScaledContinuous = ScaledContinuous('x', 2, 3)

    def test_encode(self):
        self.assertEqual(self.variable.encode(2), 0)
        self.assertEqual(self.variable.encode(5), 1)
        self.assertEqual(self.variable.encode(0), -2 / 3)

    def test_decode(self):
        self.assertEqual(self.variable.decode(0), 2)
        self.assertEqual(self.variable.decode(1), 5)
        self.assertEqual(self.variable.decode(-2 / 3), 0)

    def test_serialization_integer(self):
        variable = Integer('x', [1, 2, 3], 2, 1)
        serialized = variable.to_json()
        deserialized = Variable.from_json(serialized)
        self.assertEqual(variable, deserialized)

    def test_serialization_continuous(self):
        variable = ScaledContinuous('x', 2, 3, 1., 0.1, 10)
        serialized = variable.to_json()
        deserialized = Variable.from_json(serialized)
        self.assertEqual(variable, deserialized)


class InferFromDataFrameTestCase(unittest.TestCase):
    data: pd.DataFrame

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        data = pd.DataFrame()
        data["real"] = np.random.normal(2, 4, 100)
        data["integer"] = np.concatenate((np.random.randint(low=0, high=4, size=50), np.random.randint(7, 10, 50)))
        data["symbol"] = np.random.randint(0, 4, 100).astype(str)
        cls.data = data

    def test_types(self):
        self.assertEqual(self.data.dtypes.iloc[0], float)
        self.assertEqual(self.data.dtypes.iloc[1], int)
        self.assertEqual(self.data.dtypes.iloc[2], object)

    def test_infer_from_dataframe_with_scaling(self):
        real, integer, symbol = infer_variables_from_dataframe(self.data)
        self.assertEqual(real.name, "real")
        self.assertIsInstance(real, ScaledContinuous)
        self.assertEqual(integer.name, "integer")
        self.assertEqual(symbol.name, "symbol")
        self.assertLess(real.minimal_distance, 1.)

    def test_infer_from_dataframe_without_scaling(self):
        real, integer, symbol = infer_variables_from_dataframe(self.data, scale_continuous_types=False)
        self.assertNotIsInstance(real, ScaledContinuous)

    def test_unknown_type(self):
        df = pd.DataFrame()
        df["time"] = [datetime.now()]
        with self.assertRaises(ValueError):
            infer_variables_from_dataframe(df)


class JPTTestCase(unittest.TestCase):
    data: pd.DataFrame
    real: ScaledContinuous
    integer: Integer
    symbol: Symbolic
    model: JPT

    def setUp(self):
        np.random.seed(69)
        data = pd.DataFrame()
        data["integer"] = np.concatenate((np.random.randint(low=0, high=4, size=50), np.random.randint(7, 10, 50)))
        data["real"] = np.random.normal(2, 4, 100)
        data["symbol"] = np.random.randint(0, 4, 100).astype(str)
        self.data = data
        self.real, self.integer, self.symbol = infer_variables_from_dataframe(self.data, scale_continuous_types=False)
        self.model = JPT([self.real, self.integer, self.symbol])

    @unittest.skip("Scaled variables are kinda buggy.")
    def test_preprocess_data(self):
        preprocessed_data = self.model.preprocess_data(self.data)
        mean = preprocessed_data[:, 1].mean()
        std = preprocessed_data[:, 1].std(ddof=1)
        self.assertEqual(self.real.mean, mean)
        self.assertEqual(self.real.std, std)

        # assert that this does not throw exceptions
        for variable, column in zip(self.model.variables, preprocessed_data.T):
            if isinstance(variable, random_events.variables.Discrete):
                variable.decode_many(column.astype(int))
            else:
                variable.decode_many(column)

    def test_construct_impurity(self):
        impurity = self.model.construct_impurity()
        self.assertIsInstance(impurity, Impurity)

    def test_create_leaf_node(self):
        preprocessed_data = self.model.preprocess_data(self.data)
        leaf_node = self.model.create_leaf_node(preprocessed_data)
        self.assertEqual(len(leaf_node.children), 3)
        self.assertIsInstance(leaf_node.children[0], IntegerDistribution)
        self.assertIsInstance(leaf_node.children[1], NygaDistribution)
        self.assertIsInstance(leaf_node.children[2], SymbolicDistribution)

        # check that all likelihoods are greater than 0
        for index, row in self.data.iterrows():
            row_ = [row[variable.name] for variable in self.model.variables]
            likelihood = leaf_node.likelihood(row_)
            self.assertTrue(likelihood > 0)

    def test_fit_without_sum_units(self):
        self.model.min_impurity_improvement = 1
        self.model.fit(self.data)
        self.assertEqual(len(self.model.children), 1)
        self.assertEqual(self.model.weights, [1])
        self.assertEqual(len(self.model.children[0].children), 3)

    def test_fit(self):
        self.model._min_samples_leaf = 10
        self.model.fit(self.data)
        self.assertTrue(len(self.model.children) <= math.floor(len(self.data) / self.model.min_samples_leaf))
        self.assertTrue(all([weight > 0 for weight in self.model.weights]))

        # check that all likelihoods are greater than 0
        for index, row in self.data.iterrows():
            row_ = [row[variable.name] for variable in self.model.variables]
            likelihood = self.model.likelihood(row_)
            self.assertTrue(likelihood > 0)

    def test_fit_and_compare_to_jpt_one_leaf_only(self):
        self.model.min_impurity_improvement = 1
        self.model.fit(self.data)
        variables = old_infer_from_dataframe(self.data, scale_numeric_types=False)
        original_jpt = OldJPT(variables, min_samples_leaf=self.model.min_samples_leaf,
                              min_impurity_improvement=self.model.min_impurity_improvement)
        original_jpt.fit(self.data)
        self.assertEqual(len(self.model.children), len(original_jpt.leaves))

        for index, leaf in enumerate(original_jpt.leaves.values()):
            self.assertTrue(self.leaf_equal_to_product(leaf, self.model.children[index]))

    def leaf_equal_to_product(self, leaf: Leaf, product: DecomposableProductUnit, epsilon: float = 0.001) -> bool:
        """
        Check if a leaf is equal to a product unit.
        :param leaf: The (jpt) leaf to check.
        :param product: The product unit to check.
        :param epsilon: The epsilon to use for the comparison.
        :return: True if the leaf is equal to the product unit, False otherwise.
        """

        for variable in product.variables:
            if isinstance(variable, Continuous):
                continue
            old_distribution = leaf.distributions[variable.name]
            new_distribution: UnivariateDiscreteDistribution = \
                [child for child in product.children if child.variable == variable][0]
            for value in variable.domain:
                old_probability = old_distribution.p(value)
                new_probability = new_distribution.pdf(value)

                if abs(old_probability - new_probability) > epsilon:
                    return False

        return True

    def test_preprocessing_and_compare_to_jpt(self):
        variables = old_infer_from_dataframe(self.data, scale_numeric_types=False, precision=0.)
        original_jpt = OldJPT(variables, min_samples_leaf=self.model.min_samples_leaf,
                              min_impurity_improvement=self.model.min_impurity_improvement)
        original_preprocessing = original_jpt._preprocess_data(self.data)
        own_preprocessing = self.model.preprocess_data(self.data)

        # Symbolic columns are not preprocessed in order in JPTs. The difference is intended
        self.assertTrue(np.all(original_preprocessing[:, :-1] == own_preprocessing[:, :-1]))

    def test_fit_and_compare_to_jpt(self):
        self.model._min_samples_leaf = 10
        self.model.keep_sample_indices = True
        self.model.fit(self.data)
        variables = old_infer_from_dataframe(self.data, scale_numeric_types=False, precision=0.)
        original_jpt = OldJPT(variables, min_samples_leaf=self.model.min_samples_leaf,
                              min_impurity_improvement=self.model.min_impurity_improvement)
        original_jpt = original_jpt.learn(self.data, keep_samples=True)
        self.assertEqual(len(self.model.children), len(original_jpt.leaves))

        for original_leaf, new_leaf in zip(original_jpt.leaves.values(), self.model.children):
            self.assertSetEqual(set(original_leaf.s_indices), set(new_leaf.sample_indices))

    def test_jpt_continuous_variables_only(self):
        data = self.data[["real"]]
        variables = infer_variables_from_dataframe(data)
        model = JPT(variables)
        model.fit(data)
        self.assertEqual(len(model.children), 1)

    def test_jpt_integer_variables_only(self):
        data = self.data[["integer"]]
        variables = infer_variables_from_dataframe(data)
        model = JPT(variables)
        model.fit(data)
        self.assertEqual(len(model.children), 1)

    def test_jpt_symbolic_variables_only(self):
        data = self.data[["symbol"]]
        variables = infer_variables_from_dataframe(data)
        model = JPT(variables)
        model.fit(data)
        self.assertEqual(len(model.children), 4)

    def test_plot(self):
        self.model._min_samples_leaf = 10
        self.model.fit(self.data)
        fig = self.model.plot()
        self.assertIsNotNone(fig)  # fig.show()

    def test_variable_dependencies_to_json(self):
        serialized = self.model._variable_dependencies_to_json()
        all_variable_names = [variable.name for variable in self.model.variables]
        self.assertEqual(serialized,
                         {'real': all_variable_names, 'integer': all_variable_names, 'symbol': all_variable_names})

    def test_serialization(self):
        self.model._min_samples_leaf = 10
        self.model.fit(self.data)
        serialized = self.model.to_json()
        deserialized = Unit.from_json(serialized)
        self.assertEqual(self.model, deserialized)

    def test_dot_exporter(self):
        self.model._min_samples_leaf = 10
        self.model.fit(self.data)
        exporter = GraphVizExporter(self.model)
        dot = exporter.to_graphviz()
        self.assertIsNotNone(dot)  # dot.view(tempfile.mktemp('.gv'))


class BreastCancerTestCase(unittest.TestCase):
    data: pd.DataFrame
    model: JPT

    @classmethod
    def setUpClass(cls):
        data = sklearn.datasets.load_breast_cancer(as_frame=True)

        df = data.data
        target = data.target
        target[target == 1] = "malignant"
        target[target == 0] = "friendly"

        df["malignant"] = target
        cls.data = df

        variables = infer_variables_from_dataframe(cls.data, scale_continuous_types=False)

        cls.model = JPT(variables, min_samples_leaf=0.1)
        cls.model.fit(cls.data)

    def test_serialization(self):
        json_dict = self.model.to_json()
        model = JPT.from_json(json_dict)
        self.assertEqual(model, self.model)

        file = tempfile.NamedTemporaryFile()

        with open(file.name, "w") as f:
            json.dump(json_dict, f)

        with open(file.name, "r") as f:
            model_ = json.load(f)
        model_ = JPT.from_json(model_)
        self.assertEqual(model, model_)
        file.close()

    def test_conditional_inference(self):
        evidence = Event()
        query = Event()

        conditional_model, evidence_probability = self.model.conditional(evidence)
        self.assertAlmostEqual(1., evidence_probability)
        self.assertAlmostEqual(1., conditional_model.probability(query))

    def test_univariate_continuous_marginal(self):
        marginal = self.model.marginal(self.model.variables[:1])
        self.assertIsInstance(marginal, NygaDistribution)
        self.assertEqual(marginal.height, 1)

    def test_univariate_symbolic_marginal(self):
        variables = [v for v in self.model.variables if v.name == "malignant"]
        marginal = self.model.marginal(variables)
        self.assertIsInstance(marginal, SymbolicDistribution)
        self.assertEqual(marginal.height, 0)


class MNISTTestCase(unittest.TestCase):
    model: JPT

    @classmethod
    def setUpClass(cls):
        mnist = sklearn.datasets.load_digits(as_frame=True)
        df = mnist.data
        target = mnist.target
        df["digit"] = target
        df["digit"] = df['digit'].astype(str)

        variables = infer_variables_from_dataframe(df, scale_continuous_types=False, min_likelihood_improvement=0.01)
        cls.model = JPT(variables, min_samples_leaf=0.1)
        cls.model.fit(df)

    def test_serialization(self):
        json_dict = self.model.to_json()
        model = JPT.from_json(json_dict)
        self.assertEqual(model, self.model)

        file = tempfile.NamedTemporaryFile()

        with open(file.name, "w") as f:
            json.dump(json_dict, f)

        with open(file.name, "r") as f:
            model_ = json.load(f)
        model_ = JPT.from_json(model_)
        self.assertEqual(model, model_)
        file.close()
