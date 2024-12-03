---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Probabilistic Circuits User Guide

This tutorial walks you through the basics of using the Probabilistic Circuits (PC) from this package.
First, let's start by importing the necessary modules.

```{code-cell} ipython3
from random_events.interval import closed_open, closed
from random_events.product_algebra import Continuous, Event, SimpleEvent
from probabilistic_model.distributions import *
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import *
from probabilistic_model.probabilistic_circuit.nx.distributions.distributions import *


import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
```

Let's have a look at input units.
Input units (Leaves) are nodes that are at the end of the probabilistic circuit.
These leaves can contain any distribution and will try to call it for the respective inference tasks.
Hence, the distributions need to implement the methods required for the inference tasks.
Leaf distributions have to be wrapped in a leaf node.

Currently supported distributions are:
- (Truncated) Normal Distributions
- Uniform Distributions
- Dirac Delta Distributions
- Multinomial Distributions
- Integer Distributions

You can create them by calling the corresponding class constructors.
For this tutorial, we will stick to normal distributions.

```{code-cell} ipython3
x = Continuous("x")
y = Continuous("y")

p_x_1 = UnivariateContinuousLeaf(GaussianDistribution(x, 0, 1))
```

We can always look at the graph that we have by calling the plot_structure method.

```{code-cell} ipython3
p_x_1.probabilistic_circuit.plot_structure()
```

One node only is pretty boring, so let us create a gaussian mixture model.

```{code-cell} ipython3
p_x_2 = UnivariateContinuousLeaf(GaussianDistribution(x, 2, 1))
p_x = SumUnit()
p_x.add_subcircuit(p_x_1, 0.3)
p_x.add_subcircuit(p_x_2, 0.7)
p_x.probabilistic_circuit.plot_structure()
```

Now we got a more interesting model. We can see the nodes we created and the connections between them. Even the weights are indicated by the opacity of an edge.
Let's create a more complex model by becoming multivariate through product units.

```{code-cell} ipython3
p_y = UnivariateContinuousLeaf(GaussianDistribution(y, 1, 2))
p_xy = ProductUnit()
p_xy.add_subcircuit(p_x)
p_xy.add_subcircuit(p_y)
p_xy.probabilistic_circuit.plot_structure()
```

Now we have a model that is a bit more complex. 
We can now observe how conditioning modifies the structure of a probabilistic circuit.

```{code-cell} ipython3
e = SimpleEvent({x: closed_open(0, 1),
                 y: closed_open(0, 0.5 ) | closed(1, 1.5)}).as_composite_set()
p_xy_conditioned, _ = p_xy.probabilistic_circuit.conditional(e)
p_xy_conditioned.plot_structure()
```

We can also look at the distributions on the data level.

```{code-cell} ipython3
fig = go.Figure(p_xy_conditioned.plot(), p_xy_conditioned.plotly_layout())
fig.show()
```

We can also create a circuit from a bayesian network.

```{code-cell} ipython3
from probabilistic_model.bayesian_network.bayesian_network import *
from probabilistic_model.bayesian_network.distributions import *
from random_events.set import *
from random_events.variable import *
from random_events.interval import *
import networkx as nx

# Declare variable types and variables
class Success(SetElement):
    EMPTY_SET = -1
    FAILURE = 0
    SUCCESS = 1
    
class ObjectPosition(SetElement):
    EMPTY_SET = -1
    LEFT = 0
    RIGHT = 1
    CENTER = 2
    
class Mood(SetElement):
    EMPTY_SET = -1
    HAPPY = 0
    SAD = 1

success = Symbolic("Success", Success)
object_position = Symbolic("ObjectPosition", ObjectPosition)
mood = Symbolic("Mood", Mood)
x = Continuous("X")
y = Continuous("Y")

# construct Bayesian network
bn = BayesianNetwork()

# create root
cpd_success = RootDistribution(success, MissingDict(float, {int(Success.FAILURE): 0.8, int(Success.SUCCESS): 0.2}))
bn.add_node(cpd_success)

# create P(ObjectPosition | Success)
cpd_object_position = ConditionalProbabilityTable(object_position)
cpd_object_position.conditional_probability_distributions[int(Success.FAILURE)] = SymbolicDistribution(object_position, 
                                                                                                       MissingDict(float, {int(ObjectPosition.LEFT): 0.3, 
                                                                                                                           int(ObjectPosition.RIGHT): 0.3, 
                                                                                                                           int(ObjectPosition.CENTER): 0.4}))
cpd_object_position.conditional_probability_distributions[ int(Success.SUCCESS)] = SymbolicDistribution(object_position,
                                                                                                        MissingDict(float, {int(ObjectPosition.LEFT): 0.3, 
                                                                                                                            int(ObjectPosition.RIGHT): 0.3, 
                                                                                                                            int(ObjectPosition.CENTER): 0.4}))
bn.add_node(cpd_object_position)
bn.add_edge(cpd_success, cpd_object_position)

# create P(Mood | Success)
cpd_mood = ConditionalProbabilityTable(mood)
cpd_mood.conditional_probability_distributions[int(Success.FAILURE)] = SymbolicDistribution(mood, 
                                                                                            MissingDict(float, {int(Mood.HAPPY): 0.2, 
                                                                                                                int(Mood.SAD): 0.8}))
cpd_mood.conditional_probability_distributions[int(Success.SUCCESS)] = SymbolicDistribution(mood, 
                                                                                            MissingDict(float, {int(Mood.HAPPY): 0.9, 
                                                                                                                int(Mood.SAD): 0.1}))
bn.add_node(cpd_mood)
bn.add_edge(cpd_success, cpd_mood)

# create P(X, Y | ObjectPosition)
cpd_xy = ConditionalProbabilisticCircuit([x, y])
product_unit = ProductUnit()
product_unit.add_subcircuit(UnivariateContinuousLeaf(GaussianDistribution(x, 0, 1)))
product_unit.add_subcircuit(UnivariateContinuousLeaf(GaussianDistribution(y, 0, 1)))
default_circuit = product_unit.probabilistic_circuit

cpd_xy.conditional_probability_distributions[int(ObjectPosition.LEFT)] = default_circuit.conditional(SimpleEvent({x: closed(-np.inf, -0.5)}).as_composite_set())[0]
cpd_xy.conditional_probability_distributions[int(ObjectPosition.RIGHT)] = default_circuit.conditional(SimpleEvent({x: open(0.5, np.inf)}).as_composite_set())[0]
cpd_xy.conditional_probability_distributions[int(ObjectPosition.CENTER)] = default_circuit.conditional(SimpleEvent({x: open_closed(-0.5, 0.5)}).as_composite_set())[0]

bn.add_node(cpd_xy)
bn.add_edge(cpd_object_position, cpd_xy)

pos = nx.planar_layout(bn)
nx.draw(bn, with_labels=True, pos=pos)
```

```{code-cell} ipython3
pc_bn = bn.as_probabilistic_circuit().simplify()
pc_bn.plot_structure()

```
