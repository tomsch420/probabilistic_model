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

# Representations of Circuits

While understanding the concepts of a probabilistic circuit is subject to math, implementing it is a whole different
story.
This section discusses different approaches to represent circuits.

## the DAG (networkx) way

The easiest and naive way of implementing a circuit is using a directed acyclic graph (DAG).
The graph directly follows definition {prf:ref}`def-probabilistic-circuit`.

Let's look at an example.

```{code-cell} ipython3
import plotly
plotly.offline.init_notebook_mode()

from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit, SumUnit, ProductUnit
from probabilistic_model.probabilistic_circuit.nx.distributions.distributions import DiracDeltaDistribution
from random_events.variable import Continuous
import networkx as nx
from probabilistic_model.probabilistic_circuit.torch.pc import Layer, SumLayer, ProductLayer
from probabilistic_model.probabilistic_circuit.torch.input_layer import DiracDeltaLayer
from probabilistic_model.utils import embed_sparse_tensor_in_nan_tensor
import plotly.graph_objects as go

x = Continuous("x")
y = Continuous("y")
sum1, sum2, sum3 = SumUnit(), SumUnit(), SumUnit()
sum4, sum5 = SumUnit(), SumUnit()
prod1, prod2 = ProductUnit(), ProductUnit()
model = ProbabilisticCircuit()
model.add_node(sum1)
model.add_node(prod1)
model.add_node(prod2)
model.add_edge(sum1, prod1, weight=0.5)
model.add_edge(sum1, prod2, weight=0.5)
model.add_node(sum2)
model.add_node(sum3)
model.add_node(sum4)
model.add_node(sum5)
model.add_edge(prod1, sum2)
model.add_edge(prod1, sum4)
model.add_edge(prod2, sum3)
model.add_edge(prod2, sum5)
d_x1, d_x2  = DiracDeltaDistribution(x, 0, 1), DiracDeltaDistribution(x, 1, 2)
d_y1, d_y2 = DiracDeltaDistribution(y, 2, 3),  DiracDeltaDistribution(y, 3, 4)

model.add_node(d_y1)
model.add_node(d_x2)
model.add_node(d_y2)
model.add_node(d_x1)

model.add_edge(sum2, d_x1, weight=0.8)
model.add_edge(sum2, d_x2, weight=0.2)
model.add_edge(sum3, d_x1, weight=0.7)
model.add_edge(sum3, d_x2, weight=0.3)

model.add_edge(sum4, d_y1, weight=0.5)
model.add_edge(sum4, d_y2, weight=0.5)
model.add_edge(sum5, d_y1, weight=0.1)
model.add_edge(sum5, d_y2, weight=0.9)

pos = {sum1: (0, 0),
       prod1: (-1, -1), prod2: (1, -1),
       sum2: (-2, -2), sum3: (-1, -2), sum4: (1, -2), sum5: (2, -2),
       d_x1: (-2, -3), d_x2: (-1, -3), d_y1: (1, -3), d_y2: (2, -3)}
labels = {node: node.representation for node in model.nodes}
nx.draw(model, labels=labels, pos=pos)
```

```{code-cell} ipython3
fig = go.Figure(model.plot(), model.plotly_layout())
fig.show()
```

<!-- #region -->
The Benefits of the DAG representation are:
- Easy to understand
- Easy to implement
- Easy to install


The drawbacks are:
- Pure Python implementation is slow
- Improvement of machine learning packages do not improve the DAG approach
- Does not benefit from modern hardware acceleration
<!-- #endregion -->


## The Layered way
Modern literature suggests representing circuits in a way that is compatible with modern hardware acceleration. {cite}`liu2024scaling`, {cite}`peharz2020einsum`. 
Doing so requires a topological sorting of the circuit. In that topological sorting, each layer represents a set of nodes at the same depth (distance to the root) that can be computed in parallel.
These nodes have to be of the same type, such that their operations (weighted sum, product, density, etc.) can be computed in parallel.
The example from above would look as following:



```{code-cell} ipython3
root_sum_layer: SumLayer = Layer.from_probabilistic_circuit(model)
product_layer: ProductLayer = root_sum_layer.child_layers[0]
sum_layer_x: SumLayer = product_layer.child_layers[0]
sum_layer_y: SumLayer = product_layer.child_layers[1]
dirac_layer_x: DiracDeltaLayer = sum_layer_x.child_layers[0]
dirac_layer_y: DiracDeltaLayer = sum_layer_y.child_layers[0]
root_sum_layer
```

<!-- #region -->
The way a layered pc is structured is shown in the figure below.

![Grouped operations in a layered circuit](layered_example.png)

We can see that similar operations have been grouped together. Now they can be executed using a pytorch backend instead of a for loop in python. 
The Benefits of the layered representation are:
- Speed
- Compatible with modern frameworks for machine learning frameworks
- Compatible with modern hardware acceleration
- Most likely the future of probabilistic circuits


The drawbacks are:
- Harder to understand
- Harder to maintain
- Requires quite the overhead to install
<!-- #endregion -->
