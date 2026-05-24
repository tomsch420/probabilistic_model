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

# Pruning and Growing Probabilistic Circuits

This tutorial demonstrates how to prune and grow probabilistic circuits to optimize their structure and performance. Pruning removes less important edges to simplify the circuit, while growing adds new components to increase expressiveness. {cite}`dang2022sparse`.

## Introduction

Probabilistic circuits can become quite complex, especially when learned from data. Two important operations help manage this complexity:

- **Pruning**: Removes edges with low importance to simplify the circuit while maintaining most of its representational power
- **Growing**: Adds new components to increase the circuit's expressiveness and modeling capacity

Let's start by importing the necessary modules.

```{code-cell} ipython3
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *
from probabilistic_model.probabilistic_circuit.rx.flow_analyzer import CircuitFlowAnalyzer
from probabilistic_model.distributions import *
from random_events.product_algebra import Continuous

# Set random seed for reproducibility
np.random.seed(42)
```

## Helper Functions

First, let's define some helper functions for analyzing circuit performance.

```{code-cell} ipython3
def summarize_log_likelihood(ll_array: np.ndarray) -> tuple[float, float]:
    """
    Summarizes the log-likelihood values by computing the mean and the percentage of impossible samples.

    :param ll_array: A NumPy array of log-likelihood values.
    :return: A tuple containing the mean log-likelihood and the percentage of impossible samples.
    """
    ll_array = np.asarray(ll_array)
    finite_lls = ll_array[np.isfinite(ll_array)]
    total_samples = len(ll_array)
    impossible_count = total_samples - len(finite_lls)
    impossible_percentage = (impossible_count / total_samples) * 100 if total_samples > 0 else 0
    if len(finite_lls) == 0:
        mean_finite_ll = -np.inf
    else:
        mean_finite_ll = np.mean(finite_lls)

    return mean_finite_ll, impossible_percentage

def print_circuit_stats(circuit, dataset, name="Circuit"):
    """
    Prints the statistics of a circuit, including the number of nodes, edges, mean log-likelihood and percentage of impossible samples.

    :param circuit: The circuit object to analyze.
    :param dataset: The dataset to use for log-likelihood computation.
    :param name: An optional name for the circuit (default: "Circuit").
    """
    mean_ll, impossible_percent = summarize_log_likelihood(circuit.log_likelihood(dataset))
    print(f"\n{name} has {len(list(circuit.nodes()))} nodes and {len(list(circuit.edges()))} edges.")
    print(f"- Mean log-likelihood: {mean_ll:.3f}")
    print(f"- Impossible samples: {impossible_percent:.2f}%")
```

## Manual Circuit Construction and Pruning

Let's start by creating a simple probabilistic circuit manually and demonstrating pruning.

```{code-cell} ipython3
def create_circuit():
    """
    Creates a probabilistic circuit with a specific structure.
    """
    # Create variables
    x1 = Continuous("x1")
    x2 = Continuous("x2")
    
    # Create circuit
    circuit = ProbabilisticCircuit()
    
    # Create root sum unit
    root = SumUnit(probabilistic_circuit=circuit)
    circuit.add_node(root)

    sum1 = SumUnit(probabilistic_circuit=circuit)
    sum2 = SumUnit(probabilistic_circuit=circuit)
    sum3 = SumUnit(probabilistic_circuit=circuit)
    circuit.add_nodes_from([sum1, sum2, sum3])
    
    # Create product units
    prod1 = ProductUnit(probabilistic_circuit=circuit)
    prod2 = ProductUnit(probabilistic_circuit=circuit)
    prod3 = ProductUnit(probabilistic_circuit=circuit)
    circuit.add_nodes_from([prod1, prod2, prod3])
    
    # Create leaf distributions
    leaf1 = leaf(GaussianDistribution(x1, 0.0, 1.0), circuit)
    leaf2 = leaf(GaussianDistribution(x2, 0.0, 1.0), circuit)
    leaf3 = leaf(GaussianDistribution(x1, 2.0, 1.0), circuit)
    leaf4 = leaf(GaussianDistribution(x2, 2.0, 1.0), circuit)
    leaf5 = leaf(GaussianDistribution(x1, 4.0, 1.0), circuit)
    leaf6 = leaf(GaussianDistribution(x2, 4.0, 1.0), circuit)

    # Connect the circuit
    root.add_subcircuit(sum1, np.log(0.5))
    root.add_subcircuit(sum2, np.log(0.5))
    root.add_subcircuit(sum3, np.log(0.5))
    sum1.add_subcircuit(prod1, np.log(0.7))
    sum1.add_subcircuit(prod2, np.log(0.3))
    sum1.add_subcircuit(prod3, np.log(0.1))
    sum2.add_subcircuit(prod1, np.log(0.4))
    sum2.add_subcircuit(prod2, np.log(0.3))
    sum2.add_subcircuit(prod3, np.log(0.5))
    sum3.add_subcircuit(prod1, np.log(0.2))
    sum3.add_subcircuit(prod2, np.log(0.4))
    sum3.add_subcircuit(prod3, np.log(0.8))

    prod1.add_subcircuit(leaf1)
    prod1.add_subcircuit(leaf2)
    prod2.add_subcircuit(leaf3)
    prod2.add_subcircuit(leaf4)
    prod3.add_subcircuit(leaf5)
    prod3.add_subcircuit(leaf6)
    
    return circuit

# Create the complex circuit
original_circuit = create_circuit()
```

```{code-cell} ipython3
# Plot the original circuit structure
original_circuit.plot_structure()
plt.title("Original Circuit Structure")
plt.show()

# Generate test dataset for evaluation
test_dataset = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100)
print_circuit_stats(original_circuit, test_dataset, "Original Circuit")
```

### Pruning the Circuit

Now let's prune the circuit to remove less important edges.

```{code-cell} ipython3
# Prune 40% of the edges
pruning_percentage = 0.4
print(f"\nPruning {pruning_percentage*100}% of the edges...")

flow_analyzer = CircuitFlowAnalyzer(original_circuit)
pruned_circuit = flow_analyzer.prune(dataset=test_dataset, pruning_percentage=pruning_percentage)

# Plot the pruned circuit
pruned_circuit.plot_structure()
plt.title("Pruned Circuit Structure (40% edges removed)")
plt.show()

print_circuit_stats(pruned_circuit, test_dataset, "Pruned Circuit")
```

### Growing the Circuit

After pruning, we can grow the circuit to add new components and potentially improve its expressiveness.

```{code-cell} ipython3
# Grow the pruned circuit
noise_variance = 0.3
print(f"\nGrowing the circuit with noise variance {noise_variance}...")

grown_circuit = pruned_circuit.grow(noise_variance=noise_variance)

# Plot the grown circuit
grown_circuit.plot_structure()
plt.title("Grown Circuit Structure")
plt.show()

print_circuit_stats(grown_circuit, test_dataset, "Grown Circuit")
```

## JPT Learning with Pruning and Growing

Now let's demonstrate pruning and growing with a Joint Probability Tree (JPT) learned from real data. For more details, see the tutorial on {ref}`joint_probability_trees`.

```{code-cell} ipython3
# Create a mixture of two multivariate Gaussians
np.random.seed(69)
dist_1 = np.random.multivariate_normal(np.zeros((2,)), np.eye(2, 2), size=(200,))
dist_2 = np.random.multivariate_normal(np.array([3, 4]), [[1, 2], [1, 0]], size=(300,))
dataset = np.concatenate((dist_1, dist_2))
dataset = pd.DataFrame(dataset, columns=["x", "y"])

print("Sample dataset:")
print(dataset.head())
print(f"\nDataset shape: {dataset.shape}")
```

```{code-cell} ipython3
# Create an interactive plot of the dataset
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dist_1[:, 0], 
    y=dist_1[:, 1], 
    mode="markers", 
    name="First Component",
    marker=dict(color='blue', opacity=0.6)
))
fig.add_trace(go.Scatter(
    x=dist_2[:, 0], 
    y=dist_2[:, 1], 
    mode="markers", 
    name="Second Component",
    marker=dict(color='red', opacity=0.6)
))
fig.update_layout(
    title="Dataset: Mixture of Two Multivariate Gaussians",
    xaxis_title="x",
    yaxis_title="y",
    width=700,
    height=500
)
fig.show()
```

Next, let's visualize the learned JPT structure and distribution.

```{code-cell} ipython3
# Infer variables and learn JPT
variables = infer_variables_from_dataframe(dataset, scale_continuous_types=False)
model = JPT(variables, min_samples_leaf=500)
jpt_circuit = model.fit(dataset)

# Plot the structure
jpt_circuit.plot_structure()
plt.title("Original Learned JPT Structure")
plt.show()

print_circuit_stats(jpt_circuit, dataset.values, "Original JPT")
```

```{code-cell} ipython3
# Plot the learned distribution
figure = go.Figure(jpt_circuit.plot(number_of_samples=1000, surface=True))
figure.update_layout(
    title="Original Learned Distribution",
    width=700,
    height=500
)
figure.show()
```

### Pruning the Learned JPT

```{code-cell} ipython3
# Prune the learned JPT
pruning_percentage = 0.5
print(f"\nPruning {pruning_percentage*100}% of the JPT edges...")

jpt_circuit_copy = copy.deepcopy(jpt_circuit)
flow_analyzer = CircuitFlowAnalyzer(jpt_circuit_copy)
pruned_jpt = flow_analyzer.prune(dataset.values, pruning_percentage=pruning_percentage)

# Plot the pruned structure
pruned_jpt.plot_structure()
plt.title("Pruned JPT Structure (50% edges removed)")
plt.show()

print_circuit_stats(pruned_jpt, dataset.values, "Pruned JPT")
```

### Growing the Pruned JPT

```{code-cell} ipython3
# Grow the pruned JPT
noise_variance = 0.1
print(f"\nGrowing the pruned JPT with noise variance {noise_variance}...")

pruned_jpt_copy = copy.deepcopy(pruned_jpt)
grown_jpt = pruned_jpt_copy.grow(noise_variance=noise_variance)

# Plot the grown structure
grown_jpt.plot_structure()
plt.title("Grown JPT Structure")
plt.show()

print_circuit_stats(grown_jpt, dataset.values, "Grown JPT")
```

### Comparing All Three JPT Versions

```{code-cell} ipython3
# Generate samples from all three versions for comparison
n_samples = 500

original_samples = jpt_circuit.sample(n_samples)
pruned_samples = pruned_jpt.sample(n_samples)
grown_samples = grown_jpt.sample(n_samples)

fig = go.Figure()

# Original data
fig.add_trace(go.Scatter(
    x=dataset['x'],
    y=dataset['y'],
    mode="markers",
    name="Original Data",
    marker=dict(color='black', opacity=0.3, size=4)
))

# Original JPT samples
fig.add_trace(go.Scatter(
    x=original_samples[:, 0],
    y=original_samples[:, 1],
    mode="markers",
    name="Original JPT",
    marker=dict(color='blue', opacity=0.6, size=3)
))

# Pruned JPT samples
fig.add_trace(go.Scatter(
    x=pruned_samples[:, 0],
    y=pruned_samples[:, 1],
    mode="markers",
    name="Pruned JPT",
    marker=dict(color='red', opacity=0.6, size=3)
))

# Grown JPT samples
fig.add_trace(go.Scatter(
    x=grown_samples[:, 0],
    y=grown_samples[:, 1],
    mode="markers",
    name="Grown JPT",
    marker=dict(color='green', opacity=0.6, size=3)
))

fig.update_layout(
    title="Comparison of Original, Pruned, and Grown JPT Samples",
    xaxis_title="x",
    yaxis_title="y",
    width=800,
    height=600,
    legend=dict(x=0.02, y=0.98)
)
fig.show()
```

## Key Takeaways

In this tutorial, we demonstrated pruning and growing techniques for sparse probabilistic circuits, following the approach described in {cite}`dang2022sparse`.

**Pruning benefits:**
   * Reduces circuit complexity by removing nodes and edges
   * Maintains comparable performance while simplifying structure
   * Helps regularize the model and prevent overfitting
   * Enables faster inference due to fewer computations

**Growing benefits:**
   * Restores expressiveness after aggressive pruning
   * Improves generalization by injecting diversity
   * Recovers performance lost during pruning
   * Allows exploration of alternative circuit structures

**Trade-offs:**
   * *Pruning*: lower complexity, faster inference, but risk of underfitting
   * *Growing*: higher expressiveness, but risk of overfitting and added complexity
   * The optimal balance depends on the dataset and application

**Practical recommendations:**
   * Begin with a sufficiently expressive learned model
   * Apply pruning to discover a minimal effective structure
   * Grow selectively if pruning reduces performance too much
   * Use validation data to guide pruning and growing decisions
