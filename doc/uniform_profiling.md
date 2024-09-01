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

# Speed Evaluation of a Uniform Input Layer

Let's start with investigating the uniform distribution/layer.
First, we generate 10k uniform distributions, one time in a DAG (nx) layout and another time in the layered (torch) fashion.

```{code-cell} ipython3
import time
import torch
from random_events.product_algebra import SimpleEvent, VariableMap

from probabilistic_model.probabilistic_circuit.torch import UniformLayer
from probabilistic_model.probabilistic_circuit.nx.distributions import UniformDistribution
from random_events.interval import SimpleInterval, Bound
from random_events.variable import Continuous
from datetime import timedelta
import tabulate

x: Continuous = Continuous("x")
number_of_nodes: int = 10000

intervals = [SimpleInterval(0, 1, Bound.OPEN, Bound.OPEN) for _ in range(number_of_nodes)]
nx_uniform_distributions = [UniformDistribution(x, interval) for interval in intervals]
torch_uniform_distributions = UniformLayer(x, torch.stack([torch.zeros(number_of_nodes), torch.ones(number_of_nodes)]).double().T)
assert len(nx_uniform_distributions) == torch_uniform_distributions.number_of_nodes

def timeit(func):
    """
    Decorator to measure the time a function takes to execute.
    """
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return result, total_time

    return timeit_wrapper

def compare_times(nxt, tt):
    timedelta_nxt = timedelta(milliseconds=nxt)
    timedelta_torch = timedelta(milliseconds=tt)
    print("Time the nx process took", timedelta_nxt, "s")
    print("Time the torch process took", timedelta_torch, "s")
    print("Torch is ", nxt/tt, "times faster")
    return timedelta_nxt, timedelta_torch, nxt/tt

comparison_table = [["Inference Type", "networkx", "torch", "Speed Up"]]

```

Now we have created the uniform distributions in both `torch` and `networkx` format. Let's measure the time it takes to calculate the log likelihood of the samples.

```{code-cell} ipython3
samples_torch = torch.rand(1000, 1).double() * 3 - 1
samples_numpy = samples_torch.numpy()

@timeit
def time_log_likelihood_nx():
    return [node.log_likelihood(samples_numpy) for node in nx_uniform_distributions]

@timeit
def time_log_likelihood_torch():
    return torch_uniform_distributions.log_likelihood_of_nodes(samples_torch)

_, nx_time = time_log_likelihood_nx()

# warm up the compiler
for _ in range(2):
    time_log_likelihood_torch()
_, torch_time = time_log_likelihood_torch()

stats = compare_times(nx_time, torch_time)
comparison_table.append(["Log Likelihood", *stats])
```

Next, we investigate the probability method.

```{code-cell} ipython3
event = SimpleEvent({x: SimpleInterval(0.5, 1.5).as_composite_set() | SimpleInterval(-0.5, 0.25).as_composite_set()})

@timeit
def time_probability_nx():
    return [node.probability_of_simple_event(event) for node in nx_uniform_distributions]

@timeit
def time_probability_torch():
    return torch_uniform_distributions.probability_of_simple_event(event)

_, nx_time = time_probability_nx()
tt, torch_time = time_probability_torch()
stats = compare_times(nx_time, torch_time)
comparison_table.append(["Probability", *stats])
```

Next, conditioning!

```{code-cell} ipython3

@timeit
def time_conditioning_nx():
    return [node.log_conditional_of_simple_event(event) for node in nx_uniform_distributions]

@timeit
def time_conditioning_torch():
    return torch_uniform_distributions.log_conditional_of_simple_event(event)

_, nx_time = time_conditioning_nx()
_, torch_time = time_conditioning_torch()
stats = compare_times(nx_time, torch_time)
comparison_table.append(["Conditioning", *stats])
```

Next, sampling!

```{code-cell} ipython3
number_of_samples = 1000

@timeit
def time_sampling_nx():
    return [node.sample(number_of_samples) for node in nx_uniform_distributions]

frequencies = torch.full((torch_uniform_distributions.number_of_nodes, ), number_of_samples)

@timeit
def time_sampling_torch():
    return torch_uniform_distributions.sample_from_frequencies(frequencies)

_, nx_time = time_sampling_nx()
_, torch_time = time_sampling_torch()
stats = compare_times(nx_time, torch_time)
comparison_table.append(["Sampling", *stats])
```




Finally, moment calculation!

```{code-cell} ipython3
order, center = VariableMap({x: 1}), VariableMap({x: 0})
@timeit
def time_mean_nx():
    return [node.moment(order, center) for node in nx_uniform_distributions]
_, nx_time = time_mean_nx()


order, center = torch.tensor([1]), torch.tensor([0])
@timeit
def time_mean_torch():
    return torch_uniform_distributions.moment_of_nodes(order, center)
_, torch_time = time_mean_torch()
stats = compare_times(nx_time, torch_time)
comparison_table.append(["Moment", *stats])
```

Let's look at the entire comparison.

```{code-cell} ipython3
print(tabulate.tabulate(comparison_table))
```

<!-- #region -->
## Discussion

Apparently, it is not an across the board improvement.
The pytorch log likelihood is only one order of magnitude faster than the networkx one, while the probability is two orders of magnitude faster. 
The reason for this could be that the calculation of the uniform density does not involve a lot of maths but a data-dependent control flow, which computational graphs don't like.
Perhaps https://pytorch.org/docs/stable/cond.html offers a speed up here.  


Sampling from the torch layer is way slower than sampling from the networkx distributions. The first reason for that is that sampling the entire uniform layer at once has to use a change of variables like calculation, which involves way more steps than a default call to numpy.
The second reason might be, that the result of sampling from a layer is a sparse tensor, that has to be carefully constructed.
<!-- #endregion -->
