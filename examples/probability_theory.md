---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Probability Theory

This notebook will walk through essential concepts of probability theory with practical examples.
The aim of this notebook if to make probability theory graspable for roboticists and machine learning engineers. 

At the end of this notebook, you will understand ...

- Basic terms of probability theory
- Random events and how they interact
- Quantities of interest in probabilistic modelling
- Learning probabilistic models from data
- Applying probabilistic models to practical problems

If you are interested in a more mathematical and less engineering introduction to probability, Mario recommends the following resources:


 | Level         | Resource                                                                                             | Marios Comment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|---------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Undergraduate | {cite:p}`case2001statistical`                                                                        | A Good Introductory book on Statistical Methods from a Frequentist viewpoint. (e.g. Hypothesis Test, ANOVA etc.                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Undergraduate | {cite:p}`hogg2019introduction`                                                                       | This book provides a good introduction to pre-Machine Learning models, as an added bonus the authors redid the exercises in Python (previous versions had them in R).                                                                                                                                                                                                                                                                                                                                                                        |
| Graduate      | {cite:p}`hastie2009elements`                                                                         | Same as the previous one from the same authors, but with more math depth and understanding (e.g. optimizing the regression penalty is done by means of Lagrangian functions)                                                                                                                                                                                                                                                                                                                                                                 |
| Graduate      | {cite:p}`gelman2014bayesian`                                                                         | Solid book on Bayesian Inference, plus it provides an extensive overview on Sampling methods from the Bayesian perspective.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Graduate      | {cite:p}`bishop2006pattern`                                                                          | This is considered by many to be the Holy Bible of Machine Learning (Especially Bayesian Networks). It provides a really useful read for understanding the next 3 books in Deep Learning.                                                                                                                                                                                                                                                                                                                                                    |
| Baby          | [Deep Learning by  <br> Yoshua Bengio](https://www.deeplearningbook.org/)                            | Deep Learning. You can get away with not knowing measure theory                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Papa          | [Probabilistic Machine Learning: <br> An Introduction](https://probml.github.io/pml-book/book1.html) | Deep Learning. You can get away with not knowing measure theory                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Grandpa       | [Probabilistic Machine Learning: <br> Advanced Topics](https://probml.github.io/pml-book/book2.html) | Deep Learning. It's measure theory galore                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Bonus         | {cite:p}`mackay2003information`                                                                      | This is Probability Theory Bonanza at its finest. On a serious note, this seems to be like a really good reference manual to have once you understand mostly anything in Prob & Statistics (It starts from the basic measure theoretic arguments and builds up until it touches modern probability research areas, e.g. Malliavin Calculus and Stochastic Diff. Geometry), otherwise it's just an hipster book to show off in your library to your colleagues. Recommended Level: Alessandro's Hobo potential level after finishing his PhD. |
<!-- #endregion -->




## Concepts of Probability

The first section will walk through the essential concepts of probability that lay the foundation to understand modern probabilistic modelling.
You should get familiar with the $\sigma$-algebra first. For that, I recommend [this notebook](https://random-events.readthedocs.io/en/latest/examples/product_spaces.html).


The reason why the $\sigma$-algebra is the set of interest for probability theory is, bluntly speaking, knowing the probability of every atomic event is knowing the probability of every possible event.
Practically, the $\sigma$ algebra used in real world modelling is always the powerset of some elementary events. While it would be possible to construct a $\sigma$ algebra that is not the powerset of its elementary events, it serves, up to everything I have seen so far, no purpose to the real world.

The $\sigma$ algebra of interest for practical probabilistic reasoning is the [product $\sigma$-algebra](https://random-events.readthedocs.io/en/latest/examples/product_spaces.html). Next, we have to define probabilities of events. Formally,

<!-- #region -->
### Probability Measure
 Let $(E, \mathfrak{J} )$ be a  $\sigma$-algebra. A non-negative real function $P \rightarrow \mathbb{R}_{0, +}$
 is called a measure if it satisfies the following properties:
 
1. $P(\emptyset) = 0$
2. For any countable sequence $\{A_i \in \Im \}_{i=1,...,}$ of pairwise disjoints sets $A_i \cap A_j = \emptyset$ if $i \neq j, P$ satisfies countable additivity ($\sigma$-additivity):
$$P \left( \bigcup_{i=1}^\infty A_i  \right) = \sum_{i=1}^\infty P(A_i)$$ 
3. $P(A \cup B) = P(A) + P(B) - P(A,B)$
4. $P(E) = 1$


The probability measure just tells, that for non-intersecting sets, you can determine the probability of the union by adding the atomic probabilities. Furthermore, for intersecting sets you have to subtract the intersection, because it is added in there twice otherwise. You may realise that in the [random events package](https://random-events.readthedocs.io/en/latest/) the complex random event always contains a disjoint union of events. $\sigma$-additivity is the sole reason for that.

A common way to visually think about those things are venn diagrams. The size of the objects depicted in the diagrams is proportionate to the probability of the respective event.
<!-- #endregion -->

```python
from random_events.set import SetElement, Set
from random_events.variable import Symbolic, Continuous, Integer
from random_events.product_algebra import Event, SimpleEvent
from random_events.interval import *
import plotly
plotly.offline.init_notebook_mode()
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

x = Continuous("x")
y = Continuous("y")
```

```python
event_a = SimpleEvent({x: closed(0, 2), y: closed(0, 2)}).as_composite_set()
event_b =  SimpleEvent({x: closed(2.5, 3), y: closed(1, 2.5)}).as_composite_set()
fig = go.Figure(event_a.plot(color="red") + event_b.plot(color="blue"), event_a.plotly_layout())
fig.show()
```

In this venn diagram, we can see that event $A$ and event $B$ are not intersecting. Hence, the probability (size) of both events can be added to obtain the size of their union ($\sigma$-additivity).

```python
event_a.simple_sets[0][x] = closed(2, 2.75)
fig = go.Figure(event_a.plot(color="red") + event_b.plot(color="blue"), event_a.plotly_layout())
fig.show()
```

<!-- #region -->
In the second, venn diagram we see that the events are intersecting. Hence, the size of their union is smaller by the sum of both sizes by exactly the intersecting part (3. Axiom).


In general, thinking of probabilities as size of sets is a good intuition. Most things you can think about for geometrical sizes of shapes also apply to probability. However, probabilities never exceed unity (4. Axiom).
From the definition of the probability measure we can derive the next important theorem.
<!-- #endregion -->

### Sum Rule

From $A + \neg A = E$ we get 
$$P(A) + P(\neg A) = P(E) = 1 \text{, thus } P(A) = 1 - P(\neg A).$$
And from $A = A \cap (B + \neg B)$, using the notation $P(A, B) = P(A \cap B)$ for the joint probability of $A$ and $B$, we get the Sum Rule
$$P(A) = P(A,B) + P(A, \neg B).$$

The sum rule is a way to express a belief over only the event $A$ if said probability is not known. Instead, the joint probability of $A$ and $B$ is known and by adding (marginalizing) over all cases of $B$ we can get the probability of $A$.

```python
fig = make_subplots(rows=1, cols=2, subplot_titles=["Event A and B", "Event A and not B"])

event_a_and_b = event_a & event_b
event_a_and_not_b = event_a & ~event_b
fig.add_traces(event_a_and_b.plot(), rows=1, cols=2)
fig.add_traces(event_a_and_not_b.plot(), rows=1, cols=1)
fig.update_xaxes(range=[2, 2.75],)
fig.update_yaxes(range=[0, 2], )
fig.show()


```

In the interactive plot (play around with it as you want) you can observe, that the event $A$ is precisely described by the union of both events $A, \neg B$ and  $A, B$. Furthermore, both events are disjoint and the probability can be calculated using $\sigma$-additivity.

Before we dive into the next block of definitions, there is a little preface. As these next theorems are so strongly intertwined with each other, there is no sense of making a plot for each of them. There is an interactive plot after bayes theorem that give light to how these definitions work.
The next key definition of probabilities is the conditional probability. Formally,


### Conditional Probability

If $P(A) > 0$, the quotient
$$P(B|A) = \frac{P(A, B)}{P(A)}$$
is called the conditional probability of $B$ given $A$. It immediately gives the product rule
$$P(A,B) = P(B|A)P(A) = P(A|B)P(B).$$
It is easy to show that $P(B|A) \geq 0, P(E|A) = 1$ and for $B\cap C = \emptyset$, we have $P(B+C |A) = P(B|A) + P(C|A).$ Thus, for a fixed $A$, $(E, \Im, P(\cdot|A))$ is a probability space as well.


Instead of explaining this concept in my own words, I will refer to Todd Kemps phrasing about it:
> We often think of conditional probability intuitively in terms of a two-stage experiment. 
In the first stage of the experiment, we see whether one particular event $A$ has occurred or not. 
If it has it may influence whether a second event $B$, which we're going to measure in a second stage, 
will occur or not. Therefore, we may want to update our information on probabilities of the later 
events given information about whether the earlier event has occurred or not. 
In this language we'll refer to the original probability measure $P(B)$ as the prior probability of 
an event $B$ and after we observe that event $A$ has occurred we refer to the new updated conditional 
probability as the posterior probability of that same event $B$.
In more practical applications $A$ is referenced as the evidence since it is an event that evidently 
happened and $B$ is the query, since it is the event of interest.
There are two very elementary but extraordinarily important results that come from that line of 
thinking the so-called law of total probability and most critically, Bayes theorem.

Todd Kemp, [30.1 Conditional Probability](https://youtu.be/CX013Hfgv-o?si=RQM5e7bcHJ8wuZeb&t=121])


### Law of total probability

Let $A_1 + A_2 + \cdots + A_n = E$ and $A_i \cap A_j = \emptyset$ if $i \neq j$. Then for any $X \in \Im,$
$$P(X) = \sum_{i=1}^n P(X|A_i)P(A_i).$$

Because $X = E \cap X = \bigcup_{i=1}^n (A_i \cap X),$ we get that 
$$P(X) = \sum_{i=1}^n P(A_i, X) = \sum_{i=1}^n P(X|A_i) P(A_i).$$


### Bayes Theorem

Let $A_1 + A_2 + \cdots + A_n = E$ and $A_i \cap A_j = \emptyset$ if $i \neq j$. Then for any $X \in \Im,$
$$P(A_i|X) = \frac{P(X|A_i)P(A_i)}{\sum_{j=1}^n P(X|A_j) P(A_j)}.$$




```python
def event_trace(event: SimpleEvent, name: str) -> go.Scatter:
    """
    Create the trace for a rectangle event.
    """
    x_0 = event["x"].simple_sets[0].lower
    x_1 = event["x"].simple_sets[0].upper
    y_0 = event["y"].simple_sets[0].lower
    y_1 = event["y"].simple_sets[0].upper
    trace = go.Scatter(x=[x_0, x_0, x_1, x_1, x_0], y=[y_0, y_1, y_1, y_0, y_0], fill="toself", name=name)
    return trace

x_event = SimpleEvent({x: closed(0, 1), y: closed(0, 1)})

# create figure and add event
fig = go.Figure()
fig.add_trace(event_trace(x_event, "X"))

# add disjoint event A_1 to A_n
for index,(x_interval, y_interval) in enumerate(itertools.product([closed(-0.25, 0.5).simple_sets[0], 
                                                                   closed(0.5, 1.25).simple_sets[0]], 
                                                                  [closed(-0.25, 0.5).simple_sets[0], 
                                                                   closed(0.5, 1.25).simple_sets[0]])):
    sub_event = SimpleEvent({x: x_interval, y: y_interval})
    sub_event_trace = event_trace(sub_event, f"A_{index + 1}")
    fig.add_trace(sub_event_trace)
    intersection_trace = event_trace(sub_event.intersection_with(x_event), f"A_{index + 1} and X")
    fig.add_trace(intersection_trace)

fig.show()
```

In the interactive plot, you can see the event $X$ and the disjoint events $A_1, A_2, A_3, A_4$. The intersection of $X$ and $A_i$ is also plotted. You can see that the sum of the intersections is equal to the event $X$. This is the law of total probability. Conditioning on an event can be seen as zooming into the events and only accounting for what is inside the zoomed area. The Bayes theorem is a way to reverse the conditioning. It is a way to update the prior belief of an event given new evidence and just a consequence of the definitions given so far.


## Random Variables

While there exists a definition for random variables I find it quite confusing and disconnected to how things are approached in practice. Instead, I will give a more practical definition on what a random variable is.

A random variable is a term in a language that can take different values. The range of values of a random variable, $\text{dom}(X)$, is exhaustive and mutually exclusive. A tuple $X$ of random variables with $X=\left<X_{1}, \ldots, X_{n}\right>$  is a complex random variable with the domain $$\text{dom}(X)=\text{dom}(X_{1}) \times \ldots \times \text{dom}(X_{n}).$$ 
A possible world $x=\left<X_1=v_1,\ldots,X_n=v_n\right>$ specifies a value assignment for each random variable $X_i$ under consideration.
The set $\mathcal{X}$ of all possible worlds is called the universe.


In real world scenarios data is most of the time provided as a table, dataframe, array, etc. The rows of such an object are referred to as samples or instances and the columns as variables or features.

```python
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)

dataframe = data.data
target = data.target.astype(str)
target[target == "0"] = "Setosa"
target[target == "1"] = "Versicolour"
target[target == "2"] = "Virginica"
dataframe["plant"] = target
dataframe
```

In this example we see a dataset with 150 instances and 5 variables. Probability distributions can now be defined over events that are construct over the complex random variable of this dataset.
[This tutorial](https://random-events.readthedocs.io/en/latest/notebooks/independent_constraints.html) explains the details on how such events are constructed.
The construction of such events leads to the next important concept, independence. Formally,


### Independence

We say that $A$ is independent of $B$ is $(PA|B) = P(A)$ or equivalently that
$$P(A \cap B) = P(A) P(B).$$
Notation: $A \perp \!\!\! \perp B$. Information about $B$ does not give information about $A$ and vice versa.


In this definition $A$ and $B$ could be both events or event entire random variables.
Let's look at a practical example where we construct a joint probability distribution over colors and shapes.

```python
from probabilistic_model.distributions.multinomial import MultinomialDistribution
import numpy as np

class Color(SetElement):
    EMPTY_SET = -1
    BLUE = 0
    RED = 1
    
class Shape(SetElement):
    EMPTY_SET = -1
    CIRCLE = 0
    RECTANGLE = 1
    TRIANGLE = 2

color = Symbolic("color", Color)
shape = Symbolic("shape", Shape)

probabilities = np.array([[2/15, 1/15, 1/5],
                          [1/5, 1/10, 3/10]])

distribution = MultinomialDistribution((color, shape), probabilities)
color_event = SimpleEvent({color: Color.BLUE}).as_composite_set()
shape_event = SimpleEvent({shape: Set(Shape.CIRCLE, Shape.TRIANGLE)}).as_composite_set()
joint_event = color_event & shape_event
print(f"P({joint_event}) = {distribution.probability(joint_event)}")
print(f"P({color_event}) * P({shape_event}) = {distribution.probability(color_event) * distribution.probability(shape_event)}")
```

We can see that both events are independent as their joint probability can be decomposed to the product of the marginal probabilities.
The definition of independence can be expanded to conditional independence. Formally,


### Conditional Independence 

Two variables (events) $A$ and $B$ are conditionally independent given variable (event) $C$, if and only if their conditional distribution factorizes,
$$P(A,B|C) = P(A|C) P(B|C).$$
In that case we have $P(A|B, C) = P(A|C)$, i.e. in light of information $C$, $B$ provides no (further) information about $A$. Notation: $A \perp \!\!\! \perp B \, | \, C$.



### Continuous Random Variables

We just saw how to construct a joint probability table by enumerating all elementary events. However, regarding continuous spaces this is not possible since you cannot enumerate all real numbers.
For real spaces, the probability density function is used. Firstly though, we have to understand events in continuous spaces. In continuous spaces, we are interested in events that describe intervals on the real line. For instance, $P(0 \leq x < 1)$ is the probability that $x$ is between $0$ and $1$. Giving a formal description of the Borel algebra requires topological arguments, which I will not discuss. For most
applications it is sufficient to understand that we want to reason about intervals. Formally,


### Probability Density Function (PDF)

Let $\mathfrak{B}$ be the Borel $\sigma$-algebra on $\mathbb{R}^d$. A probability measure $P$ on $(\mathbb{R}^d, \mathfrak{B})$ has a density $p$ if $p$ is a non-negative (Borel) measurable function on $\mathbb{R^d}$ satisfying for all $B \in \mathfrak{B}$
$$P(B) = \int_B p(x) dx =: \int_B p(x_1,...,x_d) dx_1 ... dx_d.$$


Bluntly speaking, the pdf is an always positive function that is high where we think that this region is highly likely and low where we think it's not.
The integral over a pdf is called the cumulative distribution function and is the object we use to calculate the probability of an event. Formally,   


### Cumulative Distribution Function (CDF)
For probability measures $P$ on $(R^d, \mathfrak{B})$, the cumulative distribution function is the function
$$F(x) = P\left( \prod_{i=1}^d (X_i < x_i) \right).$$
If $F$ is sufficiently differentiable, then $P$ has a density given by
$$p(x) = \frac{\partial^dF}{\partial x_1, ..., \partial x_d}.$$


In this sense, the cdf over many variables describes the probability of the event that every variable is below some value.
For example, if we evaluate $F \begin{pmatrix} 1 \\ 2 \end{pmatrix}$ we calculate the probability of the event that $x_1 < 1 \wedge x_2 < 2$.

```python
from probabilistic_model.distributions.gaussian import GaussianDistribution
normal = GaussianDistribution(Continuous("x"), 0, 1)
fig = go.Figure(normal.plot(), normal.plotly_layout())
fig.for_each_trace(lambda trace: trace.update(visible='legendonly') if trace.name in ["Expectation", "Mode"] else ())
```

The figure above visualizes the objects we just discussed for a univariate normal distribution.


## Queries
The second chapter will walk you trough the most important queries in probabilistic modelling.
There are common quantities that are of interest regarding probabilities. You, most likely, have seen them in one or another form, but I will introduce them formally accompanied by some examples.


### Likelihoods
The likelihood query is the evaluation of the joint probability distribution at a point (possible world). In such a query no variable is left unassigned and no integration is performed.
Likelihood queries just calculate $p(x)$. The example below shows such a query.




```python
possible_world = np.array([[Color.BLUE, Shape.CIRCLE]])
print(f"p({color.name}={possible_world[0, 0]}, {shape.name}={possible_world[0, 1]}) = {distribution.likelihood(possible_world)}")
```

In the abstract interface to probabilistic models in this package, the likelihood query is implemented [here](https://probabilistic-model.readthedocs.io/en/latest/autoapi/probabilistic_model/probabilistic_model/index.html#probabilistic_model.probabilistic_model.ProbabilisticModel._likelihood).


### Maximum Likelihood Estimate

While likelihood query has little use in probabilistic applications, it is essential for the construction and learning of models. It is the basic formula this is evaluated a vast amount of times in maximum likelihood based learning. Formally,

$L(\theta)$ is called the likelihood function. Goal: Determine 
$$\hat{\theta}_{MLE}=\underset{\theta\in\Theta}{arg \,max} L(\theta)=\underset{\theta\in\Theta}{arg \,max}\prod_{i=1}^NP(\mathcal{D}_i | \theta),$$
which is called the Maximum Likelihood Estimate (MLE), where $\theta$ are the parameters of the distribution $p$ and $N$ being the number of observations.

As gradient descent is a common way to maximize the likelihood function, the gradient of said function is also of interest. Calculating the gradient of a big product of functions is problematic due to the complexity of the [product rule](https://en.wikipedia.org/wiki/Product_rule). Instead, the log-likelihood function is used, since the maximum of the logarithm of a function is also the maximum of the function itself. Formally,

$$\hat{\theta}_{MLE}=\underset{\theta\in\Theta}{arg \,max} log(L(\theta)) = \underset{\theta\in\Theta}{arg \,max}\sum_{i=1}^N log(P(\mathcal{D}_i | \theta))$$


### Independently and Identically Distributed

As we can see in the maximum likelihood estimate, the likelihood function is a product of many probabilities. However, multiplying probabilities to achieve a joint probability is only allowed if the events are independent, as we saw in the definition of independence. In practice, we often assume that the data is independently and identically distributed (i.i.d.) in order to allow efficient calculations. The maximum likelihood estimate can be summarized as:
>   Given some arbitrary but fixed parameter $\theta$, what is the probability that we get the
    observed data D assuming each world being independently drawn from the identical
    underlying distribution (i.i.d. assumption)
    
Since determining the exact computations needed for a maximum likelihood estimate is subject to calculus I will not annoy you with that and instead show you a practical example.
For a Normal Distribution we know that the maximum likelihood estimate for the mean is the sample mean and for the variance the sample variance.




```python
from probabilistic_model.probabilistic_circuit.distributions import GaussianDistribution
mean = dataframe["sepal length (cm)"].mean()
variance = dataframe["sepal length (cm)"].var()

sepal_length = Continuous("sepal length (cm)")
distribution = GaussianDistribution(sepal_length, mean, variance)
fig = go.Figure(distribution.plot(), distribution.plotly_layout())
fig.show()
```

The plot shows the normal distribution that is the maximum likelihood estimate for the data, if we assume the data is i. i. d. and drawn from a normal distribution.


### Marginals

The marginal query is the next interesting quantity we investigate.
A common scenario in applications of probability theory is reasoning under uncertainty. Consider the distribution from above that describes the sepal length in centimeters. How probable is it that a plant has a sepal length between 6 and 7 cm?
Answering such questions requires the integration over the described area.  In this scenario we can get the probability by the following piece of code.

```python
event = SimpleEvent({sepal_length: closed(6, 7)}).as_composite_set()
distribution.probability(event)
```

We can see that the probability of such event is approximately $34\%$.
Formally, we can describe such a query as:


Let $p(X)$ be a joint distribution over random variables $X$. The
class of marginal queries over $p$ is the set of functions that compute:
$$p(E = e, Z \in \mathcal{I}) = \int_\mathcal{I} p(z, e) dZ$$
where $e \in dom(E)$ is a partial state for any subset of random variables $E \subseteq X$, and 
$Z = X \setminus E$ is the set of $k$ random variables to be integrated over intervals 
$I = I_1 \times \cdots \times I_k$ each of which is defined over the domain of its corresponding 
random variables in $Z: I_i \subseteq dom(Z_i) $ for $ i = 1, \cdots, k.$


While this definition may be a bit weird to think about, it essentially says that marginal queries are integrations over axis aligned bounding boxes. Furthermore, marginal queries can also contain (partial) point descriptions, just as in the likelihood. [This tutorial](https://random-events.readthedocs.io/en/latest/notebooks/independent_constraints.html) dives deeper into the visualization of such events.

The intervals can also be unions of intervals, such as

```python
event = SimpleEvent({sepal_length: closed(6, 7) | closed(5, 5.5)}).as_composite_set()
distribution.probability(event)
```

### Conditionals

While marginal queries already allow for the calculations of conditional probabilities using the definition of the conditional probability, it is, in most scenarios, more interesting to consider the conditional probability space. We can construct such a thing by invoking the `conditional` method with the corresponding event.

```python
event = SimpleEvent({sepal_length: closed(6, 7)}).as_composite_set()
distribution, probability = distribution.conditional(event)
fig = go.Figure(distribution.plot(), distribution.plotly_layout())
fig.show()
```

We can see that conditioning on the event that the sepal length is between 6 and 7 cm, the resulting distribution is a zoomed in version of the original distribution. The resulting type of distribution is not a Gaussian distribution anymore, but a truncated Gaussian distribution. The probability of the event that the sepal length is between 6 and 7 cm is now $100\%$. Since the truncated Gaussian is a much more complicated object than the ordinary Gaussian distribution, you can read more about it [here](https://probabilistic-model.readthedocs.io/en/latest/examples/truncated_gaussians.html).

```python
distribution.probability(event)
```

### Marginal Distributions

The next interesting object that belongs to the marginal query class is the marginal distribution. The marginal distribution is the distribution over a subset of random variables. To achieve this, all other random variables are integrated out. The marginal distribution is the distribution over the subset of random variables. The marginal distribution can be obtained by invoking the `marginal` method with the corresponding subset of random variables.
Consider the multidimensional distribution over colors and shapes from above. We can obtain the marginal distribution over colors by the following piece of code.

```python
import tabulate
color = Symbolic("color", Color)
shape = Symbolic("shape", Shape)

probabilities = np.array([[2/15, 1/15, 1/5],
                          [1/5, 1/10, 3/10]])

distribution = MultinomialDistribution((color, shape), probabilities)
color_distribution = distribution.marginal([color])
print(tabulate.tabulate(color_distribution.to_tabulate(), tablefmt="fancy_grid"))
```

A final remark on the marginal method,  is that every inference method that is part of the marginal query class is still efficiently calculable. A problem is that, it sometimes destroys a property that is needed for finding the mode of the distribution. We will investigate later what that means.


### Moments

The final interesting object that belongs to the marginal query class are the moments of a distribution. The (central) moment of a distribution is defined as the following integral:

$$\mathbb{M}_n(x) = E((x-E(x))^n) = \int (x-\mu)^n p(x) dx$$
    
In this equation there are multiple symbols that should be discussed. Firstly, $n$ is the order of a moment. Second, $\mu$ is the center of the moment. If the center is $0$, the moment is called a central moment. The most common moments are the expectation given as

$$E(x) = \int x p(x) dx$$,

which is a moment query with $n=1$ and center $0$. The variance is the second central moment and given by

$$Var(x) = E((x-E(x))^2) = \int (x-E(x))^2 p(x) dx$$,

which is a moment query with the mean as the center and $n=2$. While higher order moments exist ($n>2$), they have little practical use and are not further discussed here.
The interface to calculate moments is the `moment` method of the distribution object. 

```python
from random_events.product_algebra import VariableMap
distribution = GaussianDistribution(sepal_length, mean, variance)
print("mean:", distribution.expectation(distribution.variables))
print("variance:", distribution.moment(VariableMap({sepal_length: 2}), distribution.expectation(distribution.variables)))
print("Raw variance:", distribution.moment(VariableMap({sepal_length: 2}), VariableMap({sepal_length: 0})))
print("Third moment with center 3:", distribution.moment(VariableMap({sepal_length: 3}), VariableMap({sepal_length: 3})))
```

As we can see, mean and variance are shortcut by their names since they are so common. Furthermore, we can calculate all moments that exist by plugging in the order and center that we want as VariableMap.

<!-- #region -->
## Mode query

The final important quantity of a probability distribution is the mode. The mode refers to the region where the denisty is maximized. Formally,

$$\hat{x} = \underset{x \in \mathcal{X}}{arg \,max} p(x). $$


While common literature describes the mode under a condition, we can omit such a description since we already defined that the conditional distribution is part of the marginal query class. Hence, the mode under a condition is just the chain of the condition and mode methods.

A common perception of the mode is, that it is the single point of highest density, such as in the example below.
<!-- #endregion -->

```python
distribution = GaussianDistribution(Continuous("x"), 0, 1)
fig = go.Figure(distribution.plot(), distribution.plotly_layout())
fig.show()
```

However, the mode is more accurately described as the set of points with the highest density. Consider the example below.

```python
condition = closed(-float("inf"), -1) | closed(1, float("inf"))
distribution, _ = distribution.conditional(SimpleEvent({distribution.variables[0]: condition}).as_composite_set())
go.Figure(distribution.plot()).show()
```

We can see that conditioning a Gaussian on such an event already creates a mode that has two points. Furthermore, modes can be sets of infinite many points, such as shown below.

```python
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
uniform = UniformDistribution(Continuous("x"), open(-1, 1).simple_sets[0])
go.Figure(uniform.plot(), uniform.plotly_layout()).show()
```

The mode of the uniform distribution is the entire interval of the uniform distribution $(-1, 1)$. The mode is particular useful when we want to find the best (most likely) solution to a problem und not just any.


## Practical Example

In practice, probabilities can be used in robotics. Consider the kitchen scenario from the [product algebra notebook](https://random-events.readthedocs.io/en/latest/notebooks/independent_constraints.html).

```python
kitchen = SimpleEvent({x: closed(0, 6.6), y: closed(0, 7)}).as_composite_set()
refrigerator = SimpleEvent({x: closed(5, 6), y: closed(6.3, 7)}).as_composite_set()
top_kitchen_island = SimpleEvent({x: closed(0, 5), y: closed(6.5, 7)}).as_composite_set()
left_cabinets = SimpleEvent({x: closed(0, 0.5), y: closed(0, 6.5)}).as_composite_set()

center_island = SimpleEvent({x: closed(2, 4), y: closed(3, 5)}).as_composite_set()

occupied_spaces = refrigerator | top_kitchen_island | left_cabinets | center_island
free_space = kitchen - occupied_spaces
fig = go.Figure(free_space.plot(), free_space.plotly_layout())
fig.show()
```

Let's now say, that we want a position to access the fridge. We want to be as close to the fridge as possible. Naturally, we need a gaussian model for that. We now construct a gaussian distribution over the free space to describe locations and their probabilities to access the fridge.

```python
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import DecomposableProductUnit
from probabilistic_model.probabilistic_circuit.distributions import GaussianDistribution
p_x = GaussianDistribution(Continuous("x"), 5.5, 0.5)
p_y = GaussianDistribution(Continuous("y"), 6.65, 0.5)
p_xy = DecomposableProductUnit()
p_xy.add_subcircuit(p_x)
p_xy.add_subcircuit(p_y)
p_xy = p_xy.probabilistic_circuit
fig = go.Figure(p_xy.plot(), p_xy.plotly_layout())
fig.show()
```

We now want to filter for all positions that are available in the kitchen. Hence, we need to condition our probability distribution on the free space of the kitchen. We can do this by invoking the `conditional` method of the distribution object.

```python
distribution, _ = p_xy.conditional(free_space)
fig = go.Figure(distribution.plot(number_of_samples=10000), distribution.plotly_layout())
fig.show()
```

As you can see, we can express our belief of good accessing positions for the fridge using probability theory. This idea scales to many more complex scenarios, such as the localization of a robot in a room or the prediction of the next state of a system. However, this is out of scope for this tutorial.
