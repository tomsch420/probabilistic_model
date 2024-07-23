# Representations of Circuits

This section discusses different approaches to represent circuits.

## the DAG (networkx) way

## The Layered way

## Nyga Distribution Log Likelihood

| Number of Parameters   | networkx    | torch       | Numpy       |
|------------------------|-------------|-------------|-------------|
| 81 Nodes, 80 Edges     | 0.000836689 | 0.001973407 | 0.000796553 |
| 790 Nodes, 789 Edges   | 0.026516185 | 0.027524014 | 0.067368826 |
| 7892 Nodes, 7891 Edges | 2.258284528 | 2.256407159 | 6.864992611 |


-> Numpy is slowest, networkx is fastest, but why?