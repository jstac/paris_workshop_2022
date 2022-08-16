---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Equilibrium in Two Dimensions

+++

#### Written for the Paris Quantitative Economics Workshop (September 2022)
#### Author: [John Stachurski](http://johnstachurski.net/)

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import newton, bisect
```

## The Market


+++

We now consider a market for two related products, good 0 and good 1, with price vector $p = (p_0, p_1)$.

Supply of good $i$ at price $p$ is 

$$ q^s_i (p) = b_i \sqrt{p_i} $$

Demand of good $i$ at price $p$ is

$$ q^d_i (p) = \exp(-a_{i0} p_0) + \exp(-a_{i1} p_1) + c_i$$

Here $c_i, b_i$ and $a_{ij}$ are parameters.

The excess demand functions are

$$ e_i(p) = q^d_i(p) - q^s_i(p), \qquad i = 1, 2 $$

An equilibrium price vector $p^*$ is one where $e_i(p^*) = 0$.  

We set

$$ 
    A = \begin{pmatrix}
            a_{11} & a_{12} \\
            a_{21} & a_{22}
        \end{pmatrix}
    \qquad \text{and} \qquad
    b = \begin{pmatrix}
            b_1 \\
            b_2
        \end{pmatrix}
$$

+++

Our default parameter values will be

```{code-cell} ipython3
a_00, a_01 = 0.9, 0.1
a_10, a_11 = 0.2, 1.1
b_0, b_1 = 1.0, 1.0
c_0, c_1 = 1.0, 1.0
```

```{code-cell} ipython3
def e_0(p_0, p_1):
    return np.exp(- a_00 * p_0) + np.exp(- a_01 * p_1) + c_0 - b_0 * np.sqrt(p_0)
```

```{code-cell} ipython3
def e_1(p_0, p_1):
    return np.exp(- a_10 * p_0) + np.exp(- a_11 * p_1) + c_1 - b_1 * np.sqrt(p_1)
```

Let us plot these functions using contour surfaces and lines.

```{code-cell} ipython3
grid_size = 100
grid_max = 10
p_grid = np.linspace(0, grid_max, grid_size)
z = np.empty((100, 100))

for i, p_1 in enumerate(p_grid):
    for j, p_2 in enumerate(p_grid):
        z[i, j] = e_0(p_1, p_2)

fig, ax = plt.subplots(figsize=(10, 5.7))
cs1 = ax.contourf(p_grid, p_grid, z.T, alpha=0.5)
ctr1 = ax.contour(p_grid, p_grid, z.T, levels=[0.0])

plt.clabel(ctr1, inline=1, fontsize=13)
plt.colorbar(cs1, ax=ax, format="%.6f")
```

```{code-cell} ipython3

for i, p_1 in enumerate(p_grid):
    for j, p_2 in enumerate(p_grid):
        z[i, j] = e_1(p_1, p_2)

fig, ax = plt.subplots(figsize=(10, 5.7))
cs1 = ax.contourf(p_grid, p_grid, z.T, alpha=0.5)
ctr1 = ax.contour(p_grid, p_grid, z.T, levels=[0.0])

plt.clabel(ctr1, inline=1, fontsize=13)
plt.colorbar(cs1, ax=ax, format="%.6f")
```

```{code-cell} ipython3
z_0 = np.empty((grid_size, grid_size))
z_1 = np.empty((grid_size, grid_size))

for i, p_1 in enumerate(p_grid):
    for j, p_2 in enumerate(p_grid):
        z_0[i, j] = e_0(p_1, p_2)
        z_1[i, j] = e_1(p_1, p_2)

fig, ax = plt.subplots(figsize=(10, 5.7))

ctr1 = ax.contour(p_grid, p_grid, z_0.T, levels=[0.0])
ctr1 = ax.contour(p_grid, p_grid, z_1.T, levels=[0.0])

plt.clabel(ctr1, inline=1, fontsize=13)
#plt.colorbar(cs1, ax=ax, format="%.6f")
```

```{code-cell} ipython3

```
