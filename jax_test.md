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

```{code-cell} ipython3
import jax
import jax.numpy as jnp
```

```{code-cell} ipython3
def tanh(x):  # Define a function
    y = jnp.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)
```

```{code-cell} ipython3
grad_tanh = jax.grad(tanh)  # Obtain its gradient function
print(grad_tanh(1.0))       # Evaluate it at x = 1.0, prints 0.4199743
```

```{code-cell} ipython3
def fwd_solver(f, z_init):
    z_prev, z = z_init, f(z_init)
    while jnp.linalg.norm(z_prev - z) > 1e-5:
        z_prev, z = z, f(z)
    return z
```

```{code-cell} ipython3

def newton_solver(f, z_init):
    f_root = lambda z: f(z) - z
    g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
    return fwd_solver(g, z_init)
```

```{code-cell} ipython3
def fixed_point_layer(solver, f, params, x):
    z_star = solver(lambda z: f(params, x, z), z_init=jnp.zeros_like(x))
    return z_star
```

```{code-cell} ipython3
f = lambda W, x, z: jnp.tanh(jnp.dot(W, z) + x)
```

```{code-cell} ipython3
ndim = 1000
W = random.normal(random.PRNGKey(0), (ndim, ndim)) / jnp.sqrt(ndim)
x = random.normal(random.PRNGKey(1), (ndim,))
```

```{code-cell} ipython3
z_star = fixed_point_layer(fwd_solver, f, W, x)
```

```{code-cell} ipython3
z_star = fixed_point_layer(newton_solver, f, W, x)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
