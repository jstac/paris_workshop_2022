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
import numpy as np

from numba import cuda, vectorize
```

```{code-cell} ipython3
@cuda.jit
def f(a, b, c):
    # like threadIdx.x + (blockIdx.x * blockDim.x)
    tid = cuda.grid(1)
    size = len(c)

    if tid < size:

        c[tid] = a[tid] + b[tid]
```

```{code-cell} ipython3
N = 5000000
a = cuda.to_device(np.random.random(N))
b = cuda.to_device(np.random.random(N))
c = cuda.device_array_like(a)
```

```{code-cell} ipython3
f.forall(len(a))(a, b, c)

c.copy_to_host()
```

```{code-cell} ipython3
type(f)
```

```{code-cell} ipython3

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
```

```{code-cell} ipython3
@vectorize(['int64(int64, int64)'], target='cuda') # Type signature and target are required for the GPU
def add_ufunc(x, y):
    return x + y
```

```{code-cell} ipython3
add_ufunc(a, b)
```

```{code-cell} ipython3

```
