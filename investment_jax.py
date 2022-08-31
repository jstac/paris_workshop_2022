"""
Re-implements the investment problem using JAX.

We test

1. VFI
2. VFI with Anderson acceleration
3. HPI
4. OPI 

"""
import numpy as np
import quantecon as qe
import jax
import jax.numpy as jnp
from jaxopt import FixedPointIteration, AndersonAcceleration

# Use 64 bit floats with JAX in order to match NumPy/Numba code
jax.config.update("jax_enable_x64", True)


def create_investment_model_jax(model):
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = model
    y_grid, z_grid, Q = map(jax.device_put, (y_grid, z_grid, Q))
    return Model(β=β, a_0=a_0, a_1=a_1, γ=γ, c=c,
                 y_size=y_size, z_size=z_size,
                 y_grid=y_grid, z_grid=z_grid, Q=Q)

def T_jax(v, jax_model):
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = jax_model
    y  = jnp.reshape(y_grid, (y_size, 1, 1))
    z  = jnp.reshape(z_grid, (1, z_size, 1))
    yp = jnp.reshape(y_grid, (1, 1, y_size))
    R = (a_0 - a_1 * y + z - c) * y - γ * (yp - y)**2

    v = jnp.reshape(v, (1, 1, y_size, z_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Q = jnp.reshape(Q, (1, z_size, 1, z_size))  # Q[j, jp]  -> Q[i, j, ip, jp]
    C = jnp.sum(v * Q, axis=3)                  # sum over jp

    return jnp.max(R + β * C, axis=2)




jax_model = create_investment_model_jax(model)
v_jax = jnp.ones((model.y_size, model.z_size))

