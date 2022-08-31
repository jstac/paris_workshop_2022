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
from solvers import successive_approx
from jaxopt import FixedPointIteration, AndersonAcceleration

from investment import Model, create_investment_model

# Use 64 bit floats with JAX in order to match NumPy/Numba code
jax.config.update("jax_enable_x64", True)


def create_investment_model_jax(model):
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = model
    y_grid, z_grid, Q = map(jax.device_put, (y_grid, z_grid, Q))
    return Model(β=β, a_0=a_0, a_1=a_1, γ=γ, c=c,
                 y_size=y_size, z_size=z_size,
                 y_grid=y_grid, z_grid=z_grid, Q=Q)

def T(v, model):
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = model
    y  = jnp.reshape(y_grid, (y_size, 1, 1))
    z  = jnp.reshape(z_grid, (1, z_size, 1))
    yp = jnp.reshape(y_grid, (1, 1, y_size))
    R = (a_0 - a_1 * y + z - c) * y - γ * (yp - y)**2

    v = jnp.reshape(v, (1, 1, y_size, z_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Q = jnp.reshape(Q, (1, z_size, 1, z_size))  # Q[j, jp]  -> Q[i, j, ip, jp]
    C = jnp.sum(v * Q, axis=3)                  # sum over jp

    return jnp.max(R + β * C, axis=2)

def T_σ(v, σ, model):
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = model
    y_size, z_size = len(y_grid), len(z_grid)

    y = jnp.reshape(y_grid, (y_size, 1))
    z = jnp.reshape(z_grid, (1, z_size))
    yp = y_grid[σ]
    r_σ = (a_0 - a_1 * y + z - c) * y - γ * (yp - y)**2

    yp_idx = jnp.arange(y_size)
    yp_idx = jnp.reshape(yp_idx, (1, 1, y_size, 1))
    σ = jnp.reshape(σ, (y_size, z_size, 1, 1))
    A = jnp.where(σ == yp_idx, 1, 0)
    Q = jnp.reshape(Q, (1, z_size, 1, z_size))
    P_σ = A * Q

    n = y_size * z_size
    P_σ = jnp.reshape(P_σ, (n, n))
    r_σ = jnp.reshape(r_σ, n)
    v = jnp.reshape(v, n)
    new_v = r_σ + β * P_σ @ v

    # Return as multi-index array
    return jnp.reshape(new_v, (y_size, z_size))

def get_greedy(v, model):
    "Compute a v-greedy policy."
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = model
    y  = jnp.reshape(y_grid, (y_size, 1, 1))
    z  = jnp.reshape(z_grid, (1, z_size, 1))
    yp = jnp.reshape(y_grid, (1, 1, y_size))
    R = (a_0 - a_1 * y + z - c) * y - γ * (yp - y)**2

    v = jnp.reshape(v, (1, 1, y_size, z_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Q = jnp.reshape(Q, (1, z_size, 1, z_size))  # Q[j, jp]  -> Q[i, j, ip, jp]
    C = jnp.sum(v * Q, axis=3)                  # sum over jp

    return jnp.argmax(R + β * C, axis=2)

def get_value(σ, model):
    "Get the value v_σ of policy σ."
    # Unpack and set up
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = model
    y_size, z_size = len(y_grid), len(z_grid)

    y = jnp.reshape(y_grid, (y_size, 1))
    z = jnp.reshape(z_grid, (1, z_size))
    yp = y_grid[σ]
    r_σ = (a_0 - a_1 * y + z - c) * y - γ * (yp - y)**2

    yp_idx = jnp.arange(y_size)
    yp_idx = jnp.reshape(yp_idx, (1, 1, y_size, 1))
    σ = jnp.reshape(σ, (y_size, z_size, 1, 1))
    A = jnp.where(σ == yp_idx, 1, 0)
    Q = jnp.reshape(Q, (1, z_size, 1, z_size))
    P_σ = A * Q

    n = y_size * z_size
    P_σ = jnp.reshape(P_σ, (n, n))
    r_σ = jnp.reshape(r_σ, n)
    v_σ = jnp.linalg.solve(np.identity(n) - β * P_σ, r_σ)
    # Return as multi-index array
    return jnp.reshape(v_σ, (y_size, z_size))


# == Solvers == #

def value_iteration(model, tol=1e-5):
    "Implements VFI."
    vz = jnp.zeros((len(model.y_grid), len(model.z_grid)))
    v_star = successive_approx(lambda v: T(v, model), vz, tolerance=tol)
    return get_greedy(v_star, model)

def policy_iteration(model):
    "Howard policy iteration routine."
    y_size, z_size = len(model.y_grid), len(model.z_grid)
    σ = jnp.ones((y_size, z_size), dtype=int)
    i, error = 0, 1.0
    while error > 0:
        v_σ = get_value(σ, model)
        σ_new = get_greedy(v_σ, model)
        error = jnp.max(np.abs(σ_new - σ))
        σ = σ_new
        i = i + 1
        print(f"Concluded loop {i} with error {error}.")
    return σ

def optimistic_policy_iteration(model, tol=1e-5, m=10):
    "Implements the OPI routine."
    v = jnp.zeros((len(model.y_grid), len(model.z_grid)))
    error = tol + 1
    while error > tol:
        last_v = v
        σ = get_greedy(v, model)
        for _ in range(m):
            v = T_σ(v, σ, model)
        error = jnp.max(np.abs(v - last_v))
    return get_greedy(v, model)

_model = create_investment_model()
model = create_investment_model_jax(_model)
print("Starting HPI.")
print(policy_iteration(model))
print("Starting VFI.")
print(value_iteration(model))
print("Starting OPI.")
print(optimistic_policy_iteration(model))

