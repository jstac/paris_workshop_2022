"""
We consider an investment problem with adjustment costs with inverse demand
curve

    P_t = a_0 - a_1 Y_t + Z_t,

where P_t is price, Y_t is output and Z_t is a demand shock.

We assume that Z_t is a discretized AR(1) process.

Current profits are 

    P_t Y_t - c Y_t - gamma (Y_{t+1} - Y_t)^2

The firm maximizes present value of expected discounted profits.  The state is

    X_t = (Y_t, Z_t)

The right-hand side of the Bellman equation is 

    B(y, z, y′) = r(y, z, y′) + β Σ_z′ v(y′, z′) Q(z, z′)."

where 

    r(y, z, y′) := (a_0 - a_1 * y + z - c) y - γ * (y′ - y)^2

"""
import numpy as np
import quantecon as qe
import jax
import jax.numpy as jnp
from jaxopt import FixedPointIteration, AndersonAcceleration
from collections import namedtuple

# Use 64 bit floats with JAX in order to match NumPy/Numba code
jax.config.update("jax_enable_x64", True)

Model = namedtuple("Model", 
                   ("β", "a_0", "a_1", "γ", "c",
                    "y_size", "z_size",
                    "y_grid", "z_grid", "Q"))

def create_investment_model(
        r=0.04,                              # Interest rate
        a_0=10.0, a_1=1.0,                   # Demand parameters
        γ=25.0, c=1.0,                       # Adjustment and unit cost 
        y_min=0.0, y_max=20.0, y_size=4,   # Grid for output
        ρ=0.9, ν=1.0,                        # AR(1) parameters
        z_size=5):                          # Grid size for shock

    β = 1/(1+r) 
    y_grid = np.linspace(y_min, y_max, y_size)  
    mc = qe.tauchen(ρ, ν, n=z_size)
    z_grid, Q = mc.state_values, mc.P

    return Model(β=β, a_0=a_0, a_1=a_1, γ=γ, c=c,
                 y_size=y_size, z_size=z_size,
                 y_grid=y_grid, z_grid=z_grid, Q=Q)

def create_investment_model_jax(model):
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = model
    y_grid, z_grid, Q = map(jax.device_put, (y_grid, z_grid, Q))
    return Model(β=β, a_0=a_0, a_1=a_1, γ=γ, c=c,
                 y_size=y_size, z_size=z_size,
                 y_grid=y_grid, z_grid=z_grid, Q=Q)

def T_vectorized(v, jax_model):
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = jax_model
    y  = jnp.reshape(y_grid, (y_size, 1, 1))
    z  = jnp.reshape(z_grid, (1, z_size, 1))
    yp = jnp.reshape(y_grid, (1, 1, y_size))
    R = (a_0 - a_1 * y + z - c) * y - γ * (yp - y)**2

    v = jnp.reshape(v, (1, 1, y_size, z_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Q = jnp.reshape(Q, (1, z_size, 1, z_size))  # Q[j, jp]  -> Q[i, j, ip, jp]
    C = jnp.sum(v * Q, axis=3)                  # sum over jp

    return jnp.max(R + β * C, axis=2)


def B(i, j, ip, v, np_model):
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = np_model
    y, z, yp = y_grid[i], z_grid[j], y_grid[ip]
    r = (a_0 - a_1 * y + z - c) * y - γ * (yp - y)**2
    return r + β * np.dot(v[ip, :], Q[j, :]) 

def T_loops(v, np_model):
    β, a_0, a_1, γ, c, y_size, z_size, y_grid, z_grid, Q = np_model
    v_new = np.empty_like(v)
    for i in range(y_size):
        for j in range(z_size):
            v_new[i, j] = np.max([B(i, j, ip, v, model) for ip in range(y_size)])
    return v_new

model = create_investment_model()
v_np = np.ones((model.y_size, model.z_size))

jax_model = create_investment_model_jax(model)
v_jax = jnp.ones((model.y_size, model.z_size))

