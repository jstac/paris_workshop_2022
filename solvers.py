import numpy as np
def successive_approx(T,                     # Operator (callable)
                      x_0,                   # Initial condition
                      tolerance=1e-6,        # Error tolerance
                      max_iter=10_000,       # Max iteration bound
                      print_step=25,         # Print at multiples
                      verbose=False):        
    """
    Implements successive approximation by iterating on the operator T.

    """
    x = x_0
    error = np.inf
    k = 1
    while error > tolerance and k <= max_iter:
        x_new = T(x)
        error = np.max(np.abs(x_new - x))
        if verbose and k % print_step == 0:
            print(f"Completed iteration {k} with error {error}.")
        x = x_new
        k += 1
    if k < max_iter:
        print(f"Terminated successfully in {k} iterations.")
    else:
        print(f"Warning: Iteration hit max_iter bound {max_iter}.")
    return x
