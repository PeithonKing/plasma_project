import numpy as np


def crank_nicolson_solve(
    u0: np.ndarray,  # initial condition
    L: int | float,  # total length
    T: int | float,  # total time
    Nl: int,         # number of spatial steps
    Nt: int,         # number of time steps
    eta: float = 4   # diffusion coefficient
):
    """Solves the differential equation using the Crank-Nicolson method.
    
    The differential equation is of the form:
    du/dt = eta * d^2u/dx^2

    -- > x
   | (b) (.) (c)
   v    \   /
   t (a)-(e)-(d)

    here e = (a + b + c + d)/4

    but, because without knowing the whole grid, you never know (d)... so... you have to solve a set of linear equations to find (d)

    Args:
        u0 (np.ndarray): Initial condition.
        L (int | float): Total length.
        T (int | float): Total time.
        Nl (int): Number of spatial steps.
        Nt (int): Number of time steps.
        eta (int, optional): Diffusion coefficient. Defaults to 4.

    Returns:
        np.ndarray: Solution to the differential equation.
        
    Example usage:
    
    ```python
    
    ```
    """
    h = L / Nl
    k = T / Nt
    alpha = k*(eta**0.5) / h**2

    u = np.zeros((Nl + 1, Nt + 1))
    u[:, 0] = u0

    u[0, :] = 0  # mostly this is the case
    u[Nl, :] = 0  # mostly this is the case

    # preparing the A and B matrices
    A = np.zeros((Nl-1, Nl-1))
    B = np.zeros((Nl-1, Nl-1))
    for i in range(Nl-1):
        A[i, i] = 2/eta + 2 * alpha
        B[i, i] = 2/eta - 2 * alpha
        if i < Nl-2:
            A[i, i+1] = -alpha
            A[i+1, i] = -alpha
            B[i, i+1] = alpha
            B[i+1, i] = alpha

    # inverting A
    A_inv = np.linalg.inv(A)

    b = np.zeros(Nl-1)
    for j in range(1,Nt+1):
        b[0]    = alpha * u[0, j-1] + alpha * u[0, j]
        b[Nl-2] = alpha * u[Nl,j-1] + alpha * u[Nl,j]
        v = B @ u[1:Nl, j-1]
        u[1:(Nl),j] = A_inv @ (v+b)

    return u.T#, alpha

def rk4(f, x, x0, y0, h=0.01):
    """Solves a first-order ordinary differential equation (ODE) using the Runge-Kutta 4th order method.

    The general form of the ODE is:
    dx/dy = f(x, y)

    Args:
        f (function): Function to be integrated.
        x (int): x value to be evaluated.
        x0 (float): Initial x value.
        y0 (float): Initial y value.
        h (float): Step size.

    Returns:
        float: Approximate solution to the ODE.
    """
    xs, ys = [x0], [y0]
    while x0 < x:
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + h / 2, y0 + k1 / 2)
        k3 = h * f(x0 + h / 2, y0 + k2 / 2)
        k4 = h * f(x0 + h, y0 + k3)
        y0 += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x0 += h
        xs.append(x0)
        ys.append(y0)
    return xs, ys, y0
