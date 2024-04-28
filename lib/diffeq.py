import numpy as np
from tqdm import tqdm, trange
from matplotlib.animation import FuncAnimation, PillowWriter
from copy import deepcopy as copy
import matplotlib.pyplot as plt


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
    print(f"alpha = {alpha}")

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

    return u.T

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

def solve(U, V, dt, dx, alpha0, omega, q=1, eta = 1, verbose = True):
    assert U.shape == V.shape, "U and V must have the same shape/resolution"
    U = -copy(U)  # copy to avoid changing the original
    V = copy(V)  # copy to avoid changing the original
    # if verbose: print(f"U shape: {U.shape}, V shape: {V.shape}")
    a = 1/dt + eta/(dx**2)
    b = eta/(2*dx**2)
    if verbose:
        print(f"Dynamo number = {alpha0*q*omega*1/eta**3}")
        print(f"a: {a}, b: {b}")

    A = np.zeros((U.shape[1]-2, U.shape[1]-2))
    for row in range(A.shape[0]):
        A[row, row] = a
        if row > 0:
            A[row, row-1] = -b
        if row < A.shape[0]-1:
            A[row, row+1] = -b

    A_inv = np.linalg.inv(A)

    if verbose: ra = trange
    else: ra = range

    z = np.linspace(-1, 1, U.shape[1]-1)

    for j in ra(1, U.shape[0]):  # for all the time steps
        # make B for U
        # print(z[:-1].shape, (V[:, 1:-1] - V[:, :-2]).shape)
        BU = (
              U[j-1, 1:-1]*(1/dt - eta/dx**2)
            # - alpha0*np.sign(z[:-1])*(V[j-1, 1:-1] - V[j-1, :-2])/dx
            - alpha0*np.sin(z[:-1]*np.pi)*(V[j-1, 1:-1] - V[j-1, :-2])/dx
            + b*(U[j-1, :-2] + U[j-1, 2:])
        )
        # print(BU.shape, U.shape)
        U[j, 1:-1] = A_inv @ BU

        BV = ( 
              V[j-1, 1:-1]*(1/dt - eta/dx**2) 
             - omega*q*U[j, 1:-1] 
             + b*(V[j-1, :-2] + V[j-1, 2:]) 
        )
        V[j, 1:-1] = A_inv @ BV.T

    return -U, V

class Gif:
    def __init__(self, labels=["$B_r$", "$B_\phi$"], skip_frame = 1, till=None, fps=25, fix_limits=True, save_dir="outputs/asgt2") -> None:
        self.labels = labels
        self.skip_frame = skip_frame
        self.till = till
        self.fps = fps
        self.fix_limits = fix_limits
        self.save_dir = save_dir

    def draw(self, x, us, name, labels=None, skip_frame = None, till=None, fps=None, fix_limits=None, save_dir=None):
        """
        Create an animated GIF of magnetic field strength vs z distance over time.

        Parameters:
            x (array_like): Distance (z) values.
            us (array_like): Magnetic field strength data with shape (2, t, n) where
                2 is the number of fields, t is the number of time steps, and n is the number of distance points.
            name (str): Name of the output GIF file.
            labels (list of str): Labels for the two magnetic fields.
            skip_frame (int, optional): Skip every `skip_frame` frames in the animation. Defaults to 1.
            till (int, optional): Limit the animation to the first `till` time steps. Defaults to None.
            fps (int, optional): Frames per second of the output GIF. Defaults to 25.
            fix_limits (bool, optional): Whether to fix the y-axis limits throughout the animation. Defaults to True.
        """
        labels = labels if labels is not None else self.labels
        skip_frame = skip_frame if skip_frame is not None else self.skip_frame
        till = till if till is not None else self.till
        fps = fps if fps is not None else self.fps
        fix_limits = fix_limits if fix_limits is not None else self.fix_limits
        save_dir = save_dir if save_dir is not None else self.save_dir
        
        us = us[:, :till:skip_frame, :] if till is not None else us[:, ::skip_frame, :]
        max_B = np.max(us)
        min_B = np.min(us)

        p = tqdm(total=us.shape[1]+1)

        colours = [
            'tab:blue',
            'tab:orange',
            'tab:green',
            'tab:red',
        ]
        fig, ax = plt.subplots()
        def update(frame):
            p.update(1)
            ax.clear()

            ax.plot(x, us[0, frame], colours[0], label=labels[0])
            ax.plot(x, us[1, frame], colours[1], label=labels[1])
            
            ax.plot(x, us[0, 0], colours[0], label=f"Initial {labels[0]}", alpha=0.4, linestyle="--")
            ax.plot(x, us[1, 0], colours[1], label=f"Initial {labels[1]}", alpha=0.4, linestyle="--")
            if fix_limits:
                plt.ylim(min_B, max_B)
            ax.set_title(f"Magnetic Field Strength vs z Distance at Time Step {frame*skip_frame}")
            ax.set_xlabel('Distance (z)')
            ax.set_ylabel('Magnetic Field Strength (B)')
            ax.legend(loc='lower right')
            ax.grid()

        animation = FuncAnimation(fig, update, frames=us.shape[1], interval=int(1000/fps), repeat=False)
        writervideo = PillowWriter(fps=fps)
        animation.save(f"{save_dir}/{name}", writer=writervideo)
        p.close()