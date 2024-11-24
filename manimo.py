from manim import *
import numpy as np
from scipy.integrate import solve_ivp

# Lorenz system definition
def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Function to solve the ODE
def ode_sol_points(initial_state, t_span, t_eval=None, sigma=10, rho=28, beta=8/3):
    # Solve the system using solve_ivp
    solution = solve_ivp(
        lorenz_system,               # The system of equations
        t_span=t_span,               # Time interval to solve over
        y0=initial_state,            # Initial state (x0, y0, z0)
        t_eval=t_eval,               # Times at which to store the solution
        args=(sigma, rho, beta),     # Parameters for the Lorenz system
        method='RK45',               # Runge-Kutta method (default)
    )
    return solution

class LorenzAttractor(ThreeDScene):
    def construct(self):
        # Create the 3D axes
        axes = ThreeDAxes(
            x_range=(-30, 30, 5),
            y_range=(-30, 30, 5),
            z_range=(0, 50, 5),
            axis_config={"color": BLUE}
        )
        
        self.add(axes)
        self.camera.set_phi(75 * DEGREES)  # Set the camera angle
        self.camera.set_theta(30 * DEGREES)  # Set the horizontal rotation of the camera
        self.camera.set_zoom(0.6)

        # Initial conditions for Lorenz attractor
        epsilon = 0.001
        states = [
            [10, 10, 10 + n * epsilon]
            for n in range(2)
        ]
        colors = [BLUE, RED]

        curves = VGroup()
        for state, color in zip(states, colors):
            solution = ode_sol_points(state, t_span=(0, 100), t_eval=np.linspace(0, 100, 2000))  
            
            points = solution.y.T  # This is a (N, 3) array where each row is [x, y, z] over time
            
            # Convert points to 3D coordinates in Manim
            if points.shape[1] != 3:
                raise ValueError(f"Points should have 3 columns, but got shape {points.shape}")

            # Manim expects 3D points as (x, y, z) for each point, hence we directly pass points
            curve = VMobject().set_points_as_corners([axes.c2p(*point) for point in points])  
            curve.set_stroke(color, 2)
            
            curve.make_smooth()
            curves.add(curve)

        # Play the animation: Create the curves
        self.play(
            AnimationGroup(*[Create(curve) for curve in curves]),
            run_time=3  # Reduced runtime for faster animation
        )
        
        # Move the camera
        self.move_camera(phi=60 * DEGREES, theta=PI * 2, run_time=8)
        self.wait(3)
