# -*- coding: utf-8 -*-
"""
Model Predictive Control for Spring-Mass System

This implementation provides MPC control for a system of M masses connected
by springs with friction, controlled by forces applied to the first and last mass.

System dynamics:
    m*x_i'' + 2*c*x_i' + h(x_i - x_{i-1}) + h(x_i - x_{i+1}) = 0

where h(r) = k*r for linear springs, or h(r) = k*r - k_nl*r^3 for nonlinear springs.

Author: MPC Spring-Mass Python Implementation
Date: November 2025
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.integrate import solve_ivp


class SpringMassMPC(object):
    """
    Model Predictive Controller for Spring-Mass System
    
    Parameters:
    -----------
    M : int
        Number of masses in the system
    m : float
        Mass of each element
    k : float
        Linear spring coefficient
    c : float
        Friction coefficient
    k_nl : float, optional
        Nonlinear spring coefficient (default: 0.01)
    u_max : float, optional
        Maximum control force (default: 5.0)
    dt : float, optional
        Time step for discretization (default: 0.1)
    N : int, optional
        Prediction horizon (number of time steps) (default: 100)
    Q : float, optional
        Control input weight (default: 1.0)
    R : float, optional
        State position weight (default: 50.0)
    is_linear : bool, optional
        Whether to use linear or nonlinear spring model (default: True)
    """
    
    def __init__(self, M=7, m=1.0, k=5.0, c=0.1, k_nl=0.01, 
                 u_max=5.0, dt=0.1, N=100, Q=1.0, R=50.0, is_linear=True):
        
        # System parameters
        self.M = M          # Number of masses
        self.m = m          # Mass of each element
        self.k = k          # Linear spring coefficient
        self.c = c          # Friction coefficient
        self.k_nl = k_nl    # Nonlinear spring coefficient
        self.u_max = u_max  # Maximum control force
        self.is_linear = is_linear
        
        # MPC parameters
        self.dt = dt        # Time discretization step
        self.N = N          # Prediction horizon
        self.Q = Q          # Control weight
        self.R = R          # State weight
        
        # Time vector for prediction horizon
        self.time = np.arange(0, N * dt, dt)
        
        # Current time step
        self.current_time = 0
        
        # History storage
        self.state_history = []
        self.control_history = []
        self.time_history = []
        
    def spring_force(self, r):
        """
        Compute spring force based on displacement
        
        Parameters:
        -----------
        r : array_like
            Displacement
            
        Returns:
        --------
        force : array_like
            Spring force
        """
        if self.is_linear:
            return self.k * r
        else:
            # Nonlinear spring: h(r) = k*r - k_nl*r^3
            return self.k * r - self.k_nl * r**3
    
    def system_dynamics(self, t, y):
        """
        Compute system dynamics dy/dt = f(t, y)
        
        Parameters:
        -----------
        t : float
            Current time
        y : array_like
            State vector [x1, ..., xM, v1, ..., vM]
            
        Returns:
        --------
        dydt : array_like
            Time derivative of state
        """
        M = self.M
        x = y[:M]
        v = y[M:2*M]
        a = np.zeros(M)
        
        # Get control inputs (if available, otherwise zero)
        u1, u2 = 0.0, 0.0
        
        # Inner masses (2 to M-1)
        for i in range(1, M-1):
            r_left = x[i] - x[i-1]
            r_right = x[i] - x[i+1]
            a[i] = -1/self.m * (2 * self.c * v[i] + 
                                self.spring_force(r_left) + 
                                self.spring_force(r_right))
        
        # First mass (controlled)
        r_right = x[0] - x[1]
        a[0] = -1/self.m * (u1 + 2 * self.c * v[0] + 
                            self.k * x[0] + 
                            self.spring_force(r_right))
        
        # Last mass (controlled)
        r_left = x[M-1] - x[M-2]
        a[M-1] = -1/self.m * (u2 + 2 * self.c * v[M-1] + 
                              self.spring_force(r_left) + 
                              self.k * x[M-1])
        
        dydt = np.concatenate([v, a])
        return dydt
    
    def discretize_dynamics(self, x_prev, v_prev, a_prev, x_next, v_next, a_next):
        """
        Discretize dynamics using trapezoidal rule
        
        Parameters:
        -----------
        x_prev, v_prev, a_prev : array_like
            Position, velocity, acceleration at time k
        x_next, v_next, a_next : array_like
            Position, velocity, acceleration at time k+1
            
        Returns:
        --------
        residual_x : array_like
            Residual for position update equation
        residual_v : array_like
            Residual for velocity update equation
        """
        dt = self.dt
        
        # x(k+1) = x(k) + 0.5 * (v(k) + v(k+1)) * dt
        residual_x = x_next - x_prev - 0.5 * (v_prev + v_next) * dt
        
        # v(k+1) = v(k) + 0.5 * (a(k) + a(k+1)) * dt
        residual_v = v_next - v_prev - 0.5 * (a_prev + a_next) * dt
        
        return residual_x, residual_v
    
    def mpc_objective(self, decision_vars, x0, v0):
        """
        MPC objective function: minimize positions and control effort
        
        Parameters:
        -----------
        decision_vars : array_like
            Decision variables [x, v, a, u]
        x0 : array_like
            Initial positions
        v0 : array_like
            Initial velocities
            
        Returns:
        --------
        cost : float
            Total cost
        """
        N = self.N
        M = self.M
        
        # Extract variables
        x = decision_vars[:N*M].reshape(N, M)
        u = decision_vars[3*N*M:].reshape(N, 2)
        
        # Cost: R * ||x||_1 + Q * ||u||_1
        position_cost = self.R * np.sum(np.abs(x))
        control_cost = self.Q * np.sum(np.abs(u))
        
        return position_cost + control_cost
    
    def mpc_constraints(self, decision_vars, x0, v0):
        """
        MPC constraints: dynamics and initial conditions
        
        Parameters:
        -----------
        decision_vars : array_like
            Decision variables [x, v, a, u]
        x0 : array_like
            Initial positions
        v0 : array_like
            Initial velocities
            
        Returns:
        --------
        constraints : array_like
            Constraint violations (should be zero)
        """
        N = self.N
        M = self.M
        
        # Extract variables
        x = decision_vars[:N*M].reshape(N, M)
        v = decision_vars[N*M:2*N*M].reshape(N, M)
        a = decision_vars[2*N*M:3*N*M].reshape(N, M)
        u = decision_vars[3*N*M:].reshape(N, 2)
        
        constraints = []
        
        # Initial conditions
        constraints.append(x[0] - x0)
        constraints.append(v[0] - v0)
        
        # Dynamics constraints for each time step
        for k in range(N-1):
            # Position and velocity updates
            res_x, res_v = self.discretize_dynamics(
                x[k], v[k], a[k], x[k+1], v[k+1], a[k+1]
            )
            constraints.append(res_x)
            constraints.append(res_v)
        
        # Force balance equations
        for k in range(N):
            # Inner masses
            for i in range(1, M-1):
                r_left = x[k, i] - x[k, i-1]
                r_right = x[k, i] - x[k, i+1]
                force_balance = (self.m * a[k, i] + 
                                2 * self.c * v[k, i] +
                                self.k * r_left + 
                                self.k * r_right)
                constraints.append(force_balance)
            
            # First mass
            r_right = x[k, 0] - x[k, 1]
            force_balance_1 = (u[k, 0] + self.m * a[k, 0] + 
                              2 * self.c * v[k, 0] +
                              self.k * x[k, 0] + 
                              self.k * r_right)
            constraints.append(force_balance_1)
            
            # Last mass
            r_left = x[k, M-1] - x[k, M-2]
            force_balance_M = (u[k, 1] + self.m * a[k, M-1] + 
                              2 * self.c * v[k, M-1] +
                              self.k * r_left + 
                              self.k * x[k, M-1])
            constraints.append(force_balance_M)
        
        return np.concatenate([np.atleast_1d(c) for c in constraints])
    
    def solve_mpc(self, x0, v0, verbose=False):
        """
        Solve the MPC optimization problem
        
        Parameters:
        -----------
        x0 : array_like
            Initial positions
        v0 : array_like
            Initial velocities
        verbose : bool, optional
            Print optimization details
            
        Returns:
        --------
        x_opt : array_like
            Optimal position trajectory
        v_opt : array_like
            Optimal velocity trajectory
        u_opt : array_like
            Optimal control inputs
        success : bool
            Whether optimization succeeded
        """
        N = self.N
        M = self.M
        
        # Total number of decision variables
        n_vars = 3*N*M + 2*N  # x, v, a (each N*M) + u (N*2)
        
        # Initial guess (zeros)
        x_init = np.zeros(n_vars)
        
        # Bounds on control inputs
        bounds = []
        # No explicit bounds on x, v, a (first 3*N*M variables)
        for _ in range(3*N*M):
            bounds.append((-np.inf, np.inf))
        # Bounds on u
        for _ in range(2*N):
            bounds.append((-self.u_max, self.u_max))
        
        # Solve using SLSQP (Sequential Least Squares Programming)
        from scipy.optimize import NonlinearConstraint
        
        constraint = NonlinearConstraint(
            lambda z: self.mpc_constraints(z, x0, v0),
            0, 0  # Equality constraints
        )
        
        result = minimize(
            lambda z: self.mpc_objective(z, x0, v0),
            x_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint,
            options={'maxiter': 1000, 'disp': verbose, 'ftol': 1e-6}
        )
        
        if not result.success and verbose:
            print(f"Optimization warning: {result.message}")
        
        # Extract solution
        x_opt = result.x[:N*M].reshape(N, M)
        v_opt = result.x[N*M:2*N*M].reshape(N, M)
        u_opt = result.x[3*N*M:].reshape(N, 2)
        
        return x_opt, v_opt, u_opt, result.success
    
    def step(self, x_current, v_current, noise_magnitude=0.0):
        """
        Execute one MPC control step
        
        Parameters:
        -----------
        x_current : array_like
            Current positions
        v_current : array_like
            Current velocities
        noise_magnitude : float, optional
            Magnitude of noise to add to control signal
            
        Returns:
        --------
        x_next : array_like
            Next positions
        v_next : array_like
            Next velocities
        u_applied : array_like
            Applied control inputs (with noise)
        """
        # Solve MPC
        x_pred, v_pred, u_pred, success = self.solve_mpc(x_current, v_current)
        
        # Take first control action (with noise)
        u_applied = u_pred[0] + noise_magnitude * (np.random.rand(2) - 0.5) * 2
        u_applied = np.clip(u_applied, -self.u_max, self.u_max)
        
        # Simulate system with applied control
        def system_with_control(t, y):
            dydt = self.system_dynamics(t, y)
            # Apply control forces
            M = self.M
            dydt[M] += u_applied[0] / self.m  # Force on first mass
            dydt[2*M-1] += u_applied[1] / self.m  # Force on last mass
            return dydt
        
        # Integrate for one time step
        y0 = np.concatenate([x_current, v_current])
        sol = solve_ivp(system_with_control, [0, self.dt], y0, 
                       method='RK45', dense_output=True)
        
        y_next = sol.y[:, -1]
        x_next = y_next[:self.M]
        v_next = y_next[self.M:]
        
        # Store history
        self.state_history.append(np.concatenate([x_next, v_next]))
        self.control_history.append(u_applied)
        self.time_history.append(self.current_time)
        self.current_time += self.dt
        
        return x_next, v_next, u_applied, x_pred
    
    def simulate_uncontrolled(self, x0, v0, T):
        """
        Simulate uncontrolled system (u=0)
        
        Parameters:
        -----------
        x0 : array_like
            Initial positions
        v0 : array_like
            Initial velocities
        T : float
            Total simulation time
            
        Returns:
        --------
        t : array_like
            Time points
        x : array_like
            Position trajectories
        """
        y0 = np.concatenate([x0, v0])
        
        sol = solve_ivp(self.system_dynamics, [0, T], y0, 
                       method='RK45', dense_output=True,
                       t_eval=np.arange(0, T, self.dt))
        
        return sol.t, sol.y[:self.M].T

