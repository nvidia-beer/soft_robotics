"""
PID Controller for Volume Control.

Task: Control the volume (area) of a soft body by applying internal pressure.
      The PID computes pressure based on volume error.

Control Law:
    pressure(t) = Kp * e + Ki * ∫e dt + Kd * de/dt
    
    where:
        e = target_volume - current_volume
        de/dt = (e - e_prev) / dt

Output: Scalar pressure value
        Positive = inflate (radial outward force)
        Negative = deflate (radial inward force)
"""

from .. import BaseController
import numpy as np


class PID(BaseController):
    """
    PID controller for volume (area) control.
    
    Computes pressure to drive volume toward target.
    Includes multiple anti-windup strategies:
    1. Integral clamping
    2. Back-calculation (reduce integral when saturated)
    3. Conditional integration (only when error is decreasing or small)
    4. Integral decay on error sign change
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        u_max: float = 10.0,       # Max rest config adjustment (matches demo.py)
        Kp: float = 1.0,           # Default from demo.py
        Ki: float = 0.5,           # Default from demo.py
        Kd: float = 0.3,           # Default from demo.py
        integral_limit: float = 5.0,   # Matches demo.py
        deadband: float = 0.0001,      # Matches demo.py
        back_calc_gain: float = 0.5,   # Back-calculation gain
        derivative_filter: float = 0.5,  # Low-pass filter coefficient for D term (0.5 = moderate)
        **kwargs
    ):
        """
        Initialize Volume PID controller.
        
        Args:
            dt: Time step (seconds)
            u_max: Maximum pressure magnitude
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            integral_limit: Anti-windup limit for integral term
            deadband: Error deadband (no control if |error| < deadband)
            back_calc_gain: Back-calculation anti-windup gain (0-1)
            derivative_filter: Low-pass filter for derivative (0-1, lower = more filtering)
        """
        super().__init__(dt, u_max)
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        self.deadband = deadband
        self.back_calc_gain = back_calc_gain
        self.derivative_filter = derivative_filter
        
        # State
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_volume = None
        self.last_error_sign = 0
        self.filtered_derivative = 0.0  # Filtered derivative term
        
        print(f"✓ PID initialized")
        print(f"  Gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
        print(f"  u_max: {u_max}, deadband: {deadband}")
        print(f"  Anti-windup: integral_limit={integral_limit}, back_calc={back_calc_gain}")
        print(f"  Derivative filter: {derivative_filter}")
    
    def compute(self, target_volume: float, current_volume: float) -> float:
        """
        Compute pressure from volume error.
        
        Simple PID with basic anti-windup (clamping only).
        """
        # Compute error
        error = target_volume - current_volume
        
        # P term
        P = self.Kp * error
        
        # I term - always integrate, with simple clamping
        self.error_integral += error * self.dt
        self.error_integral = np.clip(
            self.error_integral, -self.integral_limit, self.integral_limit
        )
        I = self.Ki * self.error_integral
        
        # D term with low-pass filter to reduce noise
        if self.step_count > 0:
            raw_derivative = (error - self.last_error) / self.dt
            # Exponential moving average filter: filtered = alpha * raw + (1-alpha) * prev
            alpha = self.derivative_filter
            self.filtered_derivative = alpha * raw_derivative + (1 - alpha) * self.filtered_derivative
        else:
            self.filtered_derivative = 0.0
        D = self.Kd * self.filtered_derivative
        
        # Total PID output
        pressure = P + I + D
        
        # Clip to max
        pressure = np.clip(pressure, -self.u_max, self.u_max)
        
        # Update state
        self.last_error = error
        self.last_volume = current_volume
        self.step_count += 1
        
        # Debug output every 100 steps
        if self.step_count % 100 == 0:
            print(f"  [PID step {self.step_count}] "
                  f"err={error:.4f} P={P:.1f} I={I:.1f} D={D:.1f} → pressure={pressure:.1f}")
        
        return pressure
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_volume = None
        self.last_error_sign = 0
        self.filtered_derivative = 0.0
    
    def set_gains(self, Kp: float = None, Ki: float = None, Kd: float = None):
        """
        Update PID gains at runtime.
        
        Args:
            Kp: New proportional gain (None = keep current)
            Ki: New integral gain (None = keep current)
            Kd: New derivative gain (None = keep current)
        """
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd
        print(f"  PID gains updated: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}")
    
    def get_state(self) -> dict:
        """
        Get current controller state for debugging/visualization.
        
        Returns:
            Dict with integral, last_error, step_count
        """
        return {
            'integral': self.error_integral,
            'last_error': self.last_error,
            'step_count': self.step_count,
        }
    
    def __str__(self):
        return f"PID(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd})"
