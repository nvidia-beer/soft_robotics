"""
NEF-Based Spiking Neural Network PD Controller (PURE - No Learning)

Based on: "Neuromorphic NEF-Based Inverse Kinematics and PID Control"
Zaidel et al., Frontiers in Neurorobotics, 2021
https://doi.org/10.3389/fnbot.2021.631159

ARCHITECTURE (4 ensembles, all with UNIFORM radius=1.0):

┌─────────────────────────────────────────────────────────┐
│                                                         │
│  in ──[÷e]──> q(t) ───┬────[Kp]────> u(t) ──[×u]──> out │
│               r=1     │               r=1               │
│                       │                ↑                │
│                       ├──> ei(t) ─[Ki]─┤                │
│                       │    r=1    ↺    │                │
│                       │                │                │
│                       └──> ed(t) ─[Kd]─┘                │
│                            r=1                          │
│                                                         │
│  Legend: [÷e]=÷error_scale  [×u]=×output_scale          │
│          All r=1 (uniform radius, signals in [-1,1])    │
│                                                         │
└─────────────────────────────────────────────────────────┘

NEF BEST PRACTICES (following Zaidel et al. 2021 & RSS 2020):
  1. ALL ensembles have SAME radius (~1.0) - uniform precision
  2. Normalize at INPUT boundary (divide by error_scale)
  3. Denormalize at OUTPUT boundary (multiply by output_scale)
  4. Gains in CONNECTION TRANSFORMS, not ensemble radii
  5. All internal signals stay in [-1, 1] range

4 ENSEMBLES (all radius=1.0):
  1. q(t)  - Normalized error [-1, 1]
  2. ei(t) - Normalized integral [-1, 1]  
  3. ed(t) - Normalized derivative [-1, 1]
  4. u(t)  - Normalized output [-1, 1]

NOTE: This is a PURE implementation with no learning/PES.
      For spiking stress-adaptive control with strain feedback, see stress.py
"""

import numpy as np
import nengo


def build_snn_pd_network(
    model,
    error_input_node,
    output_node,
    group_id: int = 0,
    prefix: str = "G0",
    Kp: float = 250.0,      # Tuned for NEF (compensates synaptic smoothing)
    Ki: float = 0.0,
    Kd: float = 80.0,       # Higher for better damping with neural differentiation
    error_scale: float = 2.0,
    output_scale: float = 500.0,
    n_neurons: int = 100,
    tau_syn: float = 0.01,
    tau_int: float = 0.1,
    tau_fast: float = 0.002,   # Faster for better derivative response (was 0.005)
    tau_slow: float = 0.02,    # Closer to tau_fast for better high-freq response (was 0.2)
    radius: float = 1.0,
):
    """
    Build a NEF-based PD controller within an existing Nengo model.
    
    Implements the architecture from Zaidel et al. 2021:
    - q(t): Error input ensemble
    - ei(t): Integrator ensemble (with recurrent connection)
    - ed(t): Differentiator ensemble (fast/slow pathway)
    - u(t): Output ensemble (sums P + I + D)
    
    All computation is FULLY NEURAL (spiking) - true to paper.
    
    Args:
        model: Parent nengo.Network to build within
        error_input_node: Node providing 2D error signal (x, y)
        output_node: Node to receive 2D control output (receives scaled forces)
        group_id: Group identifier for labeling
        prefix: Label prefix for ensembles
        Kp: Proportional gain (N/m)
        Ki: Integral gain (N/(m·s)) - set to 0 for pure PD
        Kd: Derivative gain (N·s/m)
        error_scale: Max expected error in meters (for normalization)
        output_scale: Max output force in Newtons (u_max)
        n_neurons: Neurons per dimension in ensembles
        tau_syn: Standard synaptic time constant (10ms)
        tau_int: Integrator time constant (100ms)
        tau_fast: Fast synapse for derivative (2ms - tuned for responsiveness)
        tau_slow: Slow synapse for derivative (20ms - tuned for responsiveness)
        radius: UNIFORM radius for all ensembles (textbook NEF: 1.0)
    
    Returns:
        dict with references to created ensembles
    """
    
    # Normalized gains (textbook NEF: all internal signals in [-1, 1])
    Kp_norm = Kp * error_scale / output_scale
    Ki_norm = Ki * error_scale / output_scale
    Kd_norm = Kd * error_scale / output_scale
    
    # Differentiator scale and normalization
    diff_scale = 1.0 / (tau_slow - tau_fast)  # ~5.13 for default tau values
    diff_norm = diff_scale * 0.1  # Scale down to keep derivative in [-1, 1]
    
    components = {}
    
    with model:
        # =====================================================================
        # ENSEMBLE 1: q(t) - Normalized error
        # Input is ALREADY normalized to [-1, 1] by error_input_node
        # TEXTBOOK NEF: radius=1.0
        # =====================================================================
        q_ens = nengo.Ensemble(
            n_neurons=n_neurons * 2,
            dimensions=2,
            radius=radius,  # UNIFORM radius
            label=f"{prefix}_q(t)"
        )
        nengo.Connection(error_input_node, q_ens, synapse=None)
        components['q_ens'] = q_ens
        
        # =====================================================================
        # ENSEMBLE 2: eᵢ(t) - Integral term (NEF integrator)
        # 
        # NEF INTEGRATOR: recurrent(1.0, τ) + input(τ, τ)
        # With leaky integrator to prevent windup and stay in [-1, 1]
        # TEXTBOOK NEF: radius=1.0
        # =====================================================================
        ei_ens = nengo.Ensemble(
            n_neurons=n_neurons * 2,
            dimensions=2,
            radius=radius,  # UNIFORM radius
            label=f"{prefix}_ei(t)"
        )
        # Leaky integrator: slight decay keeps values bounded
        # recurrent(0.95, τ) instead of (1.0, τ) for anti-windup
        nengo.Connection(ei_ens, ei_ens, transform=0.95, synapse=tau_int)
        nengo.Connection(q_ens, ei_ens, transform=tau_int * 0.5, synapse=tau_int)
        components['ei_ens'] = ei_ens
        
        # =====================================================================
        # ENSEMBLE 3: eᵈ(t) - Derivative term (NEF differentiator)
        # 
        # From Zaidel et al. 2021: de/dt ≈ (e(t) - e(t-Δt)) / Δt
        # Implemented via fast/slow synapse difference (high-pass filter)
        # FULLY NEURAL - all computation in spiking neurons
        # =====================================================================
        ed_ens = nengo.Ensemble(
            n_neurons=n_neurons * 4,  # More neurons for smoother derivative
            dimensions=2,
            radius=radius,  # UNIFORM radius
            label=f"{prefix}_ed(t)"
        )
        # NEF differentiator: fast - slow ≈ derivative
        # diff_norm keeps output in [-1, 1] range
        nengo.Connection(q_ens, ed_ens, transform=diff_norm, synapse=tau_fast)
        nengo.Connection(q_ens, ed_ens, transform=-diff_norm, synapse=tau_slow)
        components['ed_ens'] = ed_ens
        
        # =====================================================================
        # ENSEMBLE 4: u(t) - Output ensemble (sums P + I + D)
        # All terms already normalized, sum stays in ~[-1, 1]
        # TEXTBOOK NEF: radius=1.0
        # =====================================================================
        u_ens = nengo.Ensemble(
            n_neurons=n_neurons * 2,
            dimensions=2,
            radius=radius,  # UNIFORM radius
            label=f"{prefix}_u(t)"
        )
        # P-term: Kp_norm keeps contribution bounded
        nengo.Connection(q_ens, u_ens, transform=Kp_norm, synapse=tau_syn)
        # I-term: Ki_norm × normalized integral
        nengo.Connection(ei_ens, u_ens, transform=Ki_norm, synapse=tau_syn)
        # D-term: Kd_norm × normalized derivative
        # Compensate for diff_norm scaling in NEF differentiator
        Kd_effective = Kd_norm / diff_norm * diff_scale if diff_norm > 0 else 0
        nengo.Connection(ed_ens, u_ens, transform=Kd_effective, synapse=tau_syn)
        components['u_ens'] = u_ens
        
        # =====================================================================
        # OUTPUT: Scale from normalized [-1, 1] to real units
        # This is the ONLY place scaling happens!
        # =====================================================================
        def make_output_scale(scale):
            def scale_output(x):
                return np.clip(x * scale, -scale, scale)
            return scale_output
        
        nengo.Connection(u_ens, output_node, 
                        function=make_output_scale(output_scale),
                        synapse=tau_syn)
    
    return components


class SNN_PID_Controller:
    """
    NEF-based Spiking PD Controller for Nengo GUI integration.
    
    PURE implementation from Zaidel et al. 2021 - NO learning/PES.
    
    This class manages the creation and configuration of NEF-based PD
    controllers for trajectory tracking with multiple control groups.
    
    Designed to work within an existing Nengo model (for Nengo GUI).
    For standalone use (without GUI), see NengoPID.
    For spiking stress-adaptive control with strain feedback, see stress.py
    """
    
    def __init__(
        self,
        num_groups: int,
        Kp: float = 250.0,      # Tuned for NEF (compensates synaptic smoothing)
        Ki: float = 0.0,
        Kd: float = 80.0,       # Higher for better damping with neural differentiation
        u_max: float = 500.0,
        n_neurons: int = 100,
        error_scale: float = 2.0,
    ):
        """
        Initialize SNN PD controller configuration (Zaidel et al. 2021).
        
        Args:
            num_groups: Number of control groups
            Kp: Proportional gain (tuned for NEF)
            Ki: Integral gain (0 for pure PD)
            Kd: Derivative gain (tuned for neural differentiator)
            u_max: Maximum control force
            n_neurons: Neurons per dimension
            error_scale: Max expected error (meters)
        """
        self.num_groups = num_groups
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.u_max = u_max
        self.n_neurons = n_neurons
        self.error_scale = error_scale
        
        # Time constants (tuned for better derivative response)
        self.tau_syn = 0.01
        self.tau_int = 0.1
        self.tau_fast = 0.002   # Faster for better derivative (was 0.005)
        self.tau_slow = 0.02    # Closer to tau_fast (was 0.2)
        
        # TEXTBOOK NEF: uniform radius for all ensembles
        self.radius = 1.0
        
        # Computed values
        self.diff_scale = 1.0 / (self.tau_slow - self.tau_fast)
        self.Kp_norm = Kp * error_scale / u_max
        self.Ki_norm = Ki * error_scale / u_max
        self.Kd_norm = Kd * error_scale / u_max
        
        # Compute grid layout
        self.num_groups_per_side = int(np.sqrt(num_groups + 0.5))
        
        # Storage for built components
        self.group_components = {}
        self.all_ensembles = []
        self.error_input_nodes = []
        self.output_nodes = []
        
        # Output storage (updated by Nengo nodes)
        self._output = np.zeros(num_groups * 2, dtype=np.float32)
        
        print(f"  SNN_PID_Controller (TEXTBOOK NEF - uniform radius):")
        print(f"    Groups: {num_groups}")
        print(f"    Gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
        print(f"    Normalized: Kp_norm={self.Kp_norm:.2f}, Kd_norm={self.Kd_norm:.2f}")
        print(f"    Radius: {self.radius} (uniform for all ensembles)")
        print(f"    Neurons per group: ~{n_neurons * 10}")
    
    def build_all(
        self,
        model,
        get_error_callback,
    ):
        """
        Build all SNN PD networks for all groups within a Nengo model.
        
        Implements Zaidel et al. 2021 architecture - FULLY NEURAL.
        
        Args:
            model: Parent nengo.Network to build within
            get_error_callback: Function(group_id) -> callable that returns 2D error
                Example: get_error_callback(0) returns a function f(t) -> [err_x, err_y]
        
        Returns:
            dict with:
                - 'ensembles': list of all ensembles (for visualization)
                - 'output': reference to output array
                - 'components': dict of group components
        """
        print(f"  Building {self.num_groups} SNN PD controllers (Zaidel et al. 2021)...")
        
        with model:
            for gid in range(self.num_groups):
                group_row = gid // self.num_groups_per_side
                group_col = gid % self.num_groups_per_side
                prefix = f"G{gid}[{group_row},{group_col}]"
                
                # Create error input node
                error_fn = get_error_callback(gid)
                error_input = nengo.Node(
                    error_fn,
                    size_out=2,
                    label=f"{prefix}_ErrorIn"
                )
                self.error_input_nodes.append(error_input)
                
                # Create output store node
                def make_output_store(group_id, scale, output_array):
                    def store(t, x):
                        idx = group_id * 2
                        output_array[idx:idx+2] = np.clip(x, -scale, scale)
                    return store
                
                output_store = nengo.Node(
                    make_output_store(gid, self.u_max, self._output),
                    size_in=2,
                    label=f"{prefix}_Out"
                )
                self.output_nodes.append(output_store)
                
                # Build the PD network
                components = build_snn_pd_network(
                    model=model,
                    error_input_node=error_input,
                    output_node=output_store,
                    group_id=gid,
                    prefix=prefix,
                    Kp=self.Kp,
                    Ki=self.Ki,
                    Kd=self.Kd,
                    error_scale=self.error_scale,
                    output_scale=self.u_max,
                    n_neurons=self.n_neurons,
                    tau_syn=self.tau_syn,
                    tau_int=self.tau_int,
                    tau_fast=self.tau_fast,
                    tau_slow=self.tau_slow,
                    radius=self.radius,  # UNIFORM radius for all ensembles
                )
                
                self.group_components[gid] = components
                
                # Collect ensembles for visualization
                for key in ['q_ens', 'ei_ens', 'ed_ens', 'u_ens']:
                    if key in components:
                        self.all_ensembles.append(components[key])
                
                print(f"    {prefix}: 4 ensembles (NEF differentiator)")
        
        print(f"  ✓ {self.num_groups} SNN PDs created")
        print(f"    Total ensembles: {len(self.all_ensembles)}")
        
        return {
            'ensembles': self.all_ensembles,
            'output': self._output,
            'components': self.group_components,
        }
    
    def get_output(self):
        """Get the current control output array."""
        return self._output.copy()
    
    def get_all_ensembles(self):
        """Return list of all ensembles for visualization."""
        return self.all_ensembles
    
    def get_group_ensembles(self, group_id: int):
        """Return ensembles for a specific group."""
        return self.group_components.get(group_id, {})
    
    def __str__(self):
        return f"SNN_PID_Controller(groups={self.num_groups}, Kp={self.Kp}, Kd={self.Kd})"


class NengoPID:
    """
    NEF-based Spiking PID Controller for standalone use.
    
    PURE implementation from Zaidel et al. 2021 - NO learning/PES.
    
    This creates its own Nengo model and simulator internally,
    providing the same interface as other controllers (PID, MPC).
    
    Reuses SNN_PID_Controller internally to avoid code duplication.
    
    For spiking stress-adaptive control with strain feedback, see stress.py
    
    Based on: Zaidel et al., Frontiers in Neurorobotics, 2021
    """
    
    def __init__(
        self,
        num_groups: int,
        dt: float = 0.01,
        u_max: float = 500.0,
        Kp: float = 250.0,      # Tuned for NEF
        Ki: float = 0.0,
        Kd: float = 80.0,       # Higher for neural differentiator
        n_neurons: int = 100,
        error_scale: float = 2.0,
        device: str = 'cpu',
    ):
        """
        Initialize standalone NEF PD controller (Zaidel et al. 2021).
        
        FULLY NEURAL - all computation in spiking neurons.
        
        Args:
            num_groups: Number of control groups (each has x,y)
            dt: Simulation time step
            u_max: Maximum control force
            Kp: Proportional gain (tuned for NEF)
            Ki: Integral gain (0 for pure PD)
            Kd: Derivative gain (tuned for neural differentiator)
            n_neurons: Neurons per dimension
            error_scale: Max expected error
            device: 'cpu' or 'cuda' (note: cuda not efficient for step-by-step)
        """
        self.num_groups = num_groups
        self.dt = dt
        self.u_max = u_max
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.n_neurons = n_neurons
        self.error_scale = error_scale
        self.device = device
        
        # Control dimensions (2 per group: x, y)
        self.dim = num_groups * 2
        
        # Internal state
        self._current_error = np.zeros(self.dim)
        self._sim_time = 0.0
        
        # GPU is NOT recommended for real-time step-by-step control
        if device == 'cuda':
            print(f"  ⚠ GPU (nengo-dl) not used - slow for real-time control")
            print(f"    nengo-dl is for batch training, not step-by-step simulation")
            print(f"    Using CPU nengo instead (faster for real-time)")
        
        # Build Nengo model using SNN_PID_Controller (no duplication!)
        self._build_model()
        
        print(f"  ✓ NengoPID initialized (PURE - no learning)")
        print(f"    Groups: {num_groups}, Dim: {self.dim}")
        print(f"    Gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
        print(f"    Neurons: ~{n_neurons * 10 * num_groups}")
    
    def _build_model(self):
        """Build the internal Nengo model using SNN_PID_Controller."""
        
        # Create inner SNN_PID_Controller (reuses build_snn_pd_network)
        self._snn_controller = SNN_PID_Controller(
            num_groups=self.num_groups,
            Kp=self.Kp,
            Ki=self.Ki,
            Kd=self.Kd,
            u_max=self.u_max,
            n_neurons=self.n_neurons,
            error_scale=self.error_scale,
        )
        
        # Create error callback factory
        def make_error_callback(group_id):
            def get_error(t):
                idx = group_id * 2
                err = self._current_error[idx:idx+2]
                return err / self.error_scale
            return get_error
        
        # Build Nengo model
        self.model = nengo.Network(label="NengoPID")
        
        with self.model:
            # Build all SNN PD networks (PURE - no PES)
            self._snn_controller.build_all(
                model=self.model,
                get_error_callback=make_error_callback,
            )
        
        # Create simulator
        self._create_simulator()
    
    def _create_simulator(self):
        """Create the Nengo simulator (CPU - best for real-time control)."""
        self.sim = nengo.Simulator(self.model, dt=self.dt)
        print(f"    Nengo simulator created (dt={self.dt}s)")
    
    def get_output(self):
        """Get current control output."""
        return self._snn_controller.get_output()
    
    def compute_control(self, state_dict, get_target_fn=None):
        """
        Compute control forces using NEF-based PD.
        
        PURE implementation - no learning or stress modulation.
        For strain feedback, see stress.py
        
        Compatible interface with PID/MPC controllers.
            
        Args:
            state_dict: Dict with 'group_centroids', 'group_targets', etc.
            get_target_fn: Optional function to get target (unused, targets in state_dict)
        
        Returns:
            np.ndarray: Control forces [num_groups * 2]
        """
        # Get error from state
        centroids = state_dict.get('group_centroids', np.zeros((self.num_groups, 2)))
        targets = state_dict.get('group_targets', np.zeros((self.num_groups, 2)))
        
        error = (targets - centroids).flatten()
        self._current_error = error.astype(np.float32)
        
        # Step simulation multiple times to match physics dt
        # GUI: Nengo runs at dt=0.001, physics at 0.01 → 10 Nengo steps per physics step
        # This ensures proper neural dynamics timing
        physics_dt = state_dict.get('dt', 0.01)  # Get from state or default
        n_steps = max(1, int(round(physics_dt / self.dt)))
        for _ in range(n_steps):
            self.sim.step()
        self._sim_time += physics_dt
        
        # Get output from SNN controller
        return self.get_output()
    
    def reset(self):
        """Reset the controller state."""
        self._current_error = np.zeros(self.dim)
        self._sim_time = 0.0
        
        # Reset SNN controller output
        self._snn_controller._output[:] = 0.0
        
        # Reset simulator
        self.sim.close()
        self._create_simulator()
    
    def close(self):
        """Close the simulator."""
        if hasattr(self, 'sim'):
                self.sim.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    def __str__(self):
        return f"NengoPID(groups={self.num_groups}, Kp={self.Kp}, Kd={self.Kd})"
