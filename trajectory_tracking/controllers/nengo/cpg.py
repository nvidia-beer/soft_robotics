"""
NEF-Based Spiking Central Pattern Generator (Hopf Oscillator)

Implements Hopf oscillator dynamics as a spiking neural network using NEF.
Provides rhythmic patterns for locomotion control.

ARCHITECTURE (per oscillator):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌──────────┐                                                       │
│  │  osc[i]  │◄───── recurrent (Hopf dynamics)                       │
│  │  (x, y)  │                                                       │
│  │  2D ens  │───┬──► x = output (force)                             │
│  └──────────┘   │                                                   │
│       ▲         │                                                   │
│       │         └──► coupling to neighbors                          │
│       │                                                             │
│  neighbors ────────► coupling terms                                 │
│                                                                     │
│  HOPF DYNAMICS (Cartesian form):                                    │
│    ẋ = a(μ - r²)x - ωy    where r² = x² + y²                       │
│    ẏ = a(μ - r²)y + ωx                                              │
│                                                                     │
│  INTER-OSCILLATOR COUPLING:                                         │
│    Δx = Σⱼ wᵢⱼ(xⱼcos(φᵢⱼ) - yⱼsin(φᵢⱼ) - x)                        │
│    Δy = Σⱼ wᵢⱼ(xⱼsin(φᵢⱼ) + yⱼcos(φᵢⱼ) - y)                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Features:
- Fully spiking implementation (no Python nodes in dynamics)
- Grid-based coupling for traveling wave patterns
- RL-friendly parameter interface (amplitude, frequency, direction)
- Optional strain feedback modulation

Based on:
- Ijspeert, "Central pattern generators for locomotion control", 2008
- Hopf bifurcation oscillator theory
"""

import numpy as np
import nengo


def hopf_dynamics(x, a=5.0, mu=1.0, omega=2*np.pi*2.0):
    """
    Hopf oscillator dynamics in Cartesian coordinates.
    
    Args:
        x: State [x, y] where x = r*cos(θ), y = r*sin(θ)
        a: Convergence rate to limit cycle
        mu: Target amplitude squared (steady-state r* = √μ)
        omega: Angular frequency (rad/s)
    
    Returns:
        [ẋ, ẏ] time derivative
    
    The Hopf oscillator:
        ṙ = a(μ - r²)r     → amplitude converges to √μ
        θ̇ = ω              → constant rotation
    
    In Cartesian form:
        ẋ = a(μ - r²)x - ωy
        ẏ = a(μ - r²)y + ωx
    """
    r_sq = x[0]**2 + x[1]**2
    dx = a * (mu - r_sq) * x[0] - omega * x[1]
    dy = a * (mu - r_sq) * x[1] + omega * x[0]
    return [dx, dy]


def build_snn_cpg_network(
    model,
    output_node,
    oscillator_id: int = 0,
    prefix: str = "CPG0",
    frequency: float = 2.0,
    amplitude: float = 0.7,
    direction: float = 1.0,
    hopf_a: float = 5.0,
    n_neurons: int = 200,
    tau_syn: float = 0.01,
    radius: float = 1.5,
    initial_phase: float = 0.0,
):
    """
    Build a single Hopf oscillator as a spiking neural network.
    
    Uses NEF to implement the Hopf dynamics with recurrent connections.
    
    Args:
        model: Parent nengo.Network
        output_node: Node to receive 1D output (force)
        oscillator_id: Oscillator identifier
        prefix: Label prefix for ensembles
        frequency: Oscillation frequency (Hz)
        amplitude: Output amplitude scaling [0, 1]
        direction: +1 forward, -1 backward (reverses phase)
        hopf_a: Convergence rate to limit cycle
        n_neurons: Neurons in oscillator ensemble
        tau_syn: Synaptic time constant
        radius: Ensemble radius (should be > √mu for stability)
        initial_phase: Initial phase offset (radians)
    
    Returns:
        dict with ensemble references and parameter functions
    """
    # Hopf parameters
    mu = 1.0  # Target amplitude² (r* = 1.0)
    omega = direction * 2.0 * np.pi * frequency
    
    # Initial state on limit cycle at given phase
    x0 = np.sqrt(mu) * np.cos(initial_phase)
    y0 = np.sqrt(mu) * np.sin(initial_phase)
    
    components = {}
    
    with model:
        # =====================================================================
        # OSCILLATOR ENSEMBLE: 2D state (x, y)
        # Uses more neurons for accurate function approximation
        # =====================================================================
        osc_ens = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=2,
            radius=radius,
            label=f"{prefix}_osc"
        )
        components['osc_ens'] = osc_ens
        
        # =====================================================================
        # HOPF DYNAMICS: Recurrent connection
        # 
        # NEF recurrent dynamics: x(t+dt) = x(t) + dt * f(x(t)) / tau
        # For continuous dynamics, we need: transform = tau, function = f
        # 
        # The synapse acts as an integrator: H(s) = 1/(τs + 1)
        # Recurrent connection: x' = f(x) implemented as:
        #   connection with synapse τ and function f gives ẋ ≈ f(x)
        # =====================================================================
        def make_hopf_function(a, mu, omega):
            """Create Hopf dynamics function for NEF."""
            def hopf_fn(x):
                r_sq = x[0]**2 + x[1]**2
                dx = a * (mu - r_sq) * x[0] - omega * x[1]
                dy = a * (mu - r_sq) * x[1] + omega * x[0]
                return [tau_syn * dx, tau_syn * dy]
            return hopf_fn
        
        # Recurrent connection implementing Hopf dynamics
        nengo.Connection(
            osc_ens, osc_ens,
            function=make_hopf_function(hopf_a, mu, omega),
            synapse=tau_syn,
        )
        components['hopf_a'] = hopf_a
        components['mu'] = mu
        components['omega'] = omega
        
        # =====================================================================
        # OUTPUT: Extract x component (first dimension), scale by amplitude
        # =====================================================================
        def make_output_fn(amp):
            def output_fn(x):
                return np.clip(x[0] * amp, -1.0, 1.0)
            return output_fn
        
        nengo.Connection(
            osc_ens, output_node,
            function=make_output_fn(amplitude),
            synapse=tau_syn,
        )
        components['amplitude'] = amplitude
        
        # =====================================================================
        # INITIALIZATION: Set initial state on limit cycle
        # Uses a brief input pulse to kick-start the oscillator
        # =====================================================================
        def init_pulse(t):
            # Brief pulse at start to initialize on limit cycle
            if t < 0.1:
                return [x0 * 10, y0 * 10]  # Strong initial push
            return [0, 0]
        
        init_node = nengo.Node(init_pulse, size_out=2, label=f"{prefix}_init")
        nengo.Connection(init_node, osc_ens, synapse=0.01)
        components['init_node'] = init_node
    
    return components


class SNN_CPG_Controller:
    """
    NEF-based Spiking CPG Controller for Nengo GUI integration.
    
    Creates a network of coupled Hopf oscillators implemented as spiking
    neural networks. Designed for locomotion control with traveling wave
    patterns.
    
    Features:
    - Grid of coupled oscillators (one per 2x2 group)
    - Traveling wave patterns for locomotion
    - RL-friendly parameter interface
    - Optional strain feedback modulation
    """
    
    def __init__(
        self,
        num_groups: int,
        frequency: float = 2.0,
        amplitude: float = 0.7,
        direction: float = 1.0,
        coupling_strength: float = 2.0,
        n_neurons: int = 200,
        u_max: float = 1.0,
    ):
        """
        Initialize SNN CPG controller.
        
        Args:
            num_groups: Number of oscillators (one per 2x2 group)
            frequency: Base oscillation frequency (Hz)
            amplitude: Output amplitude [0, 1]
            direction: +1 = wave left (body right), -1 = wave right (body left)
            coupling_strength: Inter-oscillator coupling
            n_neurons: Neurons per oscillator
            u_max: Maximum output (for scaling)
        """
        self.num_groups = num_groups
        self.frequency = frequency
        self.amplitude = amplitude
        self.direction = direction
        self.coupling_strength = coupling_strength
        self.n_neurons = n_neurons
        self.u_max = u_max
        
        # Compute grid layout
        self.grid_side = int(np.sqrt(num_groups))
        
        # Hopf parameters
        self.hopf_a = 5.0  # Convergence rate
        self.mu = 1.0      # Target amplitude squared
        self.tau_syn = 0.01
        
        # Storage
        self.oscillator_ensembles = []
        self.group_components = {}
        self.output_nodes = []
        
        # Output storage
        self._output = np.zeros(num_groups, dtype=np.float32)
        
        # Build phase offsets for traveling wave
        self._build_phase_offsets()
        
        print(f"  SNN_CPG_Controller (Hopf oscillators):")
        print(f"    Groups: {num_groups} ({self.grid_side}x{self.grid_side})")
        print(f"    Frequency: {frequency} Hz")
        print(f"    Direction: {'+1 (right)' if direction > 0 else '-1 (left)'}")
        print(f"    Coupling: {coupling_strength}")
        print(f"    Neurons per oscillator: {n_neurons}")
    
    def _build_phase_offsets(self):
        """Build initial phase offsets for traveling wave pattern."""
        self.phase_offsets = np.zeros(self.num_groups)
        
        # Phase offset per column for horizontal wave
        # Negative offset so left columns lead (wave travels left, body moves right)
        for i in range(self.num_groups):
            col = i % self.grid_side
            # Left columns have larger phase (lead the wave)
            self.phase_offsets[i] = -2.0 * np.pi * col / self.grid_side
    
    def _build_neighbors(self):
        """Build 4-connected neighbor list for grid coupling."""
        neighbors = [[] for _ in range(self.num_groups)]
        side = self.grid_side
        
        for i in range(self.num_groups):
            row = i // side
            col = i % side
            
            if col > 0:
                neighbors[i].append(i - 1)      # left
            if col < side - 1:
                neighbors[i].append(i + 1)      # right
            if row > 0:
                neighbors[i].append(i - side)   # up
            if row < side - 1:
                neighbors[i].append(i + side)   # down
        
        return neighbors
    
    def build_all(self, model, strain_callback=None):
        """
        Build all CPG oscillators within a Nengo model.
        
        Args:
            model: Parent nengo.Network
            strain_callback: Optional function(group_id) -> callable returning strain
                           Strain modulates amplitude (high strain -> reduced output)
        
        Returns:
            dict with ensembles and output references
        """
        print(f"  Building {self.num_groups} Hopf oscillators...")
        
        omega = self.direction * 2.0 * np.pi * self.frequency
        neighbors = self._build_neighbors()
        
        with model:
            # Create all oscillator ensembles first
            for gid in range(self.num_groups):
                row = gid // self.grid_side
                col = gid % self.grid_side
                prefix = f"CPG{gid}[{row},{col}]"
                
                # Initial phase for traveling wave
                initial_phase = self.phase_offsets[gid]
                x0 = np.sqrt(self.mu) * np.cos(initial_phase)
                y0 = np.sqrt(self.mu) * np.sin(initial_phase)
                
                # Oscillator ensemble (2D: x, y)
                osc_ens = nengo.Ensemble(
                    n_neurons=self.n_neurons,
                    dimensions=2,
                    radius=1.5,
                    label=f"{prefix}_osc"
                )
                self.oscillator_ensembles.append(osc_ens)
                
                # Output store node
                def make_output_store(group_id, amp, output_array):
                    def store(t, x):
                        # x is the oscillator output (just the x component)
                        output_array[group_id] = np.clip(x * amp, -1.0, 1.0)
                    return store
                
                output_store = nengo.Node(
                    make_output_store(gid, self.amplitude, self._output),
                    size_in=1,
                    label=f"{prefix}_Out"
                )
                self.output_nodes.append(output_store)
                
                # Store components
                self.group_components[gid] = {
                    'osc_ens': osc_ens,
                    'output_node': output_store,
                    'initial_phase': initial_phase,
                }
                
                # Initialization pulse
                def make_init_pulse(x0_val, y0_val):
                    def init_fn(t):
                        if t < 0.1:
                            return [x0_val * 10, y0_val * 10]
                        return [0, 0]
                    return init_fn
                
                init_node = nengo.Node(
                    make_init_pulse(x0, y0), 
                    size_out=2, 
                    label=f"{prefix}_init"
                )
                nengo.Connection(init_node, osc_ens, synapse=0.01)
            
            # Add recurrent Hopf dynamics and output connections
            for gid in range(self.num_groups):
                osc_ens = self.oscillator_ensembles[gid]
                output_node = self.output_nodes[gid]
                
                # Hopf dynamics (recurrent)
                def make_hopf_fn(a, mu, w, tau):
                    def hopf_fn(x):
                        r_sq = x[0]**2 + x[1]**2
                        dx = a * (mu - r_sq) * x[0] - w * x[1]
                        dy = a * (mu - r_sq) * x[1] + w * x[0]
                        return [tau * dx, tau * dy]
                    return hopf_fn
                
                nengo.Connection(
                    osc_ens, osc_ens,
                    function=make_hopf_fn(self.hopf_a, self.mu, omega, self.tau_syn),
                    synapse=self.tau_syn,
                )
                
                # Output: extract x component
                def extract_x(state):
                    return state[0]
                
                nengo.Connection(
                    osc_ens, output_node,
                    function=extract_x,
                    synapse=self.tau_syn,
                )
            
            # Add inter-oscillator coupling
            if self.coupling_strength > 0:
                print(f"    Adding neighbor coupling (strength={self.coupling_strength})...")
                
                for i in range(self.num_groups):
                    for j in neighbors[i]:
                        # Phase difference between neighbors
                        phase_diff = self.phase_offsets[j] - self.phase_offsets[i]
                        
                        # Coupling function: rotate neighbor state by target phase diff
                        # Then pull toward that rotated state
                        def make_coupling_fn(phi, strength, tau):
                            cos_phi = np.cos(phi)
                            sin_phi = np.sin(phi)
                            def coupling_fn(x):
                                # Rotate x by phi (target relative phase)
                                x_rot = cos_phi * x[0] - sin_phi * x[1]
                                y_rot = sin_phi * x[0] + cos_phi * x[1]
                                # Return scaled coupling term
                                return [tau * strength * x_rot, tau * strength * y_rot]
                            return coupling_fn
                        
                        # Add coupling from j to i
                        nengo.Connection(
                            self.oscillator_ensembles[j],
                            self.oscillator_ensembles[i],
                            function=make_coupling_fn(phase_diff, self.coupling_strength, self.tau_syn),
                            synapse=self.tau_syn,
                        )
            
            # Optional strain feedback
            if strain_callback is not None:
                print(f"    Adding strain feedback modulation...")
                for gid in range(self.num_groups):
                    strain_fn = strain_callback(gid)
                    strain_node = nengo.Node(strain_fn, size_out=1, label=f"CPG{gid}_strain")
                    
                    # Strain inhibits oscillator (high strain -> reduced amplitude)
                    # Connect to both x and y components with negative weight
                    def strain_inhibit(s):
                        return [-0.3 * s[0], -0.3 * s[0]]
                    
                    nengo.Connection(
                        strain_node, 
                        self.oscillator_ensembles[gid],
                        function=strain_inhibit,
                        synapse=self.tau_syn,
                    )
                    
                    self.group_components[gid]['strain_node'] = strain_node
        
        print(f"  ✓ {self.num_groups} Hopf oscillators created")
        print(f"    Total oscillator ensembles: {len(self.oscillator_ensembles)}")
        
        return {
            'ensembles': self.oscillator_ensembles,
            'output': self._output,
            'components': self.group_components,
        }
    
    def get_output(self):
        """Get current CPG output array."""
        return self._output.copy()
    
    def get_all_ensembles(self):
        """Return all oscillator ensembles for visualization."""
        return self.oscillator_ensembles
    
    def __str__(self):
        return f"SNN_CPG_Controller(groups={self.num_groups}, freq={self.frequency}Hz)"


class NengoCPG:
    """
    NEF-based Spiking CPG for standalone use.
    
    Creates its own Nengo model and simulator internally.
    Provides the same interface as other controllers.
    
    The Hopf oscillator is implemented using NEF with:
    - 2D ensemble per oscillator (x, y Cartesian coordinates)
    - Recurrent connections for Hopf dynamics
    - Coupling connections for traveling wave coordination
    
    Example:
        >>> cpg = NengoCPG(num_groups=9, frequency=2.0, direction=1.0)
        >>> for _ in range(1000):
        >>>     output = cpg.step()  # Get 9D output in [-1, 1]
    """
    
    def __init__(
        self,
        num_groups: int,
        dt: float = 0.001,
        frequency: float = 2.0,
        amplitude: float = 0.7,
        direction: float = 1.0,
        coupling_strength: float = 2.0,
        n_neurons: int = 200,
        device: str = 'cpu',
    ):
        """
        Initialize standalone spiking CPG.
        
        Args:
            num_groups: Number of oscillators
            dt: Simulation time step
            frequency: Oscillation frequency (Hz)
            amplitude: Output amplitude [0, 1]
            direction: +1 = body moves right, -1 = body moves left
            coupling_strength: Inter-oscillator coupling
            n_neurons: Neurons per oscillator
            device: 'cpu' (GPU not recommended for step-by-step)
        """
        self.num_groups = num_groups
        self.dt = dt
        self.frequency = frequency
        self.amplitude = amplitude
        self.direction = direction
        self.coupling_strength = coupling_strength
        self.n_neurons = n_neurons
        self.device = device
        
        self._sim_time = 0.0
        
        # Create inner SNN CPG controller
        self._snn_cpg = SNN_CPG_Controller(
            num_groups=num_groups,
            frequency=frequency,
            amplitude=amplitude,
            direction=direction,
            coupling_strength=coupling_strength,
            n_neurons=n_neurons,
        )
        
        # Build model
        self._build_model()
        
        print(f"  ✓ NengoCPG initialized (Hopf oscillator network)")
        print(f"    Groups: {num_groups}")
        print(f"    Frequency: {frequency} Hz")
        print(f"    Total neurons: ~{n_neurons * num_groups}")
    
    def _build_model(self):
        """Build the internal Nengo model."""
        self.model = nengo.Network(label="NengoCPG")
        
        with self.model:
            self._snn_cpg.build_all(self.model)
        
        # Create simulator
        self.sim = nengo.Simulator(self.model, dt=self.dt)
        print(f"    Nengo simulator created (dt={self.dt}s)")
    
    def step(self, n_steps: int = 1) -> np.ndarray:
        """
        Step the CPG simulation.
        
        Args:
            n_steps: Number of simulation steps
        
        Returns:
            np.ndarray: CPG output [num_groups] in [-1, 1]
        """
        for _ in range(n_steps):
            self.sim.step()
        self._sim_time += n_steps * self.dt
        
        return self.get_output()
    
    def get_output(self) -> np.ndarray:
        """Get current CPG output."""
        return self._snn_cpg.get_output()
    
    def __call__(self, t: float = None) -> np.ndarray:
        """
        Nengo-compatible callable interface.
        
        Args:
            t: Current time (optional, for compatibility)
        
        Returns:
            np.ndarray: CPG output [num_groups]
        """
        # Step to reach time t if provided
        if t is not None and t > self._sim_time:
            n_steps = max(1, int((t - self._sim_time) / self.dt))
            return self.step(n_steps)
        return self.get_output()
    
    def set_rl_params(self, amplitude: float = None, frequency: float = None,
                      direction: float = None):
        """
        RL-friendly parameter interface.
        
        Note: Changing these requires rebuilding the model (expensive).
        For real-time control, use amplitude scaling instead.
        
        Args:
            amplitude: Output amplitude [0, 1]
            frequency: Oscillation frequency (Hz)
            direction: +1 or -1
        """
        rebuild = False
        
        if amplitude is not None:
            self.amplitude = np.clip(amplitude, 0.0, 1.0)
            self._snn_cpg.amplitude = self.amplitude
            rebuild = True
        
        if frequency is not None:
            self.frequency = np.clip(frequency, 0.1, 10.0)
            self._snn_cpg.frequency = self.frequency
            rebuild = True
        
        if direction is not None:
            self.direction = 1.0 if direction >= 0 else -1.0
            self._snn_cpg.direction = self.direction
            rebuild = True
        
        if rebuild:
            print(f"  ⚠ Rebuilding CPG (amp={self.amplitude}, freq={self.frequency}, dir={self.direction})")
            self.sim.close()
            self._snn_cpg = SNN_CPG_Controller(
                num_groups=self.num_groups,
                frequency=self.frequency,
                amplitude=self.amplitude,
                direction=self.direction,
                coupling_strength=self.coupling_strength,
                n_neurons=self.n_neurons,
            )
            self._build_model()
    
    def reset(self):
        """Reset the CPG to initial state."""
        self._sim_time = 0.0
        self._snn_cpg._output[:] = 0.0
        
        # Rebuild simulator
        self.sim.close()
        self.sim = nengo.Simulator(self.model, dt=self.dt)
    
    def close(self):
        """Close the simulator."""
        if hasattr(self, 'sim'):
            self.sim.close()
    
    def __del__(self):
        """Cleanup."""
        self.close()
    
    def __str__(self):
        return f"NengoCPG(groups={self.num_groups}, freq={self.frequency}Hz, dir={self.direction})"

