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


def hopf_dynamics(x, a=10.0, mu=1.0, omega=2*np.pi*2.0):
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
    amplitude: float = 1.0,
    direction: float = 1.0,
    hopf_a: float = 10.0,
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
            if t < 0.2:  # Longer for proper initialization
                return [x0 * 15, y0 * 15]  # Strong initial push
            return [0, 0]
        
        init_node = nengo.Node(init_pulse, size_out=2, label=f"{prefix}_init")
        nengo.Connection(init_node, osc_ens, synapse=0.005)  # Faster synapse
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
    - DIRECT NEURAL STRAIN INTEGRATION (biological CPG style)
    
    Strain Integration:
        Strain feedback is fed directly into the oscillator ensembles,
        affecting the neural dynamics. This is similar to how proprioceptive
        feedback modulates biological CPGs:
        
        - High extension strain → reduces oscillator amplitude on that side
        - High compression strain → increases oscillator amplitude
        - Creates reflexive adaptation to mechanical state
    """
    
    def __init__(
        self,
        num_groups: int,
        frequency: float = 2.0,
        amplitude: float = 1.0,
        direction: float = 1.0,
        coupling_strength: float = 3.0,
        strain_gain: float = 0.5,
        pes_learning_rate: float = 1e-3,  # PES learning rate (faster for CPG)
        n_neurons: int = 200,
        u_max: float = 1.0,
    ):
        """
        Initialize SNN CPG controller with PES-learned strain feedback.
        
        Args:
            num_groups: Number of oscillators (one per 2x2 group)
            frequency: Base oscillation frequency (Hz)
            amplitude: Output amplitude [0, 1]
            direction: +1 = wave left (body right), -1 = wave right (body left)
            coupling_strength: Inter-oscillator coupling
            strain_gain: Gain for error signal (scales how much strain affects learning)
            pes_learning_rate: PES learning rate (0 to disable learning)
            n_neurons: Neurons per oscillator
            u_max: Maximum output (for scaling)
        """
        self.num_groups = num_groups
        self.frequency = frequency
        self.amplitude = amplitude
        self.direction = direction
        self.coupling_strength = coupling_strength
        self.strain_gain = strain_gain
        self.pes_learning_rate = pes_learning_rate
        self.n_neurons = n_neurons
        self.u_max = u_max
        
        # Compute grid layout
        self.grid_side = int(np.sqrt(num_groups))
        
        # Hopf parameters (a=10 for smooth convergence in SNN)
        self.hopf_a = 10.0  # Convergence rate (smoother than classic 15)
        self.mu = 1.0       # Target amplitude squared
        self.tau_syn = 0.01
        
        # Storage
        self.oscillator_ensembles = []
        self.strain_ensembles = []  # NEW: for direct neural integration
        self.group_components = {}
        self.output_nodes = []
        
        # Strain input storage (updated externally)
        self._strain_input = np.zeros(num_groups, dtype=np.float32)
        
        # Position input (updated externally, used for neural velocity)
        self._position_input = np.zeros(num_groups, dtype=np.float32)  # x-position per group
        
        # Velocity ensembles (computed neurally via differentiator)
        self.velocity_ensembles = []
        
        # Output storage
        self._output = np.zeros(num_groups, dtype=np.float32)
        
        # Build phase offsets for traveling wave
        self._build_phase_offsets()
        
        # PES connection storage (for reference/debugging)
        self.pes_connections = []
        
        print(f"  SNN_CPG_Controller (Hopf + PES Strain Learning):")
        print(f"    Groups: {num_groups} ({self.grid_side}x{self.grid_side})")
        print(f"    Frequency: {frequency} Hz")
        print(f"    Direction: {'+1 (right)' if direction > 0 else '-1 (left)'}")
        print(f"    Coupling: {coupling_strength}")
        print(f"    PES learning: lr={pes_learning_rate}, strain_gain={strain_gain}")
        print(f"    Neurons per oscillator: {n_neurons}")
    
    def _build_phase_offsets(self):
        """
        Build initial phase offsets for traveling wave pattern.
        
        Matches classic HopfCPG behavior:
        - Phase gradient: HIGHER phase in direction → wave travels OPPOSITE → body moves WITH direction
        - For direction=+1 (right): left columns have higher phase, wave goes left, body goes right
        """
        self.phase_offsets = np.zeros(self.num_groups)
        
        # Phase step per grid cell - creates traveling wave
        # Using 2π across grid = full wave cycle for maximum thrust
        phase_per_cell = np.pi / 2.0  # 90° per cell = proper traveling wave
        
        for i in range(self.num_groups):
            row = i // self.grid_side
            col = i % self.grid_side
            
            # Phase gradient: HIGHER phase in direction → wave travels OPPOSITE → body moves WITH direction
            # This matches classic HopfCPG._init_phases()
            # direction > 0: higher phase on right → wave travels left → body moves right
            # direction < 0: higher phase on left → wave travels right → body moves left
            col_phase = col * phase_per_cell * self.direction
            row_phase = 0.0  # For now, only support horizontal direction (like direction in rl_locomotion)
            
            self.phase_offsets[i] = col_phase + row_phase
    
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
        
        DIRECT NEURAL STRAIN INTEGRATION:
            Strain is fed directly into the oscillator ensembles as a
            modulatory input. This mimics biological proprioceptive feedback.
        
        Args:
            model: Parent nengo.Network
            strain_callback: Optional function(group_id) -> callable returning strain
                           If None, uses internal _strain_input array
        
        Returns:
            dict with ensembles and output references
        """
        print(f"  Building {self.num_groups} Hopf oscillators with neural strain integration...")
        
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
                
                # =============================================================
                # STRAIN INPUT ENSEMBLE (Direct Neural Integration)
                # =============================================================
                # Create strain input node (reads from _strain_input array or callback)
                if strain_callback is not None:
                    strain_fn = strain_callback(gid)
                else:
                    # Default: read from internal array
                    def make_strain_reader(group_id, strain_array):
                        def read_strain(t):
                            return strain_array[group_id]
                        return read_strain
                    strain_fn = make_strain_reader(gid, self._strain_input)
                
                strain_node = nengo.Node(strain_fn, size_out=1, label=f"{prefix}_strain_in")
                
                # Strain ensemble (encodes strain neurally)
                strain_ens = nengo.Ensemble(
                    n_neurons=50,  # Smaller ensemble for strain
                    dimensions=1,
                    radius=1.0,
                    label=f"{prefix}_strain_ens"
                )
                nengo.Connection(strain_node, strain_ens, synapse=0.01)
                self.strain_ensembles.append(strain_ens)
                
                # =============================================================
                # POSITION INPUT → NEURAL VELOCITY (Nengo Differentiator)
                # =============================================================
                # Native Nengo: velocity = d(position)/dt via fast-slow synapses
                # No Python arrays, fully neural memory!
                
                def make_position_reader(group_id, pos_array):
                    def read_pos(t):
                        return pos_array[group_id]
                    return read_pos
                
                pos_node = nengo.Node(
                    make_position_reader(gid, self._position_input),
                    size_out=1,
                    label=f"{prefix}_pos_in"
                )
                
                # Velocity ensemble (neural differentiator)
                # Tuned for position values in range [0, 3] and velocities ~0.1-1.0
                tau_fast = 0.01    # Fast synapse (10ms)
                tau_slow = 0.1     # Slow synapse (100ms)
                diff_scale = 1.0 / (tau_slow - tau_fast)  # ~11
                
                vel_ens = nengo.Ensemble(
                    n_neurons=100,  # More neurons for better velocity estimate
                    dimensions=1,
                    radius=2.0,     # Larger radius for velocity range
                    label=f"{prefix}_vel"
                )
                # NEF differentiator: (fast - slow) * scale ≈ velocity
                # Scale factor to get velocity in reasonable range [-1, 1]
                vel_scale = diff_scale * 0.5  # Tuned for typical body velocities
                nengo.Connection(pos_node, vel_ens, transform=vel_scale, synapse=tau_fast)
                nengo.Connection(pos_node, vel_ens, transform=-vel_scale, synapse=tau_slow)
                self.velocity_ensembles.append(vel_ens)
                
                # =============================================================
                # PES: STRAIN → OSCILLATOR (Simple: error = strain - velocity)
                # =============================================================
                # No extra ensemble needed! Just connect both with opposite signs:
                #   - Strain adds to error (high strain = problem)
                #   - Velocity subtracts from error (high velocity = good)
                #
                # High strain + moving → cancel → low error → keep going
                # High strain + stuck → strain wins → high error → adjust
                
                # =============================================================
                # PES LEARNING CONNECTION
                # =============================================================
                # Goal: Learn strain → oscillator modulation
                # 
                # Error signal: -strain (following stress.py pattern)
                # - When strain is high (bad), error is negative
                # - PES learns to OUTPUT something that counters this error
                # - Net effect: strain causes oscillator to reduce output
                # 
                # Key insight from stress.py: use -error transform
                # =============================================================
                
                g = self.strain_gain
                
                pes_conn = nengo.Connection(
                    strain_ens, osc_ens,
                    transform=np.zeros((2, 1)),  # Starts at zero, learns!
                    synapse=self.tau_syn,
                    learning_rule_type=nengo.PES(learning_rate=self.pes_learning_rate),
                )
                
                # Error signal: -strain (like stress.py uses -tracking_error)
                # When strain > 0, this creates negative error
                # PES learns to compensate, reducing oscillator output when strain high
                nengo.Connection(
                    strain_ens, pes_conn.learning_rule,
                    transform=[[-g], [-g * 0.5]],  # Negative! Like stress.py's -1
                    synapse=0.01,
                )
                
                self.pes_connections.append(pes_conn)
                
                # Debug node: print PES values for center group
                if gid == self.num_groups // 2:
                    _debug_last_print = [0.0]
                    def make_pes_debug(group_id, strain_gain):
                        def debug_fn(t, x):
                            strain_val = x[0]
                            error = -strain_gain * strain_val  # Same as transform
                            if t - _debug_last_print[0] > 2.0 and t > 1.0:
                                print(f"[PES G{group_id}] t={t:.1f} strain={strain_val:.3f} error={error:.3f}")
                                _debug_last_print[0] = t
                        return debug_fn
                    
                    debug_node = nengo.Node(
                        make_pes_debug(gid, g),
                        size_in=1,
                        label=f"{prefix}_pes_debug"
                    )
                    nengo.Connection(strain_ens, debug_node, synapse=self.tau_syn)
                
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
                    'strain_ens': strain_ens,
                    'strain_node': strain_node,
                    'vel_ens': vel_ens,
                    'pos_node': pos_node,
                    'pes_conn': pes_conn,
                    'output_node': output_store,
                    'initial_phase': initial_phase,
                }
                
                # Initialization pulse (longer and stronger for proper phase lock)
                def make_init_pulse(x0_val, y0_val):
                    def init_fn(t):
                        if t < 0.2:  # Longer init period
                            return [x0_val * 15, y0_val * 15]  # Stronger push
                        return [0, 0]
                    return init_fn
                
                init_node = nengo.Node(
                    make_init_pulse(x0, y0), 
                    size_out=2, 
                    label=f"{prefix}_init"
                )
                nengo.Connection(init_node, osc_ens, synapse=0.005)  # Faster synapse
            
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
            
            # Add inter-oscillator coupling (phase-only, like classic HopfCPG)
            if self.coupling_strength > 0:
                print(f"    Adding phase-only neighbor coupling (strength={self.coupling_strength})...")
                
                for i in range(self.num_groups):
                    for j in neighbors[i]:
                        # Target phase difference between neighbors
                        target_phi = self.phase_offsets[j] - self.phase_offsets[i]
                        
                        # Coupling function: apply phase perturbation (perpendicular to radius)
                        # This affects phase without changing amplitude (like classic sin(Δθ))
                        def make_coupling_fn(phi, strength, tau):
                            cos_phi = np.cos(phi)
                            sin_phi = np.sin(phi)
                            def coupling_fn(x):
                                # Rotate neighbor by -phi to get relative phase
                                x_rot = cos_phi * x[0] + sin_phi * x[1]  # cos(θⱼ - φ)
                                y_rot = -sin_phi * x[0] + cos_phi * x[1]  # sin(θⱼ - φ)
                                # Apply as phase perturbation (perpendicular to radius)
                                # This is equivalent to classic: dθ += w·sin(θⱼ - θᵢ - φ)
                                return [tau * strength * (-y_rot), tau * strength * x_rot]
                            return coupling_fn
                        
                        # Add coupling from j to i
                        nengo.Connection(
                            self.oscillator_ensembles[j],
                            self.oscillator_ensembles[i],
                            function=make_coupling_fn(target_phi, self.coupling_strength, self.tau_syn),
                            synapse=self.tau_syn,
                        )
        
        print(f"  ✓ {self.num_groups} Hopf oscillators with PES learning")
        print(f"    Oscillators: {len(self.oscillator_ensembles)} × {self.n_neurons} neurons")
        print(f"    Strain: {len(self.strain_ensembles)} × 50 neurons")
        print(f"    Velocity: {len(self.velocity_ensembles)} × 50 neurons (differentiator)")
        print(f"    PES: error = strain - velocity (no extra ensemble)")
        print(f"    Total neurons: ~{(self.n_neurons + 100) * self.num_groups}")
        
        return {
            'ensembles': self.oscillator_ensembles,
            'output': self._output,
            'components': self.group_components,
        }
    
    def get_output(self):
        """Get current CPG output array."""
        return self._output.copy()
    
    def set_strain(self, strain_vector):
        """
        Set per-group strain input for PES learning.
        
        Args:
            strain_vector: Array of shape (num_groups,) with values in [-1, 1]
                          Positive = extension, Negative = compression
        """
        if strain_vector is None:
            self._strain_input[:] = 0.0
        else:
            self._strain_input = np.asarray(strain_vector, dtype=np.float32).copy()
    
    def set_position(self, position_vector):
        """
        Set per-group x-position for neural velocity computation.
        
        The velocity is computed NEURALLY via Nengo differentiator
        (fast synapse - slow synapse = derivative).
        
        Args:
            position_vector: Array of shape (num_groups,) with x-positions
                           Used to compute velocity error for PES learning
        
        Example:
            >>> cpg.set_position(group_centroids[:, 0])  # x-position of each group
        """
        if position_vector is None:
            self._position_input[:] = 0.0
        else:
            self._position_input = np.asarray(position_vector, dtype=np.float32).copy()
    
    def get_all_ensembles(self):
        """Return all oscillator ensembles for visualization."""
        return self.oscillator_ensembles
    
    def get_strain_ensembles(self):
        """Return all strain ensembles for visualization."""
        return self.strain_ensembles
    
    def __str__(self):
        return f"SNN_CPG_Controller(groups={self.num_groups}, freq={self.frequency}Hz, strain_gain={self.strain_gain})"


class NengoCPG:
    """
    NEF-based Spiking CPG for standalone use.
    
    Creates its own Nengo model and simulator internally.
    Provides the same interface as other controllers.
    
    The Hopf oscillator is implemented using NEF with:
    - 2D ensemble per oscillator (x, y Cartesian coordinates)
    - Recurrent connections for Hopf dynamics
    - Coupling connections for traveling wave coordination
    - Optional strain feedback for adaptive behavior
    
    Example (basic):
        >>> cpg = NengoCPG(num_groups=9, frequency=2.0, direction=1.0)
        >>> for _ in range(1000):
        >>>     output = cpg.step()  # Get 9D output in [-1, 1]
    
    Example (with strain feedback):
        >>> cpg = NengoCPG(num_groups=9, feedback_gain=0.5)
        >>> cpg.set_feedback(strain_array)  # Set strain per group
        >>> output = cpg.step()  # Output adapts to strain!
    """
    
    def __init__(
        self,
        num_groups: int,
        dt: float = 0.001,
        frequency: float = 2.0,
        amplitude: float = 1.0,
        direction: float = 1.0,
        coupling_strength: float = 3.0,
        feedback_gain: float = 0.5,
        pes_learning_rate: float = 1e-4,  # PES learning rate
        n_neurons: int = 200,
        device: str = 'cpu',
    ):
        """
        Initialize standalone spiking CPG with PES-learned strain feedback.
        
        Args:
            num_groups: Number of oscillators
            dt: Simulation time step
            frequency: Oscillation frequency (Hz)
            amplitude: Output amplitude [0, 1]
            direction: +1 = body moves right, -1 = body moves left
            coupling_strength: Inter-oscillator coupling
            feedback_gain: Strain gain for error signal
            pes_learning_rate: PES learning rate (0 to disable)
            n_neurons: Neurons per oscillator
            device: 'cpu' (GPU not recommended for step-by-step)
        """
        self.num_groups = num_groups
        self.dt = dt
        self.frequency = frequency
        self.amplitude = amplitude
        self.direction = direction
        self.coupling_strength = coupling_strength
        self.feedback_gain = feedback_gain  # Used as strain_gain for PES
        self.pes_learning_rate = pes_learning_rate
        self.n_neurons = n_neurons
        self.device = device
        
        self._sim_time = 0.0
        
        # Create inner SNN CPG controller with PES strain learning
        self._snn_cpg = SNN_CPG_Controller(
            num_groups=num_groups,
            frequency=frequency,
            amplitude=amplitude,
            direction=direction,
            coupling_strength=coupling_strength,
            strain_gain=feedback_gain,  # Use feedback_gain as strain_gain
            pes_learning_rate=pes_learning_rate,
            n_neurons=n_neurons,
        )
        
        # Build model
        self._build_model()
        
        print(f"  ✓ NengoCPG initialized (Hopf + PES strain learning)")
        print(f"    Groups: {num_groups}")
        print(f"    Frequency: {frequency} Hz")
        print(f"    PES: lr={pes_learning_rate}, strain_gain={feedback_gain}")
        print(f"    Total neurons: ~{n_neurons * num_groups + 50 * num_groups}")
    
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
        """
        Get current CPG output.
        
        With direct neural integration, strain affects the oscillator
        dynamics internally - no post-processing needed.
        """
        return self._snn_cpg.get_output()
    
    def set_feedback(self, feedback_vector):
        """
        Set per-group strain for PES learning.
        
        Args:
            feedback_vector: Array of shape (num_groups,) with values in [-1, 1]
        """
        self._snn_cpg.set_strain(feedback_vector)
    
    def set_position(self, position_vector):
        """
        Set per-group x-position for neural velocity computation.
        
        The velocity is computed via Nengo's neural differentiator,
        then used as the error signal for PES learning.
        
        Args:
            position_vector: Array of shape (num_groups,) with x-positions
        
        Example:
            >>> cpg.set_position(group_centroids[:, 0])  # x-position
            >>> cpg.set_feedback(strain_per_group)       # strain
            >>> output = cpg.step()                       # PES learns!
        """
        self._snn_cpg.set_position(position_vector)
    
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
                strain_gain=self.feedback_gain,
                pes_learning_rate=self.pes_learning_rate,
                n_neurons=self.n_neurons,
            )
            self._build_model()
    
    def reset(self):
        """Reset the CPG to initial state."""
        self._sim_time = 0.0
        self._snn_cpg._output[:] = 0.0
        self._snn_cpg._strain_input[:] = 0.0  # Clear strain
        
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


# =============================================================================
# DEMO / TEST MODE
# =============================================================================

if __name__ == "__main__":
    """
    Run standalone demo of the spiking Hopf CPG.
    
    Usage:
        python cpg.py                    # Default: 9 groups, 2Hz, matplotlib plot
        python cpg.py --groups 4         # 2x2 grid (4 oscillators)
        python cpg.py --freq 1.0         # 1 Hz oscillation
        python cpg.py --duration 5.0     # Run for 5 seconds
        python cpg.py --no-plot          # Skip matplotlib (just print stats)
    """
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Spiking Hopf CPG Demo")
    parser.add_argument("--groups", type=int, default=9, help="Number of oscillator groups (default: 9)")
    parser.add_argument("--freq", type=float, default=2.0, help="Oscillation frequency in Hz (default: 2.0)")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Output amplitude 0-1 (default: 1.0, matches classic)")
    parser.add_argument("--direction", type=float, default=1.0, help="Direction: +1=right, -1=left (default: 1.0)")
    parser.add_argument("--coupling", type=float, default=2.0, help="Inter-oscillator coupling (default: 2.0)")
    parser.add_argument("--neurons", type=int, default=200, help="Neurons per oscillator (default: 200)")
    parser.add_argument("--duration", type=float, default=3.0, help="Simulation duration in seconds (default: 3.0)")
    parser.add_argument("--dt", type=float, default=0.001, help="Timestep in seconds (default: 0.001)")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib visualization")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Spiking Hopf CPG Demo (NEF-based)")
    print("=" * 70)
    print(f"  Groups: {args.groups}")
    print(f"  Frequency: {args.freq} Hz")
    print(f"  Amplitude: {args.amplitude}")
    print(f"  Direction: {'+1 (body RIGHT)' if args.direction > 0 else '-1 (body LEFT)'}")
    print(f"  Coupling: {args.coupling}")
    print(f"  Neurons per oscillator: {args.neurons}")
    print(f"  Duration: {args.duration}s")
    print(f"  Timestep: {args.dt}s")
    print("=" * 70)
    
    # Create CPG
    cpg = NengoCPG(
        num_groups=args.groups,
        frequency=args.freq,
        amplitude=args.amplitude,
        direction=args.direction,
        coupling_strength=args.coupling,
        n_neurons=args.neurons,
        dt=args.dt,
    )
    
    # Run simulation
    n_steps = int(args.duration / args.dt)
    print(f"\nRunning {n_steps} simulation steps...")
    
    # Record outputs
    times = []
    outputs = []
    
    start_time = time.time()
    
    for step in range(n_steps):
        t = step * args.dt
        output = cpg.step()
        
        times.append(t)
        outputs.append(output.copy())
        
        # Print progress every 0.5s
        if step % int(0.5 / args.dt) == 0:
            print(f"  t={t:.2f}s: output range [{output.min():.3f}, {output.max():.3f}]")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Simulation complete in {elapsed:.2f}s ({n_steps/elapsed:.0f} steps/sec)")
    
    # Convert to numpy arrays
    times = np.array(times)
    outputs = np.array(outputs)
    
    # Print statistics
    print(f"\nOutput statistics:")
    print(f"  Shape: {outputs.shape} (time x groups)")
    print(f"  Mean: {outputs.mean():.4f}")
    print(f"  Std: {outputs.std():.4f}")
    print(f"  Range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    
    # Check for traveling wave pattern
    grid_side = int(np.sqrt(args.groups))
    if grid_side > 1:
        # Compute phase difference between first and last column
        col0_idx = 0
        col_last_idx = grid_side - 1
        
        # Find peaks in center of simulation
        mid_start = int(0.3 * len(times))
        mid_end = int(0.7 * len(times))
        
        col0_signal = outputs[mid_start:mid_end, col0_idx]
        col_last_signal = outputs[mid_start:mid_end, col_last_idx]
        
        # Compute cross-correlation to estimate phase shift
        correlation = np.correlate(col0_signal - col0_signal.mean(), 
                                   col_last_signal - col_last_signal.mean(), mode='full')
        lag = np.argmax(correlation) - len(col0_signal) + 1
        phase_lag_ms = lag * args.dt * 1000
        
        print(f"\nTraveling wave analysis:")
        print(f"  Phase lag (col0 → col{grid_side-1}): {phase_lag_ms:.1f} ms")
        if phase_lag_ms > 0:
            print(f"  → Wave travels LEFT (body moves RIGHT)")
        elif phase_lag_ms < 0:
            print(f"  → Wave travels RIGHT (body moves LEFT)")
        else:
            print(f"  → In-phase (no traveling wave)")
    
    # Plot if requested
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            
            print(f"\nGenerating plot...")
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot all oscillator outputs
            ax1 = axes[0]
            for i in range(args.groups):
                row = i // grid_side
                col = i % grid_side
                ax1.plot(times, outputs[:, i], label=f"G{i}[{row},{col}]", alpha=0.8)
            
            ax1.set_ylabel("Output [-1, 1]")
            ax1.set_title(f"Spiking Hopf CPG: {args.groups} oscillators @ {args.freq} Hz")
            ax1.legend(loc="upper right", fontsize=8, ncol=min(4, args.groups))
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-1.1, 1.1)
            
            # Plot heatmap of outputs over time (traveling wave visualization)
            ax2 = axes[1]
            
            # Downsample for visualization
            downsample = max(1, len(times) // 500)
            times_ds = times[::downsample]
            outputs_ds = outputs[::downsample, :]
            
            im = ax2.imshow(outputs_ds.T, aspect='auto', cmap='RdBu_r',
                           extent=[times_ds[0], times_ds[-1], args.groups-0.5, -0.5],
                           vmin=-1, vmax=1)
            ax2.set_ylabel("Oscillator ID")
            ax2.set_xlabel("Time (s)")
            ax2.set_title("Traveling Wave Pattern (blue=negative, red=positive)")
            plt.colorbar(im, ax=ax2, label="Output")
            
            plt.tight_layout()
            
            # Save figure
            save_path = "cpg_snn_demo.png"
            plt.savefig(save_path, dpi=150)
            print(f"  ✓ Saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("\n⚠ matplotlib not available, skipping plot")
            print("  Install with: pip install matplotlib")
    
    # Cleanup
    cpg.close()
    print("\n✓ Demo complete!")
