"""
NEF-Based Stress Controller with PES Feedforward Learning

Simple architecture:
- Spiking PD control
- PES learns feedforward from strain (7D per group: 5 springs + 2 FEMs)
- Feedforward connects DIRECTLY to u(t) - fully neural!

ARCHITECTURE:
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  error ──> q(t) ──┬────[Kp]────────────────> u(t) ──> output      │
│            (ens)  │                          (ens)                 │
│              │    └──> ed(t) ──────[Kd]────────┤                   │
│              │        (ens)                    │                   │
│              │                                 │                   │
│              │    strain[7D] ──> s(t) ──[PES]──┘                   │
│              │                  (ens)                              │
│              │                    ↑                                │
│              └────────────────────┘                                │
│                   -error (learning signal)                         │
│                                                                    │
│  Strain: [spring_0, spring_1, ..., spring_4, fem_0, fem_1]        │
│                                                                    │
│  u(t) = Kp*q + Kd*ed + PES_learned(strain)                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

PES learns: strain[7D] → force[2D] directly into u(t) ensemble
"""

import numpy as np
import nengo

from .pid import NengoPID, SNN_PID_Controller


# Strain dimensions per group: 5 springs + 2 FEMs
N_SPRINGS_PER_GROUP = 5
N_FEMS_PER_GROUP = 2
STRAIN_DIM = N_SPRINGS_PER_GROUP + N_FEMS_PER_GROUP  # 7


# =============================================================================
# STANDALONE CONTROLLER (creates its own simulator)
# =============================================================================

class NengoStress(NengoPID):
    """
    Stress Controller with PES Feedforward for standalone use.
    
    - Spiking PD control (from NengoPID)
    - Spiking PES feedforward (learns strain[7D] → force[2D])
    
    Strain per group: 5 springs + 2 FEMs = 7D
    
    Formula: F = F_pd + F_ff (PES learns from error)
    
    Creates its own Nengo model and simulator internally.
    """
    
    def __init__(
        self,
        num_groups: int,
        dt: float = 0.01,
        u_max: float = 500.0,
        Kp: float = 250.0,      # Tuned for NEF
        Kd: float = 80.0,       # Higher for neural differentiator
        n_neurons: int = 100,
        error_scale: float = 2.0,
        pes_learning_rate: float = 1e-4,  # Matched to GUI (faster adaptation)
        device: str = 'cpu',
    ):
        """
        Initialize stress controller with PES feedforward.
        
        Args:
            num_groups: Number of control groups
            dt: Simulation time step
            u_max: Maximum control force
            Kp: Proportional gain (+15% default to compensate synaptic filtering)
            Kd: Derivative gain (+10% default to compensate synaptic filtering)
            n_neurons: Neurons per dimension
            error_scale: Max expected error
            pes_learning_rate: PES learning rate (1e-4 default, matched to GUI)
            device: 'cpu' or 'cuda'
        """
        self.pes_learning_rate = pes_learning_rate
        self.strain_dim = STRAIN_DIM  # 7D per group
        self._current_strain = np.zeros((num_groups, STRAIN_DIM))  # (num_groups, 7)
        self._spring_groups = None  # Built on first call to _get_group_strains
        self._fem_groups = None
        
        # Initialize parent (Ki=0 for pure PD)
        super().__init__(
            num_groups=num_groups,
            dt=dt,
            u_max=u_max,
            Kp=Kp,
            Ki=0.0,
            Kd=Kd,
            n_neurons=n_neurons,
            error_scale=error_scale,
            device=device,
        )
        
        print(f"  ✓ NengoStress (spiking PD + PES feedforward)")
        print(f"    Strain dim: {STRAIN_DIM} (5 springs + 2 FEMs)")
    
    def _build_model(self):
        """Build Nengo model with PD and PES feedforward (7D strain)."""
        
        # Build parent's PD network
        super()._build_model()
        
        tau_syn = 0.01
        
        # Storage for strain ensembles and PES connections
        self._strain_ensembles = []
        
        with self.model:
            for gid in range(self.num_groups):
                prefix = f"G{gid}"
                
                # Strain input node (7D: 5 springs + 2 FEMs)
                def make_strain_input(group_id):
                    def get_strain(t):
                        return self._current_strain[group_id]  # Returns 7D array
                    return get_strain
                
                strain_node = nengo.Node(
                    make_strain_input(gid), 
                    size_out=STRAIN_DIM,  # 7D output
                    label=f"{prefix}_Strain[7D]"
                )
                
                # Strain ensemble (7D for PES source)
                # 
                # HIGH-DIMENSIONAL INTERCEPT FIX:
                # With uniform intercepts in 7D, neurons fire for only ~2-3% of inputs.
                # Triangular distribution → inverse beta transform concentrates intercepts
                # where neurons achieve ~50% firing probability, maximizing capacity.
                # See: Zaidel et al. 2021 (NEF-based IK/PID on Loihi).
                #
                n_strain_neurons = self.n_neurons * STRAIN_DIM
                triangular_samples = np.random.triangular(0.3, 0.5, 0.7, n_strain_neurons)  # GUI values
                strain_intercepts = nengo.dists.CosineSimilarity(STRAIN_DIM + 2).ppf(1 - triangular_samples)
                
                s_ens = nengo.Ensemble(
                    n_neurons=n_strain_neurons,
                    dimensions=STRAIN_DIM,  # 7D
                    max_rates=nengo.dists.Uniform(100, 200),  # Match GUI
                    intercepts=strain_intercepts,  # Triangular for 7D
                    neuron_type=nengo.LIF(),  # Match GUI
                    radius=1.0,
                    label=f"{prefix}_s(t)[7D]"
                )
                nengo.Connection(strain_node, s_ens, synapse=tau_syn)
                self._strain_ensembles.append(s_ens)
                
                # PES feedforward connection (directly to u_ens - fully neural!)
                if self.pes_learning_rate > 0:
                    q_ens = self._snn_controller.group_components[gid]['q_ens']
                    u_ens = self._snn_controller.group_components[gid]['u_ens']
                    
                    # PES: strain[7D] → u_ens (feedforward added to output neurally)
                    ff_conn = nengo.Connection(
                        s_ens, u_ens,  # Direct to output ensemble!
                        transform=np.zeros((2, STRAIN_DIM)),
                        synapse=tau_syn,
                        learning_rule_type=nengo.PES(learning_rate=self.pes_learning_rate),
                    )
                    
                    # Learning signal: -error
                    nengo.Connection(q_ens, ff_conn.learning_rule, transform=-1)
        
        print(f"    + PES feedforward: lr={self.pes_learning_rate}, strain_dim={STRAIN_DIM}")
    
    def compute_control(self, state_dict, get_target_fn=None):
        """
        Compute control: F = u(t) which includes P + I + D + FF (fully neural)
        
        Args:
            state_dict: Dict with centroids, targets, strains
            get_target_fn: Optional (unused)
        
        Returns:
            np.ndarray: Control forces [num_groups * 2]
        """
        # Update strain input
        self._current_strain = self._get_group_strains(state_dict)
        
        # Update error input
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
        
        # Output includes PD + PES feedforward (all in u_ens)
        return np.clip(self._snn_controller.get_output(), -self.u_max, self.u_max)
    
    def _get_group_strains(self, state_dict):
        """
        Extract 7D strain per group from state dict using spatial assignment.
        
        Uses the SAME geometry-based logic as the GUI:
        - Springs/FEMs are assigned to groups based on which particles they connect
        - A spring belongs to a group if BOTH its particles are in that group's 2x2 region
        - An FEM belongs to a group if ALL 3 vertices are in that group's 2x2 region
        
        Returns:
            np.ndarray: Shape (num_groups, 7) where each row is:
                [spring_0, spring_1, spring_2, spring_3, spring_4, fem_0, fem_1]
        """
        # Build group assignments on first call
        if not hasattr(self, '_spring_groups') or self._spring_groups is None:
            self._build_strain_group_assignments(state_dict)
        
        spring_strains = state_dict.get('spring_strains', None)
        fem_strains = state_dict.get('fem_strains', None)
        
        # Initialize output: (num_groups, 7)
        group_strains = np.zeros((self.num_groups, STRAIN_DIM))
        
        # Get spring strains using spatial assignment
        if spring_strains is not None and len(spring_strains) > 0:
            spring_strains = np.array(spring_strains)
            for g in range(self.num_groups):
                group_spring_indices = self._spring_groups.get(g, [])[:N_SPRINGS_PER_GROUP]
                for i, spring_idx in enumerate(group_spring_indices):
                    if spring_idx < len(spring_strains):
                        group_strains[g, i] = np.clip(spring_strains[spring_idx], -1.0, 1.0)
        
        # Get FEM strains using spatial assignment
        if fem_strains is not None and len(fem_strains) > 0:
            fem_strains = np.array(fem_strains)
            for g in range(self.num_groups):
                group_fem_indices = self._fem_groups.get(g, [])[:N_FEMS_PER_GROUP]
                for i, fem_idx in enumerate(group_fem_indices):
                    if fem_idx < len(fem_strains):
                        group_strains[g, N_SPRINGS_PER_GROUP + i] = np.clip(fem_strains[fem_idx], -1.0, 1.0)
        
        return group_strains
    
    def _build_strain_group_assignments(self, state_dict):
        """
        Build spatial assignments of springs/FEMs to groups (same logic as GUI).
        
        A 2x2 group at (row, col) contains particles:
          - (row, col), (row, col+1), (row+1, col), (row+1, col+1)
        
        A spring belongs to a group if BOTH endpoints are in that group.
        An FEM belongs to a group if ALL 3 vertices are in that group.
        """
        # Get grid size from positions
        positions = state_dict.get('positions', None)
        if positions is None:
            self._spring_groups = {g: [] for g in range(self.num_groups)}
            self._fem_groups = {g: [] for g in range(self.num_groups)}
            return
        
        # Infer grid size N from num_particles = N*N
        n_particles = len(positions)
        N = int(np.sqrt(n_particles))
        num_groups_per_side = N - 1  # (N-1) x (N-1) groups
        
        def particle_to_grid(idx):
            return (idx // N, idx % N)
        
        def get_particle_groups(idx):
            """Return all groups that contain this particle."""
            row, col = particle_to_grid(idx)
            groups = []
            # A particle at (row, col) is in groups (r, c) where:
            # - r <= row <= r+1 and c <= col <= c+1
            if row < N-1 and col < N-1:
                groups.append((row, col))      # top-left of group
            if row < N-1 and col > 0:
                groups.append((row, col-1))    # top-right of group
            if row > 0 and col < N-1:
                groups.append((row-1, col))    # bottom-left of group
            if row > 0 and col > 0:
                groups.append((row-1, col-1))  # bottom-right of group
            return groups
        
        # Initialize group assignments
        self._spring_groups = {g: [] for g in range(self.num_groups)}
        self._fem_groups = {g: [] for g in range(self.num_groups)}
        
        # Get model indices from state (if available via tracking_env)
        # We need spring_indices and tri_indices from the model
        # These should be passed through state_dict or cached
        group_info = state_dict.get('group_info', None)
        
        # For now, use the model's spring/FEM topology from tracking_env
        # The tracking_env provides this via the 'positions' array structure
        # We'll compute based on the standard grid topology
        
        # Standard grid springs: horizontal, vertical, diagonal
        spring_idx = 0
        # Horizontal springs
        for row in range(N):
            for col in range(N - 1):
                p0 = row * N + col
                p1 = row * N + col + 1
                groups0 = set(get_particle_groups(p0))
                groups1 = set(get_particle_groups(p1))
                common = groups0 & groups1
                for (gr, gc) in common:
                    gid = gr * num_groups_per_side + gc
                    if gid < self.num_groups:
                        self._spring_groups[gid].append(spring_idx)
                spring_idx += 1
        
        # Vertical springs
        for row in range(N - 1):
            for col in range(N):
                p0 = row * N + col
                p1 = (row + 1) * N + col
                groups0 = set(get_particle_groups(p0))
                groups1 = set(get_particle_groups(p1))
                common = groups0 & groups1
                for (gr, gc) in common:
                    gid = gr * num_groups_per_side + gc
                    if gid < self.num_groups:
                        self._spring_groups[gid].append(spring_idx)
                spring_idx += 1
        
        # Diagonal springs (both directions)
        for row in range(N - 1):
            for col in range(N - 1):
                # Diagonal 1: (row,col) -> (row+1,col+1)
                p0 = row * N + col
                p1 = (row + 1) * N + col + 1
                groups0 = set(get_particle_groups(p0))
                groups1 = set(get_particle_groups(p1))
                common = groups0 & groups1
                for (gr, gc) in common:
                    gid = gr * num_groups_per_side + gc
                    if gid < self.num_groups:
                        self._spring_groups[gid].append(spring_idx)
                spring_idx += 1
                
                # Diagonal 2: (row,col+1) -> (row+1,col)
                p0 = row * N + col + 1
                p1 = (row + 1) * N + col
                groups0 = set(get_particle_groups(p0))
                groups1 = set(get_particle_groups(p1))
                common = groups0 & groups1
                for (gr, gc) in common:
                    gid = gr * num_groups_per_side + gc
                    if gid < self.num_groups:
                        self._spring_groups[gid].append(spring_idx)
                spring_idx += 1
        
        # FEM triangles (2 per cell, checkerboard pattern)
        fem_idx = 0
        for row in range(N - 1):
            for col in range(N - 1):
                # Two triangles per cell
                p00 = row * N + col
                p01 = row * N + col + 1
                p10 = (row + 1) * N + col
                p11 = (row + 1) * N + col + 1
                
                # Triangle patterns depend on checkerboard
                if (row + col) % 2 == 0:
                    # Upper-left triangle: p00, p01, p10
                    # Lower-right triangle: p01, p11, p10
                    tri1 = [p00, p01, p10]
                    tri2 = [p01, p11, p10]
                else:
                    # Upper-right triangle: p00, p01, p11
                    # Lower-left triangle: p00, p11, p10
                    tri1 = [p00, p01, p11]
                    tri2 = [p00, p11, p10]
                
                for tri in [tri1, tri2]:
                    groups_all = [set(get_particle_groups(p)) for p in tri]
                    common = groups_all[0] & groups_all[1] & groups_all[2]
                    for (gr, gc) in common:
                        gid = gr * num_groups_per_side + gc
                        if gid < self.num_groups:
                            self._fem_groups[gid].append(fem_idx)
                    fem_idx += 1
        
        # Debug output
        total_springs = sum(len(v) for v in self._spring_groups.values())
        total_fems = sum(len(v) for v in self._fem_groups.values())
        print(f"    Strain groups built: {total_springs} spring assignments, {total_fems} FEM assignments")
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self._current_strain = np.zeros((self.num_groups, STRAIN_DIM))  # (num_groups, 7)
        self._spring_groups = None  # Rebuild on next call
        self._fem_groups = None
    
    def __str__(self):
        return f"NengoStress(groups={self.num_groups}, Kp={self.Kp}, Kd={self.Kd}, PES={self.pes_learning_rate})"


# =============================================================================
# GUI CONTROLLER (uses external Nengo model/simulator)
# =============================================================================

class SNN_Stress_Controller(SNN_PID_Controller):
    """
    Stress Controller with PES Feedforward for Nengo GUI.
    
    - Spiking PD control (from parent)
    - Spiking PES feedforward (learns strain[7D] → force[2D])
    
    Strain per group: 5 springs + 2 FEMs = 7D
    
    Formula: F = F_pd + F_ff
    
    Can either:
    - REUSE existing strain ensembles from GUI (auto-detects dimensions)
    - CREATE own 7D strain ensembles if none provided
    """
    
    def __init__(
        self,
        num_groups: int,
        Kp: float = 250.0,      # Tuned for NEF
        Kd: float = 80.0,       # Higher for neural differentiator
        u_max: float = 500.0,
        n_neurons: int = 100,
        error_scale: float = 2.0,
        pes_learning_rate: float = 1e-4,  # Matched to GUI (faster adaptation)
    ):
        """
        Initialize stress controller for GUI (Zaidel et al. 2021 + PES).
        
        Args:
            num_groups: Number of control groups
            Kp: Proportional gain (tuned for NEF)
            Kd: Derivative gain (tuned for neural differentiator)
            u_max: Maximum control force
            n_neurons: Neurons per dimension
            error_scale: Max expected error
            pes_learning_rate: PES learning rate (0 to disable)
        """
        self.pes_learning_rate = pes_learning_rate
        self.tau_syn = 0.01
        self.strain_dim = STRAIN_DIM  # 7D per group
        
        # Initialize parent (Ki=0)
        super().__init__(
            num_groups=num_groups,
            Kp=Kp,
            Ki=0.0,
            Kd=Kd,
            u_max=u_max,
            n_neurons=n_neurons,
            error_scale=error_scale,
        )
        
        # Strain storage for own ensembles (if created)
        self._current_strain = np.zeros((num_groups, STRAIN_DIM))
        self._strain_ensembles = []
        
        # PES connections (for reference)
        self.pes_connections = []
        
        print(f"  + PES feedforward: lr={pes_learning_rate}, strain_dim={STRAIN_DIM}")
    
    def build_all(self, model, get_error_callback, strain_ensembles=None, 
                  get_strain_callback=None, dt=0.01):
        """
        Build PD networks and PES feedforward (7D strain).
        
        FULLY NEURAL architecture (Zaidel et al. 2021).
        
        Can either:
        - REUSE strain_ensembles from GUI (auto-detects dimensions)
        - CREATE own 7D strain ensembles using get_strain_callback
        
        Args:
            model: Parent nengo.Network
            get_error_callback: Function(group_id) → callable returning 2D error
            strain_ensembles: Existing strain ensembles from GUI (optional)
            get_strain_callback: Function(group_id) → callable returning 7D strain (optional)
            dt: Time step (unused, kept for compatibility)
        """
        # Build parent's PD networks (fully neural)
        result = super().build_all(model, get_error_callback)
        
        if self.pes_learning_rate <= 0:
            print(f"    PES disabled (lr=0)")
            return result
        
        # Determine strain source
        use_external = strain_ensembles is not None and len(strain_ensembles) > 0
        use_callback = get_strain_callback is not None
        
        if not use_external and not use_callback:
            print(f"    ⚠ No strain source - PES disabled")
            print(f"       Provide strain_ensembles or get_strain_callback")
            return result
        
        print(f"  Building PES feedforward (lr={self.pes_learning_rate})...")
        
        with model:
            for gid in range(self.num_groups):
                q_ens = self.group_components[gid]['q_ens']
                prefix = f"G{gid}"
                
                # Determine strain ensemble and dimensions
                if use_external and gid < len(strain_ensembles):
                    # Use provided strain ensemble
                    s_ens = strain_ensembles[gid]
                    strain_dim = s_ens.dimensions
                    print(f"    {prefix}: Using external s_ens (dim={strain_dim})")
                elif use_callback:
                    # Create own 7D strain ensemble
                    strain_fn = get_strain_callback(gid)
                    strain_node = nengo.Node(
                        strain_fn,
                        size_out=STRAIN_DIM,
                        label=f"{prefix}_Strain[7D]"
                    )
                    # Triangular intercepts for 7D (matched to GUI: 0.3, 0.5, 0.7)
                    n_s_neurons = self.n_neurons * STRAIN_DIM
                    tri_samples = np.random.triangular(0.3, 0.5, 0.7, n_s_neurons)  # GUI values
                    s_intercepts = nengo.dists.CosineSimilarity(STRAIN_DIM + 2).ppf(1 - tri_samples)
                    
                    s_ens = nengo.Ensemble(
                        n_neurons=n_s_neurons,
                        dimensions=STRAIN_DIM,
                        max_rates=nengo.dists.Uniform(100, 200),  # Match GUI
                        intercepts=s_intercepts,  # Triangular for 7D
                        neuron_type=nengo.LIF(),  # Match GUI
                        radius=1.0,
                        label=f"{prefix}_s(t)[7D]"
                    )
                    nengo.Connection(strain_node, s_ens, synapse=self.tau_syn)
                    self._strain_ensembles.append(s_ens)
                    strain_dim = STRAIN_DIM
                    print(f"    {prefix}: Created own s_ens (dim={strain_dim})")
                else:
                    print(f"    {prefix}: ⚠ No strain source, skipping")
                    continue
                
                # Get the output ensemble u(t) from PD controller
                u_ens = self.group_components[gid]['u_ens']
                
                # PES: strain[N-D] → u_ens[2D] (learns feedforward directly to output!)
                # Transform shape: (2, strain_dim) - starts at zero, PES learns
                ff_conn = nengo.Connection(
                    s_ens, u_ens,  # Connect to OUTPUT ensemble, not a Python node!
                    transform=np.zeros((2, strain_dim)),
                    synapse=self.tau_syn,
                    learning_rule_type=nengo.PES(learning_rate=self.pes_learning_rate),
                )
                
                # Learning signal: -error (positive error → increase force)
                nengo.Connection(q_ens, ff_conn.learning_rule, transform=-1)
                
                # Store connection reference
                self.pes_connections.append(ff_conn)
                
                print(f"    {prefix}: s(t)[{strain_dim}D] ──[PES]──> u(t)")
        
        print(f"  ✓ PES feedforward connected")
        return result
    
    def get_output(self):
        """
        Get output: F = u(t) which includes P + I + D + FF (PES feedforward)
        
        The feedforward is now part of u(t) - fully neural!
        """
        return np.clip(self._output, -self.u_max, self.u_max)
    
    def __str__(self):
        return f"SNN_Stress_Controller(groups={self.num_groups}, PES={self.pes_learning_rate})"
