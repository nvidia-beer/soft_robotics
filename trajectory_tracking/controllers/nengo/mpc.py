"""
NengoMPC - SNN-enhanced MPC (Halaly & Tsur 2024)

Two models:
1. Learning model: PES learns to correct predictions
2. Prediction model: Uses learned weights for MPC optimization
"""

from ..mpc import MPC
import numpy as np
import nengo
from scipy.optimize import minimize

try:
    import nengo_dl
    NENGO_DL_AVAILABLE = True
except ImportError:
    NENGO_DL_AVAILABLE = False


class ControllableNode(nengo.Node):
    """Node whose output can be set directly."""
    def __init__(self, size_out, label=None):
        self._value = np.zeros(size_out)
        super().__init__(lambda t: self._value, size_out=size_out, label=label)
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, v):
        self._value = np.asarray(v)


class NengoMPC(MPC):
    """MPC with SNN prediction learning."""
    
    def __init__(
        self,
        num_groups: int,
        dt: float = 0.01,
        u_max: float = 500.0,
        horizon: int = 10,
        spring_k: float = 5.0,
        damping: float = 0.1,
        mass: float = 1.0,
        Q: float = 500.0,
        R: float = 0.001,
        snn_neurons: int = 200,
        snn_learning_rate: float = 1e-5,
        snn_tau: float = 0.05,
        use_gpu: bool = True,
        learning_enabled: bool = True,
        **kwargs
    ):
        super().__init__(
            num_groups=num_groups, dt=dt, u_max=u_max, horizon=horizon,
            spring_k=spring_k, damping=damping, mass=mass, Q=Q, R=R, **kwargs
        )
        
        self.learning_enabled = learning_enabled
        self.nengo_dt = dt
        self.tau = dt * 2
        self.snn_neurons = snn_neurons
        self.snn_learning_rate = snn_learning_rate
        self.use_gpu = use_gpu
        
        # State/action dimensions
        self.local_state_size = 4 * num_groups
        self.local_action_size = 2 * num_groups
        self.dynamics_size = 4 * num_groups
        
        # Normalization
        self.max_stat = np.tile([0.5, 0.5, 2.0, 2.0], num_groups)
        self.action_scale = 1.0 / u_max
        self.dynamics_scale = 10.0
        
        self.learn_error = np.zeros(self.dynamics_size)
        self.last_state = None
        self.last_action = None
        self.prev_u = np.zeros(self.local_action_size * horizon)
        
        self._build_learning_model()
        self._build_prediction_model()
        
        print(f"  ✓ NengoMPC: neurons={snn_neurons}, lr={snn_learning_rate}, learning={'ON' if learning_enabled else 'OFF'}")
    
    def _static_dynamics(self, x):
        """Static predictor using parent's _step method."""
        actions_scaled = x[:self.local_action_size]
        states_scaled = x[self.local_action_size:]
        
        group_states = np.zeros((self.num_groups, 4))
        u_t = np.zeros(self.num_groups * 2)
        
        for g in range(self.num_groups):
            u_t[g*2] = actions_scaled[g*2] / self.action_scale
            u_t[g*2+1] = actions_scaled[g*2+1] / self.action_scale
            s = g * 4
            group_states[g] = states_scaled[s:s+4] * self.max_stat[s:s+4]
        
        next_states = self._step(group_states, u_t)
        
        dynamics = []
        for g in range(self.num_groups):
            d = (next_states[g] - group_states[g]) * self.dynamics_scale
            dynamics.extend(d)
        return dynamics
    
    def _build_model(self, trainable):
        """Build Nengo network. trainable=True adds PES learning."""
        input_size = self.local_action_size + self.local_state_size
        
        model = nengo.Network(seed=0)
        with model:
            action_node = ControllableNode(self.local_action_size)
            state_node = ControllableNode(self.local_state_size)
            
            static = nengo.Ensemble(1, input_size, neuron_type=nengo.Direct())
            nengo.Connection(action_node, static[:self.local_action_size], synapse=None)
            nengo.Connection(state_node, static[self.local_action_size:], synapse=None)
            
            adaptive = nengo.Ensemble(
                self.snn_neurons, input_size,
                radius=np.sqrt(input_size), neuron_type=nengo.LIF(), seed=0
            )
            nengo.Connection(action_node, adaptive[:self.local_action_size],
                           transform=self.action_scale, synapse=None)
            nengo.Connection(state_node, adaptive[self.local_action_size:],
                           transform=np.diag(1.0/self.max_stat), synapse=None)
            
            output = nengo.Node(size_in=self.dynamics_size)
            nengo.Connection(static, output, function=self._static_dynamics, synapse=self.tau)
            
            if trainable:
                error_node = nengo.Node(lambda t: self.learn_error, size_out=self.dynamics_size)
                conn = nengo.Connection(
                    adaptive, output,
                    function=lambda x: [0]*self.dynamics_size, synapse=self.tau,
                    learning_rule_type=nengo.PES(learning_rate=self.snn_learning_rate)
                )
                nengo.Connection(error_node, conn.learning_rule, synapse=None)
            else:
                conn = nengo.Connection(
                    adaptive.neurons, output,
                    transform=np.zeros((self.dynamics_size, self.snn_neurons)),
                    synapse=self.tau, learning_rule_type=nengo.PES()
                )
            
            probe = nengo.Probe(output, synapse=None)
        
        return model, action_node, state_node, adaptive, conn, probe
    
    def _build_learning_model(self):
        m, self.learn_action, self.learn_state, self.learn_adaptive, self.learn_conn, self.learn_probe = self._build_model(True)
        self.learn_model = m
        self.learn_sim = self._create_simulator(m)
    
    def _build_prediction_model(self):
        m, self.pred_action, self.pred_state, self.pred_adaptive, self.pred_conn, self.pred_probe = self._build_model(False)
        self.pred_model = m
        self.pred_sim = self._create_simulator(m)
    
    def _create_simulator(self, model):
        if self.use_gpu and self.snn_neurons >= 500 and NENGO_DL_AVAILABLE:
            try:
                import tensorflow as tf
                if tf.config.list_physical_devices('GPU'):
                    return nengo_dl.Simulator(model, dt=self.nengo_dt, progress_bar=False, device="/gpu:0")
            except:
                pass
        return nengo.Simulator(model, dt=self.nengo_dt, progress_bar=False)
    
    def _copy_weights(self):
        """Copy learned weights to prediction model."""
        try:
            w = self.learn_sim.signals[self.learn_sim.model.sig[self.learn_conn]['weights']]
            self.pred_sim.signals[self.pred_sim.model.sig[self.pred_conn]['weights']] = w.copy()
        except:
            pass
    
    def _predict_one_step(self, state, action):
        """Predict next state using SNN."""
        self.pred_action.value = action * self.action_scale
        self.pred_state.value = state / self.max_stat
        self.pred_sim.run(self.nengo_dt, progress_bar=False)
        
        if len(self.pred_sim.data[self.pred_probe]) > 0:
            dynamics = self.pred_sim.data[self.pred_probe][-1] / self.dynamics_scale
        else:
            dynamics = np.zeros(self.dynamics_size)
        return state + dynamics
    
    def _predict_states(self, state, actions):
        """Predict states over horizon."""
        states = []
        
        if self.learning_enabled:
            self.pred_sim.reset()
            self._copy_weights()
            for t in range(self.horizon):
                action = actions[t*self.local_action_size : (t+1)*self.local_action_size]
                state = self._predict_one_step(state, action)
                states.append(state.copy())
        else:
            for t in range(self.horizon):
                action = actions[t*self.local_action_size : (t+1)*self.local_action_size]
                group_states = state.reshape(self.num_groups, 4)
                state = self._step(group_states, action).flatten()
                states.append(state.copy())
        return states
    
    def _cost_function(self, u, current_state, targets, prev_action):
        """MPC cost: tracking + control + smoothness."""
        predicted = self._predict_states(current_state, u)
        cost = 0.0
        
        for t, state in enumerate(predicted):
            for g in range(self.num_groups):
                pos = state[g*4 : g*4+2]
                tgt = targets[t][g*2 : g*2+2]
                cost += self.Q * np.sum((pos - tgt)**2)
            
            action = u[t*self.local_action_size : (t+1)*self.local_action_size]
            cost += self.R * np.sum(action**2)
            
            prev = prev_action if t == 0 else u[(t-1)*self.local_action_size : t*self.local_action_size]
            cost += 0.1 * np.sum((action - prev)**2)
        
        return cost
    
    def _run_learning(self, state, action, actual_dynamics):
        """Update SNN with observed dynamics."""
        self.learn_action.value = action * self.action_scale
        self.learn_state.value = state / self.max_stat
        
        if len(self.learn_sim.data[self.learn_probe]) > 0:
            predicted = self.learn_sim.data[self.learn_probe][-1]
        else:
            predicted = np.zeros(self.dynamics_size)
        
        self.learn_error = predicted - actual_dynamics * self.dynamics_scale
        self.learn_sim.run(self.nengo_dt, progress_bar=False)
    
    def compute_control(self, state_dict, target_func):
        """Compute MPC control for all groups."""
        centroids = state_dict['group_centroids']
        velocities = state_dict['group_velocities']
        targets = state_dict['group_targets']
        initial = state_dict['initial_group_centroids']
        
        # Build state vector
        current_state = np.zeros(self.local_state_size)
        for g in range(self.num_groups):
            rel = centroids[g] - initial[g]
            current_state[g*4 : g*4+4] = [rel[0], rel[1], velocities[g, 0], velocities[g, 1]]
        
        # Learning step
        if self.learning_enabled and self.last_state is not None and self.last_action is not None:
            self._run_learning(self.last_state, self.last_action, current_state - self.last_state)
        
        # Target sequence
        target_seq = []
        for _ in range(self.horizon):
            t = np.zeros(self.num_groups * 2)
            for g in range(self.num_groups):
                t[g*2 : g*2+2] = targets[g] - initial[g]
            target_seq.append(t)
        
        prev = self.last_action if self.last_action is not None else np.zeros(self.local_action_size)
        
        # Optimize
        result = minimize(
            self._cost_function, self.prev_u,
            args=(current_state, target_seq, prev),
            bounds=[(-self.u_max, self.u_max)] * len(self.prev_u),
            method='SLSQP',
            options={'maxiter': 20, 'ftol': 1e-4}
        )
        
        self.prev_u = result.x
        u = result.x[:self.local_action_size].astype(np.float32)
        
        if not np.isfinite(u).all():
            u = np.zeros(self.local_action_size, dtype=np.float32)
        
        self.last_state = current_state.copy()
        self.last_action = u.copy()
        self.step_count += 1
        
        return u
    
    def reset(self):
        super().reset()
        self.last_state = None
        self.last_action = None
        self.learn_error = np.zeros(self.dynamics_size)
        self.prev_u = np.zeros(self.local_action_size * self.horizon)
        self.learn_sim.reset()
        self.pred_sim.reset()
    
    def close(self):
        try: self.learn_sim.close()
        except: pass
        try: self.pred_sim.close()
        except: pass
    
    def __del__(self):
        self.close()
    
    def __str__(self):
        return f"NengoMPC({self.snn_neurons}n, h={self.horizon}, learn={self.learning_enabled})"
