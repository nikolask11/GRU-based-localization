#====================================================================
#2D Robot Localization: Particle Filter vs Neural Network Estimator
#====================================================================


import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List

# =============================================================================
# Configuration
# =============================================================================

# World parameters
WORLD_SIZE = 20.0  # 20x20 unit world

# Landmark positions (8 fixed landmarks)
LANDMARKS = np.array([
    [4.0, 4.0],
    [4.0, 16.0],
    [16.0, 4.0],
    [16.0, 16.0],
    [10.0, 2.0],
    [10.0, 18.0],
    [2.0, 10.0],
    [18.0, 10.0],
])
NUM_LANDMARKS = len(LANDMARKS)

# Motion noise standard deviations
SIGMA_V = 0.1      # Linear velocity noise
SIGMA_OMEGA = 0.05  # Angular velocity noise

# Sensor noise standard deviations
SIGMA_RANGE = 0.5    # Range measurement noise
SIGMA_BEARING = 0.1  # Bearing measurement noise

# Particle filter parameters
NUM_PARTICLES = 500

# Simulation parameters
TIMESTEPS = 30  # Shorter sequences for better NN training
DT = 1.0  # Time step (abstract units)

# Velocity bounds (for normalization)
MAX_VELOCITY = 1.0      # Maximum linear velocity
MAX_ANGULAR_VEL = 0.5   # Maximum angular velocity

# Neural network training parameters
NUM_TRAINING_TRAJECTORIES = 500  # More trajectories for better generalization
HIDDEN_SIZE = 64
NUM_EPOCHS = 100  # More epochs for convergence
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def normalize_angle_array(angles: np.ndarray) -> np.ndarray:
    """Normalize array of angles to [-pi, pi] range."""
    return np.arctan2(np.sin(angles), np.cos(angles))


def angular_difference(angle1: float, angle2: float) -> float:
    """Compute the smallest angular difference between two angles."""
    diff = angle1 - angle2
    return abs(normalize_angle(diff))


# =============================================================================
# Robot Class
# =============================================================================

class Robot:
    """
    Simulates a robot in a 2D world.
    
    State: (x, y, theta) where theta is heading in radians.
    The robot senses range and bearing to landmarks.
    """
    
    def __init__(self, x: float = None, y: float = None, theta: float = None):
        """Initialize robot at given pose or random pose."""
        self.x = x if x is not None else np.random.uniform(0, WORLD_SIZE)
        self.y = y if y is not None else np.random.uniform(0, WORLD_SIZE)
        self.theta = theta if theta is not None else np.random.uniform(-np.pi, np.pi)
    
    def move(self, v: float, omega: float, add_noise: bool = True) -> Tuple[float, float]:
        """
        Move the robot according to velocity commands.
        
        Args:
            v: Linear velocity
            omega: Angular velocity
            add_noise: Whether to add motion noise
            
        Returns:
            Tuple of (actual_v, actual_omega) used for movement
        """
        # Add motion noise if requested
        if add_noise:
            v_actual = v + np.random.normal(0, SIGMA_V)
            omega_actual = omega + np.random.normal(0, SIGMA_OMEGA)
        else:
            v_actual = v
            omega_actual = omega
        
        # Update pose using motion model
        self.x += v_actual * np.cos(self.theta)
        self.y += v_actual * np.sin(self.theta)
        self.theta = normalize_angle(self.theta + omega_actual)
        
        # Wrap position to stay in world (toroidal world)
        self.x = self.x % WORLD_SIZE
        self.y = self.y % WORLD_SIZE
        
        return v_actual, omega_actual
    
    def sense(self, add_noise: bool = True) -> np.ndarray:
        """
        Sense range and bearing to all landmarks.
        
        Returns:
            Array of shape (NUM_LANDMARKS, 2) with [range, bearing] for each landmark
        """
        measurements = np.zeros((NUM_LANDMARKS, 2))
        
        for i, landmark in enumerate(LANDMARKS):
            # Compute true range and bearing
            dx = landmark[0] - self.x
            dy = landmark[1] - self.y
            
            true_range = np.sqrt(dx**2 + dy**2)
            true_bearing = normalize_angle(np.arctan2(dy, dx) - self.theta)
            
            # Add sensor noise if requested
            if add_noise:
                measured_range = true_range + np.random.normal(0, SIGMA_RANGE)
                measured_bearing = normalize_angle(true_bearing + np.random.normal(0, SIGMA_BEARING))
            else:
                measured_range = true_range
                measured_bearing = true_bearing
            
            measurements[i] = [measured_range, measured_bearing]
        
        return measurements
    
    def get_pose(self) -> np.ndarray:
        """Return current pose as numpy array [x, y, theta]."""
        return np.array([self.x, self.y, self.theta])
    
    def set_pose(self, pose: np.ndarray):
        """Set robot pose from numpy array [x, y, theta]."""
        self.x = pose[0]
        self.y = pose[1]
        self.theta = normalize_angle(pose[2])


# =============================================================================
# Particle Filter (Monte Carlo Localization)
# =============================================================================

class ParticleFilter:
    """
    Monte Carlo Localization using a particle filter.
    
    Each particle represents a hypothesis for the robot's pose (x, y, theta).
    Particles are weighted based on how well they explain sensor observations.
    """
    
    def __init__(self, num_particles: int = NUM_PARTICLES):
        """Initialize particles uniformly across the world."""
        self.num_particles = num_particles
        self.particles = np.zeros((num_particles, 3))  # [x, y, theta]
        self.weights = np.ones(num_particles) / num_particles
        
        # Initialize particles uniformly
        self.particles[:, 0] = np.random.uniform(0, WORLD_SIZE, num_particles)  # x
        self.particles[:, 1] = np.random.uniform(0, WORLD_SIZE, num_particles)  # y
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)  # theta
    
    def predict(self, v: float, omega: float):
        """
        Prediction step: move all particles according to motion model with noise.
        
        Args:
            v: Commanded linear velocity
            omega: Commanded angular velocity
        """
        # Add noise to controls for each particle
        v_noisy = v + np.random.normal(0, SIGMA_V, self.num_particles)
        omega_noisy = omega + np.random.normal(0, SIGMA_OMEGA, self.num_particles)
        
        # Update particle positions
        self.particles[:, 0] += v_noisy * np.cos(self.particles[:, 2])
        self.particles[:, 1] += v_noisy * np.sin(self.particles[:, 2])
        self.particles[:, 2] += omega_noisy
        
        # Normalize angles
        self.particles[:, 2] = np.arctan2(
            np.sin(self.particles[:, 2]),
            np.cos(self.particles[:, 2])
        )
        
        # Wrap positions to world bounds
        self.particles[:, 0] = self.particles[:, 0] % WORLD_SIZE
        self.particles[:, 1] = self.particles[:, 1] % WORLD_SIZE
    
    def update(self, measurements: np.ndarray):
        """
        Update step: compute particle weights based on sensor measurements.
        
        Args:
            measurements: Array of shape (NUM_LANDMARKS, 2) with [range, bearing]
        """
        # Compute likelihood for each particle
        for i in range(self.num_particles):
            weight = 1.0
            
            for j, landmark in enumerate(LANDMARKS):
                # Expected measurement for this particle
                dx = landmark[0] - self.particles[i, 0]
                dy = landmark[1] - self.particles[i, 1]
                expected_range = np.sqrt(dx**2 + dy**2)
                expected_bearing = normalize_angle(
                    np.arctan2(dy, dx) - self.particles[i, 2]
                )
                
                # Actual measurement
                measured_range = measurements[j, 0]
                measured_bearing = measurements[j, 1]
                
                # Compute likelihood using Gaussian
                range_likelihood = np.exp(
                    -0.5 * ((measured_range - expected_range) / SIGMA_RANGE) ** 2
                )
                bearing_diff = normalize_angle(measured_bearing - expected_bearing)
                bearing_likelihood = np.exp(
                    -0.5 * (bearing_diff / SIGMA_BEARING) ** 2
                )
                
                weight *= range_likelihood * bearing_likelihood
            
            self.weights[i] = weight
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def resample(self):
        """
        Resample particles based on weights (low variance resampling).
        """
        # Use systematic resampling for lower variance
        positions = (np.arange(self.num_particles) + np.random.uniform()) / self.num_particles
        cumulative_sum = np.cumsum(self.weights)
        
        indices = np.searchsorted(cumulative_sum, positions)
        indices = np.clip(indices, 0, self.num_particles - 1)
        
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def estimate(self) -> np.ndarray:
        """
        Compute weighted mean estimate of robot pose.
        
        Returns:
            Estimated pose [x, y, theta]
        """
        # Weighted mean for x and y
        x_est = np.sum(self.weights * self.particles[:, 0])
        y_est = np.sum(self.weights * self.particles[:, 1])
        
        # Circular mean for theta
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        theta_est = np.arctan2(sin_sum, cos_sum)
        
        return np.array([x_est, y_est, theta_est])


# =============================================================================
# Neural Network Estimator (GRU-based)
# =============================================================================

class LocalizationGRU(nn.Module):
    """
    GRU-based neural network for robot localization.
    
    NN CHANGES:
    ---------------
    1. GRU over LSTM: Simpler architecture with fewer parameters. GRUs have
       comparable performance to LSTMs on many tasks and train more stably
       with limited data.
       
    2. Single layer: Avoids overfitting on this relatively simple task.
       Deep networks need more data to generalize well.
       
    3. Final hidden state only: The network is trained as a filter, outputting
       only the final pose estimate. This is more aligned with the localization
       task (estimate current pose given history) vs sequence-to-sequence
       prediction which is harder.
       
    4. Sin/cos angle representation: Outputs (sin θ, cos θ) instead of θ directly.
       This avoids discontinuity at ±π and allows proper angle learning.
    """
    
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE):
        """
        Initialize the GRU network.
        
        Args:
            input_size: Size of input features (controls + flattened sensor readings)
            hidden_size: Number of GRU hidden units
        """
        super(LocalizationGRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Single-layer GRU (simpler, more stable for this task)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Output layers - separate heads for position and angle
        self.fc_position = nn.Linear(hidden_size, 2)  # x, y (normalized)
        self.fc_angle = nn.Linear(hidden_size, 2)     # sin(theta), cos(theta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - predicts only the final pose.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            predictions: Tensor of shape (batch, 3) with [x_norm, y_norm, theta]
                        Position outputs are normalized [0, 1], theta in radians
        """
        # GRU forward - we only use the final hidden state
        _, h_n = self.gru(x)  # h_n: (1, batch, hidden_size)
        
        # Get the last hidden state
        final_hidden = h_n.squeeze(0)  # (batch, hidden_size)
        
        # Predict normalized position (x, y in [0, 1])
        position = torch.sigmoid(self.fc_position(final_hidden))  # (batch, 2)
        
        # Predict angle as sin/cos components
        angle_components = self.fc_angle(final_hidden)  # (batch, 2)
        
        # Normalize to unit circle and convert to angle
        # This ensures valid sin/cos values
        norm = torch.norm(angle_components, dim=1, keepdim=True).clamp(min=1e-6)
        angle_normalized = angle_components / norm
        theta = torch.atan2(angle_normalized[:, 0], angle_normalized[:, 1])  # (batch,)
        
        # Combine predictions: [x_norm, y_norm, theta]
        predictions = torch.cat([position, theta.unsqueeze(-1)], dim=-1)  # (batch, 3)
        
        return predictions


def compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute combined loss for position and orientation.
    
    LOSS DESIGN:
    ------------
    - Position: MSE loss on normalized coordinates
    - Angle: Cosine-based loss that handles wraparound correctly
      Loss = 1 - cos(pred_θ - target_θ)
      This is 0 when angles match, and increases smoothly as they diverge,
      with maximum at π radians difference.
    
    Args:
        predictions: Predicted poses (batch, 3) - [x_norm, y_norm, theta]
        targets: Ground truth poses (batch, 3) - [x_norm, y_norm, theta]
        
    Returns:
        Combined loss value
    """
    # Position loss (MSE on normalized coordinates)
    position_loss = nn.MSELoss()(predictions[:, :2], targets[:, :2])
    
    # Angular loss (cosine-based for proper wraparound handling)
    pred_theta = predictions[:, 2]
    target_theta = targets[:, 2]
    
    # cos(angle_diff) is 1 when angles match, -1 when opposite
    # Loss = 1 - cos(diff) is 0 when match, 2 when opposite
    angle_diff = pred_theta - target_theta
    angle_loss = torch.mean(1 - torch.cos(angle_diff))
    
    # Combined loss - weight angle loss less since position is primary concern
    # and angle has different scale (0-2 vs ~0-0.5 for position MSE)
    total_loss = position_loss + 0.1 * angle_loss
    
    return total_loss


# =============================================================================
# Data Generation with Normalization
# =============================================================================

def normalize_inputs(controls: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Normalize all inputs to similar scales for better training.
    
    NORMALIZATION SCHEME:
    ---------------------
    - Linear velocity v: divide by MAX_VELOCITY → [0, ~1]
    - Angular velocity ω: divide by MAX_ANGULAR_VEL → [-1, ~1]
    - Range measurements: divide by WORLD_SIZE → [0, ~1.5] (max range ≈ √2 * WORLD_SIZE)
    - Bearing measurements: kept in radians [-π, π] (already bounded)
    
    This prevents features with larger magnitudes from dominating the learning.
    
    Args:
        controls: (timesteps, 2) array of [v, omega]
        observations: (timesteps, NUM_LANDMARKS * 2) array of [range, bearing, ...]
        
    Returns:
        Normalized input array (timesteps, input_size)
    """
    # Normalize controls
    v_norm = controls[:, 0:1] / MAX_VELOCITY
    omega_norm = controls[:, 1:2] / MAX_ANGULAR_VEL
    
    # Reshape observations for easier processing
    obs_reshaped = observations.reshape(-1, NUM_LANDMARKS, 2)
    
    # Normalize ranges (divide by world size)
    ranges_norm = obs_reshaped[:, :, 0] / WORLD_SIZE
    
    # Bearings stay in radians (already in [-π, π])
    bearings = obs_reshaped[:, :, 1]
    
    # Flatten back
    obs_norm = np.concatenate([
        ranges_norm.reshape(-1, NUM_LANDMARKS),
        bearings.reshape(-1, NUM_LANDMARKS)
    ], axis=1)
    
    # Combine all
    return np.concatenate([v_norm, omega_norm, obs_norm], axis=1)


def normalize_target(pose: np.ndarray) -> np.ndarray:
    """
    Normalize target pose for training.
    
    Args:
        pose: [x, y, theta]
        
    Returns:
        Normalized pose [x/WORLD_SIZE, y/WORLD_SIZE, theta]
    """
    return np.array([
        pose[0] / WORLD_SIZE,
        pose[1] / WORLD_SIZE,
        pose[2]  # theta stays in radians
    ])


def denormalize_pose(normalized_pose: np.ndarray) -> np.ndarray:
    """
    Convert normalized pose back to world coordinates.
    
    Args:
        normalized_pose: [x_norm, y_norm, theta]
        
    Returns:
        World pose [x, y, theta]
    """
    return np.array([
        normalized_pose[0] * WORLD_SIZE,
        normalized_pose[1] * WORLD_SIZE,
        normalized_pose[2]
    ])


def generate_trajectory(timesteps: int = TIMESTEPS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single robot trajectory with controls and sensor readings.
    
    Returns:
        Tuple of:
        - controls: Array of shape (timesteps, 2) with [v, omega]
        - observations: Array of shape (timesteps, NUM_LANDMARKS * 2)
        - ground_truth: Array of shape (timesteps, 3) with [x, y, theta]
    """
    robot = Robot()
    
    controls = np.zeros((timesteps, 2))
    observations = np.zeros((timesteps, NUM_LANDMARKS * 2))
    ground_truth = np.zeros((timesteps, 3))
    
    for t in range(timesteps):
        # Generate smooth control commands
        # Velocity varies smoothly over time
        v = 0.5 + 0.3 * np.sin(t * 0.2)
        omega = 0.3 * np.sin(t * 0.15) + 0.1 * np.cos(t * 0.1)
        
        # Clamp to valid ranges
        v = np.clip(v, 0, MAX_VELOCITY)
        omega = np.clip(omega, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        
        controls[t] = [v, omega]
        
        # Get sensor observations
        measurements = robot.sense(add_noise=True)
        observations[t] = measurements.flatten()
        
        # Store ground truth
        ground_truth[t] = robot.get_pose()
        
        # Move robot
        robot.move(v, omega, add_noise=True)
    
    return controls, observations, ground_truth


def generate_training_data(num_trajectories: int = NUM_TRAINING_TRAJECTORIES) -> Tuple[
    np.ndarray, np.ndarray
]:
    """
    Generate training data from multiple trajectories.
    
    The network is trained to predict ONLY THE FINAL POSE given the
    sequence of controls and observations. This is the key change from
    sequence-to-sequence prediction.
    
    Returns:
        Tuple of:
        - inputs: Array of shape (num_trajectories, timesteps, input_size) - NORMALIZED
        - targets: Array of shape (num_trajectories, 3) - NORMALIZED final poses
    """
    # Calculate input size: v, omega, NUM_LANDMARKS ranges, NUM_LANDMARKS bearings
    input_size = 2 + NUM_LANDMARKS * 2
    
    all_inputs = np.zeros((num_trajectories, TIMESTEPS, input_size))
    all_targets = np.zeros((num_trajectories, 3))  # Only final pose!
    
    print(f"Generating {num_trajectories} training trajectories...")
    print(f"  Sequence length: {TIMESTEPS} timesteps")
    print(f"  Target: final pose only (not full sequence)")
    
    for i in range(num_trajectories):
        controls, observations, ground_truth = generate_trajectory()
        
        # Normalize inputs
        all_inputs[i] = normalize_inputs(controls, observations)
        
        # Target is ONLY the final pose, normalized
        all_targets[i] = normalize_target(ground_truth[-1])
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_trajectories} trajectories")
    
    return all_inputs, all_targets


# =============================================================================
# Training
# =============================================================================

def train_neural_network(
    inputs: np.ndarray,
    targets: np.ndarray,
    num_epochs: int = NUM_EPOCHS
) -> LocalizationGRU:
    """
    Train the GRU localization network.
    
    Args:
        inputs: Training inputs (num_samples, seq_len, input_size) - normalized
        targets: Training targets (num_samples, 3) - normalized final poses
        num_epochs: Number of training epochs
        
    Returns:
        Trained model
    """
    # Convert to tensors
    X = torch.FloatTensor(inputs)
    Y = torch.FloatTensor(targets)
    
    # Split into train/validation
    n_train = int(0.9 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]
    
    # Create data loader
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    input_size = inputs.shape[2]
    model = LocalizationGRU(input_size=input_size)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nTraining neural network for {num_epochs} epochs...")
    print(f"  Architecture: GRU (single layer, {HIDDEN_SIZE} hidden units)")
    print(f"  Input size: {input_size}")
    print(f"  Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Objective: Predict final pose only")
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            
            predictions = model(batch_X)
            loss = compute_loss(predictions, batch_Y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = compute_loss(val_pred, Y_val).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{num_epochs}, "
                  f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    
    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_methods(
    model: LocalizationGRU,
    num_eval_trajectories: int = 10
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Evaluate both Particle Filter and Neural Network on test trajectories.
    
    For fair comparison, both methods estimate the pose at each timestep:
    - PF: Updates incrementally as usual
    - NN: Uses a sliding window of the last TIMESTEPS observations
    
    Args:
        model: Trained GRU model
        num_eval_trajectories: Number of trajectories to evaluate on
        
    Returns:
        Lists of error arrays for each trajectory:
        - pf_position_errors
        - pf_angle_errors  
        - nn_position_errors
        - nn_angle_errors
    """
    model.eval()
    
    all_pf_pos_errors = []
    all_pf_ang_errors = []
    all_nn_pos_errors = []
    all_nn_ang_errors = []
    
    # For evaluation, use longer trajectories
    eval_timesteps = TIMESTEPS * 3
    
    print(f"\nEvaluating on {num_eval_trajectories} test trajectories...")
    print(f"  Evaluation trajectory length: {eval_timesteps} timesteps")
    print(f"  NN uses sliding window of {TIMESTEPS} steps for each estimate")
    
    for traj_idx in range(num_eval_trajectories):
        # Generate longer test trajectory
        controls, observations, ground_truth = generate_trajectory(eval_timesteps)
        
        # Initialize particle filter
        pf = ParticleFilter()
        
        # Arrays to store errors
        pf_pos_errors = np.zeros(eval_timesteps)
        pf_ang_errors = np.zeros(eval_timesteps)
        nn_pos_errors = np.zeros(eval_timesteps)
        nn_ang_errors = np.zeros(eval_timesteps)
        
        # Run particle filter step by step
        for t in range(eval_timesteps):
            v, omega = controls[t]
            measurements = observations[t].reshape(NUM_LANDMARKS, 2)
            true_pose = ground_truth[t]
            
            # Particle filter update
            if t > 0:
                pf.predict(v, omega)
            pf.update(measurements)
            pf.resample()
            
            pf_estimate = pf.estimate()
            
            # Neural network estimate using sliding window
            if t >= TIMESTEPS - 1:
                # Use last TIMESTEPS steps
                start_idx = t - TIMESTEPS + 1
                window_controls = controls[start_idx:t+1]
                window_obs = observations[start_idx:t+1]
                
                # Normalize and predict
                nn_input = normalize_inputs(window_controls, window_obs)
                nn_input_tensor = torch.FloatTensor(nn_input).unsqueeze(0)
                
                with torch.no_grad():
                    nn_pred_norm = model(nn_input_tensor).squeeze(0).numpy()
                
                nn_estimate = denormalize_pose(nn_pred_norm)
            else:
                # Not enough history yet - pad with zeros
                # This gives the NN a disadvantage early on (fair, as it needs history)
                padded_controls = np.zeros((TIMESTEPS, 2))
                padded_obs = np.zeros((TIMESTEPS, NUM_LANDMARKS * 2))
                padded_controls[-(t+1):] = controls[:t+1]
                padded_obs[-(t+1):] = observations[:t+1]
                
                nn_input = normalize_inputs(padded_controls, padded_obs)
                nn_input_tensor = torch.FloatTensor(nn_input).unsqueeze(0)
                
                with torch.no_grad():
                    nn_pred_norm = model(nn_input_tensor).squeeze(0).numpy()
                
                nn_estimate = denormalize_pose(nn_pred_norm)
            
            # Compute errors
            pf_pos_errors[t] = np.sqrt(
                (pf_estimate[0] - true_pose[0])**2 +
                (pf_estimate[1] - true_pose[1])**2
            )
            pf_ang_errors[t] = angular_difference(pf_estimate[2], true_pose[2])
            
            nn_pos_errors[t] = np.sqrt(
                (nn_estimate[0] - true_pose[0])**2 +
                (nn_estimate[1] - true_pose[1])**2
            )
            nn_ang_errors[t] = angular_difference(nn_estimate[2], true_pose[2])
        
        all_pf_pos_errors.append(pf_pos_errors)
        all_pf_ang_errors.append(pf_ang_errors)
        all_nn_pos_errors.append(nn_pos_errors)
        all_nn_ang_errors.append(nn_ang_errors)
        
        print(f"  Trajectory {traj_idx + 1}: "
              f"PF pos err={np.mean(pf_pos_errors):.3f}, "
              f"NN pos err={np.mean(nn_pos_errors):.3f}")
    
    return all_pf_pos_errors, all_pf_ang_errors, all_nn_pos_errors, all_nn_ang_errors


def plot_results(
    pf_pos_errors: List[np.ndarray],
    pf_ang_errors: List[np.ndarray],
    nn_pos_errors: List[np.ndarray],
    nn_ang_errors: List[np.ndarray]
):
    """
    Plot comparison of estimation errors over time.
    
    Creates a figure with position and orientation error plots.
    """
    eval_timesteps = len(pf_pos_errors[0])
    
    # Average errors across trajectories
    avg_pf_pos = np.mean(pf_pos_errors, axis=0)
    avg_pf_ang = np.mean(pf_ang_errors, axis=0)
    avg_nn_pos = np.mean(nn_pos_errors, axis=0)
    avg_nn_ang = np.mean(nn_ang_errors, axis=0)
    
    # Standard deviation for confidence bands
    std_pf_pos = np.std(pf_pos_errors, axis=0)
    std_nn_pos = np.std(nn_pos_errors, axis=0)
    
    timesteps = np.arange(eval_timesteps)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Position error plot
    ax1 = axes[0]
    ax1.plot(timesteps, avg_pf_pos, 'b-', label='Particle Filter', linewidth=2)
    ax1.fill_between(timesteps, 
                     np.maximum(0, avg_pf_pos - std_pf_pos), 
                     avg_pf_pos + std_pf_pos,
                     alpha=0.2, color='blue')
    ax1.plot(timesteps, avg_nn_pos, 'r-', label='Neural Network (GRU)', linewidth=2)
    ax1.fill_between(timesteps, 
                     np.maximum(0, avg_nn_pos - std_nn_pos), 
                     avg_nn_pos + std_nn_pos,
                     alpha=0.2, color='red')
    ax1.axvline(x=TIMESTEPS-1, color='gray', linestyle='--', alpha=0.5, 
                label=f'NN full history ({TIMESTEPS} steps)')
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Position Error (units)', fontsize=12)
    ax1.set_title('Position Error vs Time', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, eval_timesteps - 1)
    
    # Orientation error plot
    ax2 = axes[1]
    ax2.plot(timesteps, np.degrees(avg_pf_ang), 'b-', label='Particle Filter', linewidth=2)
    ax2.plot(timesteps, np.degrees(avg_nn_ang), 'r-', label='Neural Network (GRU)', linewidth=2)
    ax2.axvline(x=TIMESTEPS-1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Orientation Error (degrees)', fontsize=12)
    ax2.set_title('Orientation Error vs Time', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, eval_timesteps - 1)
    
    plt.tight_layout()
    plt.savefig('localization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to 'localization_comparison.png'")


def plot_sample_trajectory(model: LocalizationGRU):
    """
    Plot a sample trajectory showing ground truth, PF estimate, and NN estimate.
    """
    model.eval()
    
    # Generate longer trajectory for visualization
    eval_timesteps = TIMESTEPS * 3
    controls, observations, ground_truth = generate_trajectory(eval_timesteps)
    
    # Run particle filter
    pf = ParticleFilter()
    pf_estimates = np.zeros((eval_timesteps, 3))
    nn_estimates = np.zeros((eval_timesteps, 3))
    
    for t in range(eval_timesteps):
        v, omega = controls[t]
        measurements = observations[t].reshape(NUM_LANDMARKS, 2)
        
        if t > 0:
            pf.predict(v, omega)
        pf.update(measurements)
        pf.resample()
        pf_estimates[t] = pf.estimate()
        
        # NN estimate
        if t >= TIMESTEPS - 1:
            start_idx = t - TIMESTEPS + 1
            window_controls = controls[start_idx:t+1]
            window_obs = observations[start_idx:t+1]
        else:
            padded_controls = np.zeros((TIMESTEPS, 2))
            padded_obs = np.zeros((TIMESTEPS, NUM_LANDMARKS * 2))
            padded_controls[-(t+1):] = controls[:t+1]
            padded_obs[-(t+1):] = observations[:t+1]
            window_controls = padded_controls
            window_obs = padded_obs
        
        nn_input = normalize_inputs(window_controls, window_obs)
        nn_input_tensor = torch.FloatTensor(nn_input).unsqueeze(0)
        
        with torch.no_grad():
            nn_pred_norm = model(nn_input_tensor).squeeze(0).numpy()
        nn_estimates[t] = denormalize_pose(nn_pred_norm)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot landmarks
    ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1], 
               c='green', s=200, marker='^', label='Landmarks', zorder=5)
    
    # Plot trajectories
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], 
            'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(pf_estimates[:, 0], pf_estimates[:, 1], 
            'b--', linewidth=2, label='Particle Filter', alpha=0.8)
    ax.plot(nn_estimates[:, 0], nn_estimates[:, 1], 
            'r--', linewidth=2, label='Neural Network (GRU)', alpha=0.8)
    
    # Mark start and end
    ax.scatter([ground_truth[0, 0]], [ground_truth[0, 1]], 
               c='black', s=150, marker='o', zorder=6, label='Start')
    ax.scatter([ground_truth[-1, 0]], [ground_truth[-1, 1]], 
               c='black', s=150, marker='x', zorder=6, label='End')
    
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_xlabel('X (units)', fontsize=12)
    ax.set_ylabel('Y (units)', fontsize=12)
    ax.set_title('Sample Trajectory Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Trajectory plot saved to 'trajectory_comparison.png'")


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Main function that runs the complete simulation.
    
    1. Generates training data
    2. Trains the neural network
    3. Evaluates both methods
    4. Plots comparison results
    """
    print("=" * 70)
    print("2D Robot Localization: Particle Filter vs Neural Network (GRU)")
    print("=" * 70)
    
    print(f"\nWorld Configuration:")
    print(f"  World size: {WORLD_SIZE} x {WORLD_SIZE} units")
    print(f"  Number of landmarks: {NUM_LANDMARKS}")
    print(f"  Training sequence length: {TIMESTEPS} timesteps")
    print(f"\nNoise Parameters:")
    print(f"  Motion noise (v): σ = {SIGMA_V}")
    print(f"  Motion noise (ω): σ = {SIGMA_OMEGA}")
    print(f"  Sensor noise (range): σ = {SIGMA_RANGE}")
    print(f"  Sensor noise (bearing): σ = {SIGMA_BEARING}")
    
    print("\n" + "=" * 70)
    print("KEY IMPROVEMENTS IN THIS VERSION:")
    print("=" * 70)
    print("1. Final-pose prediction only (not sequence-to-sequence)")
    print("2. Input/output normalization for stable gradients")
    print("3. GRU instead of LSTM (simpler, more stable)")
    print("4. Cosine-based angular loss for proper angle handling")
    print("5. More training data with shorter sequences")
    
    # Step 1: Generate training data
    print("\n" + "-" * 40)
    print("STEP 1: Generating Training Data")
    print("-" * 40)
    train_inputs, train_targets = generate_training_data(NUM_TRAINING_TRAJECTORIES)
    
    # Step 2: Train neural network
    print("\n" + "-" * 40)
    print("STEP 2: Training Neural Network")
    print("-" * 40)
    model = train_neural_network(train_inputs, train_targets, NUM_EPOCHS)
    
    # Step 3: Evaluate both methods
    print("\n" + "-" * 40)
    print("STEP 3: Evaluating Methods")
    print("-" * 40)
    pf_pos, pf_ang, nn_pos, nn_ang = evaluate_methods(model, num_eval_trajectories=10)
    
    # Print summary statistics
    print("\n" + "-" * 40)
    print("RESULTS SUMMARY")
    print("-" * 40)
    
    avg_pf_pos = np.mean([np.mean(e) for e in pf_pos])
    avg_nn_pos = np.mean([np.mean(e) for e in nn_pos])
    avg_pf_ang = np.mean([np.mean(e) for e in pf_ang])
    avg_nn_ang = np.mean([np.mean(e) for e in nn_ang])
    
    print(f"\nAverage Position Error:")
    print(f"  Particle Filter: {avg_pf_pos:.3f} units")
    print(f"  Neural Network:  {avg_nn_pos:.3f} units")
    
    print(f"\nAverage Orientation Error:")
    print(f"  Particle Filter: {np.degrees(avg_pf_ang):.2f} degrees")
    print(f"  Neural Network:  {np.degrees(avg_nn_ang):.2f} degrees")
    
    # Analysis
    print("\n" + "-" * 40)
    print("ANALYSIS")
    print("-" * 40)
    if avg_pf_pos < avg_nn_pos:
        ratio = avg_nn_pos / avg_pf_pos
        print(f"\nThe Particle Filter outperforms the Neural Network by {ratio:.1f}x")
        print("This is expected because:")
        print("  - PF has access to the exact sensor noise model")
        print("  - PF updates incrementally with each observation")
        print("  - NN must learn the motion/sensor model from data")
        print("  - NN has a limited context window")
    else:
        print("\nThe Neural Network matches or outperforms the Particle Filter!")
        print("This suggests the NN has learned the underlying dynamics well.")
    
    # Step 4: Plot results
    print("\n" + "-" * 40)
    print("STEP 4: Generating Plots")
    print("-" * 40)
    
    plot_results(pf_pos, pf_ang, nn_pos, nn_ang)
    plot_sample_trajectory(model)
    
    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - localization_comparison.png: Error comparison over time")
    print("  - trajectory_comparison.png: Sample trajectory visualization")


if __name__ == "__main__":
    main()
