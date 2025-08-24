"""
CTDE (Centralized Training Decentralized Execution) Policy for Soccer AI

This module implements a custom policy architecture for multi-agent soccer training
that supports centralized training with decentralized execution. Each agent has its
own actor network but shares parameters, while a centralized critic observes the
global game state.

**Key Features**:
- Individual actor networks for each agent (decentralized execution)
- Shared parameters across agents for sample efficiency
- Centralized critic with global observations
- Role-based feature processing
- Compatible with stable-baselines3 PPO
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution
from gymnasium import spaces


class CTDEFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for CTDE that processes agent-specific observations.
    Handles role embeddings and agent identity features.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Expected observation dimension per agent (68 dims)
        # The input will be individual agent observations after reshaping
        self.obs_dim = 68
        
        # Network layers for feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.feature_net(observations)


class CTDEActorNetwork(nn.Module):
    """
    Individual actor network for each agent in CTDE architecture.
    Processes agent-specific observations and outputs action probabilities.
    """
    
    def __init__(self, features_dim: int, action_dim: int = 7):
        super().__init__()
        
        self.action_net = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.action_net(features)


class CTDECriticNetwork(nn.Module):
    """
    Centralized critic network that processes global observations.
    Used during training to provide centralized value estimates.
    """
    
    def __init__(self, global_obs_dim: int = 102):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(global_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, global_observations: torch.Tensor) -> torch.Tensor:
        return self.value_net(global_observations)


class CTDEPolicy(ActorCriticPolicy):
    """
    CTDE Policy implementation compatible with stable-baselines3.
    
    **Architecture**:
    - 5 individual actor networks (one per agent) with shared parameters
    - 1 centralized critic network with global observations
    - Custom observation processing for agent-specific features
    
    **Training**: Uses centralized critic for value estimation
    **Execution**: Uses decentralized actors for action selection
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.MultiDiscrete,
        lr_schedule,
        net_arch: List[int] = None,
        activation_fn = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class = CTDEFeaturesExtractor,
        features_extractor_kwargs = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs = None,
        n_agents: int = 5,
        global_obs_dim: int = 102,
    ):
        # Store CTDE-specific parameters
        self.n_agents = n_agents
        self.global_obs_dim = global_obs_dim
        # For flattened observations: total_dim = n_agents * individual_agent_obs_dim
        self.total_obs_dim = observation_space.shape[0]
        self.agent_obs_dim = self.total_obs_dim // n_agents  # Should be 68
        
        # Initialize parent class with all required parameters
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs or {},
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        """Build the feature extractor and networks."""
        
        # The features_extractor is already created by parent class
        features_dim = self.features_extractor.features_dim
        
        # Individual actor networks (shared parameters)
        self.actor_net = CTDEActorNetwork(features_dim, action_dim=7)
        
        # Centralized critic network
        self.critic_net = CTDECriticNetwork(self.global_obs_dim)
        
        # Action distribution
        self.action_dist = CategoricalDistribution(7)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for CTDE policy.
        
        Args:
            obs: Agent observations [batch_size, obs_dim] or [batch_size, n_agents, obs_dim]
            deterministic: Whether to use deterministic actions
            
        Returns:
            actions: Selected actions for each agent
            values: Value estimates (requires global observations)
            log_probs: Log probabilities of selected actions
        """
        
        # Process flattened agent observations
        batch_size = obs.shape[0]
        # Reshape flattened observations back to [batch_size, n_agents, agent_obs_dim]
        obs_reshaped = obs.reshape(batch_size, self.n_agents, self.agent_obs_dim)
        # Flatten for processing: [batch_size * n_agents, agent_obs_dim]
        obs_flat = obs_reshaped.reshape(-1, self.agent_obs_dim)
        
        # Extract features
        features = self.features_extractor(obs_flat)
        
        # Get action logits for all agents
        action_logits = self.actor_net(features)
        
        # Reshape back to [batch_size, n_agents, action_dim]
        action_logits = action_logits.reshape(batch_size, self.n_agents, -1)
        
        # Sample actions
        self.action_dist.proba_distribution(action_logits)
        actions = self.action_dist.get_actions(deterministic=deterministic)
        log_probs = self.action_dist.log_prob(actions)
        
        # For values, we need global observations (handled separately in training)
        values = torch.zeros(batch_size, 1)  # Placeholder - will be computed with global obs
        
        return actions, values, log_probs
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, global_obs: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training (used by PPO).
        
        Args:
            obs: Agent observations
            actions: Actions to evaluate  
            global_obs: Global observations for critic
            
        Returns:
            values: Value estimates from centralized critic
            log_probs: Log probabilities of actions
            entropy: Action entropy
        """
        
        # Process flattened agent observations for actors
        batch_size = obs.shape[0]
        # Reshape flattened observations back to [batch_size, n_agents, agent_obs_dim]
        obs_reshaped = obs.reshape(batch_size, self.n_agents, self.agent_obs_dim)
        # Flatten for processing: [batch_size * n_agents, agent_obs_dim]
        obs_flat = obs_reshaped.reshape(-1, self.agent_obs_dim)
        
        # Handle actions - flatten if needed
        if len(actions.shape) == 2:  # [batch_size, n_agents]
            actions_flat = actions.reshape(-1)
        else:
            actions_flat = actions
        
        # Extract features and get action logits
        features = self.features_extractor(obs_flat)
        action_logits = self.actor_net(features)
        
        # Compute log probabilities and entropy
        self.action_dist.proba_distribution(action_logits)
        log_probs = self.action_dist.log_prob(actions_flat)
        entropy = self.action_dist.entropy()
        
        # Compute values using centralized critic
        if global_obs is not None:
            values = self.critic_net(global_obs)
        else:
            # Fallback to zeros if no global observations provided
            values = torch.zeros(batch_size, 1)
        
        return values, log_probs, entropy
    
    def predict_values(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Predict values using centralized critic.
        
        Args:
            global_obs: Global observations for critic
            
        Returns:
            values: Value estimates
        """
        return self.critic_net(global_obs)
    
    def get_distribution(self, obs: torch.Tensor) -> CategoricalDistribution:
        """Get action distribution for given observations."""
        
        # Process flattened observations
        batch_size = obs.shape[0]
        # Reshape flattened observations back to [batch_size, n_agents, agent_obs_dim]
        obs_reshaped = obs.reshape(batch_size, self.n_agents, self.agent_obs_dim)
        # Flatten for processing: [batch_size * n_agents, agent_obs_dim]
        obs_flat = obs_reshaped.reshape(-1, self.agent_obs_dim)
        
        # Get action logits
        features = self.features_extractor(obs_flat)
        action_logits = self.actor_net(features)
        
        # Create distribution
        self.action_dist.proba_distribution(action_logits)
        return self.action_dist


def create_ctde_policy_kwargs() -> Dict:
    """
    Create policy kwargs for CTDE training.
    
    Returns:
        dict: Policy configuration for PPO training
    """
    return {
        "policy_class": CTDEPolicy,
        "policy_kwargs": {
            "features_extractor_class": CTDEFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "n_agents": 5,
            "global_obs_dim": 102,
            "net_arch": [256, 256, 128],
        }
    }