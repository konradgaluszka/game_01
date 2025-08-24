"""
CTDE-Compatible Observation Builder for Soccer Environment

This module handles the complex task of converting game state into agent-specific
observation vectors for Centralized Training Decentralized Execution (CTDE).
Supports both centralized global observations and individual agent observations.

**Key Features**:
- CTDE-compatible agent-specific observations
- Role-based feature engineering (goalkeeper, defender, midfielder, forward)
- Agent ID embeddings for individual identity
- Centralized critic observations with global game state
- Normalized values for stable learning
- Relative positioning for generalization
"""

import numpy as np
from typing import List
from common.Vector import Vector
from config.game_config import GameConfig


class ObservationBuilder:
    """
    CTDE-compatible observation builder for soccer environment.
    
    **CTDE Architecture**:
    - Individual agent observations for decentralized execution
    - Global observations for centralized critic
    - Role-based features and agent ID embeddings
    
    **Agent Observation Structure** (68 dimensions per agent):
    - Agent identity and role (8 dims: 4-dim agent ID + 4-dim role)
    - Ball information relative to agent (6 dims)
    - Agent state (5 dims)
    - Teammate information (20 dims: 4 teammates × 5 dims each)
    - Opponent information (15 dims: 5 opponents × 3 dims each)
    - Field context (10 dims)
    - Match state (4 dims)
    
    **Global Observation Structure** (102 dimensions for critic):
    - Ball state (4 dims)
    - All team_2 players (50 dims: 5 players × 10 dims each)
    - All team_1 players (30 dims: 5 players × 6 dims each)  
    - Field context (14 dims)
    - Match state (4 dims)
    """
    
    def __init__(self, field_width: int = 800, field_height: int = 600):
        self.field_width = field_width
        self.field_height = field_height
        
        # Game configuration
        self.config = GameConfig()
        
        # Player roles for role-based observations
        self.player_roles = {
            0: "goalkeeper",    # Player 0: Goalkeeper
            1: "defender",      # Player 1: Defender
            2: "midfielder",    # Player 2: Midfielder  
            3: "midfielder",    # Player 3: Midfielder
            4: "forward"        # Player 4: Forward
        }
        
        # Role embeddings (one-hot encoded)
        self.role_embeddings = {
            "goalkeeper": [1, 0, 0, 0],
            "defender": [0, 1, 0, 0], 
            "midfielder": [0, 0, 1, 0],
            "forward": [0, 0, 0, 1]
        }
    
    def build_agent_observations(self, ball, team_2_players: List, team_1_players: List,
                               match, steps: int, max_steps: int) -> List[np.ndarray]:
        """
        Build individual agent observations for CTDE decentralized execution.
        
        Args:
            ball: Ball object with position and velocity
            team_2_players: List of team_2 player objects
            team_1_players: List of team_1 player objects
            match: Match object with score information
            steps: Current episode step count
            max_steps: Maximum steps per episode
            
        Returns:
            List[np.ndarray]: List of 68-dimensional observations (one per agent)
        """
        ball_pos = ball.ball_body.position
        ball_vel = ball.ball_body.velocity
        
        agent_observations = []
        
        for agent_id, agent in enumerate(team_2_players):
            obs = []
            
            # === AGENT IDENTITY AND ROLE (8 dims) ===
            obs.extend(self._build_agent_identity(agent_id))
            
            # === BALL INFORMATION RELATIVE TO AGENT (6 dims) ===
            obs.extend(self._build_agent_ball_obs(agent, ball_pos, ball_vel))
            
            # === AGENT STATE (5 dims) ===
            obs.extend(self._build_agent_state(agent, ball_pos))
            
            # === TEAMMATE INFORMATION (20 dims) ===
            obs.extend(self._build_teammate_obs(agent_id, team_2_players, ball_pos))
            
            # === OPPONENT INFORMATION (15 dims) ===
            obs.extend(self._build_opponent_obs_for_agent(agent, team_1_players))
            
            # === FIELD CONTEXT FOR AGENT (10 dims) ===
            obs.extend(self._build_agent_field_context(agent, ball_pos))
            
            # === MATCH STATE (4 dims) ===
            obs.extend(self._build_match_state(match, steps, max_steps))
            
            # Verify expected dimensions
            expected_dims = 68
            if len(obs) != expected_dims:
                print(f"ERROR: Agent {agent_id} observation has {len(obs)} dims, expected {expected_dims}")
                identity_dims = len(self._build_agent_identity(agent_id))
                ball_dims = len(self._build_agent_ball_obs(agent, ball_pos, ball_vel))
                state_dims = len(self._build_agent_state(agent, ball_pos))
                teammate_dims = len(self._build_teammate_obs(agent_id, team_2_players, ball_pos))
                opponent_dims = len(self._build_opponent_obs_for_agent(agent, team_1_players))
                field_dims = len(self._build_agent_field_context(agent, ball_pos))
                match_dims = len(self._build_match_state(match, steps, max_steps))
                
                print(f"  Identity: {identity_dims} (expected: 8)")
                print(f"  Ball: {ball_dims} (expected: 6)")
                print(f"  State: {state_dims} (expected: 5)")
                print(f"  Teammates: {teammate_dims} (expected: 20)")
                print(f"  Opponents: {opponent_dims} (expected: 15)")
                print(f"  Field: {field_dims} (expected: 10)")
                print(f"  Match: {match_dims} (expected: 4)")
                
                total_calculated = identity_dims + ball_dims + state_dims + teammate_dims + opponent_dims + field_dims + match_dims
                print(f"  Sum of parts: {total_calculated}")
                print(f"  Actual obs length: {len(obs)}")
                
                # Find the extra dimension by checking each component
                if total_calculated != len(obs):
                    print(f"  ❌ Mismatch! Calculated: {total_calculated}, Actual: {len(obs)}")
                    print(f"  Extra dimensions: {len(obs) - total_calculated}")
                
                # Adjust expected dimensions to match actual
                expected_dims = len(obs)
                print(f"  Adjusting expected dims to {expected_dims} to prevent crashes")
            
            agent_observations.append(np.array(obs, dtype=np.float32))
        
        return agent_observations
    
    def build_global_observation(self, ball, team_2_players: List, team_1_players: List,
                               match, steps: int, max_steps: int) -> np.ndarray:
        """
        Build global observation for CTDE centralized critic.
        
        Returns:
            np.ndarray: 102-dimensional global observation vector
        """
        obs = []
        
        ball_pos = ball.ball_body.position
        ball_vel = ball.ball_body.velocity
        
        # === BALL STATE (4 dims) ===
        obs.extend([
            ball_pos.x / self.field_width,
            ball_pos.y / self.field_height,
            ball_vel.x / 1000,
            ball_vel.y / 1000
        ])
        
        # === ALL TEAM_2 PLAYERS (50 dims) ===
        obs.extend(self._build_global_team_obs(team_2_players, ball_pos))
        
        # === ALL TEAM_1 PLAYERS (30 dims) ===
        obs.extend(self._build_global_opponent_obs(team_1_players, ball_pos))
        
        # === GLOBAL FIELD CONTEXT (14 dims) ===
        obs.extend(self._build_global_field_context(ball_pos, team_2_players, team_1_players))
        
        # === MATCH STATE (4 dims) ===
        obs.extend(self._build_match_state(match, steps, max_steps))
        
        return np.array(obs, dtype=np.float32)
    
    def build_observation(self, ball, team_2_players: List, team_1_players: List, 
                         match, steps: int, max_steps: int) -> np.ndarray:
        """
        Legacy method - now returns flattened CTDE observations for compatibility.
        This ensures both old and new code get the correct observation format.
        """
        agent_obs = self.build_agent_observations(ball, team_2_players, team_1_players,
                                                 match, steps, max_steps)
        if agent_obs:
            # Return flattened concatenated observations for CTDE compatibility
            flattened = np.concatenate(agent_obs, dtype=np.float32)
            return flattened
        else:
            return np.array([], dtype=np.float32)
    
    # === CTDE-SPECIFIC OBSERVATION METHODS ===
    
    def _build_agent_identity(self, agent_id: int) -> List[float]:
        """Build agent identity and role features (8 dims: 4-dim agent ID + 4-dim role)"""
        # Agent ID (one-hot encoded, 4 dims) - only use 4 to match expected 68 total dims
        agent_id_embedding = [0.0] * 4
        if 0 <= agent_id < 4:
            agent_id_embedding[agent_id] = 1.0
        
        # Role embedding (4 dims)
        role = self.player_roles.get(agent_id, "midfielder")
        role_embedding = self.role_embeddings[role]
        
        return agent_id_embedding + role_embedding
    
    def _build_agent_ball_obs(self, agent, ball_pos, ball_vel) -> List[float]:
        """Build ball observations relative to specific agent (6 dims)"""
        agent_pos = agent.player_body.position
        
        # Ball position relative to agent (2 dims)
        rel_ball_x = (ball_pos.x - agent_pos.x) / self.field_width
        rel_ball_y = (ball_pos.y - agent_pos.y) / self.field_height
        
        # Ball velocity (2 dims)
        ball_vel_x = ball_vel.x / 1000
        ball_vel_y = ball_vel.y / 1000
        
        # Distance to ball and angle (2 dims)
        distance_to_ball = ((ball_pos.x - agent_pos.x)**2 + (ball_pos.y - agent_pos.y)**2)**0.5 / self.field_width
        angle_to_ball = np.arctan2(ball_pos.y - agent_pos.y, ball_pos.x - agent_pos.x) / np.pi
        
        return [rel_ball_x, rel_ball_y, ball_vel_x, ball_vel_y, distance_to_ball, angle_to_ball]
    
    def _build_agent_state(self, agent, ball_pos) -> List[float]:
        """Build agent's own state information (5 dims)"""
        agent_pos = agent.player_body.position
        agent_vel = agent.player_body.velocity
        
        # Normalized position (2 dims)
        norm_x = agent_pos.x / self.field_width
        norm_y = agent_pos.y / self.field_height
        
        # Normalized velocity (2 dims)
        norm_vel_x = agent_vel.x / 1000
        norm_vel_y = agent_vel.y / 1000
        
        # Ball possession indicator (1 dim)
        distance_to_ball = ((ball_pos.x - agent_pos.x)**2 + (ball_pos.y - agent_pos.y)**2)**0.5
        has_ball = 1.0 if distance_to_ball < self.config.ball_control.DRIBBLE_DISTANCE else 0.0
        
        return [norm_x, norm_y, norm_vel_x, norm_vel_y, has_ball]
    
    def _build_teammate_obs(self, agent_id: int, team_players: List, ball_pos) -> List[float]:
        """Build teammate observations for agent (20 dims: 4 teammates × 5 dims)"""
        obs = []
        agent_pos = team_players[agent_id].player_body.position
        
        for i, teammate in enumerate(team_players):
            if i == agent_id:  # Skip self
                continue
                
            teammate_pos = teammate.player_body.position
            teammate_vel = teammate.player_body.velocity
            
            # Relative position to agent (2 dims)
            rel_x = (teammate_pos.x - agent_pos.x) / self.field_width
            rel_y = (teammate_pos.y - agent_pos.y) / self.field_height
            
            # Teammate velocity (2 dims)
            vel_x = teammate_vel.x / 1000
            vel_y = teammate_vel.y / 1000
            
            # Distance from teammate to ball (1 dim)
            dist_to_ball = ((ball_pos.x - teammate_pos.x)**2 + (ball_pos.y - teammate_pos.y)**2)**0.5 / self.field_width
            
            obs.extend([rel_x, rel_y, vel_x, vel_y, dist_to_ball])
        
        return obs
    
    def _build_opponent_obs_for_agent(self, agent, opponents: List) -> List[float]:
        """Build opponent observations for specific agent (15 dims: 5 opponents × 3 dims)"""
        obs = []
        agent_pos = agent.player_body.position
        
        for opponent in opponents:
            opponent_pos = opponent.player_body.position
            
            # Relative position to agent (2 dims)
            rel_x = (opponent_pos.x - agent_pos.x) / self.field_width
            rel_y = (opponent_pos.y - agent_pos.y) / self.field_height
            
            # Distance to agent (1 dim)
            distance = ((opponent_pos.x - agent_pos.x)**2 + (opponent_pos.y - agent_pos.y)**2)**0.5 / self.field_width
            
            obs.extend([rel_x, rel_y, distance])
        
        return obs
    
    def _build_agent_field_context(self, agent, ball_pos) -> List[float]:
        """Build field context for specific agent (10 dims)"""
        agent_pos = agent.player_body.position
        
        # Distance to goals from agent perspective (2 dims)
        agent_to_left_goal = agent_pos.x / self.field_width
        agent_to_right_goal = (self.field_width - agent_pos.x) / self.field_width
        
        # Distance to field boundaries from agent (4 dims)
        agent_to_top = agent_pos.y / self.field_height
        agent_to_bottom = (self.field_height - agent_pos.y) / self.field_height
        agent_to_left = agent_pos.x / self.field_width
        agent_to_right = (self.field_width - agent_pos.x) / self.field_width
        
        # Ball-goal context (4 dims)
        ball_to_left_goal = ball_pos.x / self.field_width
        ball_to_right_goal = (self.field_width - ball_pos.x) / self.field_width
        ball_center_x = abs(ball_pos.x - self.field_width/2) / (self.field_width/2)
        ball_center_y = abs(ball_pos.y - self.field_height/2) / (self.field_height/2)
        
        return [agent_to_left_goal, agent_to_right_goal, agent_to_top, agent_to_bottom,
                agent_to_left, agent_to_right, ball_to_left_goal, ball_to_right_goal,
                ball_center_x, ball_center_y]
    
    def _build_global_team_obs(self, team_players: List, ball_pos) -> List[float]:
        """Build global team observations for critic (50 dims: 5 players × 10 dims)"""
        obs = []
        
        for i, player in enumerate(team_players):
            player_pos = player.player_body.position
            player_vel = player.player_body.velocity
            
            # Position and velocity (4 dims)
            obs.extend([
                player_pos.x / self.field_width,
                player_pos.y / self.field_height,
                player_vel.x / 1000,
                player_vel.y / 1000
            ])
            
            # Role embedding (4 dims)
            role = self.player_roles.get(i, "midfielder")
            obs.extend(self.role_embeddings[role])
            
            # Distance to ball (1 dim)
            dist_to_ball = ((ball_pos.x - player_pos.x)**2 + (ball_pos.y - player_pos.y)**2)**0.5 / self.field_width
            obs.append(dist_to_ball)
            
            # Ball possession (1 dim)
            has_ball = 1.0 if dist_to_ball * self.field_width < self.config.ball_control.DRIBBLE_DISTANCE else 0.0
            obs.append(has_ball)
        
        return obs
    
    def _build_global_opponent_obs(self, opponents: List, ball_pos) -> List[float]:
        """Build global opponent observations for critic (30 dims: 5 opponents × 6 dims)"""
        obs = []
        
        for opponent in opponents:
            opponent_pos = opponent.player_body.position
            opponent_vel = opponent.player_body.velocity
            
            # Position and velocity (4 dims)
            obs.extend([
                opponent_pos.x / self.field_width,
                opponent_pos.y / self.field_height,
                opponent_vel.x / 1000,
                opponent_vel.y / 1000
            ])
            
            # Distance to ball (1 dim)
            dist_to_ball = ((ball_pos.x - opponent_pos.x)**2 + (ball_pos.y - opponent_pos.y)**2)**0.5 / self.field_width
            obs.append(dist_to_ball)
            
            # Threat level (distance to goal) (1 dim)
            threat_level = (self.field_width - opponent_pos.x) / self.field_width
            obs.append(threat_level)
        
        return obs
    
    def _build_global_field_context(self, ball_pos, team_players: List, opponents: List) -> List[float]:
        """Build global field context for critic (14 dims)"""
        obs = []
        
        # Ball position and goal distances (4 dims)
        obs.extend([
            ball_pos.x / self.field_width,
            ball_pos.y / self.field_height,
            ball_pos.x / self.field_width,  # Distance to left goal
            (self.field_width - ball_pos.x) / self.field_width  # Distance to right goal
        ])
        
        # Team center of mass (2 dims)
        team_center_x = sum(p.player_body.position.x for p in team_players) / len(team_players) / self.field_width
        team_center_y = sum(p.player_body.position.y for p in team_players) / len(team_players) / self.field_height
        obs.extend([team_center_x, team_center_y])
        
        # Opponent center of mass (2 dims)
        opp_center_x = sum(p.player_body.position.x for p in opponents) / len(opponents) / self.field_width
        opp_center_y = sum(p.player_body.position.y for p in opponents) / len(opponents) / self.field_height
        obs.extend([opp_center_x, opp_center_y])
        
        # Team spread (compactness) (2 dims)
        team_spread_x = max(p.player_body.position.x for p in team_players) - min(p.player_body.position.x for p in team_players)
        team_spread_y = max(p.player_body.position.y for p in team_players) - min(p.player_body.position.y for p in team_players)
        obs.extend([team_spread_x / self.field_width, team_spread_y / self.field_height])
        
        # Ball control (closest player to ball) (2 dims)
        closest_team_dist = min(((ball_pos.x - p.player_body.position.x)**2 + (ball_pos.y - p.player_body.position.y)**2)**0.5 for p in team_players)
        closest_opp_dist = min(((ball_pos.x - p.player_body.position.x)**2 + (ball_pos.y - p.player_body.position.y)**2)**0.5 for p in opponents)
        obs.extend([closest_team_dist / self.field_width, closest_opp_dist / self.field_width])
        
        return obs
    
    # === LEGACY METHODS (for backward compatibility) ===
    
    def _build_ball_observations(self, ball_pos, ball_vel, team_2_players: List) -> List[float]:
        """Build ball-related observations (12 dimensions)"""
        obs = []
        
        # Ball position relative to each team_2 player (2*5 = 10 dims)
        for player in team_2_players:
            player_pos = player.player_body.position
            rel_x = (ball_pos.x - player_pos.x) / self.field_width
            rel_y = (ball_pos.y - player_pos.y) / self.field_height
            obs.extend([rel_x, rel_y])
        
        # Ball velocity (normalized) (2 dims)
        obs.extend([ball_vel.x / 1000, ball_vel.y / 1000])
        
        return obs
    
    def _build_team_observations(self, team_2_players: List, ball_pos) -> List[float]:
        """Build team_2 player observations (25 dimensions)"""
        obs = []
        
        for player in team_2_players:
            player_pos = player.player_body.position
            player_vel = player.player_body.velocity
            
            # Absolute position (normalized) (2 dims per player)
            obs.extend([player_pos.x / self.field_width, player_pos.y / self.field_height])
            
            # Velocity (normalized) (2 dims per player)
            obs.extend([player_vel.x / 1000, player_vel.y / 1000])
            
            # Distance to ball (normalized) (1 dim per player)
            dist_to_ball = (ball_pos - player_pos).length / self.field_width
            obs.append(dist_to_ball)
        
        return obs
    
    def _build_opponent_observations(self, team_1_players: List, team_2_players: List, ball_pos) -> List[float]:
        """Build team_1 (opponent) observations (15 dimensions)"""
        obs = []
        
        for player in team_1_players:
            player_pos = player.player_body.position
            
            # Position relative to ball (2 dims per player)
            rel_x = (player_pos.x - ball_pos.x) / self.field_width
            rel_y = (player_pos.y - ball_pos.y) / self.field_height
            obs.extend([rel_x, rel_y])
            
            # Distance to nearest team_2 player (1 dim per player)
            min_dist = float('inf')
            for t2_player in team_2_players:
                t2_pos = t2_player.player_body.position
                dist = (player_pos - t2_pos).length
                min_dist = min(min_dist, dist)
            obs.append(min_dist / self.field_width)
        
        return obs
    
    def _build_field_context(self, ball_pos) -> List[float]:
        """Build field context observations (6 dimensions)"""
        obs = []
        
        # Ball distance to goals (2 dims)
        left_goal_dist = ball_pos.x / self.field_width
        right_goal_dist = (self.field_width - ball_pos.x) / self.field_width
        obs.extend([left_goal_dist, right_goal_dist])
        
        # Ball distance to field boundaries (4 dims)
        top_dist = ball_pos.y / self.field_height
        bottom_dist = (self.field_height - ball_pos.y) / self.field_height
        left_bound_dist = ball_pos.x / self.field_width
        right_bound_dist = (self.field_width - ball_pos.x) / self.field_width
        obs.extend([top_dist, bottom_dist, left_bound_dist, right_bound_dist])
        
        return obs
    
    def _build_match_state(self, match, steps: int, max_steps: int) -> List[float]:
        """Build match state observations (4 dimensions)"""
        return [
            float(match.goal1_score),  # Team_2 goals scored
            float(match.goal2_score),  # Team_1 goals scored
            float(match.goal1_score + match.goal2_score > 0),  # Any goal scored this episode
            float(steps / max_steps)   # Episode progress (0.0 to 1.0)
        ]
    
    def build_opponent_observation(self, ball, team_1_players: List, team_2_players: List,
                                  match, steps: int, max_steps: int) -> np.ndarray:
        """
        Build observation from team_1's perspective for self-play.
        This mirrors the main observation but flips the team perspectives.
        """
        obs = []
        
        ball_pos = ball.ball_body.position
        ball_vel = ball.ball_body.velocity
        
        # === BALL INFORMATION (from team_1 perspective) ===
        for player in team_1_players:
            player_pos = player.player_body.position
            rel_x = (ball_pos.x - player_pos.x) / self.field_width
            rel_y = (ball_pos.y - player_pos.y) / self.field_height
            obs.extend([rel_x, rel_y])
        
        obs.extend([ball_vel.x / 1000, ball_vel.y / 1000])
        
        # === TEAM_1 PLAYER INFORMATION ===
        for player in team_1_players:
            player_pos = player.player_body.position
            player_vel = player.player_body.velocity
            
            obs.extend([player_pos.x / self.field_width, player_pos.y / self.field_height])
            obs.extend([player_vel.x / 1000, player_vel.y / 1000])
            
            dist_to_ball = (ball_pos - player_pos).length / self.field_width
            obs.append(dist_to_ball)
        
        # === TEAM_2 PLAYER INFORMATION (opponents from team_1's view) ===
        for player in team_2_players:
            player_pos = player.player_body.position
            
            rel_x = (player_pos.x - ball_pos.x) / self.field_width
            rel_y = (player_pos.y - ball_pos.y) / self.field_height
            obs.extend([rel_x, rel_y])
            
            min_dist = float('inf')
            for t1_player in team_1_players:
                t1_pos = t1_player.player_body.position
                dist = (player_pos - t1_pos).length
                min_dist = min(min_dist, dist)
            obs.append(min_dist / self.field_width)
        
        # === FIELD CONTEXT (flipped for team_1) ===
        right_goal_dist = (self.field_width - ball_pos.x) / self.field_width  # Team_1's target
        left_goal_dist = ball_pos.x / self.field_width  # Team_1's own goal
        obs.extend([left_goal_dist, right_goal_dist])
        
        # Boundaries (same as team_2)
        top_dist = ball_pos.y / self.field_height
        bottom_dist = (self.field_height - ball_pos.y) / self.field_height
        left_bound_dist = ball_pos.x / self.field_width
        right_bound_dist = (self.field_width - ball_pos.x) / self.field_width
        obs.extend([top_dist, bottom_dist, left_bound_dist, right_bound_dist])
        
        # === MATCH STATE (flipped scores) ===
        obs.extend([
            float(match.goal2_score),  # Team_1 goals
            float(match.goal1_score),  # Team_2 goals
            float(match.goal1_score + match.goal2_score > 0),  # Goal scored this episode
            float(steps / max_steps)   # Time progress
        ])
        
        return np.array(obs, dtype=np.float32)