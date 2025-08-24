"""
Debug script to check observation dimensions
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.soccer_env_ctde import SoccerEnvCTDE

def debug_observation_dimensions():
    """Debug the actual observation dimensions being produced"""
    print("=== Debugging Observation Dimensions ===")
    
    env = SoccerEnvCTDE(observation_mode='agent')
    obs, _ = env.reset()
    
    print(f"Total flattened observation shape: {obs.shape}")
    print(f"Expected: (340,) for 5 agents × 68 dims each")
    
    # Get individual agent observations
    agent_obs_list = env.get_agent_observations()
    print(f"\nIndividual agent observations:")
    for i, agent_obs in enumerate(agent_obs_list):
        print(f"  Agent {i}: shape {agent_obs.shape}")
    
    # Check each component
    obs_builder = env.observation_builder
    ball = env.ball
    team_2_players = env.team_2.players()
    team_1_players = env.team_1.players()
    match = env.match
    
    print(f"\n=== Component Dimensions ===")
    for agent_id, agent in enumerate(team_2_players):
        print(f"\nAgent {agent_id} ({obs_builder.player_roles[agent_id]}):")
        
        # Test each component
        identity = obs_builder._build_agent_identity(agent_id)
        print(f"  Identity: {len(identity)} dims (expected: 8)")
        
        ball_obs = obs_builder._build_agent_ball_obs(agent, ball.ball_body.position, ball.ball_body.velocity)
        print(f"  Ball obs: {len(ball_obs)} dims (expected: 6)")
        
        agent_state = obs_builder._build_agent_state(agent, ball.ball_body.position)
        print(f"  Agent state: {len(agent_state)} dims (expected: 5)")
        
        teammate_obs = obs_builder._build_teammate_obs(agent_id, team_2_players, ball.ball_body.position)
        print(f"  Teammate obs: {len(teammate_obs)} dims (expected: 20)")
        
        opponent_obs = obs_builder._build_opponent_obs_for_agent(agent, team_1_players)
        print(f"  Opponent obs: {len(opponent_obs)} dims (expected: 15)")
        
        field_context = obs_builder._build_agent_field_context(agent, ball.ball_body.position)
        print(f"  Field context: {len(field_context)} dims (expected: 10)")
        
        match_state = obs_builder._build_match_state(match, env.steps, env.max_steps)
        print(f"  Match state: {len(match_state)} dims (expected: 4)")
        
        total_dims = len(identity) + len(ball_obs) + len(agent_state) + len(teammate_obs) + len(opponent_obs) + len(field_context) + len(match_state)
        print(f"  Total for agent {agent_id}: {total_dims} dims (expected: 68)")
        
        # Only check first agent to avoid spam
        if agent_id == 0:
            break
    
    env.close()

if __name__ == "__main__":
    debug_observation_dimensions()