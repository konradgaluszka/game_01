"""
Test script for the refactored SoccerEnv to ensure functionality is maintained.

This script tests:
1. Environment initialization
2. Basic step/reset functionality  
3. Observation space correctness
4. Reward calculation
5. Different phases and opponent types
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ai.soccer_env import SoccerEnv


def test_basic_functionality():
    """Test basic environment functionality"""
    print("Testing basic functionality...")
    
    # Create environment
    env = SoccerEnv()
    print(f"✓ Environment created successfully")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful, observation shape: {obs.shape}")
    
    # Verify observation dimension
    assert obs.shape == (62,), f"Expected observation shape (62,), got {obs.shape}"
    print(f"✓ Observation dimension correct")
    
    # Test step
    action = [0, 0, 0, 0, 0]  # Do nothing for all players
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful, reward: {reward}")
    
    # Test action space
    assert env.action_space.nvec.tolist() == [7, 7, 7, 7, 7], "Action space incorrect"
    print(f"✓ Action space correct")
    
    env.close()
    print("✓ Basic functionality test passed!\n")


def test_different_phases():
    """Test different curriculum phases"""
    print("Testing different phases...")
    
    phases = ["ball_awareness", "basic_soccer", "competitive_soccer"]
    
    for phase in phases:
        print(f"  Testing phase: {phase}")
        env = SoccerEnv(phase=phase, curriculum=True)
        
        # Verify phase is set correctly
        assert env.current_phase == phase, f"Phase not set correctly: {env.current_phase} != {phase}"
        
        # Test episode length varies by phase
        episode_length = env.max_steps
        print(f"    Episode length: {episode_length}")
        
        # Test observation and reward
        obs, _ = env.reset()
        action = [1, 2, 3, 4, 5]  # Different actions for each player
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"    Reward: {reward}")
        
        env.close()
    
    print("✓ Phase testing passed!\n")


def test_opponent_types():
    """Test different opponent types"""
    print("Testing opponent types...")
    
    # Test opponent AI
    print("  Testing OpponentAI...")
    env = SoccerEnv()
    assert env.opponent_type == "opponent_ai"
    
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0])
    print(f"    OpponentAI reward: {reward}")
    env.close()
    
    # Test self-play (without model)
    print("  Testing self-play (no model)...")
    env = SoccerEnv(self_play=True)
    assert env.opponent_type == "self_play"
    
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0])
    print(f"    Self-play reward: {reward}")
    env.close()
    
    print("✓ Opponent type testing passed!\n")


def test_training_info():
    """Test additional helper methods"""
    print("Testing training info methods...")
    
    env = SoccerEnv(phase="basic_soccer", curriculum=True)
    
    # Test training info
    info = env.get_training_info()
    print(f"  Current phase: {info['current_phase']}")
    print(f"  Episode steps: {info['episode_steps']}")
    print(f"  Max steps: {info['max_steps']}")
    
    # Test phase switching
    env.set_phase("competitive_soccer")
    assert env.current_phase == "competitive_soccer"
    
    # Test opponent switching  
    env.set_opponent_type("phase_based")
    assert env.opponent_type == "phase_based"
    
    env.close()
    print("✓ Training info methods passed!\n")


def test_curriculum_progression():
    """Test curriculum learning progression"""
    print("Testing curriculum progression...")
    
    # Test automatic phase progression
    env1 = SoccerEnv(curriculum=True, total_timesteps=10000)  # Should be ball_awareness
    assert env1.current_phase == "ball_awareness"
    
    env2 = SoccerEnv(curriculum=True, total_timesteps=50000)  # Should be basic_soccer  
    assert env2.current_phase == "basic_soccer"
    
    env3 = SoccerEnv(curriculum=True, total_timesteps=100000)  # Should be competitive_soccer
    assert env3.current_phase == "competitive_soccer"
    
    print(f"  10k steps -> {env1.current_phase}")
    print(f"  50k steps -> {env2.current_phase}")  
    print(f"  100k steps -> {env3.current_phase}")
    
    env1.close()
    env2.close()
    env3.close()
    
    print("✓ Curriculum progression passed!\n")


def test_observation_consistency():
    """Test that observations are consistent and reasonable"""
    print("Testing observation consistency...")
    
    env = SoccerEnv()
    obs, _ = env.reset()
    
    # Check for NaN or inf values
    assert not np.any(np.isnan(obs)), "Observation contains NaN values"
    assert not np.any(np.isinf(obs)), "Observation contains infinite values"
    
    # Check observation is in reasonable range (normalized)
    assert np.all(obs >= -10) and np.all(obs <= 10), "Observation values outside reasonable range"
    
    # Test multiple steps
    for i in range(10):
        action = np.random.randint(0, 7, size=5)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert not np.any(np.isnan(obs)), f"NaN in observation at step {i}"
        assert not np.isnan(reward), f"NaN reward at step {i}"
        
        if terminated:
            obs, _ = env.reset()
    
    env.close()
    print("✓ Observation consistency passed!\n")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("TESTING REFACTORED SOCCER ENVIRONMENT")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_different_phases()
        test_opponent_types()
        test_training_info()
        test_curriculum_progression()
        test_observation_consistency()
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The refactored SoccerEnv maintains full functionality.")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)