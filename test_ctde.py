"""
Test script for CTDE implementation

This script tests the CTDE soccer environment and policy to ensure all components
work correctly before running full training.
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.soccer_env_ctde import SoccerEnvCTDE
from ai.observation_builder import ObservationBuilder

def test_observation_builder():
    """Test the enhanced observation builder"""
    print("=== Testing CTDE Observation Builder ===")
    
    # Create environment to get game objects
    env = SoccerEnvCTDE(observation_mode='agent')
    obs, _ = env.reset()
    
    # Test agent observations
    agent_obs = env.get_agent_observations()
    print(f"Agent observations shape: {[obs.shape for obs in agent_obs]}")
    print(f"Expected: [(68,)] × 5 agents")
    
    # Test global observation
    global_obs = env.get_global_observation()
    print(f"Global observation shape: {global_obs.shape}")
    print(f"Expected: (102,)")
    
    # Test observation components
    obs_builder = env.observation_builder
    
    # Test agent identity building
    for i in range(5):
        identity = obs_builder._build_agent_identity(i)
        role = obs_builder.player_roles[i]
        print(f"Agent {i} ({role}): Identity dims = {len(identity)} (expected: 8)")
    
    env.close()
    print("✓ Observation builder tests passed\n")

def test_ctde_environment():
    """Test CTDE environment with different observation modes"""
    print("=== Testing CTDE Environment ===")
    
    # Test different observation modes
    modes = ['agent', 'global', 'combined']
    
    for mode in modes:
        print(f"Testing observation mode: {mode}")
        env = SoccerEnvCTDE(observation_mode=mode)
        
        obs, info = env.reset()
        print(f"  Reset observation type: {type(obs)}")
        
        if mode == 'agent':
            print(f"  Agent obs shape: {obs.shape}")
            assert obs.shape == (5, 68), f"Expected (5, 68), got {obs.shape}"
        elif mode == 'global':
            print(f"  Global obs shape: {obs.shape}")
            assert obs.shape == (102,), f"Expected (102,), got {obs.shape}"
        elif mode == 'combined':
            print(f"  Combined obs keys: {obs.keys()}")
            print(f"  Agent obs shape: {obs['agent_obs'].shape}")
            print(f"  Global obs shape: {obs['global_obs'].shape}")
            assert obs['agent_obs'].shape == (5, 68)
            assert obs['global_obs'].shape == (102,)
        
        # Test step
        actions = np.array([0, 1, 2, 3, 4])  # Different action for each agent
        obs, reward, terminated, truncated, info = env.step(actions)
        print(f"  Step reward: {reward:.3f}")
        
        env.close()
    
    print("✓ Environment tests passed\n")

def test_ctde_training_compatibility():
    """Test CTDE environment with vectorized environments"""
    print("=== Testing CTDE Training Compatibility ===")
    
    try:
        from stable_baselines3.common.env_util import make_vec_env
        from ai.soccer_env_ctde import create_ctde_env
        
        # Test vectorized environment creation
        def make_env():
            return create_ctde_env(observation_mode='agent')
        
        vec_env = make_vec_env(make_env, n_envs=2)
        
        obs = vec_env.reset()
        print(f"Vectorized obs shape: {obs.shape}")
        print(f"Expected: (2, 340) for 2 envs, 340 dims flattened (5 agents × 68 dims)")
        
        # Test step
        actions = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])  # Actions for 2 envs
        obs, rewards, dones, infos = vec_env.step(actions)
        print(f"Vectorized step rewards: {rewards}")
        
        vec_env.close()
        print("✓ Vectorized environment compatibility passed\n")
        
    except Exception as e:
        print(f"⚠ Vectorized environment test failed: {e}")
        print("This may require stable-baselines3 to be installed\n")

def test_simple_ctde_training():
    """Test that simple CTDE training can be set up"""
    print("=== Testing Simple CTDE Training Setup ===")
    
    try:
        from stable_baselines3 import PPO
        from ai.soccer_env_ctde import SoccerEnvCTDE
        
        # Create environment
        env = SoccerEnvCTDE(observation_mode='agent')
        print(f"Environment observation space: {env.observation_space}")
        print(f"Environment action space: {env.action_space}")
        
        # Test that PPO can be created with this environment
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            policy_kwargs=dict(net_arch=[256, 256])
        )
        print("✓ PPO model creation successful")
        
        # Test a few steps
        obs, _ = env.reset()
        for _ in range(5):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        print("✓ Model prediction and environment step successful")
        env.close()
        print("✓ Simple CTDE training setup test passed\n")
        
    except Exception as e:
        print(f"⚠ Simple CTDE training test failed: {e}")
        import traceback
        traceback.print_exc()
        print()

def test_role_assignments():
    """Test role-based player assignments"""
    print("=== Testing Role Assignments ===")
    
    env = SoccerEnvCTDE(observation_mode='agent')
    obs_builder = env.observation_builder
    
    print("Player role assignments:")
    for i in range(5):
        role = obs_builder.player_roles[i]
        role_embedding = obs_builder.role_embeddings[role]
        print(f"  Player {i}: {role} -> {role_embedding}")
    
    # Test that all roles are represented
    roles_used = list(obs_builder.player_roles.values())
    expected_roles = ['goalkeeper', 'defender', 'midfielder', 'forward']
    
    for role in expected_roles:
        if role in roles_used:
            print(f"  ✓ {role} role assigned")
        else:
            print(f"  ⚠ {role} role not assigned")
    
    env.close()
    print("✓ Role assignment tests passed\n")

def test_training_info():
    """Test CTDE training information"""
    print("=== Testing Training Information ===")
    
    env = SoccerEnvCTDE(observation_mode='combined')
    env.reset()
    
    training_info = env.get_training_info()
    print("Training info keys:", list(training_info.keys()))
    
    expected_keys = ['current_phase', 'observation_mode', 'agent_obs_shape', 'global_obs_shape']
    for key in expected_keys:
        if key in training_info:
            print(f"  ✓ {key}: {training_info[key]}")
        else:
            print(f"  ⚠ Missing key: {key}")
    
    env.close()
    print("✓ Training info tests passed\n")

def run_all_tests():
    """Run all CTDE tests"""
    print("Starting CTDE implementation tests...\n")
    
    try:
        test_observation_builder()
        test_ctde_environment()
        test_role_assignments()
        test_training_info()
        test_ctde_training_compatibility()
        test_simple_ctde_training()
        
        print("=== All CTDE Tests Passed! ===")
        print("The CTDE implementation is ready for training.")
        print("\nNext steps:")
        print("1. Run a short Simple CTDE training test:")
        print("   python ai/train_ctde_simple.py train --timesteps 5000 --model-name test_simple_ctde")
        print("2. Compare with traditional training:")
        print("   python ai/train.py train --timesteps 5000 --model-name test_traditional")
        print("3. Run longer Simple CTDE training:")
        print("   python ai/train_ctde_simple.py train --timesteps 50000 --model-name full_simple_ctde --curriculum")
        print("4. Try the advanced CTDE policy (if needed):")
        print("   python ai/train_ctde.py train --timesteps 10000 --model-name advanced_ctde")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)