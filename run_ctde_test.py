"""
Simple test script to diagnose the dimension issue
"""

import sys
sys.path.append('.')

try:
    from ai.soccer_env_ctde import SoccerEnvCTDE
    import numpy as np
    
    print("Creating CTDE environment...")
    env = SoccerEnvCTDE(observation_mode='agent')
    
    print("Resetting environment...")
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Expected: (340,)")
    
    if obs.shape[0] != 340:
        print(f"DIMENSION MISMATCH! Got {obs.shape[0]}, expected 340")
        print("This will cause the neural network error.")
        
        # Get individual agent observations for debugging
        agent_obs = env.get_agent_observations()
        print(f"Number of agents: {len(agent_obs)}")
        for i, agent_ob in enumerate(agent_obs):
            print(f"Agent {i} dims: {agent_ob.shape[0]}")
    
    env.close()
    print("Test completed successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("This suggests the virtual environment is not activated.")
    print("Please run: venv\\Scripts\\activate")
    
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()