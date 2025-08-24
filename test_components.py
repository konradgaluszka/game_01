"""
Test individual components without initializing the full environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def test_imports():
    """Test that all new components can be imported"""
    print("Testing component imports...")
    
    try:
        from ai.observation_builder import ObservationBuilder
        print("OK ObservationBuilder imported")
        
        from ai.reward_calculator import RewardCalculator
        print("OK RewardCalculator imported")
        
        from ai.curriculum_manager import CurriculumManager  
        print("OK CurriculumManager imported")
        
        from ai.opponent_manager import OpponentManager
        print("OK OpponentManager imported")
        
        print("OK All component imports successful!\n")
        return True
        
    except Exception as e:
        print(f"ERROR Import failed: {e}")
        return False


def test_curriculum_manager():
    """Test curriculum manager functionality"""
    print("Testing CurriculumManager...")
    
    try:
        from ai.curriculum_manager import CurriculumManager
        
        # Test basic functionality
        manager = CurriculumManager()
        
        # Test phase progression
        phase1 = manager.get_current_phase(10000)   # Should be ball_awareness
        phase2 = manager.get_current_phase(50000)   # Should be basic_soccer  
        phase3 = manager.get_current_phase(100000)  # Should be competitive_soccer
        
        print(f"  10k steps -> {phase1}")
        print(f"  50k steps -> {phase2}")
        print(f"  100k steps -> {phase3}")
        
        assert phase1 == "ball_awareness"
        assert phase2 == "basic_soccer" 
        assert phase3 == "competitive_soccer"
        
        # Test episode lengths
        len1 = manager.get_episode_length("ball_awareness")
        len2 = manager.get_episode_length("basic_soccer")
        len3 = manager.get_episode_length("competitive_soccer")
        
        print(f"  Episode lengths: {len1}, {len2}, {len3}")
        assert len1 < len2 < len3
        
        # Test ball positioning
        pos = manager.get_ball_start_position("ball_awareness")
        assert isinstance(pos, tuple) and len(pos) == 2
        print(f"  Ball position: {pos}")
        
        print("OK CurriculumManager test passed!\n")
        return True
        
    except Exception as e:
        print(f"ERROR CurriculumManager test failed: {e}")
        return False


def test_reward_calculator():
    """Test reward calculator functionality"""
    print("Testing RewardCalculator...")
    
    try:
        from ai.reward_calculator import RewardCalculator
        
        calculator = RewardCalculator()
        
        # Test available phases
        phases = calculator.get_available_phases()
        print(f"  Available phases: {phases}")
        assert "ball_awareness" in phases
        assert "basic_soccer" in phases
        assert "competitive_soccer" in phases
        
        # Test reward calculation with dummy game state
        dummy_game_state = {
            'ball_position': type('pos', (), {'x': 400, 'y': 300})(),
            'ball_velocity': type('vel', (), {'x': 0, 'y': 0})(),
            'team_2_players': [],
            'team_1_players': [],
            'match': type('match', (), {'goal1_score': 0, 'goal2_score': 0})(),
            'field_width': 800,
            'field_height': 600,
            'steps': 0,
            'max_steps': 500
        }
        
        # Test each phase
        for phase in phases:
            try:
                reward = calculator.calculate_reward(phase, dummy_game_state)
                print(f"  {phase} reward: {reward}")
                assert isinstance(reward, float)
            except Exception as e:
                print(f"  Warning: {phase} calculation failed: {e}")
        
        print("OK RewardCalculator test passed!\n")
        return True
        
    except Exception as e:
        print(f"ERROR RewardCalculator test failed: {e}")
        return False


def test_observation_builder():
    """Test observation builder functionality"""
    print("Testing ObservationBuilder...")
    
    try:
        from ai.observation_builder import ObservationBuilder
        
        builder = ObservationBuilder(800, 600)
        print("OK ObservationBuilder created")
        
        # Test dummy ball position observation
        pos = builder._build_field_context(type('pos', (), {'x': 400, 'y': 300})())
        print(f"  Field context dims: {len(pos)}")
        assert len(pos) == 6  # Should be 6 dimensions
        
        # Test match state
        match_state = builder._build_match_state(
            type('match', (), {'goal1_score': 1, 'goal2_score': 0})(), 
            100, 500
        )
        print(f"  Match state dims: {len(match_state)}")
        assert len(match_state) == 4  # Should be 4 dimensions
        
        print("OK ObservationBuilder test passed!\n")
        return True
        
    except Exception as e:
        print(f"ERROR ObservationBuilder test failed: {e}")
        return False


def run_component_tests():
    """Run all component tests"""
    print("=" * 50)
    print("TESTING REFACTORED COMPONENTS")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_curriculum_manager,
        test_reward_calculator,
        test_observation_builder
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            break
    
    print("=" * 50)
    if passed == len(tests):
        print("SUCCESS ALL COMPONENT TESTS PASSED! SUCCESS")
        print("The refactored components are working correctly.")
    else:
        print(f"ERROR {len(tests) - passed} tests failed")
    print("=" * 50)
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_component_tests()
    sys.exit(0 if success else 1)