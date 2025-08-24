"""
Curriculum Learning Manager for Soccer Training

This module handles the progressive difficulty system that guides AI training
from basic ball awareness to complex strategic soccer gameplay. It manages
phase transitions, episode length adjustments, and ball positioning strategies.

**Key Features**:
- Automatic phase progression based on timesteps
- Manual phase override for targeted training
- Phase-specific episode lengths and ball positioning
- Configurable transition thresholds
"""

from typing import Tuple, Optional
import random


class CurriculumManager:
    """
    Manages curriculum learning progression for soccer AI training.
    
    **Training Phases**:
    1. **Ball Awareness** (0-25k steps): Learn to approach and control ball
    2. **Basic Soccer** (25k-75k steps): Add shooting, passing, basic tactics
    3. **Competitive Soccer** (75k+ steps): Full strategic gameplay
    
    **Phase Features**:
    - Episode length increases with complexity
    - Ball starting positions adapt to phase goals
    - Reward complexity scales with player capability
    """
    
    def __init__(self, manual_phase: Optional[str] = None, enable_curriculum: bool = True):
        """
        Initialize curriculum manager.
        
        Args:
            manual_phase: Force specific phase ("ball_awareness", "basic_soccer", "competitive_soccer")
            enable_curriculum: Whether to use curriculum progression (False = jump to competitive)
        """
        self.manual_phase = manual_phase
        self.enable_curriculum = enable_curriculum
        
        # Phase transition thresholds (in timesteps)
        self.phase_thresholds = {
            "ball_awareness": 0,
            "basic_soccer": 25000,
            "competitive_soccer": 75000
        }
        
        # Episode length per phase (in steps at 60 FPS)
        self.episode_lengths = {
            "ball_awareness": 300,    # 5 seconds - quick feedback
            "basic_soccer": 400,      # 6.7 seconds - more time for actions  
            "competitive_soccer": 500  # 8.3 seconds - full strategic play
        }
        
        # Field dimensions for ball positioning
        self.field_width = 800
        self.field_height = 600
        self.field_center = (self.field_width // 2, self.field_height // 2)
    
    def get_current_phase(self, total_timesteps: int) -> str:
        """
        Determine current training phase based on timesteps or manual override.
        
        Args:
            total_timesteps: Total training timesteps completed
            
        Returns:
            str: Current phase name
        """
        # Manual phase override
        if self.manual_phase:
            return self.manual_phase
        
        # Skip curriculum - go straight to competitive
        if not self.enable_curriculum:
            return "competitive_soccer"
        
        # Automatic phase progression
        if total_timesteps < self.phase_thresholds["basic_soccer"]:
            return "ball_awareness"
        elif total_timesteps < self.phase_thresholds["competitive_soccer"]:
            return "basic_soccer"
        else:
            return "competitive_soccer"
    
    def get_episode_length(self, phase: str) -> int:
        """
        Get maximum episode length for the current phase.
        
        Args:
            phase: Current training phase
            
        Returns:
            int: Maximum steps per episode
        """
        return self.episode_lengths.get(phase, 500)
    
    def get_ball_start_position(self, phase: str) -> Tuple[float, float]:
        """
        Generate appropriate ball starting position for the current phase.
        
        Args:
            phase: Current training phase
            
        Returns:
            Tuple[float, float]: (x, y) ball starting position
        """
        if phase == "ball_awareness":
            return self._get_ball_awareness_position()
        elif phase == "basic_soccer":
            return self._get_basic_soccer_position()
        else:
            return self._get_competitive_position()
    
    def _get_ball_awareness_position(self) -> Tuple[float, float]:
        """
        Ball positioning for phase 1: Close to center but with some variation
        to encourage active ball-seeking behavior.
        """
        # Start ball near center field with moderate variation
        # This forces team_2 players to actively move toward it
        center_x, center_y = self.field_center
        
        # Smaller variation to keep ball visible and reachable
        offset_x = random.uniform(-80, 80)
        offset_y = random.uniform(-80, 80)
        
        x = max(100, min(self.field_width - 100, center_x + offset_x))
        y = max(100, min(self.field_height - 100, center_y + offset_y))
        
        return (x, y)
    
    def _get_basic_soccer_position(self) -> Tuple[float, float]:
        """
        Ball positioning for phase 2: More variation to practice different
        scenarios but still reasonable for developing skills.
        """
        center_x, center_y = self.field_center
        
        # Larger variation to practice different field positions
        offset_x = random.uniform(-120, 120)
        offset_y = random.uniform(-80, 80)
        
        x = max(80, min(self.field_width - 80, center_x + offset_x))
        y = max(80, min(self.field_height - 80, center_y + offset_y))
        
        return (x, y)
    
    def _get_competitive_position(self) -> Tuple[float, float]:
        """
        Ball positioning for phase 3: Full field variation to simulate
        realistic game conditions and challenge advanced skills.
        """
        center_x, center_y = self.field_center
        
        # Maximum variation for full game scenario training
        offset_x = random.uniform(-150, 150)
        offset_y = random.uniform(-100, 100)
        
        x = max(60, min(self.field_width - 60, center_x + offset_x))
        y = max(60, min(self.field_height - 60, center_y + offset_y))
        
        return (x, y)
    
    def get_phase_info(self, phase: str) -> dict:
        """
        Get detailed information about a training phase.
        
        Args:
            phase: Phase name
            
        Returns:
            dict: Phase configuration and description
        """
        phase_info = {
            "ball_awareness": {
                "name": "Ball Awareness",
                "description": "Learn basic ball-seeking and control behavior",
                "episode_length": self.episode_lengths["ball_awareness"],
                "timestep_range": "0 - 25,000",
                "key_skills": ["Ball proximity", "Movement toward ball", "Basic control", "Boundary avoidance"],
                "reward_focus": ["Ball control bonus", "Proximity rewards", "Movement alignment", "Boundary penalties"]
            },
            "basic_soccer": {
                "name": "Basic Soccer",
                "description": "Add shooting, passing, and basic positioning",
                "episode_length": self.episode_lengths["basic_soccer"],
                "timestep_range": "25,000 - 75,000",
                "key_skills": ["Shooting", "Passing", "Team spread", "Action rewards"],
                "reward_focus": ["Action execution", "Team coordination", "Strategic positioning", "Enhanced goals"]
            },
            "competitive_soccer": {
                "name": "Competitive Soccer",
                "description": "Full strategic and tactical gameplay",
                "episode_length": self.episode_lengths["competitive_soccer"],
                "timestep_range": "75,000+",
                "key_skills": ["Strategic passing", "Role-based positioning", "Formation play", "Advanced tactics"],
                "reward_focus": ["Forward passes", "Shot accuracy", "Formation rewards", "Positional play"]
            }
        }
        
        return phase_info.get(phase, phase_info["competitive_soccer"])
    
    def should_transition_phase(self, current_phase: str, total_timesteps: int) -> Tuple[bool, str]:
        """
        Check if it's time to transition to the next phase.
        
        Args:
            current_phase: Current training phase
            total_timesteps: Total timesteps completed
            
        Returns:
            Tuple[bool, str]: (should_transition, next_phase)
        """
        # No transitions if using manual phase or curriculum disabled
        if self.manual_phase or not self.enable_curriculum:
            return False, current_phase
        
        # Check for phase transitions
        if (current_phase == "ball_awareness" and 
            total_timesteps >= self.phase_thresholds["basic_soccer"]):
            return True, "basic_soccer"
        
        if (current_phase == "basic_soccer" and 
            total_timesteps >= self.phase_thresholds["competitive_soccer"]):
            return True, "competitive_soccer"
        
        return False, current_phase
    
    def get_training_summary(self, total_timesteps: int) -> dict:
        """
        Get summary of current training progress and curriculum status.
        
        Args:
            total_timesteps: Total timesteps completed
            
        Returns:
            dict: Training progress summary
        """
        current_phase = self.get_current_phase(total_timesteps)
        phase_info = self.get_phase_info(current_phase)
        
        # Calculate progress within current phase
        if current_phase == "ball_awareness":
            phase_progress = min(1.0, total_timesteps / self.phase_thresholds["basic_soccer"])
            next_milestone = self.phase_thresholds["basic_soccer"]
        elif current_phase == "basic_soccer":
            phase_start = self.phase_thresholds["basic_soccer"]
            phase_length = self.phase_thresholds["competitive_soccer"] - phase_start
            phase_progress = min(1.0, (total_timesteps - phase_start) / phase_length)
            next_milestone = self.phase_thresholds["competitive_soccer"]
        else:
            phase_progress = 1.0  # Completed all phases
            next_milestone = None
        
        return {
            "current_phase": current_phase,
            "phase_info": phase_info,
            "total_timesteps": total_timesteps,
            "phase_progress": phase_progress,
            "next_milestone": next_milestone,
            "curriculum_enabled": self.enable_curriculum,
            "manual_override": self.manual_phase is not None,
            "episode_length": self.get_episode_length(current_phase)
        }