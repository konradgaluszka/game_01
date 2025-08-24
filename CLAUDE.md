# Claude Code Project Instructions

## Project Overview
This is a soccer AI training project that uses reinforcement learning to train an AI team to play soccer against a rule-based opponent. The main game is human vs AI, with team_1 (red, left side) controlled by keyboard and team_2 (blue, right side) controlled by trained AI models.

## Key Project Commands

### Training Commands
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# NEW: OpponentAI training (recommended) - Uses intelligent rule-based opponent
python ai/train.py train --timesteps 100000 --model-name opponent_ai_training

# Full curriculum training (with OpponentAI)
python ai/train.py train --timesteps 100000 --model-name curriculum_soccer_ai --curriculum

# Self-play training (AI vs AI) - Most Effective!
python ai/train.py train --timesteps 50000 --model-name selfplay_ai --self-play --opponent-model ai/models/improved_ball_chasing_final.zip
python ai/train.py train --timesteps 100000 --model-name selfplay_curriculum --curriculum --self-play --opponent-model ai/models/curriculum_ai_final.zip

# Iterative self-play (opponent updates during training) - ULTIMATE!
python ai/train.py train --timesteps 50000 --model-name iterative_selfplay --self-play --iterative-selfplay --opponent-model ai/models/improved_ball_chasing_final.zip --update-freq 5000
python ai/train.py train --timesteps 100000 --model-name iterative_curriculum --curriculum --self-play --iterative-selfplay --opponent-model ai/models/basic_ai_final.zip --update-freq 8000

# Phase-specific training
python ai/train.py train --timesteps 25000 --model-name phase1_ai --phase ball_awareness
python ai/train.py train --timesteps 50000 --model-name phase2_ai --phase basic_soccer --load ai/models/phase1_ai_final.zip
python ai/train.py train --timesteps 50000 --model-name phase3_ai --phase competitive_soccer --load ai/models/phase2_ai_final.zip

# Traditional training (no curriculum)
python ai/train.py train --timesteps 100000 --model-name traditional_ai

# NEW: CTDE (Centralized Training Decentralized Execution) Training - ADVANCED!
python ai/train_ctde.py train --timesteps 100000 --model-name ctde_soccer_ai --n-envs 4

# CTDE with curriculum learning
python ai/train_ctde.py train --timesteps 100000 --model-name ctde_curriculum --curriculum --n-envs 4

# CTDE self-play training
python ai/train_ctde.py train --timesteps 50000 --model-name ctde_selfplay --self-play --opponent-model ai/models/ctde_soccer_ai_final.zip --n-envs 4

# CTDE iterative self-play (most advanced)
python ai/train_ctde.py train --timesteps 100000 --model-name ctde_iterative --curriculum --self-play --iterative-selfplay --opponent-model ai/models/ctde_curriculum_final.zip --update-freq 10000 --n-envs 4

# RECOMMENDED: Simple CTDE training (easier to use, works reliably)
python ai/train_ctde_simple.py train --timesteps 50000 --model-name simple_ctde --curriculum --n-envs 4

# Simple CTDE self-play
python ai/train_ctde_simple.py train --timesteps 50000 --model-name simple_ctde_selfplay --self-play --opponent-model ai/models/simple_ctde_simple_ctde_final.zip --n-envs 4
```

### Testing Commands
```bash
# Run the main game (loads trained AI automatically)
python main.py

# Test training environment directly
python ai/soccer_env.py

# Test CTDE implementation
python test_ctde.py

# Evaluate CTDE models
python ai/train_ctde.py evaluate --model-path ai/models/ctde_soccer_ai_final.zip --episodes 5

# Evaluate Simple CTDE models (recommended)
python ai/train_ctde_simple.py evaluate --model-path ai/models/simple_ctde_simple_ctde_final.zip --episodes 5
```

### Lint/TypeCheck Commands
```bash
# No specific linting configured yet
# TODO: Add flake8, black, or similar Python linting tools
```

## Project Structure

### Core Game Files
- `main.py` - Main game entry point with human vs AI gameplay
- `game/match.py` - Match logic, scoring, and game state management
- `stadium/pitch.py` - Soccer field with physics boundaries and goals
- `stadium/goal.py` - Goal posts and scoring detection
- `player/player.py` - Individual player physics and ball control
- `player/team.py` - Team formation and player management
- `player/ball.py` - Ball physics object
- `common/Vector.py` - 2D vector mathematics utility

### AI Training System
- `ai/soccer_env.py` - Gymnasium environment for RL training
- `ai/train.py` - PPO training script using stable-baselines3
- `ai/ai_controller.py` - Real-time AI controller for live gameplay
- `ai/models/` - Directory for saved trained models (.zip files)

### Configuration Files
- `requirements.txt` - Python dependencies
- `venv/` - Virtual environment directory

## Training Details

### Action Space
- 7 actions per player × 5 players = 35-dimensional multi-discrete action space
- Actions: 0=nothing, 1=up, 2=down, 3=left, 4=right, 5=shoot, 6=pass

### Observation Space
- 62-dimensional continuous observation vector
- Includes: ball position/velocity, player positions/velocities, relative positions, field context

### Reward Function
- Goal scoring: +1000 (team_2 scores), -500 (team_1 scores)
- Ball control: +15 for possession, +5 for proximity
- Ball progress: +8 for ball near goal, +10 for ball moving toward goal
- Boundary penalties: -15 for edges, -20 for corners
- Shooting opportunities: +25 for close-range, +12 for shooting range
- Passing opportunities: +15 for forward passes

### Training Parameters
- Algorithm: PPO (Proximal Policy Optimization)
- Episode length: 500 steps (8.3 seconds at 60 FPS)
- Default training: 100,000 timesteps
- Model saves: Every 10,000 timesteps

## Key Technical Details

### Dependencies
- pygame: Game rendering and input
- pymunk: 2D physics simulation
- stable-baselines3: Reinforcement learning algorithms
- gymnasium: RL environment interface
- numpy: Numerical computations
- tensorboard: Training visualization

### Ball Control System
- 4-spring dribbling system for natural ball control
- Players automatically control ball when within DRIBBLE_DISTANCE
- Shooting/passing removes springs and applies impulse

### Opponent AI
- **NEW OpponentAI System**: Sophisticated rule-based opponent with role-based behaviors
  - Goalkeeper: Stays in goal area, intelligent positioning, ball clearing
  - Defenders: Maintain defensive line, pressure opponents, pass to attackers 
  - Attackers: Push forward, attempt shots, make attacking runs
  - Uses same player physics as human control for fair competition
  
- **Legacy Opponent AI**: Simple rule-based AI during training
  - Attack mode: Shoots, passes, and advances toward goal
  - Defense mode: Pressures ball carrier and takes defensive positions
  - Contest mode: Competes for loose balls

### NEW: CTDE (Centralized Training Decentralized Execution) Architecture
- **Individual Agent Observations**: 68-dimensional per agent with role-based features
  - Agent identity and role embeddings (8 dims)
  - Ball information relative to agent (6 dims)  
  - Agent state and ball possession (5 dims)
  - Teammate information (20 dims: 4 teammates × 5 dims)
  - Opponent information (15 dims: 5 opponents × 3 dims)
  - Field context for agent (10 dims)
  - Match state (4 dims)

- **Global Critic Observations**: 102-dimensional for centralized training
  - Ball state (4 dims)
  - All team_2 players with roles (50 dims: 5 players × 10 dims)
  - All team_1 players (30 dims: 5 players × 6 dims)
  - Global field context and team statistics (14 dims)
  - Match state (4 dims)

- **Role-Based Player Assignments**:
  - Player 0: Goalkeeper (defensive specialist)
  - Player 1: Defender (defensive positioning)
  - Player 2-3: Midfielders (ball control and passing)
  - Player 4: Forward (attacking specialist)

- **Training Benefits**:
  - Individual player specialization through role embeddings
  - Better team coordination via centralized critic
  - More sample-efficient training with shared parameters
  - Enhanced self-play capabilities with role diversity

## Common Issues & Solutions

### Training Issues
- **Edge-seeking behavior**: Fixed with strong boundary penalties (-15 to -20)
- **No shooting/passing**: Fixed with explicit action rewards (+25 to +45)
- **Weak opponent**: Fixed with competitive opponent AI (0.8-0.9x force)

### Technical Issues
- **Import errors**: Ensure virtual environment is activated
- **Pygame initialization**: Always call pygame.init() even in headless mode
- **Vec2d mutability**: Create new Vec2d objects instead of modifying existing ones

### Performance Issues
- **Slow training**: Reduce observation space or use headless mode
- **Memory issues**: Reduce buffer size or timesteps per training run

## Development Guidelines

### Code Style
- Add comprehensive docstrings to all functions and classes
- Use type hints where appropriate
- Follow existing naming conventions
- Comment complex physics or RL logic

### Testing
- Test changes with short training runs (1000-5000 timesteps)
- Verify AI behavior in live gameplay after training
- Check reward function balance with tensorboard logs

### Model Management
- Save models with descriptive names including timesteps
- Keep successful models backed up
- Document significant hyperparameter changes

## Future Improvements

### Potential Enhancements
- Add more sophisticated opponent strategies
- Implement curriculum learning (start easy, increase difficulty)
- Add team formations and tactical variations
- Implement multi-agent training (train both teams)
- Add different field sizes and game modes

### Technical Debt
- Add proper linting and code formatting
- Implement unit tests for critical components
- Add configuration files for hyperparameters
- Improve error handling and logging

## Notes for Claude Code Sessions

### Always Remember
1. Activate virtual environment before running any training
2. Use `python ai/train.py` for training, `python main.py` for gameplay
3. Check tensorboard logs to monitor training progress
4. Boundary penalties must be strong enough to prevent edge-seeking
5. Opponent AI needs to be competitive for effective training

### Quick Development Workflow
1. Make code changes
2. Test with short training run (1000 timesteps)
3. Verify behavior in live gameplay
4. If successful, run longer training (10000+ timesteps)
5. Save successful models with descriptive names

### Debugging Training Issues
1. Check reward function balance in tensorboard
2. Verify action space is being used correctly
3. Test opponent AI behavior in isolation
4. Examine observation space for NaN or inf values
5. Monitor episode lengths and termination conditions