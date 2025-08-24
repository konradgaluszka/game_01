# Virtual Environment Setup Complete



## Quick Start:

### Windows:
```cmd
# Double-click activate_venv.bat or run:
activate_venv.bat
```

### Unix/Linux/Mac:
```bash
chmod +x activate_venv.sh
./activate_venv.sh
```

### Manual Activation:
```bash
# Windows (Git Bash/MSYS2)
source venv/Scripts/activate

# Linux/Mac
source venv/bin/activate
```

## Available Commands (after activation):
```bash
python main.py                       # Run the soccer game
python ai/train.py train [steps]     # Train AI model  
python ai/train.py evaluate [model]  # Evaluate AI model
python setup_ai.py                   # Setup and train initial model
```

## Notes:
- The virtual environment isolates all AI dependencies
- Main game runs with or without AI dependencies
- When no trained model exists, team_2 plays without AI
- When a trained model exists at `ai/models/soccer_ai_final.zip`, team_2 uses AI

## ✅ SETUP COMPLETE!

The virtual environment has been created and all dependencies installed successfully. You can now:

1. **Train AI models:** `python ai/train.py train [timesteps]`
2. **Play against AI:** `python main.py` (team_1 = keyboard, team_2 = AI)
3. **Evaluate models:** `python ai/train.py evaluate [model_path]`

The initial model has been trained and saved to `ai/models/soccer_ai_final.zip`.

python ai/train_ctde.py train --timesteps 100000 --model-name ctde_iterative --curriculum --self-play --iterative-selfplay --opponent-model ai/models/ctde_curriculum_final.zip --update-freq 10000 --n-envs 4