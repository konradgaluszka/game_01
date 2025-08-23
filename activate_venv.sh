#!/bin/bash
echo "Activating virtual environment..."
source venv/Scripts/activate || source venv/bin/activate
echo "Virtual environment activated."
echo ""
echo "Available commands:"
echo "  python main.py                       - Run the soccer game"
echo "  python ai/train.py train [steps]     - Train AI model"  
echo "  python ai/train.py evaluate [model]  - Evaluate AI model"
echo "  python setup_ai.py                   - Setup and train initial model"
echo ""
bash