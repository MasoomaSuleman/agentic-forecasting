#!/bin/bash
# Run from the repo root. Sets up the workspace venv and launches Jupyter.

if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    uv venv .venv
fi

uv sync --dev

echo "Virtual environment activated and dependencies synced."

# Install Jupyter kernel
uv run ipython kernel install --user --name=aieng-forecasting --display-name "AIEng Forecasting"
echo "Jupyter kernel installed."

# Start Jupyter lab
echo "Starting Jupyter lab..."
uv run jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token=''
