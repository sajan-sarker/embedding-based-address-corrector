#!/bin/bash

# This script sets up the environment for the embedding-based address corrector project.

if ! command -v python &> /dev/null
then
    echo "Python could not be found. Please install Python3 before running this script."
    exit
fi

if ! command -v pip &> /dev/null
then
    echo "pip could not be found. Please install pip before running this script."
    exit
fi

echo "Python version:"
python --version

echo "pip version:"
pip --version

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/Scripts/activate

# check if the virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Failed to activate virtual environment. Please check your Python installation."
    exit 1
else
    echo "Virtual environment activated."
fi

# install required packages
echo "Installing required packages..."
python -m pip install -r requirements.txt 