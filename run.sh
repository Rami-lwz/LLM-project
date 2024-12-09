#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask application using gunicorn
exec gunicorn -w 4 -b 0.0.0.0:8000 app:app