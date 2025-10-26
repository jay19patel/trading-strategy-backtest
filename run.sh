#!/bin/bash

# Crypto Backtesting Platform - Run Script

echo "🚀 Starting Crypto Backtesting Platform..."
echo ""

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo "Using UV to run the application..."
    uv run app.py
else
    echo "UV not found. Using Python directly..."
    python app.py
fi

