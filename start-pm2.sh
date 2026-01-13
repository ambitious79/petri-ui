#!/bin/bash

echo "Starting Petri-UI with PM2..."
echo ""

# Activate virtual environment(venv) if it exists
if [ -d "venv" ]; then
    echo "✓ Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "✓ Activating virtual environment (../venv)..."
    source ../venv/bin/activate
fi

# Activate virtual environment(.venv) if it exists
if [ -d ".venv" ]; then
    echo "✓ Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# Load .env file if it exists
if [ -f ".env" ]; then
    echo "✓ Loading environment variables from .env file..."
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Check PM2
if ! command -v pm2 &> /dev/null; then
    echo "❌ PM2 not installed. Run: npm install -g pm2"
    exit 1
fi
echo "✓ PM2 is installed"

# Check Python packages
echo ""
echo "Checking dependencies..."

# Check inspect-ai
if ! python3 -c "import inspect_ai" 2>/dev/null; then
    echo "❌ inspect-ai not installed."
    echo "   Run: pip install -r requirements.txt"
    exit 1
fi
echo "✓ inspect-ai is installed"

# Check and update petri and astro-petri from local repository
PETRI_LOCAL_PATH="/home/trishool/petri"
ASTRO_PETRI_LOCAL_PATH="/home/trishool/astro-petri"

# Install petri (if available)
if [ -d "$PETRI_LOCAL_PATH" ]; then
    echo "✓ Local petri repository found at $PETRI_LOCAL_PATH"
    echo "  Installing petri in editable mode..."
    pip install -e "$PETRI_LOCAL_PATH" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ petri installed from local repository (editable mode)"
        echo "  Your local changes will be automatically picked up!"
    else
        echo "⚠️  Failed to install from local repository, checking if already installed..."
        if ! python3 -c "import petri" 2>/dev/null; then
            echo "❌ petri package not installed."
            echo "   Run: pip install -r requirements.txt"
            exit 1
        fi
        echo "✓ petri package is installed (from pip)"
    fi
else
    # Fallback: check if petri is installed via pip
    if ! python3 -c "import petri" 2>/dev/null; then
        echo "❌ petri package not installed."
        echo "   Run: pip install -r requirements.txt"
        exit 1
    fi
    echo "✓ petri package is installed (from pip)"
fi

# Install astro-petri (provides standard petri/audit task)
if [ -d "$ASTRO_PETRI_LOCAL_PATH" ]; then
    echo "✓ Local astro-petri repository found at $ASTRO_PETRI_LOCAL_PATH"
    echo "  Installing astro-petri in editable mode (with fresh reinstall)..."
    
    # Clear Python cache to ensure fresh code is loaded
    find "$ASTRO_PETRI_LOCAL_PATH" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
    find "$ASTRO_PETRI_LOCAL_PATH" -type f -name "*.pyc" -delete 2>/dev/null
    
    # Use uv if available, otherwise fall back to pip
    if command -v uv &> /dev/null; then
        uv pip install --reinstall --no-deps -e "$ASTRO_PETRI_LOCAL_PATH"
        if [ $? -eq 0 ]; then
            echo "✓ astro-petri installed from local repository (editable mode with uv)"
            echo "  Your local changes will be automatically picked up!"
        else
            echo "⚠️  Failed to install astro-petri with uv"
        fi
    else
        pip install --force-reinstall --no-deps -e "$ASTRO_PETRI_LOCAL_PATH"
        if [ $? -eq 0 ]; then
            echo "✓ astro-petri installed from local repository (editable mode with pip)"
            echo "  Your local changes will be automatically picked up!"
        else
            echo "⚠️  Failed to install astro-petri with pip"
        fi
    fi
else
    # Check if astro-petri is available
    if ! python3 -c "import astro_petri" 2>/dev/null; then
        echo "⚠️  astro-petri not found in local path: $ASTRO_PETRI_LOCAL_PATH"
        echo "   Note: This app uses the standard petri/audit task from astro-petri"
    else
        echo "✓ astro-petri package is installed"
    fi
fi

# Check Flask
if ! python3 -c "import flask" 2>/dev/null; then
    echo "❌ flask not installed."
    echo "   Run: pip install -r requirements.txt"
    exit 1
fi
echo "✓ flask is installed"

# Check API key
echo ""
if [ -z "$CHUTES_API_KEY" ]; then
    echo "⚠️  Warning: CHUTES_API_KEY not set"
    echo "   Evaluations will fail without API key."
    echo ""
    if [ -f ".env" ]; then
        echo "   .env file exists but CHUTES_API_KEY not found."
        echo "   Make sure your .env file contains:"
        echo "   CHUTES_API_KEY=your_actual_key_here"
        echo "   (no quotes, no spaces around =)"
    else
        echo "   Create a .env file with:"
        echo "   CHUTES_API_KEY=your_actual_key_here"
        echo "   Or set it with: export CHUTES_API_KEY=your_key"
    fi
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ CHUTES_API_KEY is set"
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p logs outputs temp
echo "✓ Directories created"

# Start or restart
echo ""
if pm2 list | grep -q "petri-ui"; then
    echo "Restarting petri-ui..."
    pm2 restart petri-ui
else
    echo "Starting petri-ui..."
    pm2 start ecosystem.config.js
fi

pm2 save

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✓ Petri-UI is running!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Server:      http://localhost:5000"
echo "View logs:   pm2 logs petri-ui"
echo "Stop:        pm2 stop petri-ui"
echo "Restart:     pm2 restart petri-ui"
echo ""
echo "Target Models (3):"
echo "  • openai-api/chutes/openai/gpt-oss-120b-TEE"
echo "  • openai-api/chutes/moonshotai/Kimi-K2-Thinking-TEE"
echo "  • openai-api/chutes/NousResearch/Hermes-4-405B-FP8-TEE"
echo ""
echo "Auditor: openai-api/chutes/Qwen/Qwen3-235B-A22B-Instruct-2507"
echo "Judge:   openai-api/chutes/Qwen/Qwen3-235B-A22B-Instruct-2507"
echo ""
echo "To enable auto-start on boot:"
echo "  pm2 startup"
echo "  # Follow instructions, then run: pm2 save"
echo "═══════════════════════════════════════════════════════════"

