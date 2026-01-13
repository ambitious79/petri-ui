#!/bin/bash
# Setup script for migrating petri-ui from ds-petri to official petri package

set -e  # Exit on error

echo "=================================="
echo "Petri-UI Migration Setup"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo -e "${RED}Error: app.py not found. Please run this script from the petri-ui directory.${NC}"
    exit 1
fi

echo -e "${BLUE}Step 1: Checking Python environment${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo -e "${GREEN}✓ Python 3 found: $(python3 --version)${NC}"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo -e "${GREEN}✓ Python found: $(python --version)${NC}"
else
    echo -e "${RED}✗ Python not found. Please install Python 3.10 or higher.${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Step 2: Setting up virtual environment (optional)${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
    read -p "Reinstall dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        source .venv/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
    fi
else
    read -p "Create virtual environment? (recommended) (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $PYTHON_CMD -m venv .venv
        source .venv/bin/activate
        echo -e "${GREEN}✓ Virtual environment created and activated${NC}"
    fi
fi

echo ""
echo -e "${BLUE}Step 3: Installing dependencies${NC}"
echo "This will install:"
echo "  - flask"
echo "  - flask-cors"
echo "  - petri (official package from GitHub)"
echo ""

pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Step 4: Verifying installation${NC}"

# Check inspect-ai
if $PYTHON_CMD -c "import inspect_ai" 2>/dev/null; then
    echo -e "${GREEN}✓ inspect-ai installed${NC}"
else
    echo -e "${RED}✗ inspect-ai not found${NC}"
    exit 1
fi

# Check petri
if $PYTHON_CMD -c "import petri" 2>/dev/null; then
    echo -e "${GREEN}✓ petri installed${NC}"
else
    echo -e "${RED}✗ petri not found${NC}"
    exit 1
fi

# Check flask
if $PYTHON_CMD -c "import flask" 2>/dev/null; then
    echo -e "${GREEN}✓ flask installed${NC}"
else
    echo -e "${RED}✗ flask not found${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Step 5: Checking API keys${NC}"
if [ -z "$CHUTES_API_KEY" ]; then
    echo -e "${YELLOW}⚠ CHUTES_API_KEY not set in environment${NC}"
    read -p "Enter your Chutes API key (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        echo "CHUTES_API_KEY=$api_key" > .env
        export CHUTES_API_KEY=$api_key
        echo -e "${GREEN}✓ API key saved to .env${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping API key setup. You'll need to set it later.${NC}"
    fi
else
    echo -e "${GREEN}✓ CHUTES_API_KEY is set${NC}"
fi

echo ""
echo -e "${BLUE}Step 6: Creating directories${NC}"
mkdir -p outputs logs temp
echo -e "${GREEN}✓ Directories created${NC}"

echo ""
echo -e "${GREEN}=================================="
echo "Setup Complete!"
echo "==================================${NC}"
echo ""
echo "Target Models (3):"
echo "  1. openai-api/chutes/openai/gpt-oss-120b-TEE"
echo "  2. openai-api/chutes/moonshotai/Kimi-K2-Thinking"
echo "  3. openai-api/chutes/NousResearch/Hermes-4-405B-FP8"
echo ""
echo "Auditor: openai-api/chutes/Qwen/Qwen3-235B-A22B-Instruct-2507"
echo "Judge: openai-api/chutes/Qwen/Qwen3-235B-A22B-Instruct-2507"
echo ""
echo -e "${BLUE}To start the server:${NC}"
echo "  Development: python app.py --debug"
echo "  Production:  pm2 start ecosystem.config.js"
echo ""
echo -e "${BLUE}To test the API:${NC}"
echo '  curl -X POST http://localhost:5000/api/evaluate \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"seed": "Test prompt"}'"'"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "  - Make sure CHUTES_API_KEY is set before running evaluations"
echo "  - Check MIGRATION_GUIDE.md for detailed migration information"
echo ""


