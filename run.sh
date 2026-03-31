#!/usr/bin/env bash
# Launch Revision Radar using the bwater virtual environment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".env" ]; then
  echo "⚠️  No .env file found. Creating from .env.example..."
  cp .env.example .env
  echo "   → Edit .env and add your ANTHROPIC_API_KEY before running again."
  exit 1
fi

echo "🚀  Starting Revision Radar..."
bwater/bin/streamlit run app.py --server.port 8501 --server.headless false
