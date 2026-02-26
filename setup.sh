#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  setup.sh  —  One-shot setup for the RAG CLI
#  Run once:  bash setup.sh
# ═══════════════════════════════════════════════════════════════
set -e

GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; RESET="\033[0m"; BOLD="\033[1m"
ok()   { echo -e "${GREEN}  ✓  $*${RESET}"; }
warn() { echo -e "${YELLOW}  ⚠  $*${RESET}"; }
info() { echo -e "${CYAN}  ›  $*${RESET}"; }
err()  { echo -e "${RED}  ✗  $*${RESET}"; exit 1; }

echo -e "\n${BOLD}${CYAN}  RAG CLI — Setup${RESET}\n"

# ── 1. Python ────────────────────────────────────────────────────────────────
info "Checking Python ≥ 3.10…"
PY=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
MAJOR=$(echo $PY | cut -d. -f1); MINOR=$(echo $PY | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
  err "Python $PY found — need 3.10+. Install from https://python.org"
fi
ok "Python $PY"

# ── 2. Virtual env ───────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  info "Creating virtual environment (.venv)…"
  python3 -m venv .venv
  ok "Virtual environment created"
else
  ok "Virtual environment already exists"
fi

source .venv/bin/activate
info "Activated .venv"

# ── 3. pip upgrade ───────────────────────────────────────────────────────────
info "Upgrading pip…"
pip install --quiet --upgrade pip

# ── 4. Python deps ───────────────────────────────────────────────────────────
info "Installing Python packages (this takes ~2 min first time)…"
pip install --quiet \
  "llama-index>=0.10.0" \
  "llama-index-vector-stores-pinecone>=0.1.0" \
  "llama-index-llms-ollama>=0.1.0" \
  "llama-index-embeddings-huggingface>=0.2.0" \
  "pinecone-client>=3.0.0" \
  "sentence-transformers>=2.7.0" \
  "transformers>=4.40.0" \
  "torch>=2.2.0" \
  "pdfplumber>=0.10.0" \
  "python-docx>=1.1.0" \
  "python-dotenv>=1.0.0" \
  "httpx>=0.27.0" \
  "rich>=13.0.0"
ok "Python packages installed"

# ── 5. .env ──────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  info "Creating .env from template…"
  cp .env.example .env
  warn ".env created — open it and add your PINECONE_API_KEY"
else
  ok ".env already exists"
fi

# ── 6. Ollama ────────────────────────────────────────────────────────────────
info "Checking Ollama…"
if ! command -v ollama &>/dev/null; then
  warn "Ollama not installed. Installing now…"
  if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &>/dev/null; then
      brew install --cask ollama
    else
      warn "Homebrew not found. Download Ollama from https://ollama.com/download"
    fi
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    warn "Windows: download from https://ollama.com/download"
  fi
else
  ok "Ollama $(ollama --version 2>/dev/null | head -1)"
fi

# ── 7. Pull llama3 ────────────────────────────────────────────────────────────
info "Pulling llama3 model (4.7 GB — skip if already done)…"
if ollama list 2>/dev/null | grep -q "llama3"; then
  ok "llama3 already pulled"
else
  ollama pull llama3
  ok "llama3 pulled"
fi

# ── done ─────────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}${GREEN}  ═══════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}    Setup complete!${RESET}"
echo -e "${BOLD}${GREEN}  ═══════════════════════════════════════${RESET}"
echo
echo -e "  Next steps:"
echo -e "  ${YELLOW}1.${RESET}  Edit ${CYAN}.env${RESET} → add your ${CYAN}PINECONE_API_KEY${RESET}"
echo -e "  ${YELLOW}2.${RESET}  Start Ollama:   ${CYAN}ollama serve${RESET}"
echo -e "  ${YELLOW}3.${RESET}  Run the CLI:    ${CYAN}source .venv/bin/activate && python rag_cli.py${RESET}"
echo
