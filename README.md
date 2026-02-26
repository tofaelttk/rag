# RAG CLI — Terminal Document Intelligence

Ask anything about your PDF/TXT/DOCX files — entirely from the terminal.
**No browser. No server. No frontend.**

```
Stack: LlamaIndex · Pinecone · Ollama (Llama3) · HuggingFace BGE Embeddings
```

---

## What you need

| Thing | Why |
|---|---|
| Python 3.10+ | runs the CLI |
| Pinecone account (free tier works) | vector store |
| Ollama installed locally | runs Llama3 LLM |
| ~5 GB disk | for llama3 model |

---

## 1 — Get the code

```bash
git clone https://github.com/CamillyShrestha/Lrag.git
cd Lrag
```

---

## 2 — Python setup

```bash
# create virtual environment
python3 -m venv .venv

# activate it
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# install all packages
pip install \
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
  "httpx>=0.27.0"
```

OR run the one-shot setup script:
```bash
bash setup.sh
```

---

## 3 — Pinecone API key

1. Go to **https://app.pinecone.io** — sign up free
2. Dashboard → API Keys → copy your key

```bash
cp .env.example .env
nano .env        # or: code .env  /  vim .env
```

Your `.env`:
```
PINECONE_API_KEY=pcsk_xxxxxxxxxx...

# optional (defaults are fine)
PINECONE_INDEX_NAME=rag-demo
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
```

---

## 4 — Install Ollama + pull model

### macOS
```bash
brew install --cask ollama
# OR download from https://ollama.com/download
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download from **https://ollama.com/download**

### Start + pull model
```bash
# Terminal 1 — keep running
ollama serve

# Terminal 2 — one time only (~4.7 GB)
ollama pull llama3
ollama list    # verify
```

Want a smaller model?
```bash
ollama pull llama3.2    # 2 GB, faster
ollama pull mistral     # 4.1 GB, alternative
# then set OLLAMA_MODEL=llama3.2 in .env
```

---

## 5 — Run the CLI

### Interactive mode (prompts for files)
```bash
source .venv/bin/activate
python rag_cli.py
```

```
  File 1 path (or Enter to finish): /path/to/report.pdf
  ✓ Added: report.pdf  (842 KB)
  File 2 path (or Enter to finish):     ← press Enter to finish

  ... indexes files ...

  You ❯ What are the main findings?
  Assistant ❯  The report concludes that...
```

### Pass files directly
```bash
python rag_cli.py report.pdf
python rag_cli.py paper.pdf notes.txt meeting.docx
python rag_cli.py ~/Documents/thesis.pdf
```

### Reuse existing Pinecone index (skip re-ingestion)
```bash
python rag_cli.py --reuse
```

---

## Chat commands

| Command | What it does |
|---|---|
| Type + Enter | Ask a question |
| `/sources` | Toggle showing retrieved chunks |
| `/top 8` | Change Top-K retrieval (default 5) |
| `/help` | Show commands |
| `/exit` `/quit` `/q` | Quit |
| `Ctrl+C` | Also quits |

---

## Troubleshooting

**`PINECONE_API_KEY not set`**
```bash
cat .env   # check key is there, no spaces around =
```

**`Ollama not reachable`**
```bash
ollama serve    # run in a separate terminal, leave it open
```

**`llama3 not found`**
```bash
ollama pull llama3
```

**Slow on first run** — BGE model downloads ~130 MB once, then cached.

**Apple Silicon OOM** — add to `.env`:
```
FORCE_CPU_EMBEDDINGS=1
```

---

## Quick reference

```bash
# SETUP (one time)
git clone https://github.com/CamillyShrestha/Lrag.git && cd Lrag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env      # add PINECONE_API_KEY
ollama serve &
ollama pull llama3

# RUN
python rag_cli.py                      # interactive
python rag_cli.py doc.pdf              # ingest + chat
python rag_cli.py a.pdf b.txt c.docx   # multiple files
python rag_cli.py --reuse              # reuse existing index
```

---

## Repo structure

```
Lrag/
├── rag_cli.py        ← the CLI (this is all you need)
├── .env.example      ← copy to .env, add PINECONE_API_KEY
├── setup.sh          ← one-shot install script
├── requirements.txt  ← pip packages
└── README.md
```
