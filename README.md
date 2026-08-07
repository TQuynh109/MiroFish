<div align="center">

<img src="/home/anman/intern/quynhht/MiroFish/static/figs/SimPetro_logo_compressed.jpeg" alt="SimPetro Logo" width="75%"/>

A Simple and Universal Swarm Intelligence Engine, Predicting Anything

<a href="https://www.shanda.com/" target="_blank"><img src="/home/anman/intern/quynhht/MiroFish/static/figs/shanda_logo.png" alt="Shanda" height="40"/></a>

</div>

## ⚡ Overview

**SimPetro** is a new-generation AI prediction engine built on multi-agent technology. By extracting seed information from the real world (breaking news, policy drafts, financial signals), it automatically constructs a high-fidelity parallel digital world. Inside this space, thousands of agents — each with an independent persona, long-term memory, and behavioral logic — interact freely and evolve socially. You can inject variables dynamically from a "god's-eye view" to precisely simulate how the future unfolds — **let the future play out in a digital sandbox, so decisions win after a hundred simulated battles**.

> All you need to do: upload seed material (a data analysis report or an interesting story) and describe your prediction goal in natural language.
> SimPetro returns: a detailed prediction report, plus a high-fidelity digital world you can interact with in depth.

### Our Vision

SimPetro aims to build a swarm-intelligence mirror of reality. By capturing the group-level emergence that arises from individual interactions, it breaks through the limits of traditional prediction:

- **At the macro level**: a rehearsal lab for decision-makers, letting policy and PR fail safely at zero risk
- **At the micro level**: a creative sandbox for individual users — whether simulating a novel's ending or exploring a wild idea, it stays fun, playful, and within reach

From serious forecasting to playful simulation, we let every "what if" show its result, and make predicting anything possible.

## 🔄 Workflow

1. **Graph construction**: real-world seed extraction & individual/collective memory injection & GraphRAG construction
2. **Environment setup**: entity-relation extraction & persona generation & simulation parameters injected by the environment-config agent
3. **Run simulation**: dual-platform parallel simulation & automatic parsing of the prediction goal & dynamic updating of temporal memory
4. **Report generation**: the ReportAgent uses a rich toolset to interact deeply with the post-simulation environment
5. **Deep interaction**: chat with any character in the simulated world & chat with the ReportAgent

## 🚀 Quick Start

### Option 1: Run from source (recommended)

#### Prerequisites

| Tool | Version | Purpose | Check |
|------|---------|---------|-------|
| **Node.js** | 18+ | Frontend runtime, includes npm | `node -v` |
| **Python** | ≥3.11, ≤3.12 | Backend runtime | `python --version` |
| **uv** | latest | Python package manager | `uv --version` |

#### 1. Configure environment variables

```bash
# Copy the example config file
cp .env.example .env

# Edit .env and fill in the required API keys
```

**Required environment variables:**

```env
# LLM API config (any LLM API compatible with the OpenAI SDK format)
# Recommended: the qwen-plus model on Alibaba Bailian: https://bailian.console.aliyun.com/
# Note: token usage can be heavy — try a run with fewer than 40 rounds first
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus

# Zep Cloud config
# The free monthly quota is enough for simple usage: https://app.getzep.com/
ZEP_API_KEY=your_zep_api_key
```

#### 2. Install dependencies

```bash
# Install everything in one command (root + frontend + backend)
npm run setup:all
```

Or step by step:

```bash
# Install Node dependencies (root + frontend)
npm run setup

# Install Python dependencies (backend, virtualenv created automatically)
npm run setup:backend
```

#### 3. Start the services

```bash
# Start frontend and backend together (run from the project root)
npm run dev
```

**Service URLs:**
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:5001`

**Start individually:**

```bash
npm run backend   # backend only
npm run frontend  # frontend only
```

### Option 2: Docker deployment

```bash
# 1. Configure environment variables (same as source deployment)
cp .env.example .env

# 2. Pull the images and start
docker compose up -d
```

By default it reads the `.env` in the project root and maps ports `3000 (frontend) / 5001 (backend)`.

> Mirror registry addresses for faster pulls are provided as comments in `docker-compose.yml`; swap them in as needed.

## Contributing
We sincerely thank all team members for their dedication and time in developing this project. Your collaboration and sense of responsibility have contributed to achieving the best possible results.

| **Name**              | **Major**     | **University**                 |
|-----------------------|---------------|--------------------------------|
| Huynh Thao Quynh      | Data Science  | University of Science (VNUHCM) |
| Nguyen Ngoc Thanh Thu | Data Science  | University of Science (VNUHCM) |
| Kieu Thi Ngoc Vui     | Data Science  | University of Science (VNUHCM) |