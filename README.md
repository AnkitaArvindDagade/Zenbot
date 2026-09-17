# Zenbot 🌿 — Mental Health Support Chatbot (MERN stack)

Zenbot has been rebuilt from the original Streamlit app into a full **MERN**
stack app (MongoDB, Express, React, Node.js), with a login/signup flow, saved
chat history per user, and a calmer, more polished animated UI.

## What was actually broken, and how it was fixed

The old app called a Hugging Face **`hf-inference`** provider directly with a
hardcoded model id. Hugging Face frequently changes which models are
available on that free serverless provider, so the app broke with:

```
Bad request: Model not supported by provider hf-inference
```

The new backend (`backend/utils/botEngine.js`) never hard-fails like that:

1. If you set an `OPENAI_API_KEY` in `backend/.env`, Zenbot calls a real LLM
   for natural, context-aware replies. It works with OpenAI itself, or any
   OpenAI-*compatible* endpoint (Groq, OpenRouter, Together AI, a local
   Ollama server, etc.) — just change `OPENAI_BASE_URL` / `OPENAI_MODEL`.
2. If no key is set, or that call fails for any reason (rate limit, bad
   model name, network hiccup), it automatically falls back to a **built-in,
   rule-based empathetic reply engine** — so the chatbot always responds
   instead of showing a red error box. **No API key is required to run the
   app at all.**
3. Messages that mention self-harm or suicide are detected and answered with
   a caring message plus crisis-line numbers, regardless of which engine is
   active.

## Retrieval-augmented generation (RAG)

Zenbot also does real RAG, and — like the reply engine — it needs **no API
key** to work:

- `backend/resources/*.md` is a small hand-written knowledge base (coping
  techniques for anxiety, stress, low mood, sleep, loneliness, anger,
  grounding/breathing, self-care, and when to seek professional help).
- On first startup, `backend/server.js` automatically ingests these files:
  each `## Section` becomes one chunk, embedded with a **local** embedding
  model (`@xenova/transformers`, `Xenova/all-MiniLM-L6-v2`) that runs
  entirely on your machine — no OpenAI key needed for this part. The
  chunks + embeddings are stored in MongoDB (`KnowledgeChunk` collection).
  This only happens once; later restarts skip it automatically.
- On every message, `backend/utils/rag.js` embeds the user's message and
  finds the closest matching chunk(s) by cosine similarity.
  - If you've configured `OPENAI_API_KEY`, the matched chunks are injected
    into the LLM's system prompt as grounding context (it's told to weave
    them in naturally, in its own words, not quote them).
  - If you haven't, the rule-based engine appends the single best-matching
    chunk directly under its normal reply, when the match is strong enough.
- If retrieval fails for any reason (e.g. Mongo briefly unavailable),
  Zenbot just answers without it — RAG is a bonus layer, not a dependency.

**Note:** the very first server start needs internet access once, to
download the ~90MB local embedding model (same as any other npm package —
after that it's cached on disk and runs fully offline). To re-run
ingestion manually (e.g. after editing the docs in `backend/resources/`):

```bash
cd backend
npm run ingest:force
```

## Project structure

```
zenbot-mern/
├── backend/                 Express API + MongoDB models
│   ├── config/db.js
│   ├── middleware/auth.js
│   ├── models/User.js
│   ├── models/Chat.js
│   ├── models/KnowledgeChunk.js  RAG chunk + embedding storage
│   ├── routes/authRoutes.js
│   ├── routes/chatRoutes.js
│   ├── utils/botEngine.js        the reply engine described above
│   ├── utils/rag.js              retrieval (finds relevant chunks)
│   ├── utils/embeddings.js       local embedding model wrapper
│   ├── scripts/ingestKnowledge.js
│   ├── resources/*.md            the knowledge base itself
│   ├── server.js
│   └── .env.example
└── frontend/                 React (Vite) UI
    └── src/
        ├── pages/Login.jsx, Signup.jsx, ChatApp.jsx
        ├── components/       Sidebar, message bubbles, composer, etc.
        └── context/AuthContext.jsx
```

## 1. Prerequisites

- **Node.js 18+** (needed for the built-in `fetch` used to call an AI
  provider) — check with `node -v`
- **MongoDB** running somewhere reachable — either:
  - installed locally ([macOS/Windows/Linux instructions](https://www.mongodb.com/docs/manual/installation/)), or
  - a free [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) cluster (no local install needed)

## 2. Setup

```bash
# from the zenbot-mern folder
npm run install:all
```

This installs dependencies for both `backend/` and `frontend/`.

Then configure the backend:

```bash
cd backend
cp .env.example .env
```

Open `backend/.env` and set:

- `MONGO_URI` — your local Mongo URL (default works if you installed Mongo
  locally with default settings) or your Atlas connection string
- `JWT_SECRET` — any long random string
- `OPENAI_API_KEY` — **optional**, leave blank to use the built-in reply
  engine with no external API at all

## 3. Run it

From the **root** `zenbot-mern` folder, run both servers together:

```bash
npm run dev
```

- Backend API → http://localhost:5000
- Frontend app → http://localhost:5173 (Vite proxies `/api` calls to the backend automatically)

Open **http://localhost:5173**, create an account, and start chatting.

(You can also run them in two separate terminals with `npm run dev:backend`
and `npm run dev:frontend` if you prefer.)

## 4. Building for production

```bash
cd frontend
npm run build
```

This outputs static files to `frontend/dist`, which you can serve with any
static host (Vercel, Netlify, Nginx, etc.). Deploy `backend/` separately
(Render, Railway, Fly.io, a VPS, …) and point the frontend's API calls at
its public URL, or serve `frontend/dist` from the Express app itself.

## Notes on the mental health content

Zenbot is a supportive listening companion, **not** a licensed therapist or
a diagnostic tool — the UI says this explicitly under the message box. If
you or someone you know is in crisis, please contact a local emergency
number or a crisis line (a few are listed in the app's automatic response to
messages that mention self-harm).
