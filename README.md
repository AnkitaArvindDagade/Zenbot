# Zenbot 🌿 — Mental Health Support Chatbot 

Zenbot is a mental-health support chatbot implemented as a MERN stack
application (MongoDB, Express, React, Node.js). It offers a private chat
space per user, persistent chat history, and a responsive React UI built
with Vite. The app can run with or without a hosted AI key: when an
`OPENAI_API_KEY` is present the backend uses a hosted LLM; otherwise it
falls back to a rule-based empathetic reply engine. Optional RAG
capabilities let the bot ground replies from a small curated knowledge
base stored in MongoDB.

Below you'll find the architecture, workflow, a diagram, the tech stack,
and the live repository/deployment link.

## Project Info

- Name: Zenbot — Mental Health Support Chatbot
- Purpose: Supportive listening, coping techniques, and lightweight
  retrieval-augmented replies for non-clinical mental health support.
- Repo: https://github.com/AnkitaArvindDagade/Zenbot.git
- URL: https://zenbot-teal.vercel.app

## Architecture

Zenbot consists of two primary components:

- Frontend: Vite + React app that handles user authentication, UI, and
  communicates with the backend API under `/api/*`.
- Backend: Express server that exposes the REST API, stores data in
  MongoDB, handles auth with JWT, runs the reply engine and optional
  RAG ingestion.

The backend may call a hosted LLM/embeddings provider (OpenAI or
compatible) when `OPENAI_API_KEY` is set, or use a local embedding model
(`@xenova/transformers`) when running fully local.

## Workflow

1. User logs in / signs up via the frontend.
2. The frontend creates or selects a chat and posts user messages to
   `/api/chats/:id/messages`.
3. Backend appends the user message to the chat, calls `getBotReply()` and
   saves the bot reply.
   - If `OPENAI_API_KEY` is set, the backend sends a chat completion
     request to the configured `OPENAI_BASE_URL` with a system prompt and
     optional retrieved context.
   - If no key or the API call fails, the backend uses a rule-based
     engine that replies with empathetic prompts and optional RAG snippets.
   - If the message matches crisis patterns, the backend returns a
     crisis reply with helpline numbers.
4. Frontend renders the chat messages for the user.
# Zenbot 🌿 — Mental Health Support Chatbot 

Zenbot is a mental-health support chatbot implemented as a MERN stack
application (MongoDB, Express, React, Node.js). It offers a private chat
space per user, persistent chat history, and a responsive React UI built
with Vite. The app can run with or without a hosted AI key: when an
`OPENAI_API_KEY` is present the backend uses a hosted LLM; otherwise it
falls back to a rule-based empathetic reply engine. Optional RAG
capabilities let the bot ground replies from a small curated knowledge
base stored in MongoDB.

Below you'll find the architecture, workflow, a diagram, the tech stack,
and the live repository/deployment link.

## Project Info

- Name: Zenbot — Mental Health Support Chatbot
- Purpose: Supportive listening, coping techniques, and lightweight
  retrieval-augmented replies for non-clinical mental health support.
- Repo: https://github.com/AnkitaArvindDagade/Zenbot.git
- URL: https://zenbot-teal.vercel.app

## Architecture

Zenbot consists of two primary components:

- Frontend: Vite + React app that handles user authentication, UI, and
  communicates with the backend API under `/api/*`.
- Backend: Express server that exposes the REST API, stores data in
  MongoDB, handles auth with JWT, runs the reply engine and optional
  RAG ingestion.

The backend may call a hosted LLM/embeddings provider (OpenAI or
compatible) when `OPENAI_API_KEY` is set, or use a local embedding model
(`@xenova/transformers`) when running fully local.

## Workflow

1. User logs in / signs up via the frontend.
2. The frontend creates or selects a chat and posts user messages to
   `/api/chats/:id/messages`.
3. Backend appends the user message to the chat, calls `getBotReply()` and
   saves the bot reply.
   - If `OPENAI_API_KEY` is set, the backend sends a chat completion
     request to the configured `OPENAI_BASE_URL` with a system prompt and
     optional retrieved context.
   - If no key or the API call fails, the backend uses a rule-based
     engine that replies with empathetic prompts and optional RAG snippets.
   - If the message matches crisis patterns, the backend returns a
     crisis reply with helpline numbers.
4. Frontend renders the chat messages for the user.

## Built With

- React, Vite (frontend)
- Node.js, Express, Mongoose (backend)
- JWT for auth (`jsonwebtoken`)
- Optional local embeddings: `@xenova/transformers`
- Dev tools: `nodemon`, `concurrently`

## Live / Deployment

- Repository: https://github.com/AnkitaArvindDagade/Zenbot.git
- Frontend: https://zenbot-teal.vercel.app
- Backend: https://zenbot-api-dx8t.onrender.com/api/health

## Deployment

- Frontend: Vercel (Hobby / free tier) — hosts the built `frontend/dist`.
- Backend: Render (Free web service) — deploy the Express API using the
  provided `render.yaml` blueprint.
- Database: MongoDB Atlas M0 (free cluster) — use as the app's `MONGO_URI`.

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

This installs dependencies for both `backend/` and `frontend`.

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

## 4. Building for production

```bash
cd frontend
npm run build
```

## Notes on the mental health content

Zenbot is a supportive listening companion, **not** a licensed therapist or
a diagnostic tool — the UI says this explicitly under the message box. If
you or someone you know is in crisis, please contact a local emergency
number or a crisis line (a few are listed in the app's automatic response to
messages that mention self-harm).
