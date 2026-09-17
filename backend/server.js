require('dotenv').config();
const express = require('express');
const cors = require('cors');
const connectDB = require('./config/db');
const authRoutes = require('./routes/authRoutes');
const chatRoutes = require('./routes/chatRoutes');
const { ingestKnowledgeBase } = require('./scripts/ingestKnowledge');

const app = express();

const allowedOrigins = (process.env.CORS_ORIGIN || '')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

app.use(
  cors({
    origin(origin, callback) {
      if (!origin || allowedOrigins.length === 0 || allowedOrigins.includes(origin)) {
        return callback(null, true);
      }
      return callback(new Error('Origin is not allowed by CORS'));
    },
  })
);
app.use(express.json());

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'zenbot-backend' });
});

app.use('/api/auth', authRoutes);
app.use('/api/chats', chatRoutes);

// Fallback for unknown API routes
app.use('/api', (req, res) => {
  res.status(404).json({ message: 'API route not found' });
});

const PORT = process.env.PORT || 5000;

connectDB().then(async () => {
  // Seed the RAG knowledge base on first run only (no-op if already populated).
  // This can take a little while the very first time, since it downloads a
  // small local embedding model — subsequent starts are instant.
  if (process.env.RAG_ENABLED !== 'false') try {
    await ingestKnowledgeBase();
  } catch (err) {
    console.warn('⚠️  Knowledge base ingestion failed — Zenbot will still run without RAG:', err.message);
  }

  app.listen(PORT, () => {
    console.log(`🚀 Zenbot backend running on http://localhost:${PORT}`);
  });
});
