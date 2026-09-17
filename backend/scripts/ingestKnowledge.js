const fs = require('fs');
const path = require('path');
const KnowledgeChunk = require('../models/KnowledgeChunk');
const { embed } = require('../utils/embeddings');

const RESOURCES_DIR = path.join(__dirname, '..', 'resources');

/**
 * Splits one markdown file into chunks: the "# Doc Title" becomes docTitle,
 * and every "## Section Heading" becomes its own retrievable chunk.
 */
function parseMarkdownIntoChunks(filename, raw) {
  const lines = raw.split('\n');
  let docTitle = filename;
  const chunks = [];
  let currentTitle = null;
  let currentLines = [];

  const flush = () => {
    if (currentTitle && currentLines.length) {
      const text = currentLines.join('\n').trim();
      if (text) chunks.push({ sectionTitle: currentTitle, text });
    }
    currentLines = [];
  };

  for (const line of lines) {
    if (line.startsWith('# ')) {
      docTitle = line.replace(/^#\s+/, '').trim();
    } else if (line.startsWith('## ')) {
      flush();
      currentTitle = line.replace(/^##\s+/, '').trim();
    } else {
      currentLines.push(line);
    }
  }
  flush();

  return { docTitle, chunks };
}

async function ingestKnowledgeBase({ force = false } = {}) {
  const existingCount = await KnowledgeChunk.countDocuments();
  if (existingCount > 0 && !force) {
    console.log(`ℹ️  Knowledge base already has ${existingCount} chunks — skipping ingestion.`);
    return { skipped: true, count: existingCount };
  }

  if (!fs.existsSync(RESOURCES_DIR)) {
    console.warn('⚠️  No resources/ folder found — skipping RAG ingestion.');
    return { skipped: true, count: 0 };
  }

  const files = fs.readdirSync(RESOURCES_DIR).filter((f) => f.endsWith('.md'));
  if (files.length === 0) {
    console.warn('⚠️  resources/ folder is empty — skipping RAG ingestion.');
    return { skipped: true, count: 0 };
  }

  console.log(`🧠 Ingesting knowledge base from ${files.length} file(s)... (first run downloads a small local embedding model, ~90MB)`);

  await KnowledgeChunk.deleteMany({});

  let total = 0;
  for (const file of files) {
    const raw = fs.readFileSync(path.join(RESOURCES_DIR, file), 'utf-8');
    const { docTitle, chunks } = parseMarkdownIntoChunks(file, raw);

    for (const chunk of chunks) {
      const embedding = await embed(`${chunk.sectionTitle}. ${chunk.text}`);
      await KnowledgeChunk.create({
        source: file,
        docTitle,
        sectionTitle: chunk.sectionTitle,
        text: chunk.text,
        embedding,
      });
      total += 1;
    }
  }

  console.log(`✅ Knowledge base ready: ${total} chunks embedded from ${files.length} document(s).`);
  return { skipped: false, count: total };
}

module.exports = { ingestKnowledgeBase };

// Allow running directly: `node scripts/ingestKnowledge.js [--force]`
if (require.main === module) {
  require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
  const mongoose = require('mongoose');
  const connectDB = require('../config/db');
  const force = process.argv.includes('--force');

  connectDB()
    .then(() => ingestKnowledgeBase({ force }))
    .then(() => mongoose.disconnect())
    .then(() => process.exit(0))
    .catch((err) => {
      console.error('❌ Ingestion failed:', err);
      process.exit(1);
    });
}
