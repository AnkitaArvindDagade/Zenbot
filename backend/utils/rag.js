const KnowledgeChunk = require('../models/KnowledgeChunk');
const { embed, cosineSimilarity } = require('./embeddings');

const MIN_SCORE = 0.35; // below this, the match is too weak to be useful
const TOP_K = 2;

/**
 * @param {string} query - the user's latest message
 * @returns {Promise<Array<{docTitle: string, sectionTitle: string, text: string, score: number}>>}
 */
async function retrieveRelevant(query) {
  if (process.env.RAG_ENABLED === 'false') return [];

  try {
    const chunks = await KnowledgeChunk.find({}).lean();
    if (!chunks.length) return [];

    const queryVector = await embed(query);

    const scored = chunks
      .map((c) => ({
        docTitle: c.docTitle,
        sectionTitle: c.sectionTitle,
        text: c.text,
        score: cosineSimilarity(queryVector, c.embedding),
      }))
      .sort((a, b) => b.score - a.score);

    return scored.filter((c) => c.score >= MIN_SCORE).slice(0, TOP_K);
  } catch (err) {
    console.warn('RAG retrieval failed, continuing without it:', err.message);
    return [];
  }
}

module.exports = { retrieveRelevant };
