/**
 * Generates text embeddings fully locally using transformers.js, so RAG
 * retrieval works with zero API keys and zero cost. The model
 * (Xenova/all-MiniLM-L6-v2, ~90MB) downloads once on first use and is then
 * cached on disk by the library — after that, embedding runs offline.
 *
 * If you'd rather use OpenAI's hosted embeddings instead (e.g. to avoid
 * the one-time local model download), you can swap the implementation of
 * embed() below for a fetch() call to `${OPENAI_BASE_URL}/embeddings`.
 */

let extractorPromise = null;

function getExtractor() {
  if (!extractorPromise) {
    // Dynamic import because @xenova/transformers is an ESM-only package.
    extractorPromise = import('@xenova/transformers').then(({ pipeline }) =>
      pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
    );
  }
  return extractorPromise;
}

/**
 * @param {string} text
 * @returns {Promise<number[]>} a normalized embedding vector
 */
async function embed(text) {
  const extractor = await getExtractor();
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  // vectors from embed() are already normalized, so dot product == cosine similarity
  return dot;
}

module.exports = { embed, cosineSimilarity };
