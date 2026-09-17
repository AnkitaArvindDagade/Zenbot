const mongoose = require('mongoose');

const KnowledgeChunkSchema = new mongoose.Schema(
  {
    source: { type: String, required: true }, // filename, e.g. "anxiety.md"
    docTitle: { type: String, required: true }, // top-level "# Title" of the doc
    sectionTitle: { type: String, required: true }, // the "## Heading" of this chunk
    text: { type: String, required: true },
    embedding: { type: [Number], required: true },
  },
  { timestamps: true }
);

module.exports = mongoose.model('KnowledgeChunk', KnowledgeChunkSchema);
