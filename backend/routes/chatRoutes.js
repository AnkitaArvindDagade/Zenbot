const express = require('express');
const Chat = require('../models/Chat');
const protect = require('../middleware/auth');
const { getBotReply } = require('../utils/botEngine');

const router = express.Router();
router.use(protect);

// GET /api/chats — list this user's chats, newest first (sidebar "Chat History")
router.get('/', async (req, res) => {
  const chats = await Chat.find({ user: req.user._id })
    .sort({ updatedAt: -1 })
    .select('title createdAt updatedAt');
  res.json({ chats });
});

// POST /api/chats — start a new chat
router.post('/', async (req, res) => {
  const chat = await Chat.create({
    user: req.user._id,
    title: 'New chat',
    messages: [
      {
        sender: 'bot',
        text:
          "Hi, I'm Zenbot 🌿 This is a private space to talk about how you're feeling. " +
          "There's no judgment here — what's on your mind today?",
        mood: 'greeting',
      },
    ],
  });
  res.status(201).json({ chat });
});

// GET /api/chats/:id — full chat with messages
router.get('/:id', async (req, res) => {
  const chat = await Chat.findOne({ _id: req.params.id, user: req.user._id });
  if (!chat) return res.status(404).json({ message: 'Chat not found' });
  res.json({ chat });
});

// DELETE /api/chats/:id
router.delete('/:id', async (req, res) => {
  const chat = await Chat.findOneAndDelete({ _id: req.params.id, user: req.user._id });
  if (!chat) return res.status(404).json({ message: 'Chat not found' });
  res.json({ message: 'Chat deleted' });
});

// POST /api/chats/:id/messages — send a user message, get Zenbot's reply
router.post('/:id/messages', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || !text.trim()) {
      return res.status(400).json({ message: 'Message text is required' });
    }

    const chat = await Chat.findOne({ _id: req.params.id, user: req.user._id });
    if (!chat) return res.status(404).json({ message: 'Chat not found' });

    chat.messages.push({ sender: 'user', text: text.trim() });

    const { reply, mood } = await getBotReply(chat.messages, text.trim());
    chat.messages.push({ sender: 'bot', text: reply, mood });

    // Auto-title the chat from the first user message
    if (chat.title === 'New chat') {
      chat.title = text.trim().slice(0, 40) + (text.trim().length > 40 ? '…' : '');
    }

    await chat.save();
    res.json({ chat });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Zenbot had trouble replying. Please try again.' });
  }
});

module.exports = router;
