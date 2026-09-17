/**
 * Zenbot reply engine.
 *
 * The original Streamlit project called the Hugging Face "hf-inference"
 * provider directly for a chat model, which is exactly why it broke
 * ("Model not supported by provider hf-inference") — HF regularly changes
 * which providers/models are available on the free tier, so a hardcoded
 * model id goes stale.
 *
 * This engine never hard-fails:
 *   1. If OPENAI_API_KEY is set, it tries a real LLM call (works with
 *      OpenAI or any OpenAI-compatible endpoint — Groq, OpenRouter,
 *      Together, a local Ollama server, etc. via OPENAI_BASE_URL).
 *   2. If no key is set, or the API call fails for any reason, it falls
 *      back to a built-in empathetic rule-based engine, so the chatbot
 *      always responds instead of showing a red error box.
 */

const { retrieveRelevant } = require('./rag');

const CRISIS_PATTERNS = [
  /suicid/i,
  /kill myself/i,
  /end my life/i,
  /want to die/i,
  /hurt myself/i,
  /self[\s-]?harm/i,
  /no reason to live/i,
];

const CRISIS_REPLY =
  "I'm really glad you told me this, and I want you to know your life matters and you don't have to go through this alone. " +
  "I'm not able to provide the kind of help you need right now, but a trained counselor can — please reach out to one of these right away:\n\n" +
  "• India — Kiran Mental Health Helpline: 1800-599-0019 (24/7)\n" +
  "• India — iCall: +91 9152987821\n" +
  "• US & Canada — 988 Suicide & Crisis Lifeline\n" +
  "• UK & ROI — Samaritans: 116 123\n" +
  "• Anywhere — findahelpline.com to find a local line\n\n" +
  "If you're in immediate danger, please contact your local emergency number or go to the nearest emergency room. " +
  "Would you like to tell me a little more about what's been happening? I'm here to listen.";

const CATEGORIES = [
  {
    tag: 'greeting',
    test: /\b(hi|hello|hey|good morning|good evening|good afternoon)\b/i,
    replies: [
      "Hi there 👋 I'm Zenbot. I'm really glad you're here. How are you feeling today?",
      "Hello! This is a safe space to talk about anything that's on your mind. What's going on with you today?",
      "Hey, welcome back. How has your day been treating you so far?",
    ],
  },
  {
    tag: 'sad',
    test: /\b(sad|down|depress|unhappy|crying|cry|hopeless|empty|numb)\b/i,
    replies: [
      "I'm sorry you're feeling this way — that sounds really heavy to carry. Do you want to tell me a bit more about what's making you feel sad?",
      "Thank you for sharing that with me. Sadness can feel so isolating. What do you think has been weighing on you the most lately?",
      "That sounds really tough. You don't have to have it all figured out — I'm just here to listen. What happened?",
    ],
  },
  {
    tag: 'anxious',
    test: /\b(anxious|anxiety|panic|nervous|worried|worry|overwhelm)\b/i,
    replies: [
      "Anxiety can feel so overwhelming in the moment. Let's slow down together — can you try taking one slow breath in for 4 counts, hold for 4, and out for 4?",
      "That racing, on-edge feeling is exhausting. What's the thought that keeps circling in your mind right now?",
      "It makes sense to feel worried sometimes. Is there something specific that triggered this feeling today, or does it feel more general?",
    ],
  },
  {
    tag: 'stressed',
    test: /\b(stress|stressed|pressure|burnt? out|burnout|exhausted|tired)\b/i,
    replies: [
      "It sounds like you're carrying a lot right now. What's the biggest source of pressure on you at the moment?",
      "Running on empty is really hard. Have you been able to take any breaks for yourself recently, even small ones?",
      "That sounds draining. Let's break it down together — what's one thing on your plate that feels the most urgent?",
    ],
  },
  {
    tag: 'angry',
    test: /\b(angry|anger|mad|furious|frustrat|irritat)\b/i,
    replies: [
      "That frustration sounds really valid. What happened that brought this on?",
      "It's okay to feel angry — it's often a sign that something important to you was affected. Want to talk through it?",
      "I hear you. Sometimes anger is protecting a softer feeling underneath, like hurt or disappointment. Does that resonate at all?",
    ],
  },
  {
    tag: 'lonely',
    test: /\b(lonely|alone|isolated|no one understands|no friends)\b/i,
    replies: [
      "Feeling lonely is genuinely painful, even when people are around you. I'm glad you reached out here — what does a typical day look like for you right now?",
      "You're not alone in this conversation, at least — I'm here with you. What's been making you feel most disconnected lately?",
      "Loneliness can creep in even in a crowded room. Is there someone in your life you feel like you could reach out to, even just to say hi?",
    ],
  },
  {
    tag: 'sleep',
    test: /\b(sleep|insomnia|can'?t sleep|tired all the time|nightmare)\b/i,
    replies: [
      "Sleep struggles can really wear a person down. What does your mind tend to do when you're lying there trying to fall asleep?",
      "That sounds exhausting. Sometimes a wind-down routine — dim lights, no screens, slow breathing — can help signal to your body it's time to rest. Have you tried anything like that?",
    ],
  },
  {
    tag: 'positive',
    test: /\b(good|great|happy|excited|grateful|thankful|better|okay|fine)\b/i,
    replies: [
      "That's really lovely to hear 😊 What's been going well for you?",
      "I'm glad to hear that! Want to share what's contributing to the good feeling?",
      "That's wonderful. Celebrating the good moments matters just as much as working through the hard ones.",
    ],
  },
  {
    tag: 'gratitude',
    test: /\b(thank you|thanks|appreciate)\b/i,
    replies: [
      "You're very welcome. I'm really glad this helped, even a little. I'm here whenever you need to talk. 💜",
      "Anytime. Taking care of your mental health is something to be proud of — you're doing that right now just by talking about it.",
    ],
  },
  {
    tag: 'bye',
    test: /\b(bye|goodbye|see you|talk later|good night)\b/i,
    replies: [
      "Take care of yourself. Remember, I'll be right here whenever you want to talk again. 🌙",
      "Goodbye for now — be gentle with yourself today. You did the right thing by talking it out.",
    ],
  },
];

const DEFAULT_REPLIES = [
  "Thank you for sharing that. Can you tell me a little more about how that's been affecting you?",
  "I'm listening. What's been the hardest part about that for you?",
  "That sounds important. How long have you been feeling this way?",
  "I hear you. What do you think would help you feel even a little bit better right now?",
  "Thanks for opening up. Is this something that's been on your mind a lot lately?",
];

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function isCrisisMessage(text) {
  return CRISIS_PATTERNS.some((re) => re.test(text));
}

function ruleBasedReply(text) {
  for (const category of CATEGORIES) {
    if (category.test.test(text)) {
      return { reply: pick(category.replies), mood: category.tag };
    }
  }
  return { reply: pick(DEFAULT_REPLIES), mood: 'neutral' };
}

const SYSTEM_PROMPT =
  'You are Zenbot, a warm, empathetic mental-health support companion (not a licensed therapist). ' +
  'Respond with active listening, validation, and gentle, practical coping suggestions. ' +
  'Keep replies concise (2-4 sentences), avoid clinical jargon, never diagnose, and if the user mentions ' +
  'self-harm or suicide, gently encourage them to contact a crisis line or emergency services.';

function buildContextBlock(retrieved) {
  if (!retrieved || retrieved.length === 0) return '';
  const lines = retrieved.map((c) => `- (${c.sectionTitle}) ${c.text}`).join('\n');
  return (
    '\n\nRelevant guidance retrieved from Zenbot\'s knowledge base ' +
    '(weave this in naturally if it fits, in your own words — do not quote it verbatim or list it as bullet points):\n' +
    lines
  );
}

async function tryOpenAICompatible(history, latestMessage, retrieved) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return null;

  const baseUrl = (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/$/, '');
  const model = process.env.OPENAI_MODEL || 'gpt-4o-mini';

  const messages = [
    { role: 'system', content: SYSTEM_PROMPT + buildContextBlock(retrieved) },
    ...history.slice(-8).map((m) => ({
      role: m.sender === 'user' ? 'user' : 'assistant',
      content: m.text,
    })),
    { role: 'user', content: latestMessage },
  ];

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);

  try {
    const res = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ model, messages, temperature: 0.7, max_tokens: 300 }),
      signal: controller.signal,
    });

    if (!res.ok) {
      console.warn('AI provider returned non-OK status:', res.status, await res.text());
      return null;
    }

    const data = await res.json();
    const reply = data?.choices?.[0]?.message?.content?.trim();
    return reply || null;
  } catch (err) {
    console.warn('AI provider call failed, falling back to rule-based engine:', err.message);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * @param {Array<{sender: 'user'|'bot', text: string}>} history - prior messages in the chat
 * @param {string} latestMessage - the newest user message
 * @returns {Promise<{reply: string, mood: string, source: 'ai'|'rules'|'crisis'}>}
 */
async function getBotReply(history, latestMessage) {
  if (isCrisisMessage(latestMessage)) {
    return { reply: CRISIS_REPLY, mood: 'crisis', source: 'crisis' };
  }

  // Retrieve relevant knowledge-base chunks once, used by whichever path answers.
  const retrieved = await retrieveRelevant(latestMessage);

  const aiReply = await tryOpenAICompatible(history, latestMessage, retrieved);
  if (aiReply) {
    return { reply: aiReply, mood: 'ai', source: 'ai', retrieved: retrieved.map((r) => r.sectionTitle) };
  }

  const { reply, mood } = ruleBasedReply(latestMessage);
  const topMatch = retrieved[0];

  // Blend in one retrieved tip, in the knowledge base's own words, only when
  // it's a genuinely close match — otherwise leave the reply as-is.
  const blended = topMatch && topMatch.score >= 0.45 ? `${reply}\n\nSomething that sometimes helps: ${topMatch.text}` : reply;

  return {
    reply: blended,
    mood,
    source: 'rules',
    retrieved: topMatch ? [topMatch.sectionTitle] : [],
  };
}

module.exports = { getBotReply, isCrisisMessage };
