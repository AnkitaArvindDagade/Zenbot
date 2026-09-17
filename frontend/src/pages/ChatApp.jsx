import { useEffect, useRef, useState, useCallback } from 'react';
import { AnimatePresence } from 'framer-motion';
import api from '../api/axios';
import AmbientBackground from '../components/AmbientBackground.jsx';
import Sidebar from '../components/Sidebar.jsx';
import MessageBubble from '../components/MessageBubble.jsx';
import TypingIndicator from '../components/TypingIndicator.jsx';
import Composer from '../components/Composer.jsx';

export default function ChatApp() {
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const [loadingChats, setLoadingChats] = useState(true);
  const scrollRef = useRef(null);

  const loadChats = useCallback(async () => {
    const res = await api.get('/chats');
    setChats(res.data.chats);
    return res.data.chats;
  }, []);

  const openChat = useCallback(async (id) => {
    const res = await api.get(`/chats/${id}`);
    setActiveChat(res.data.chat);
  }, []);

  const createChat = useCallback(async () => {
    const res = await api.post('/chats');
    setActiveChat(res.data.chat);
    setChats((prev) => [{ ...res.data.chat }, ...prev]);
  }, []);

  useEffect(() => {
    (async () => {
      setLoadingChats(true);
      const list = await loadChats();
      if (list.length > 0) {
        await openChat(list[0]._id);
      } else {
        await createChat();
      }
      setLoadingChats(false);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [activeChat, sending]);

  const handleSelect = async (id) => {
    if (activeChat?._id === id) return;
    await openChat(id);
  };

  const handleDelete = async (id) => {
    await api.delete(`/chats/${id}`);
    const remaining = chats.filter((c) => c._id !== id);
    setChats(remaining);
    if (activeChat?._id === id) {
      if (remaining.length > 0) {
        await openChat(remaining[0]._id);
      } else {
        await createChat();
      }
    }
  };

  const handleSend = async () => {
    const text = draft.trim();
    if (!text || !activeChat || sending) return;

    setDraft('');
    setActiveChat((prev) => ({
      ...prev,
      messages: [...prev.messages, { _id: `temp-${Date.now()}`, sender: 'user', text }],
    }));
    setSending(true);

    try {
      const res = await api.post(`/chats/${activeChat._id}/messages`, { text });
      setActiveChat(res.data.chat);
      setChats((prev) => {
        const updated = prev.map((c) =>
          c._id === res.data.chat._id ? { ...c, title: res.data.chat.title } : c
        );
        const found = updated.find((c) => c._id === res.data.chat._id);
        return [found, ...updated.filter((c) => c._id !== res.data.chat._id)];
      });
    } catch (err) {
      setActiveChat((prev) => ({
        ...prev,
        messages: [
          ...prev.messages,
          {
            _id: `err-${Date.now()}`,
            sender: 'bot',
            text: "I'm having trouble responding right now. Please try sending that again in a moment.",
          },
        ],
      }));
    } finally {
      setSending(false);
    }
  };

  if (loadingChats) {
    return <div className="page-loader">Setting up your space…</div>;
  }

  return (
    <div className="app-shell">
      <AmbientBackground />
      <Sidebar
        chats={chats}
        activeChatId={activeChat?._id}
        onSelect={handleSelect}
        onNewChat={createChat}
        onDelete={handleDelete}
      />

      <main className="chat-main">
        <div className="chat-header">
          <h2>{activeChat?.title || 'New chat'}</h2>
          <span className="status">
            <span className="pulse" /> Zenbot is here
          </span>
        </div>

        <div className="messages" ref={scrollRef}>
          {(!activeChat || activeChat.messages.length === 0) && (
            <div className="empty-state">
              <span className="leaf">🌿</span>
              <h3>Take a breath</h3>
              <p>Whenever you're ready, share what's on your mind. There's no rush here.</p>
            </div>
          )}

          <AnimatePresence initial={false}>
            {activeChat?.messages.map((m) => (
              <MessageBubble key={m._id} sender={m.sender} text={m.text} />
            ))}
            {sending && <TypingIndicator key="typing" />}
          </AnimatePresence>
        </div>

        <Composer value={draft} onChange={setDraft} onSend={handleSend} disabled={sending} />
      </main>
    </div>
  );
}
