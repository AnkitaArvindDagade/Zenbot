import { motion } from 'framer-motion';

export default function MessageBubble({ sender, text }) {
  const isBot = sender === 'bot';
  return (
    <motion.div
      className={`msg-row ${isBot ? 'bot' : 'user'}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
    >
      <div className="msg-avatar">{isBot ? '🌿' : '🙂'}</div>
      <div className="bubble">{text}</div>
    </motion.div>
  );
}
