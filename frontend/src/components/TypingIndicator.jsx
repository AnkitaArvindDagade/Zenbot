import { motion } from 'framer-motion';

export default function TypingIndicator() {
  return (
    <motion.div
      className="msg-row bot typing-row"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
    >
      <div className="msg-avatar">🌿</div>
      <div className="typing-dots">
        <span />
        <span />
        <span />
      </div>
    </motion.div>
  );
}
