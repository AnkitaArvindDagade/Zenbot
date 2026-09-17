import { motion } from 'framer-motion';
import AmbientBackground from './AmbientBackground.jsx';

export default function AuthLayout({ children }) {
  return (
    <div className="auth-shell">
      <AmbientBackground />

      <div className="auth-brand">
        <div className="mark">
          <span className="leaf">🌿</span> Zenbot
        </div>
        <h1>
          A calmer place <em>to talk.</em>
        </h1>
        <p className="lede">
          Zenbot is a private space to put words to how you're feeling — no
          judgment, no waiting room, no pressure to have it all figured out.
        </p>
        <div className="auth-features">
          <div className="auth-feature">
            <span className="dot" />
            Your conversations stay private to your account
          </div>
          <div className="auth-feature">
            <span className="dot" />
            Here any hour of the day or night
          </div>
          <div className="auth-feature">
            <span className="dot" />
            A gentle, listening ear — not a replacement for professional care
          </div>
        </div>
      </div>

      <div className="auth-form-wrap">
        <motion.div
          className="auth-card"
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        >
          {children}
        </motion.div>
      </div>
    </div>
  );
}
