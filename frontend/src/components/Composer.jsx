import { useRef } from 'react';

export default function Composer({ value, onChange, onSend, disabled }) {
  const textareaRef = useRef(null);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const handleChange = (e) => {
    onChange(e.target.value);
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
    }
  };

  return (
    <div className="composer">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSend();
        }}
      >
        <textarea
          ref={textareaRef}
          rows={1}
          placeholder="Type how you're feeling…"
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button className="send-btn" type="submit" disabled={disabled || !value.trim()} aria-label="Send message">
          ➤
        </button>
      </form>
      <p className="hint">Zenbot offers supportive listening, not medical advice or diagnosis.</p>
    </div>
  );
}
