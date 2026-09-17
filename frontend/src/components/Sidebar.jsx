import { useAuth } from '../context/AuthContext.jsx';

export default function Sidebar({ chats, activeChatId, onSelect, onNewChat, onDelete }) {
  const { user, logout } = useAuth();
  const initial = (user?.name || '?').trim().charAt(0).toUpperCase();

  return (
    <aside className="sidebar">
      <div className="mark">🌿 Zenbot</div>

      <button className="new-chat-btn" onClick={onNewChat} type="button">
        <span>＋</span> New chat
      </button>

      <div className="sidebar-label">Chat history</div>

      <div className="chat-list">
        {chats.length === 0 && (
          <p style={{ color: 'var(--text-faint)', fontSize: '0.85rem', padding: '0.5rem 0.3rem' }}>
            Your past conversations will show up here.
          </p>
        )}
        {chats.map((chat) => (
          <button
            key={chat._id}
            type="button"
            className={`chat-list-item ${chat._id === activeChatId ? 'active' : ''}`}
            onClick={() => onSelect(chat._id)}
          >
            <span className="title">{chat.title || 'New chat'}</span>
            <span
              className="remove"
              role="button"
              tabIndex={-1}
              onClick={(e) => {
                e.stopPropagation();
                onDelete(chat._id);
              }}
              title="Delete chat"
            >
              ✕
            </span>
          </button>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="user-chip">
          <div className="avatar">{initial}</div>
          <div className="name">{user?.name}</div>
        </div>
        <button className="logout-btn" onClick={logout} type="button">
          Log out
        </button>
      </div>
    </aside>
  );
}
