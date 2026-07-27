import { useState } from 'react'

export default function Message({ message }) {
  const [showSources, setShowSources] = useState(false)
  const isAssistant = message.role === 'assistant'

  return (
    <div className={`message ${message.role}`}>
      <div className="message-bubble">{message.content}</div>

      {isAssistant && (
        <div className="message-meta">
          {message.mode && (
            <span className={`mode-badge ${message.mode}`}>
              {message.mode === 'multi' ? '🔀 Multi-source' : '📄 Single-source'}
            </span>
          )}
          {message.sources?.length > 0 && (
            <button
              className="sources-toggle"
              onClick={() => setShowSources((v) => !v)}
            >
              {showSources ? 'Hide sources' : 'Show sources'}
            </button>
          )}
        </div>
      )}

      {showSources && message.sources?.length > 0 && (
        <div className="sources-panel">
          {message.sources.map((src, i) => (
            <div key={i} className="source-item">
              <div className="source-kb-name">📚 {src.kb_name}</div>
              <div className="source-answer">{src.answer}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
