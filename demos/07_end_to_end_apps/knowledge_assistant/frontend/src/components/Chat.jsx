import { useEffect, useRef, useState } from 'react'
import Message from './Message'

export default function Chat({ messages, loading, connected, selectedKBs, onSend }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const canChat = connected && selectedKBs.length > 0

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim() || !canChat || loading) return
    onSend(input.trim())
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="chat-area">
      <div className="chat-header">
        <h2>Chat</h2>
        {selectedKBs.length > 0 ? (
          <p>
            Querying: <strong>{selectedKBs.join(', ')}</strong>
            {selectedKBs.length > 1 && ' — multi-source mode'}
          </p>
        ) : (
          <p>No knowledge bases selected</p>
        )}
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">💬</div>
            <h3>Ask anything</h3>
            <p>Upload documents to a knowledge base, select it in the sidebar, then ask a question.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <Message key={i} message={msg} />
        ))}

        {loading && (
          <div className="message assistant">
            <div className="message-bubble">
              <div className="loading-dots">
                <span />
                <span />
                <span />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        {!connected && (
          <div className="alert alert-info">
            Connect to a Llama Stack server using the sidebar to get started.
          </div>
        )}
        {connected && selectedKBs.length === 0 && (
          <div className="alert alert-info">
            Select at least one knowledge base in the sidebar.
          </div>
        )}
        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            className="chat-input"
            placeholder={canChat ? 'Ask a question… (Enter to send)' : 'Connect and select a knowledge base first…'}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!canChat || loading}
          />
          <button
            className="btn btn-primary"
            type="submit"
            disabled={!canChat || loading || !input.trim()}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}
