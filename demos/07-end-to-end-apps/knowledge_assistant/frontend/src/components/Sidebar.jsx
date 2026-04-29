import { useState } from 'react'
import { listKBFiles, deleteFile } from '../api'

export default function Sidebar({
  connected,
  serverInfo,
  knowledgeBases,
  selectedKBs,
  onConnect,
  onCreateKB,
  onUploadFiles,
  onAddURL,
  onToggleKB,
}) {
  const [host, setHost] = useState('localhost')
  const [port, setPort] = useState(8321)
  const [connecting, setConnecting] = useState(false)
  const [connectError, setConnectError] = useState('')

  const [newKBName, setNewKBName] = useState('')
  const [uploadTarget, setUploadTarget] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')

  const [urlInput, setUrlInput] = useState('')
  const [addingUrl, setAddingUrl] = useState(false)
  const [urlError, setUrlError] = useState('')

  const [deletingFile, setDeletingFile] = useState(null) // file_id being deleted

  const [expandedKBs, setExpandedKBs] = useState({})
  const [kbFiles, setKBFiles] = useState({})

  const handleToggleExpand = async (e, kbName) => {
    e.stopPropagation()
    const nowExpanded = !expandedKBs[kbName]
    setExpandedKBs((prev) => ({ ...prev, [kbName]: nowExpanded }))
    if (nowExpanded && !kbFiles[kbName]) {
      try {
        const files = await listKBFiles(kbName)
        setKBFiles((prev) => ({ ...prev, [kbName]: files }))
      } catch {
        setKBFiles((prev) => ({ ...prev, [kbName]: [] }))
      }
    }
  }

  const refreshFileList = async (kbName) => {
    try {
      const files = await listKBFiles(kbName)
      setKBFiles((prev) => ({ ...prev, [kbName]: files }))
    } catch { /* ignore */ }
  }

  const handleConnect = async (e) => {
    e.preventDefault()
    setConnecting(true)
    setConnectError('')
    try {
      await onConnect(host, Number(port))
    } catch (err) {
      setConnectError(err.message)
    } finally {
      setConnecting(false)
    }
  }

  const handleCreateKB = async (e) => {
    e.preventDefault()
    if (!newKBName.trim()) return
    try {
      await onCreateKB(newKBName.trim())
      if (!uploadTarget) setUploadTarget(newKBName.trim())
      setNewKBName('')
    } catch (err) {
      console.error(err)
    }
  }

  const handleDeleteFile = async (kbName, fileId) => {
    setDeletingFile(fileId)
    try {
      await deleteFile(kbName, fileId)
      await refreshFileList(kbName)
    } catch (err) {
      console.error(err)
    } finally {
      setDeletingFile(null)
    }
  }

  const handleAddURL = async (e) => {
    e.preventDefault()
    if (!urlInput.trim() || !uploadTarget) return
    setAddingUrl(true)
    setUrlError('')
    try {
      await onAddURL(uploadTarget, urlInput.trim())
      setUrlInput('')
      await refreshFileList(uploadTarget)
    } catch (err) {
      setUrlError(err.message)
    } finally {
      setAddingUrl(false)
    }
  }

  const handleFileChange = async (e) => {
    const files = Array.from(e.target.files)
    if (!files.length || !uploadTarget) return
    setUploading(true)
    setUploadError('')
    try {
      await onUploadFiles(uploadTarget, files)
      e.target.value = ''
      await refreshFileList(uploadTarget)
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const shieldId = serverInfo?.shield_id ?? null
  const availableModels = serverInfo?.available_models ?? []

  const handleModelChange = async (e) => {
    try {
      await onConnect(host, Number(port), e.target.value)
    } catch (err) {
      console.error(err)
    }
  }

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>📚 Knowledge Assistant</h1>
        {connected && (
          <div className="connected-badge">
            {shieldId && <span className="shield-badge" title={`Safety shield: ${shieldId}`}>🛡️</span>}
          </div>
        )}
        {connected && availableModels.length > 0 && (
          <select
            className="form-input"
            value={serverInfo?.model_id ?? ''}
            onChange={handleModelChange}
            style={{ marginTop: 8, fontSize: 12 }}
          >
            {availableModels.map((m) => (
              <option key={m} value={m}>{m.split('/').pop()}</option>
            ))}
          </select>
        )}
      </div>

      {/* Connection form — shown only when disconnected */}
      {!connected && (
        <div className="sidebar-section">
          <h2>Connection</h2>
          <form onSubmit={handleConnect}>
            <div className="form-group">
              <label className="form-label">Host</label>
              <input
                className="form-input"
                value={host}
                onChange={(e) => setHost(e.target.value)}
                placeholder="localhost"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Port</label>
              <input
                className="form-input"
                type="number"
                value={port}
                onChange={(e) => setPort(e.target.value)}
              />
            </div>
            {connectError && <div className="alert alert-error">{connectError}</div>}
            <button className="btn btn-primary btn-full" disabled={connecting}>
              {connecting ? 'Connecting…' : 'Connect'}
            </button>
          </form>
        </div>
      )}

      {/* Knowledge bases */}
      {connected && (
        <>
          <div className="sidebar-section">
            <h2>Knowledge Bases</h2>

            {knowledgeBases.length > 0 && (
              <div className="kb-list">
                {knowledgeBases.map((kb) => (
                  <div key={kb.name}>
                    <div className="kb-item" onClick={() => onToggleKB(kb.name)}>
                      <input
                        type="checkbox"
                        checked={selectedKBs.includes(kb.name)}
                        onChange={() => onToggleKB(kb.name)}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <span className="kb-item-name">{kb.name}</span>
                      <span className="kb-item-count">
                        {kb.doc_count} doc{kb.doc_count !== 1 ? 's' : ''}
                      </span>
                      <button
                        className="kb-expand-btn"
                        onClick={(e) => handleToggleExpand(e, kb.name)}
                        title={expandedKBs[kb.name] ? 'Hide files' : 'Show files'}
                      >
                        {expandedKBs[kb.name] ? '▾' : '▸'}
                      </button>
                    </div>

                    {expandedKBs[kb.name] && (
                      <div className="kb-file-list">
                        {!kbFiles[kb.name] && (
                          <div className="kb-file-loading">Loading…</div>
                        )}
                        {kbFiles[kb.name]?.length === 0 && (
                          <div className="kb-file-loading">No files yet.</div>
                        )}
                        {kbFiles[kb.name]?.map((f) => (
                          <div key={f.id} className="kb-file-item">
                            <span className={`kb-file-dot ${f.status}`} />
                            <span className="kb-file-name" title={f.name}>{f.name}</span>
                            <button
                              className="kb-file-delete"
                              onClick={() => handleDeleteFile(kb.name, f.id)}
                              disabled={deletingFile === f.id}
                              title="Remove from knowledge base"
                            >
                              {deletingFile === f.id ? '…' : '×'}
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            <form onSubmit={handleCreateKB} className="kb-create-row">
              <input
                className="form-input"
                placeholder="New knowledge base…"
                value={newKBName}
                onChange={(e) => setNewKBName(e.target.value)}
              />
              <button className="btn btn-secondary" type="submit">
                +
              </button>
            </form>
          </div>

          {/* File upload — shown only when at least one KB exists */}
          {knowledgeBases.length > 0 && (
            <div className="sidebar-section">
              <h2>Upload Documents</h2>
              <div className="form-group">
                <label className="form-label">Upload to</label>
                <select
                  className="form-input"
                  value={uploadTarget}
                  onChange={(e) => setUploadTarget(e.target.value)}
                >
                  <option value="">Select knowledge base…</option>
                  {knowledgeBases.map((kb) => (
                    <option key={kb.name} value={kb.name}>
                      {kb.name}
                    </option>
                  ))}
                </select>
              </div>

              {uploadTarget && (
                <>
                  <form onSubmit={handleAddURL} style={{ marginBottom: 8 }}>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <input
                        className="form-input"
                        placeholder="https://…"
                        value={urlInput}
                        onChange={(e) => setUrlInput(e.target.value)}
                        disabled={addingUrl}
                        style={{ flex: 1 }}
                      />
                      <button className="btn btn-secondary" type="submit" disabled={addingUrl || !urlInput.trim()}>
                        {addingUrl ? '…' : 'Add'}
                      </button>
                    </div>
                    {urlError && <div className="alert alert-error" style={{ marginTop: 6 }}>{urlError}</div>}
                  </form>

                  <label className="upload-label">
                    {uploading ? 'Ingesting…' : 'Choose files (.txt, .md, .pdf)'}
                    <input
                      type="file"
                      multiple
                      accept=".txt,.md,.pdf"
                      style={{ display: 'none' }}
                      onChange={handleFileChange}
                      disabled={uploading}
                    />
                  </label>
                  {uploadError && (
                    <div className="alert alert-error" style={{ marginTop: 8 }}>
                      {uploadError}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
