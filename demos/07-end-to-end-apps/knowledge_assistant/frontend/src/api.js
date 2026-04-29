const BASE = '/api'

async function _post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail ?? 'Request failed')
  return data
}

export async function connect(host, port, modelId = null) {
  return _post('/connect', { host, port, ...(modelId ? { model_id: modelId } : {}) })
}

export async function listKnowledgeBases() {
  const res = await fetch(`${BASE}/knowledge-bases`)
  if (!res.ok) throw new Error('Failed to list knowledge bases')
  return res.json()
}

export async function createKnowledgeBase(name) {
  return _post('/knowledge-bases', { name })
}

export async function uploadFiles(kbName, files) {
  const form = new FormData()
  for (const f of files) form.append('files', f)

  const res = await fetch(`${BASE}/knowledge-bases/${encodeURIComponent(kbName)}/files`, {
    method: 'POST',
    body: form,
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail ?? 'Upload failed')
  return data
}

export async function listKBFiles(kbName) {
  const res = await fetch(`${BASE}/knowledge-bases/${encodeURIComponent(kbName)}/files`)
  if (!res.ok) throw new Error('Failed to list files')
  return res.json()
}

export async function addURL(kbName, url) {
  return _post(`/knowledge-bases/${encodeURIComponent(kbName)}/urls`, { url })
}

export async function deleteFile(kbName, fileId) {
  const res = await fetch(`${BASE}/knowledge-bases/${encodeURIComponent(kbName)}/files/${encodeURIComponent(fileId)}`, {
    method: 'DELETE',
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail ?? 'Delete failed')
  return data
}

export async function* streamChat(question, kbNames) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, kb_names: kbNames }),
  })

  if (!res.ok) {
    const data = await res.json()
    throw new Error(data.detail ?? 'Chat request failed')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() // keep incomplete last line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const payload = line.slice(6).trim()
      if (payload === '[DONE]') return
      try {
        yield JSON.parse(payload)
      } catch {
        // skip malformed lines
      }
    }
  }
}
