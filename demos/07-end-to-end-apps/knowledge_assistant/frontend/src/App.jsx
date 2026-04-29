import { useState } from 'react'
import Sidebar from './components/Sidebar'
import Chat from './components/Chat'
import { connect, createKnowledgeBase, listKnowledgeBases, uploadFiles, addURL, streamChat } from './api'

export default function App() {
  const [connected, setConnected] = useState(false)
  const [serverInfo, setServerInfo] = useState(null)
  const [knowledgeBases, setKnowledgeBases] = useState([])
  const [selectedKBs, setSelectedKBs] = useState([])
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  const handleConnect = async (host, port, modelId = null) => {
    const info = await connect(host, port, modelId)
    setServerInfo(info)
    setKnowledgeBases(info.knowledge_bases)
    setSelectedKBs(info.knowledge_bases.map((kb) => kb.name))
    setConnected(true)
  }

  const handleCreateKB = async (name) => {
    const kb = await createKnowledgeBase(name)
    setKnowledgeBases((prev) => [...prev, kb])
  }

  const handleUploadFiles = async (kbName, files) => {
    await uploadFiles(kbName, files)
    const updated = await listKnowledgeBases()
    setKnowledgeBases(updated)
  }

  const handleAddURL = async (kbName, url) => {
    await addURL(kbName, url)
    const updated = await listKnowledgeBases()
    setKnowledgeBases(updated)
  }

  const handleToggleKB = (name) => {
    setSelectedKBs((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    )
  }

  const handleSend = async (question) => {
    setMessages((prev) => [...prev, { role: 'user', content: question }])
    setLoading(true)
    try {
      for await (const event of streamChat(question, selectedKBs)) {
        if (event.type === 'answer') {
          setMessages((prev) => [
            ...prev,
            {
              role: 'assistant',
              content: event.content,
              mode: event.mode,
              sources: event.sources,
            },
          ])
        } else if (event.type === 'error') {
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: `Error: ${event.message}`, mode: 'single', sources: [] },
          ])
        }
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}`, mode: 'single', sources: [] },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Sidebar
        connected={connected}
        serverInfo={serverInfo}
        knowledgeBases={knowledgeBases}
        selectedKBs={selectedKBs}
        onConnect={handleConnect}
        onCreateKB={handleCreateKB}
        onUploadFiles={handleUploadFiles}
        onAddURL={handleAddURL}
        onToggleKB={handleToggleKB}
      />
      <Chat
        messages={messages}
        loading={loading}
        connected={connected}
        selectedKBs={selectedKBs}
        onSend={handleSend}
      />
    </>
  )
}
