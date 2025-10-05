import React, { useState } from 'react'
import axios from 'axios'

export default function App() {
  const [messages, setMessages] = useState([{
    role: 'bot', content: 'Hi — ask me something about the uploaded documents.'
  }])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  async function send() {
    if (!input.trim()) return
    const userMsg = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    // Use Server-Sent Events for streaming
  const params = new URLSearchParams({ question: input })
  const evtSrc = new EventSource(`http://localhost:8000/stream_chat?${params.toString()}`)

  // append an empty bot message which we'll fill as chunks arrive
  setMessages(prev => [...prev, { role: 'bot', content: '' }])

    evtSrc.addEventListener('context', e => {
      try {
        const payload = JSON.parse(e.data)
        // push context as a small message for visibility (optional)
        setMessages(prev => [...prev, { role: 'bot', content: `(context) ${payload.text}` }])
      } catch (err) {
        console.error('context parse error', err)
      }
    })

    evtSrc.addEventListener('chunk', e => {
      try {
        const payload = JSON.parse(e.data)
        setMessages(prev => {
          // update the last bot message by appending text
          const copy = [...prev]
          const last = copy.length - 1
          copy[last] = { ...copy[last], content: (copy[last].content || '') + payload.text }
          return copy
        })
      } catch (err) {
        console.error('chunk parse error', err)
      }
    })

    evtSrc.addEventListener('error', e => {
      setMessages(prev => [...prev, { role: 'bot', content: 'Error from server.' }])
      setLoading(false)
      evtSrc.close()
    })

    evtSrc.addEventListener('done', () => {
      setLoading(false)
      evtSrc.close()
      setInput('')
      // finished
    })
  }

  return (
    <div className="chat-root">
      <div className="chat-window">
        {messages.map((m, i) => (
          <div key={i} className={m.role === 'user' ? 'msg user' : 'msg bot'}>
            <pre>{m.content}</pre>
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && send()} placeholder="Type a question..." />
        <button onClick={send} disabled={loading}>{loading ? '...' : 'Send'}</button>
      </div>
    </div>
  )
}
