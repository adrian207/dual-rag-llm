/**
 * Main Chat Interface Component with Streaming Support
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { useState, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { apiClient } from '@/lib/api'
import Message from './Message'

export default function ChatInterface() {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { messages, addMessage, loading, setLoading, selectedModel, webSearchEnabled, githubRepo } = useAppStore()

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    addMessage({ role: 'user', content: userMessage })
    setLoading(true)

    try {
      let fullResponse = ''
      addMessage({ role: 'assistant', content: '' })

      const stream = apiClient.streamQuery({
        question: userMessage,
        model_override: selectedModel || undefined,
        use_web_search: webSearchEnabled,
        github_repo: githubRepo || undefined,
      })

      for await (const chunk of stream) {
        fullResponse += chunk
        useAppStore.getState().updateLastMessage(fullResponse)
      }
    } catch (error) {
      console.error('Query error:', error)
      addMessage({
        role: 'assistant',
        content: 'Sorry, an error occurred while processing your query.',
        error: true,
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="flex-1 flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-2xl">
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                Welcome to Dual RAG LLM
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-8">
                Ask me anything about coding, development, or technical topics. I can search the web, explore GitHub repositories, and provide intelligent answers.
              </p>
              <div className="grid grid-cols-2 gap-4 text-left">
                {QUICK_EXAMPLES.map((example, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(example.prompt)}
                    className="p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-primary-500 transition-colors text-sm"
                  >
                    <div className="font-medium text-gray-900 dark:text-white mb-1">
                      {example.title}
                    </div>
                    <div className="text-gray-500 dark:text-gray-400 text-xs">
                      {example.prompt}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              disabled={loading}
              className="flex-1 px-4 py-3 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Thinking...</span>
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  <span>Send</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </main>
  )
}

const QUICK_EXAMPLES = [
  {
    title: 'Python Async',
    prompt: 'How do I implement async functions in Python?',
  },
  {
    title: 'C# Features',
    prompt: 'What are the new features in C# 12?',
  },
  {
    title: 'FastAPI Middleware',
    prompt: 'How do I add middleware in FastAPI?',
  },
  {
    title: 'PowerShell Errors',
    prompt: 'How do I handle errors in PowerShell?',
  },
]

