/**
 * Global Application State Store (Zustand)
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Message, SystemStats, AppSettings, Model } from '@/types'

interface AppState {
  // Messages
  messages: Message[]
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  clearMessages: () => void
  updateLastMessage: (content: string) => void

  // System Stats
  stats: SystemStats | null
  setStats: (stats: SystemStats) => void

  // Models
  models: Model[]
  setModels: (models: Model[]) => void
  selectedModel: string | null
  setSelectedModel: (model: string | null) => void

  // Settings
  settings: AppSettings
  updateSettings: (settings: Partial<AppSettings>) => void

  // UI State
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
  loading: boolean
  setLoading: (loading: boolean) => void
  
  // Tools
  webSearchEnabled: boolean
  setWebSearchEnabled: (enabled: boolean) => void
  githubRepo: string
  setGithubRepo: (repo: string) => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Messages
      messages: [],
      addMessage: (message) => {
        const newMessage: Message = {
          ...message,
          id: Math.random().toString(36).substring(7),
          timestamp: new Date(),
        }
        set({ messages: [...get().messages, newMessage] })
      },
      clearMessages: () => set({ messages: [] }),
      updateLastMessage: (content) => {
        const messages = get().messages
        if (messages.length === 0) return
        
        const lastMessage = messages[messages.length - 1]
        if (lastMessage.role === 'assistant') {
          lastMessage.content = content
          set({ messages: [...messages] })
        }
      },

      // System Stats
      stats: null,
      setStats: (stats) => set({ stats }),

      // Models
      models: [],
      setModels: (models) => set({ models }),
      selectedModel: null,
      setSelectedModel: (model) => set({ selectedModel: model }),

      // Settings
      settings: {
        theme: 'dark',
        language: 'en',
        autoScroll: true,
        soundEnabled: false,
        compactMode: false,
      },
      updateSettings: (newSettings) =>
        set({ settings: { ...get().settings, ...newSettings } }),

      // UI State
      sidebarOpen: true,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      loading: false,
      setLoading: (loading) => set({ loading }),

      // Tools
      webSearchEnabled: false,
      setWebSearchEnabled: (enabled) => set({ webSearchEnabled: enabled }),
      githubRepo: '',
      setGithubRepo: (repo) => set({ githubRepo: repo }),
    }),
    {
      name: 'dual-rag-llm-storage',
      partialize: (state) => ({
        settings: state.settings,
        selectedModel: state.selectedModel,
      }),
    }
  )
)

