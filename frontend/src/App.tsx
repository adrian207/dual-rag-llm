/**
 * Main App Component
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useAppStore } from '@/store/useAppStore'
import { apiClient } from '@/lib/api'
import ChatInterface from '@/components/ChatInterface'
import Sidebar from '@/components/Sidebar'
import Header from '@/components/Header'

function App() {
  const { settings, setStats, setModels } = useAppStore()

  // Apply theme
  useEffect(() => {
    const root = window.document.documentElement
    root.classList.remove('light', 'dark')

    if (settings.theme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light'
      root.classList.add(systemTheme)
    } else {
      root.classList.add(settings.theme)
    }
  }, [settings.theme])

  // Fetch system stats
  useQuery({
    queryKey: ['stats'],
    queryFn: apiClient.getStats,
    refetchInterval: 5000,
    onSuccess: (data) => setStats(data),
  })

  // Fetch models
  useQuery({
    queryKey: ['models'],
    queryFn: apiClient.listModels,
    onSuccess: (data) => setModels(data),
  })

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
      <Header />
      <div className="flex-1 flex overflow-hidden">
        <Sidebar />
        <ChatInterface />
      </div>
    </div>
  )
}

export default App

