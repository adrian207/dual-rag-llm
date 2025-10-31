/**
 * Header Component
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { Menu, Moon, Sun, Monitor } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'

export default function Header() {
  const { setSidebarOpen, sidebarOpen, settings, updateSettings } = useAppStore()

  const cycleTheme = () => {
    const themes: Array<'light' | 'dark' | 'system'> = ['light', 'dark', 'system']
    const currentIndex = themes.indexOf(settings.theme)
    const nextTheme = themes[(currentIndex + 1) % themes.length]
    updateSettings({ theme: nextTheme })
  }

  const ThemeIcon = {
    light: Sun,
    dark: Moon,
    system: Monitor,
  }[settings.theme]

  return (
    <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">
              Dual RAG LLM System
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Enterprise AI Assistant • v1.13.0
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={cycleTheme}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            title={`Theme: ${settings.theme}`}
          >
            <ThemeIcon className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  )
}

