/**
 * Sidebar Component with System Stats
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { Database, TrendingUp, Globe, Github, Trash2 } from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '@/store/useAppStore'
import { apiClient } from '@/lib/api'
import clsx from 'clsx'

export default function Sidebar() {
  const { sidebarOpen, stats, clearMessages } = useAppStore()
  const queryClient = useQueryClient()

  const clearCacheMutation = useMutation({
    mutationFn: apiClient.clearCache,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['stats'] })
    },
  })

  if (!sidebarOpen) return null

  return (
    <aside className="w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
          System Info
        </h2>
        
        {stats && (
          <div className="space-y-3">
            <StatItem
              icon={<Database className="w-4 h-4" />}
              label="Total Queries"
              value={stats.cache.total_queries.toLocaleString()}
            />
            <StatItem
              icon={<TrendingUp className="w-4 h-4" />}
              label="Cache Hit Rate"
              value={`${(stats.cache.hit_rate * 100).toFixed(1)}%`}
              color="text-green-600 dark:text-green-400"
            />
            <StatItem
              icon={<Globe className="w-4 h-4" />}
              label="Web Searches"
              value={stats.tools.web_searches.toLocaleString()}
            />
            <StatItem
              icon={<Github className="w-4 h-4" />}
              label="GitHub Queries"
              value={stats.tools.github_queries.toLocaleString()}
            />
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Knowledge Bases
          </div>
          <StatusBadge
            label="MS Docs"
            status={stats?.ms_index_loaded ?? false}
          />
          <StatusBadge
            label="OSS Docs"
            status={stats?.oss_index_loaded ?? false}
          />
        </div>

        {stats && stats.models_cached.length > 0 && (
          <div className="mt-6">
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Cached Models
            </div>
            <div className="space-y-1">
              {stats.models_cached.map((model) => (
                <div
                  key={model}
                  className="text-xs text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded"
                >
                  {model}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="p-4 border-t border-gray-200 dark:border-gray-700 space-y-2">
        <button
          onClick={() => clearCacheMutation.mutate()}
          disabled={clearCacheMutation.isPending}
          className="w-full px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          <Database className="w-4 h-4" />
          Clear Cache
        </button>
        <button
          onClick={clearMessages}
          className="w-full px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          <Trash2 className="w-4 h-4" />
          Clear Chat
        </button>
      </div>
    </aside>
  )
}

interface StatItemProps {
  icon: React.ReactNode
  label: string
  value: string
  color?: string
}

function StatItem({ icon, label, value, color = 'text-primary-600 dark:text-primary-400' }: StatItemProps) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
        {icon}
        <span>{label}</span>
      </div>
      <span className={clsx('text-sm font-semibold', color)}>{value}</span>
    </div>
  )
}

interface StatusBadgeProps {
  label: string
  status: boolean
}

function StatusBadge({ label, status }: StatusBadgeProps) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-gray-600 dark:text-gray-400">{label}</span>
      <span
        className={clsx(
          'px-2 py-0.5 rounded-full text-xs font-medium',
          status
            ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
            : 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300'
        )}
      >
        {status ? 'Loaded' : 'Not Loaded'}
      </span>
    </div>
  )
}

