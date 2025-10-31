/**
 * System Overview Component
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { Activity, Database, TrendingUp, AlertTriangle, Zap, Globe, Github } from 'lucide-react'

interface SystemOverviewProps {
  health: any
  stats: any
}

export default function SystemOverview({ health, stats }: SystemOverviewProps) {
  const metrics = [
    {
      label: 'Total Queries',
      value: stats?.cache?.total_queries?.toLocaleString() || '0',
      change: '+12.5%',
      trend: 'up',
      icon: Activity,
      color: 'blue'
    },
    {
      label: 'Cache Hit Rate',
      value: `${((stats?.cache?.hit_rate || 0) * 100).toFixed(1)}%`,
      change: '+5.2%',
      trend: 'up',
      icon: Zap,
      color: 'green'
    },
    {
      label: 'Web Searches',
      value: stats?.tools?.web_searches?.toLocaleString() || '0',
      change: '+8.3%',
      trend: 'up',
      icon: Globe,
      color: 'purple'
    },
    {
      label: 'GitHub Queries',
      value: stats?.tools?.github_queries?.toLocaleString() || '0',
      change: '+15.7%',
      trend: 'up',
      icon: Github,
      color: 'orange'
    },
  ]
  
  return (
    <div className="space-y-6">
      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric) => {
          const Icon = metric.icon
          const colorClasses = {
            blue: 'bg-blue-500',
            green: 'bg-green-500',
            purple: 'bg-purple-500',
            orange: 'bg-orange-500',
          }[metric.color]
          
          return (
            <div
              key={metric.label}
              className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`p-3 rounded-lg ${colorClasses} bg-opacity-10`}>
                  <Icon className={`w-6 h-6 text-${metric.color}-600 dark:text-${metric.color}-400`} />
                </div>
                <span className="text-sm font-medium text-green-600 dark:text-green-400">
                  {metric.change}
                </span>
              </div>
              
              <div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metric.value}
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  {metric.label}
                </div>
              </div>
            </div>
          )
        })}
      </div>
      
      {/* System Health */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Services Status */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Services Status
          </h3>
          
          <div className="space-y-3">
            <ServiceStatus
              name="Ollama LLM"
              status={health?.ollama_connected ? 'online' : 'offline'}
              uptime="99.9%"
            />
            <ServiceStatus
              name="MS Documentation"
              status={stats?.ms_index_loaded ? 'online' : 'offline'}
              uptime="100%"
            />
            <ServiceStatus
              name="OSS Documentation"
              status={stats?.oss_index_loaded ? 'online' : 'offline'}
              uptime="100%"
            />
            <ServiceStatus
              name="Redis Cache"
              status="online"
              uptime="99.8%"
            />
          </div>
        </div>
        
        {/* Recent Activity */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Recent Activity
          </h3>
          
          <div className="space-y-3">
            <ActivityItem
              type="query"
              message="Query processed successfully"
              time="2 minutes ago"
            />
            <ActivityItem
              type="cache"
              message="Cache hit rate increased to 85%"
              time="15 minutes ago"
            />
            <ActivityItem
              type="model"
              message="Model switched to deepseek-coder"
              time="1 hour ago"
            />
            <ActivityItem
              type="system"
              message="System health check completed"
              time="2 hours ago"
            />
          </div>
        </div>
      </div>
      
      {/* Cached Models */}
      {stats?.models_cached && stats.models_cached.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Cached Models
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {stats.models_cached.map((model: string) => (
              <div
                key={model}
                className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
              >
                <div className="w-2 h-2 bg-green-500 rounded-full" />
                <span className="text-sm font-mono text-gray-700 dark:text-gray-300">
                  {model}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function ServiceStatus({ name, status, uptime }: { name: string, status: string, uptime: string }) {
  const isOnline = status === 'online'
  
  return (
    <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
      <div className="flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-500' : 'bg-red-500'}`} />
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          {name}
        </span>
      </div>
      
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {uptime}
        </span>
        <span className={`text-xs font-medium ${isOnline ? 'text-green-600' : 'text-red-600'}`}>
          {status.toUpperCase()}
        </span>
      </div>
    </div>
  )
}

function ActivityItem({ type, message, time }: { type: string, message: string, time: string }) {
  const icons = {
    query: Activity,
    cache: Database,
    model: TrendingUp,
    system: AlertTriangle,
  }
  
  const Icon = icons[type as keyof typeof icons] || Activity
  
  return (
    <div className="flex items-start gap-3 p-3 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg transition-colors">
      <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded">
        <Icon className="w-4 h-4 text-blue-600 dark:text-blue-400" />
      </div>
      
      <div className="flex-1 min-w-0">
        <p className="text-sm text-gray-700 dark:text-gray-300">
          {message}
        </p>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          {time}
        </p>
      </div>
    </div>
  )
}

