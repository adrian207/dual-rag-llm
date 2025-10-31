/**
 * Analytics Dashboard Component
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { TrendingUp, Activity, Clock, AlertCircle, BarChart3 } from 'lucide-react'
import { format } from 'date-fns'

type Period = 'hour' | 'day' | 'week' | 'month'

export default function AnalyticsDashboard() {
  const [period, setPeriod] = useState<Period>('day')
  
  // Fetch analytics report
  const { data: report, isLoading } = useQuery({
    queryKey: ['analytics-report', period],
    queryFn: async () => {
      const response = await fetch(`/api/analytics/report?period=${period}`)
      return response.json()
    },
    refetchInterval: 30000,
  })
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-gray-500">Loading analytics...</div>
      </div>
    )
  }
  
  const { query_analytics, model_analytics, performance, insights } = report || {}
  
  return (
    <div className="space-y-6">
      {/* Period Selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Analytics Report
        </h2>
        
        <div className="flex gap-2">
          {(['hour', 'day', 'week', 'month'] as Period[]).map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`
                px-4 py-2 rounded-lg text-sm font-medium transition-colors
                ${period === p
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }
              `}
            >
              {p.charAt(0).toUpperCase() + p.slice(1)}
            </button>
          ))}
        </div>
      </div>
      
      {/* Insights */}
      {insights && insights.length > 0 && (
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border border-blue-200 dark:border-blue-800">
          <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-200 mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Key Insights
          </h3>
          
          <ul className="space-y-2">
            {insights.map((insight: string, index: number) => (
              <li key={index} className="flex items-start gap-2 text-blue-800 dark:text-blue-300">
                <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                <span>{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {/* Query Analytics */}
      {query_analytics && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Query Analytics
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              label="Total Queries"
              value={query_analytics.total_queries.toLocaleString()}
              icon={Activity}
            />
            <MetricCard
              label="Queries/Hour"
              value={query_analytics.queries_per_hour.toFixed(1)}
              icon={TrendingUp}
            />
            <MetricCard
              label="Avg Response Time"
              value={`${query_analytics.avg_response_time_ms.toFixed(0)}ms`}
              icon={Clock}
            />
            <MetricCard
              label="Success Rate"
              value={`${(query_analytics.success_rate * 100).toFixed(1)}%`}
              icon={query_analytics.success_rate > 0.95 ? Activity : AlertCircle}
              status={query_analytics.success_rate > 0.95 ? 'success' : 'warning'}
            />
          </div>
          
          {/* Peak Hours */}
          {query_analytics.peak_hours && query_analytics.peak_hours.length > 0 && (
            <div className="mt-6">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                Peak Hours
              </h4>
              <div className="flex gap-2">
                {query_analytics.peak_hours.map((hour: number) => (
                  <div
                    key={hour}
                    className="px-3 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded-lg text-sm font-medium"
                  >
                    {hour}:00
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Top Query Types */}
          {query_analytics.top_query_types && Object.keys(query_analytics.top_query_types).length > 0 && (
            <div className="mt-6">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                Top Query Types
              </h4>
              <div className="space-y-2">
                {Object.entries(query_analytics.top_query_types).map(([type, count]: [string, any]) => (
                  <div key={type} className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">{type}</span>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Performance Analytics */}
      {performance && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Performance Metrics
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            <MetricCard
              label="Avg Latency"
              value={`${performance.avg_api_latency_ms.toFixed(0)}ms`}
              icon={Clock}
            />
            <MetricCard
              label="P50 Latency"
              value={`${performance.p50_latency_ms.toFixed(0)}ms`}
              icon={Clock}
            />
            <MetricCard
              label="P95 Latency"
              value={`${performance.p95_latency_ms.toFixed(0)}ms`}
              icon={Clock}
              status={performance.p95_latency_ms > 5000 ? 'warning' : 'success'}
            />
            <MetricCard
              label="P99 Latency"
              value={`${performance.p99_latency_ms.toFixed(0)}ms`}
              icon={Clock}
              status={performance.p99_latency_ms > 10000 ? 'warning' : 'success'}
            />
            <MetricCard
              label="Uptime"
              value={`${performance.uptime_percentage.toFixed(2)}%`}
              icon={Activity}
              status={performance.uptime_percentage > 99 ? 'success' : 'warning'}
            />
          </div>
        </div>
      )}
      
      {/* Top Models */}
      {model_analytics && model_analytics.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Top Models
          </h3>
          
          <div className="space-y-3">
            {model_analytics.slice(0, 5).map((model: any) => (
              <div
                key={model.model_name}
                className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
              >
                <div>
                  <div className="font-mono text-sm font-medium text-gray-900 dark:text-white">
                    {model.model_name}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {model.usage_count} queries • {model.avg_response_time_ms.toFixed(0)}ms avg
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">
                    {(model.success_rate * 100).toFixed(1)}% success
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {(model.cache_hit_rate * 100).toFixed(0)}% cached
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, icon: Icon, status = 'info' }: any) {
  const statusColors = {
    success: 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20',
    warning: 'border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20',
    error: 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20',
    info: 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800',
  }
  
  return (
    <div className={`p-4 rounded-lg border ${statusColors[status]}`}>
      <div className="flex items-center justify-between mb-2">
        <Icon className="w-4 h-4 text-gray-400" />
      </div>
      <div className="text-2xl font-bold text-gray-900 dark:text-white">
        {value}
      </div>
      <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
        {label}
      </div>
    </div>
  )
}

