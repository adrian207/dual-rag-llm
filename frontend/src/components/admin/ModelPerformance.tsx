/**
 * Model Performance Dashboard
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { useQuery } from '@tanstack/react-query'
import { Activity, TrendingUp, Zap, Clock, CheckCircle, AlertCircle } from 'lucide-react'
import { format } from 'date-fns'

export default function ModelPerformance() {
  // Fetch model analytics
  const { data: analytics, isLoading } = useQuery({
    queryKey: ['model-analytics'],
    queryFn: async () => {
      const response = await fetch('/api/analytics/models?period=day')
      return response.json()
    },
    refetchInterval: 10000,
  })
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-gray-500">Loading model performance...</div>
      </div>
    )
  }
  
  const models = analytics || []
  
  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <SummaryCard
          icon={Activity}
          label="Total Models"
          value={models.length.toString()}
          color="blue"
        />
        <SummaryCard
          icon={TrendingUp}
          label="Total Queries"
          value={models.reduce((sum: number, m: any) => sum + m.usage_count, 0).toLocaleString()}
          color="green"
        />
        <SummaryCard
          icon={Zap}
          label="Avg Cache Hit"
          value={`${(models.reduce((sum: number, m: any) => sum + m.cache_hit_rate, 0) / models.length * 100 || 0).toFixed(0)}%`}
          color="purple"
        />
        <SummaryCard
          icon={CheckCircle}
          label="Avg Success"
          value={`${(models.reduce((sum: number, m: any) => sum + m.success_rate, 0) / models.length * 100 || 0).toFixed(1)}%`}
          color="emerald"
        />
      </div>
      
      {/* Model Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {models.map((model: any) => (
          <ModelCard key={model.model_name} model={model} />
        ))}
      </div>
      
      {models.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          No model performance data available yet
        </div>
      )}
    </div>
  )
}

function SummaryCard({ icon: Icon, label, value, color }: any) {
  const colors = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500',
    emerald: 'bg-emerald-500',
  }
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-lg ${colors[color]} bg-opacity-10`}>
          <Icon className={`w-6 h-6 text-${color}-600 dark:text-${color}-400`} />
        </div>
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

function ModelCard({ model }: { model: any }) {
  const successRate = model.success_rate * 100
  const cacheHitRate = model.cache_hit_rate * 100
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white font-mono">
            {model.model_name}
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            {model.usage_count.toLocaleString()} queries
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          {successRate >= 95 ? (
            <CheckCircle className="w-5 h-5 text-green-500" />
          ) : (
            <AlertCircle className="w-5 h-5 text-yellow-500" />
          )}
        </div>
      </div>
      
      {/* Metrics */}
      <div className="space-y-3">
        <MetricRow
          icon={Clock}
          label="Avg Response Time"
          value={`${model.avg_response_time_ms.toFixed(0)}ms`}
          percentage={null}
        />
        
        <MetricRow
          icon={CheckCircle}
          label="Success Rate"
          value={`${successRate.toFixed(1)}%`}
          percentage={successRate}
        />
        
        <MetricRow
          icon={Zap}
          label="Cache Hit Rate"
          value={`${cacheHitRate.toFixed(1)}%`}
          percentage={cacheHitRate}
        />
        
        <MetricRow
          icon={Activity}
          label="Total Tokens"
          value={model.total_tokens.toLocaleString()}
          percentage={null}
        />
      </div>
    </div>
  )
}

function MetricRow({ icon: Icon, label, value, percentage }: any) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <Icon className="w-4 h-4 text-gray-400" />
        <span className="text-sm text-gray-600 dark:text-gray-400">{label}</span>
      </div>
      
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-gray-900 dark:text-white">
          {value}
        </span>
        
        {percentage !== null && (
          <div className="w-16 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full ${
                percentage >= 90 ? 'bg-green-500' :
                percentage >= 70 ? 'bg-yellow-500' :
                'bg-red-500'
              }`}
              style={{ width: `${Math.min(percentage, 100)}%` }}
            />
          </div>
        )}
      </div>
    </div>
  )
}

