/**
 * Cost Tracking Dashboard
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { DollarSign, TrendingUp, TrendingDown, AlertCircle, Calendar, PieChart } from 'lucide-react'
import { format, subDays } from 'date-fns'

export default function CostDashboard() {
  const [dateRange, setDateRange] = useState(30)
  const [forecastDays, setForecastDays] = useState(30)
  
  const startDate = subDays(new Date(), dateRange).toISOString()
  const endDate = new Date().toISOString()
  
  // Fetch cost data
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['cost-summary', startDate, endDate],
    queryFn: async () => {
      const response = await fetch(`/api/costs/summary?start_date=${startDate}&end_date=${endDate}`)
      return response.json()
    },
    refetchInterval: 30000,
  })
  
  const { data: breakdown } = useQuery({
    queryKey: ['cost-breakdown', startDate, endDate],
    queryFn: async () => {
      const response = await fetch(`/api/costs/breakdown?start_date=${startDate}&end_date=${endDate}`)
      return response.json()
    },
    refetchInterval: 30000,
  })
  
  const { data: forecast } = useQuery({
    queryKey: ['cost-forecast', forecastDays],
    queryFn: async () => {
      const response = await fetch(`/api/costs/forecast?days=${forecastDays}`)
      return response.json()
    },
    refetchInterval: 60000,
  })
  
  const { data: alerts } = useQuery({
    queryKey: ['budget-alerts'],
    queryFn: async () => {
      const response = await fetch('/api/costs/alerts')
      return response.json()
    },
  })
  
  const { data: pricing } = useQuery({
    queryKey: ['model-pricing'],
    queryFn: async () => {
      const response = await fetch('/api/costs/pricing')
      return response.json()
    },
  })
  
  if (summaryLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-gray-500">Loading cost data...</div>
      </div>
    )
  }
  
  const totalCost = summary?.total_cost || 0
  const avgCostPerRequest = summary?.average_cost_per_request || 0
  const totalRequests = summary?.total_requests || 0
  
  // Calculate savings (if using Ollama vs cloud)
  const cloudCost = totalRequests * 0.002 // Estimated cloud cost
  const savings = cloudCost - totalCost
  
  return (
    <div className="space-y-6">
      {/* Date Range Selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Cost Tracking
        </h2>
        
        <div className="flex gap-2">
          {[7, 30, 90, 365].map((days) => (
            <button
              key={days}
              onClick={() => setDateRange(days)}
              className={`
                px-4 py-2 rounded-lg text-sm font-medium transition-colors
                ${dateRange === days
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }
              `}
            >
              {days === 365 ? '1 Year' : `${days} Days`}
            </button>
          ))}
        </div>
      </div>
      
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <CostCard
          icon={DollarSign}
          label="Total Cost"
          value={`$${totalCost.toFixed(4)}`}
          change={null}
          color="blue"
        />
        <CostCard
          icon={TrendingUp}
          label="Avg Cost/Request"
          value={`$${avgCostPerRequest.toFixed(6)}`}
          change={null}
          color="green"
        />
        <CostCard
          icon={Calendar}
          label="Total Requests"
          value={totalRequests.toLocaleString()}
          change={null}
          color="purple"
        />
        <CostCard
          icon={TrendingDown}
          label="Savings vs Cloud"
          value={`$${savings.toFixed(2)}`}
          change="+100%"
          color="emerald"
        />
      </div>
      
      {/* Cost by Model */}
      {summary?.cost_by_model && Object.keys(summary.cost_by_model).length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <PieChart className="w-5 h-5" />
            Cost by Model
          </h3>
          
          <div className="space-y-3">
            {Object.entries(summary.cost_by_model)
              .sort(([, a]: any, [, b]: any) => b - a)
              .map(([model, cost]: [string, any]) => {
                const percentage = (cost / totalCost * 100) || 0
                return (
                  <div key={model} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-mono text-gray-700 dark:text-gray-300">{model}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          {percentage.toFixed(1)}%
                        </span>
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          ${cost.toFixed(4)}
                        </span>
                      </div>
                    </div>
                    <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-600"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                )
              })}
          </div>
        </div>
      )}
      
      {/* Cost Breakdown */}
      {breakdown && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* By Type */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Cost Breakdown
            </h3>
            
            <div className="space-y-3">
              <BreakdownItem label="Input Tokens" value={breakdown.input_tokens_cost} />
              <BreakdownItem label="Output Tokens" value={breakdown.output_tokens_cost} />
              <BreakdownItem label="Request Fees" value={breakdown.request_cost} />
              <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                <BreakdownItem label="Total" value={breakdown.total_cost} bold />
              </div>
            </div>
          </div>
          
          {/* Top Cost Queries */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Top Cost Queries
            </h3>
            
            <div className="space-y-2">
              {breakdown.top_cost_queries?.slice(0, 5).map((query: any, index: number) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded"
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-mono text-gray-700 dark:text-gray-300 truncate">
                      {query.model}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {query.tokens.toLocaleString()} tokens
                    </div>
                  </div>
                  <div className="text-sm font-medium text-gray-900 dark:text-white">
                    ${query.cost.toFixed(6)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      
      {/* Cost Trend */}
      {summary?.cost_by_day && summary.cost_by_day.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Daily Cost Trend
          </h3>
          
          <div className="space-y-2">
            {summary.cost_by_day.slice(-14).map((day: any) => {
              const maxCost = Math.max(...summary.cost_by_day.map((d: any) => d.cost))
              const width = (day.cost / maxCost * 100) || 0
              
              return (
                <div key={day.date} className="flex items-center gap-3">
                  <span className="text-xs text-gray-500 dark:text-gray-400 w-20">
                    {format(new Date(day.date), 'MMM dd')}
                  </span>
                  <div className="flex-1 h-6 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
                    <div
                      className="h-full bg-blue-600 flex items-center px-2"
                      style={{ width: `${Math.max(width, 5)}%` }}
                    >
                      {width > 15 && (
                        <span className="text-xs text-white font-medium">
                          ${day.cost.toFixed(4)}
                        </span>
                      )}
                    </div>
                  </div>
                  {width <= 15 && (
                    <span className="text-xs text-gray-600 dark:text-gray-400 w-16 text-right">
                      ${day.cost.toFixed(4)}
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
      
      {/* Forecast */}
      {forecast && forecast.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Cost Forecast ({forecastDays} days)
            </h3>
            <select
              value={forecastDays}
              onChange={(e) => setForecastDays(parseInt(e.target.value))}
              className="px-3 py-1 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded text-sm"
            >
              <option value="7">7 days</option>
              <option value="30">30 days</option>
              <option value="90">90 days</option>
            </select>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="text-sm text-blue-600 dark:text-blue-400 mb-1">Predicted Total</div>
              <div className="text-2xl font-bold text-blue-900 dark:text-blue-200">
                ${forecast.reduce((sum: number, f: any) => sum + f.predicted_cost, 0).toFixed(2)}
              </div>
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="text-sm text-green-600 dark:text-green-400 mb-1">Best Case</div>
              <div className="text-2xl font-bold text-green-900 dark:text-green-200">
                ${forecast.reduce((sum: number, f: any) => sum + f.confidence_interval_low, 0).toFixed(2)}
              </div>
            </div>
            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
              <div className="text-sm text-orange-600 dark:text-orange-400 mb-1">Worst Case</div>
              <div className="text-2xl font-bold text-orange-900 dark:text-orange-200">
                ${forecast.reduce((sum: number, f: any) => sum + f.confidence_interval_high, 0).toFixed(2)}
              </div>
            </div>
          </div>
          
          <div className="text-sm text-gray-500 dark:text-gray-400">
            Based on historical trends and usage patterns
          </div>
        </div>
      )}
      
      {/* Budget Alerts */}
      {alerts && alerts.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            Budget Alerts
          </h3>
          
          <div className="space-y-3">
            {alerts.map((alert: any) => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border ${
                  alert.enabled
                    ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
                    : 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900 dark:text-white">{alert.name}</span>
                  <span className={`text-sm ${alert.enabled ? 'text-green-600' : 'text-gray-500'}`}>
                    {alert.enabled ? 'Active' : 'Disabled'}
                  </span>
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Budget: ${alert.budget_amount} / {alert.period} • Alert at {alert.threshold_percentage}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Model Pricing */}
      {pricing && pricing.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Model Pricing
          </h3>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Model
                  </th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Input ($/1K)
                  </th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Output ($/1K)
                  </th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Per Request
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {pricing.map((price: any) => (
                  <tr key={price.model_name} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                    <td className="px-4 py-3 text-sm font-mono text-gray-900 dark:text-white">
                      {price.model_name}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-gray-600 dark:text-gray-400">
                      ${price.cost_per_1k_input_tokens.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-gray-600 dark:text-gray-400">
                      ${price.cost_per_1k_output_tokens.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-gray-600 dark:text-gray-400">
                      ${price.cost_per_request.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function CostCard({ icon: Icon, label, value, change, color }: any) {
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
        {change && (
          <span className="text-sm font-medium text-green-600 dark:text-green-400">
            {change}
          </span>
        )}
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

function BreakdownItem({ label, value, bold = false }: any) {
  return (
    <div className="flex items-center justify-between">
      <span className={`text-sm ${bold ? 'font-semibold' : ''} text-gray-600 dark:text-gray-400`}>
        {label}
      </span>
      <span className={`text-sm ${bold ? 'font-bold' : 'font-medium'} text-gray-900 dark:text-white`}>
        ${value.toFixed(4)}
      </span>
    </div>
  )
}

