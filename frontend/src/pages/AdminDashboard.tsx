/**
 * Admin Dashboard - Main Management Interface
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  LayoutDashboard, 
  Activity, 
  Shield, 
  Database, 
  AlertTriangle,
  TrendingUp,
  Users,
  Settings,
  FileText,
  Lock,
  BarChart3
} from 'lucide-react'
import { apiClient } from '@/lib/api'
import SystemOverview from '@/components/admin/SystemOverview'
import AuditLogViewer from '@/components/admin/AuditLogViewer'
import EncryptionPanel from '@/components/admin/EncryptionPanel'
import ModelPerformance from '@/components/admin/ModelPerformance'
import ConfigurationEditor from '@/components/admin/ConfigurationEditor'

type DashboardTab = 'overview' | 'audit' | 'encryption' | 'models' | 'config' | 'analytics'

export default function AdminDashboard() {
  const [activeTab, setActiveTab] = useState<DashboardTab>('overview')
  
  // Fetch system health
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: apiClient.getHealth,
    refetchInterval: 5000,
  })
  
  // Fetch system stats
  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: apiClient.getStats,
    refetchInterval: 10000,
  })
  
  const tabs = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'audit', label: 'Audit Logs', icon: FileText },
    { id: 'encryption', label: 'Encryption', icon: Lock },
    { id: 'models', label: 'Models', icon: Activity },
    { id: 'config', label: 'Configuration', icon: Settings },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  ]
  
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Admin Dashboard
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                System Management & Monitoring
              </p>
            </div>
            
            <div className="flex items-center gap-4">
              {/* System Status */}
              <div className="flex items-center gap-2 px-4 py-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-green-700 dark:text-green-400">
                  System Online
                </span>
              </div>
              
              {/* Version */}
              <div className="px-4 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <span className="text-sm font-medium text-blue-700 dark:text-blue-400">
                  v1.16.0
                </span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Tabs */}
        <div className="px-6 flex gap-1 border-t border-gray-200 dark:border-gray-700">
          {tabs.map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as DashboardTab)}
                className={`
                  flex items-center gap-2 px-4 py-3 border-b-2 transition-colors
                  ${isActive 
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                <span className="font-medium text-sm">{tab.label}</span>
              </button>
            )
          })}
        </div>
      </header>
      
      {/* Content */}
      <main className="p-6">
        {activeTab === 'overview' && <SystemOverview health={health} stats={stats} />}
        {activeTab === 'audit' && <AuditLogViewer />}
        {activeTab === 'encryption' && <EncryptionPanel />}
        {activeTab === 'models' && <ModelPerformance />}
        {activeTab === 'config' && <ConfigurationEditor />}
        {activeTab === 'analytics' && (
          <div className="text-center py-12 text-gray-500">
            Analytics Dashboard - Coming Soon
          </div>
        )}
      </main>
    </div>
  )
}

