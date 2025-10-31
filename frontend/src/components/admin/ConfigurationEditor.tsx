/**
 * Configuration Editor Component
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Settings, Save, RefreshCw, AlertTriangle } from 'lucide-react'

export default function ConfigurationEditor() {
  const [activeSection, setActiveSection] = useState<'analytics' | 'audit' | 'language'>('analytics')
  
  // Fetch configurations
  const { data: analyticsConfig, refetch: refetchAnalytics } = useQuery({
    queryKey: ['analytics-config'],
    queryFn: async () => {
      const response = await fetch('/api/analytics/config')
      return response.json()
    },
  })
  
  const { data: auditConfig, refetch: refetchAudit } = useQuery({
    queryKey: ['audit-config'],
    queryFn: async () => {
      const response = await fetch('/api/audit/config')
      return response.json()
    },
  })
  
  const { data: languageConfig, refetch: refetchLanguage } = useQuery({
    queryKey: ['language-config'],
    queryFn: async () => {
      const response = await fetch('/api/language/config')
      return response.json()
    },
  })
  
  const sections = [
    { id: 'analytics', label: 'Analytics', config: analyticsConfig },
    { id: 'audit', label: 'Audit Logging', config: auditConfig },
    { id: 'language', label: 'Languages', config: languageConfig },
  ]
  
  return (
    <div className="space-y-6">
      {/* Section Tabs */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="flex border-b border-gray-200 dark:border-gray-700">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id as any)}
              className={`
                flex-1 px-6 py-4 text-sm font-medium transition-colors
                ${activeSection === section.id
                  ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700'
                }
              `}
            >
              {section.label}
            </button>
          ))}
        </div>
      </div>
      
      {/* Configuration Content */}
      {activeSection === 'analytics' && analyticsConfig && (
        <AnalyticsConfig config={analyticsConfig} refetch={refetchAnalytics} />
      )}
      
      {activeSection === 'audit' && auditConfig && (
        <AuditConfig config={auditConfig} refetch={refetchAudit} />
      )}
      
      {activeSection === 'language' && languageConfig && (
        <LanguageConfig config={languageConfig} refetch={refetchLanguage} />
      )}
    </div>
  )
}

function AnalyticsConfig({ config, refetch }: any) {
  const [formData, setFormData] = useState(config)
  
  const updateMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await fetch('/api/analytics/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
      return response.json()
    },
    onSuccess: () => {
      refetch()
      alert('Analytics configuration updated successfully!')
    },
  })
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
        Analytics Configuration
      </h3>
      
      <div className="space-y-4">
        <ToggleField
          label="Analytics Enabled"
          checked={formData.enabled}
          onChange={(checked) => setFormData({ ...formData, enabled: checked })}
        />
        
        <ToggleField
          label="Track Queries"
          checked={formData.track_queries}
          onChange={(checked) => setFormData({ ...formData, track_queries: checked })}
        />
        
        <ToggleField
          label="Track API Calls"
          checked={formData.track_api_calls}
          onChange={(checked) => setFormData({ ...formData, track_api_calls: checked })}
        />
        
        <ToggleField
          label="Track Model Usage"
          checked={formData.track_model_usage}
          onChange={(checked) => setFormData({ ...formData, track_model_usage: checked })}
        />
        
        <NumberField
          label="Retention Days"
          value={formData.retention_days}
          onChange={(value) => setFormData({ ...formData, retention_days: value })}
        />
        
        <NumberField
          label="Aggregation Interval (minutes)"
          value={formData.aggregation_interval_minutes}
          onChange={(value) => setFormData({ ...formData, aggregation_interval_minutes: value })}
        />
      </div>
      
      <div className="mt-6 flex gap-3">
        <button
          onClick={() => updateMutation.mutate(formData)}
          disabled={updateMutation.isPending}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 transition-colors flex items-center gap-2"
        >
          <Save className="w-4 h-4" />
          {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
        </button>
        
        <button
          onClick={() => setFormData(config)}
          className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Reset
        </button>
      </div>
    </div>
  )
}

function AuditConfig({ config, refetch }: any) {
  const [formData, setFormData] = useState(config)
  
  const updateMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await fetch('/api/audit/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
      return response.json()
    },
    onSuccess: () => {
      refetch()
      alert('Audit configuration updated successfully!')
    },
  })
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
        Audit Logging Configuration
      </h3>
      
      <div className="space-y-4">
        <ToggleField
          label="Audit Logging Enabled"
          checked={formData.enabled}
          onChange={(checked) => setFormData({ ...formData, enabled: checked })}
        />
        
        <ToggleField
          label="Log API Requests"
          checked={formData.log_api_requests}
          onChange={(checked) => setFormData({ ...formData, log_api_requests: checked })}
        />
        
        <ToggleField
          label="Log Queries"
          checked={formData.log_queries}
          onChange={(checked) => setFormData({ ...formData, log_queries: checked })}
        />
        
        <NumberField
          label="Retention Days"
          value={formData.retention_days}
          onChange={(value) => setFormData({ ...formData, retention_days: value })}
        />
      </div>
      
      <div className="mt-6 flex gap-3">
        <button
          onClick={() => updateMutation.mutate(formData)}
          disabled={updateMutation.isPending}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 transition-colors flex items-center gap-2"
        >
          <Save className="w-4 h-4" />
          {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
        </button>
        
        <button
          onClick={() => setFormData(config)}
          className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Reset
        </button>
      </div>
    </div>
  )
}

function LanguageConfig({ config, refetch }: any) {
  const [formData, setFormData] = useState(config)
  
  const updateMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await fetch('/api/language/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
      return response.json()
    },
    onSuccess: () => {
      refetch()
      alert('Language configuration updated successfully!')
    },
  })
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
        Language Configuration
      </h3>
      
      <div className="space-y-4">
        <ToggleField
          label="Translation Enabled"
          checked={formData.translation_enabled}
          onChange={(checked) => setFormData({ ...formData, translation_enabled: checked })}
        />
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Default Language
          </label>
          <select
            value={formData.default_language}
            onChange={(e) => setFormData({ ...formData, default_language: e.target.value })}
            className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
          >
            {config.supported_languages?.map((lang: string) => (
              <option key={lang} value={lang}>{lang.toUpperCase()}</option>
            ))}
          </select>
        </div>
      </div>
      
      <div className="mt-6 flex gap-3">
        <button
          onClick={() => updateMutation.mutate(formData)}
          disabled={updateMutation.isPending}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 transition-colors flex items-center gap-2"
        >
          <Save className="w-4 h-4" />
          {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
        </button>
        
        <button
          onClick={() => setFormData(config)}
          className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Reset
        </button>
      </div>
    </div>
  )
}

function ToggleField({ label, checked, onChange }: any) {
  return (
    <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
        {label}
      </label>
      <button
        onClick={() => onChange(!checked)}
        className={`
          relative inline-flex h-6 w-11 items-center rounded-full transition-colors
          ${checked ? 'bg-blue-600' : 'bg-gray-300 dark:bg-gray-600'}
        `}
      >
        <span
          className={`
            inline-block h-4 w-4 transform rounded-full bg-white transition-transform
            ${checked ? 'translate-x-6' : 'translate-x-1'}
          `}
        />
      </button>
    </div>
  )
}

function NumberField({ label, value, onChange }: any) {
  return (
    <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
        {label}
      </label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full px-3 py-2 bg-white dark:bg-gray-600 border border-gray-300 dark:border-gray-500 rounded-lg"
      />
    </div>
  )
}

