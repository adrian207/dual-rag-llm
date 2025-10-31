/**
 * Encryption Management Panel
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */
import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Lock, Key, RefreshCw, Shield, AlertTriangle, CheckCircle, Copy } from 'lucide-react'
import { apiClient } from '@/lib/api'

export default function EncryptionPanel() {
  const [encryptText, setEncryptText] = useState('')
  const [decryptText, setDecryptText] = useState('')
  const [result, setResult] = useState<string>('')
  const [showKeyGenerated, setShowKeyGenerated] = useState(false)
  
  // Fetch encryption status
  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ['encryption-status'],
    queryFn: apiClient.getEncryptionStatus,
  })
  
  // Encrypt mutation
  const encryptMutation = useMutation({
    mutationFn: (data: string) => apiClient.encryptData({ data }),
    onSuccess: (response) => {
      setResult(response.encrypted_data)
    },
  })
  
  // Decrypt mutation
  const decryptMutation = useMutation({
    mutationFn: (data: string) => apiClient.decryptData({ encrypted_data: data }),
    onSuccess: (response) => {
      setResult(response.decrypted_data)
    },
  })
  
  // Key rotation mutation
  const rotateMutation = useMutation({
    mutationFn: () => apiClient.rotateKey({}),
    onSuccess: () => {
      refetchStatus()
      alert('Key rotated successfully! Old data will need re-encryption.')
    },
  })
  
  // Generate key mutation
  const generateKeyMutation = useMutation({
    mutationFn: () => apiClient.generateKey(),
    onSuccess: (response) => {
      setResult(response.key)
      setShowKeyGenerated(true)
    },
  })
  
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    alert('Copied to clipboard!')
  }
  
  return (
    <div className="space-y-6">
      {/* Encryption Status */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <Shield className="w-5 h-5" />
            Encryption Status
          </h3>
          <button
            onClick={() => refetchStatus()}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <StatusCard
            label="Encryption Enabled"
            value={status?.encryption_enabled ? 'Yes' : 'No'}
            status={status?.encryption_enabled ? 'success' : 'warning'}
          />
          <StatusCard
            label="Algorithm"
            value={status?.algorithm || 'N/A'}
            status="info"
          />
          <StatusCard
            label="Key Initialized"
            value={status?.key_initialized ? 'Yes' : 'No'}
            status={status?.key_initialized ? 'success' : 'error'}
          />
          <StatusCard
            label="Key Age"
            value={status?.key_age_days ? `${status.key_age_days} days` : 'N/A'}
            status={status?.key_age_days && status.key_age_days > 90 ? 'warning' : 'success'}
          />
        </div>
        
        {status?.key_age_days && status.key_age_days > 90 && (
          <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                Key Rotation Recommended
              </p>
              <p className="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
                Your encryption key is {status.key_age_days} days old. Consider rotating it for enhanced security.
              </p>
            </div>
          </div>
        )}
      </div>
      
      {/* Encrypt/Decrypt Tools */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Encrypt */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Lock className="w-5 h-5" />
            Encrypt Data
          </h3>
          
          <div className="space-y-4">
            <textarea
              value={encryptText}
              onChange={(e) => setEncryptText(e.target.value)}
              placeholder="Enter text to encrypt..."
              className="w-full h-32 px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg resize-none"
            />
            
            <button
              onClick={() => encryptMutation.mutate(encryptText)}
              disabled={!encryptText || encryptMutation.isPending}
              className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {encryptMutation.isPending ? 'Encrypting...' : 'Encrypt'}
            </button>
          </div>
        </div>
        
        {/* Decrypt */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Key className="w-5 h-5" />
            Decrypt Data
          </h3>
          
          <div className="space-y-4">
            <textarea
              value={decryptText}
              onChange={(e) => setDecryptText(e.target.value)}
              placeholder="Enter encrypted text to decrypt..."
              className="w-full h-32 px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg resize-none"
            />
            
            <button
              onClick={() => decryptMutation.mutate(decryptText)}
              disabled={!decryptText || decryptMutation.isPending}
              className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {decryptMutation.isPending ? 'Decrypting...' : 'Decrypt'}
            </button>
          </div>
        </div>
      </div>
      
      {/* Result Display */}
      {result && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Result
            </h3>
            <button
              onClick={() => copyToClipboard(result)}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            >
              <Copy className="w-4 h-4" />
            </button>
          </div>
          
          <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg font-mono text-sm break-all">
            {result}
          </div>
        </div>
      )}
      
      {/* Key Management */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Key Management
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={() => {
              if (confirm('Rotate encryption key? This will require re-encrypting all data.')) {
                rotateMutation.mutate()
              }
            }}
            disabled={rotateMutation.isPending}
            className="px-4 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-lg disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            {rotateMutation.isPending ? 'Rotating...' : 'Rotate Key'}
          </button>
          
          <button
            onClick={() => {
              if (confirm('Generate new encryption key? Save it securely!')) {
                generateKeyMutation.mutate()
              }
            }}
            disabled={generateKeyMutation.isPending}
            className="px-4 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
          >
            <Key className="w-4 h-4" />
            {generateKeyMutation.isPending ? 'Generating...' : 'Generate New Key'}
          </button>
        </div>
        
        {showKeyGenerated && (
          <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-800 dark:text-red-200">
                  ⚠️ Save This Key Immediately!
                </p>
                <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                  Store the key shown above in a secure location. It cannot be recovered if lost.
                  Set it as the ENCRYPTION_MASTER_KEY environment variable.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function StatusCard({ label, value, status }: { label: string, value: string, status: 'success' | 'warning' | 'error' | 'info' }) {
  const colors = {
    success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-800 dark:text-green-200',
    warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-200',
    error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-800 dark:text-red-200',
    info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-200',
  }
  
  const icons = {
    success: CheckCircle,
    warning: AlertTriangle,
    error: AlertTriangle,
    info: Shield,
  }
  
  const Icon = icons[status]
  
  return (
    <div className={`p-4 rounded-lg border ${colors[status]}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm opacity-80">{label}</p>
          <p className="text-lg font-semibold mt-1">{value}</p>
        </div>
        <Icon className="w-5 h-5" />
      </div>
    </div>
  )
}

