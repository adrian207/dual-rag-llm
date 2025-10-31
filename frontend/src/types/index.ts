/**
 * TypeScript Type Definitions
 * 
 * @author Adrian Johnson <adrian207@gmail.com>
 */

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  model?: string
  cached?: boolean
  error?: boolean
}

export interface SystemStats {
  cache: CacheStats
  tools: ToolStats
  models_cached: string[]
  ms_index_loaded: boolean
  oss_index_loaded: boolean
}

export interface CacheStats {
  total_queries: number
  cache_hits: number
  cache_misses: number
  hit_rate: number
}

export interface ToolStats {
  web_searches: number
  github_queries: number
}

export interface QueryRequest {
  question: string
  file_ext?: string
  github_repo?: string
  use_web_search?: boolean
  model_override?: string
  compare_models?: boolean
  language?: string
}

export interface QueryResponse {
  answer: string
  model: string
  source: string
  chunks_retrieved: number
  cached?: boolean
  tools_used?: string[]
}

export interface Model {
  name: string
  size: string
  modified_at: string
  digest: string
  details?: {
    format: string
    family: string
    families: string[]
    parameter_size: string
    quantization_level: string
  }
}

export interface Language {
  code: string
  name: string
  native_name: string
}

export type Theme = 'light' | 'dark' | 'system'

export interface AppSettings {
  theme: Theme
  language: string
  autoScroll: boolean
  soundEnabled: boolean
  compactMode: boolean
}

