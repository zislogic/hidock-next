/**
 * TranscriptionControls Component
 *
 * Per-recording language and model/provider selector with (re-)transcribe button.
 * Replaces the simple Transcribe button in SourceReader with richer controls.
 */

import { useState, useEffect } from 'react'
import { Wand2, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Select, SelectTrigger, SelectValue, SelectContent, SelectItem
} from '@/components/ui/select'
import type { UnifiedRecording } from '@/types/unified-recording'
import { hasLocalPath } from '@/types/unified-recording'

const LANGUAGES = [
  { value: 'auto', label: 'Auto' },
  { value: 'en', label: 'English' },
  { value: 'de', label: 'German' },
  { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },
  { value: 'it', label: 'Italian' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ja', label: 'Japanese' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ko', label: 'Korean' },
]

interface ModelOption {
  value: string // e.g. "whisper:base", "whisper:large-v3-turbo", "gemini:gemini-3-pro-preview"
  label: string
  provider: string
}

interface TranscriptionControlsProps {
  recording: UnifiedRecording
  hasTranscript: boolean
  onTranscribe: (overrides: { provider?: string; model?: string; language?: string }) => void
  disabled?: boolean
}

export function TranscriptionControls({
  recording,
  hasTranscript,
  onTranscribe,
  disabled = false
}: TranscriptionControlsProps) {
  const [language, setLanguage] = useState('auto')
  const [selectedModel, setSelectedModel] = useState('')
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([])
  const [loading, setLoading] = useState(true)

  const isProcessing = recording.transcriptionStatus === 'processing'
  const isPending = recording.transcriptionStatus === 'pending'
  const isBusy = isProcessing || isPending

  // Load available models and defaults on mount
  useEffect(() => {
    let cancelled = false

    async function loadModels() {
      setLoading(true)
      const options: ModelOption[] = []

      // Get downloaded whisper models
      try {
        const result = await window.electronAPI.whisper.getDownloadedModels()
        if (result.success && result.models) {
          const modelLabels: Record<string, string> = {
            tiny: 'Whisper Tiny',
            base: 'Whisper Base',
            small: 'Whisper Small',
            medium: 'Whisper Medium',
            'large-v3-turbo': 'Whisper Large V3 Turbo',
            'large-v3': 'Whisper Large V3'
          }
          for (const model of result.models) {
            options.push({
              value: `whisper:${model}`,
              label: modelLabels[model] || `Whisper ${model}`,
              provider: 'whisper'
            })
          }
        }
      } catch (e) {
        console.error('Failed to get whisper models:', e)
      }

      // Check if Gemini is available
      try {
        const result = await window.electronAPI.config.getValue('transcription.geminiApiKey')
        const apiKey = result?.success ? result.data : null
        if (apiKey && typeof apiKey === 'string' && apiKey.trim() !== '') {
          const modelResult = await window.electronAPI.config.getValue('transcription.geminiModel')
          const geminiModel = modelResult?.success ? modelResult.data : 'gemini-3-pro-preview'
          options.push({
            value: `gemini:${geminiModel}`,
            label: `Gemini (${geminiModel})`,
            provider: 'gemini'
          })
        }
      } catch (e) {
        console.error('Failed to check Gemini config:', e)
      }

      if (cancelled) return

      setModelOptions(options)

      // Set defaults from config
      try {
        const providerResult = await window.electronAPI.config.getValue('transcription.provider')
        const provider = providerResult?.success ? providerResult.data : 'whisper'

        if (provider === 'whisper') {
          const modelResult = await window.electronAPI.config.getValue('transcription.whisperModelSize')
          const modelSize = modelResult?.success ? modelResult.data : 'base'
          const defaultValue = `whisper:${modelSize}`
          // Only set if available
          if (options.some(o => o.value === defaultValue)) {
            setSelectedModel(defaultValue)
          } else if (options.length > 0) {
            setSelectedModel(options[0].value)
          }

          const langResult = await window.electronAPI.config.getValue('transcription.whisperLanguage')
          const defaultLang = langResult?.success ? langResult.data : 'auto'
          setLanguage(defaultLang as string)
        } else {
          const geminiOption = options.find(o => o.provider === 'gemini')
          if (geminiOption) {
            setSelectedModel(geminiOption.value)
          } else if (options.length > 0) {
            setSelectedModel(options[0].value)
          }

          const langResult = await window.electronAPI.config.getValue('transcription.language')
          const defaultLang = langResult?.success && langResult.data ? langResult.data : 'auto'
          setLanguage(defaultLang as string)
        }
      } catch (e) {
        if (options.length > 0) setSelectedModel(options[0].value)
      }

      setLoading(false)
    }

    loadModels()
    return () => { cancelled = true }
  }, [])

  if (!hasLocalPath(recording)) return null
  if (loading && modelOptions.length === 0) return null

  const handleTranscribe = () => {
    if (!selectedModel) return

    const [provider, model] = selectedModel.split(':')
    onTranscribe({
      provider,
      model,
      language: language !== 'auto' ? language : undefined
    })
  }

  const buttonLabel = isBusy
    ? (isProcessing ? 'In Progress' : 'Queued')
    : (hasTranscript ? 'Re-Transcribe' : 'Transcribe')

  const buttonIcon = isBusy
    ? <RefreshCw className={`h-4 w-4 ${isProcessing ? 'animate-spin' : ''}`} />
    : <Wand2 className="h-4 w-4" />

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {/* Language selector */}
      <Select value={language} onValueChange={setLanguage} disabled={isBusy || disabled}>
        <SelectTrigger className="w-[100px] h-8 text-xs">
          <SelectValue placeholder="Language" />
        </SelectTrigger>
        <SelectContent>
          {LANGUAGES.map((lang) => (
            <SelectItem key={lang.value} value={lang.value}>
              {lang.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Model/Provider selector */}
      {modelOptions.length > 0 && (
        <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isBusy || disabled}>
          <SelectTrigger className="w-[180px] h-8 text-xs">
            <SelectValue placeholder="Select model" />
          </SelectTrigger>
          <SelectContent>
            {modelOptions.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}

      {/* Transcribe / Re-Transcribe button */}
      <Button
        variant="outline"
        size="sm"
        onClick={handleTranscribe}
        disabled={isBusy || disabled || !selectedModel}
        className="gap-2"
        title={isBusy ? buttonLabel : hasTranscript ? 'Re-transcribe with selected model' : 'Start transcription'}
      >
        {buttonIcon}
        {buttonLabel}
      </Button>
    </div>
  )
}
