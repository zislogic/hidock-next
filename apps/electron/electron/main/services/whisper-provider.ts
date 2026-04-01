import { initWhisper, initWhisperVad } from '@fugood/whisper.node'
import type { WhisperContext, WhisperVadContext, TranscribeResult } from '@fugood/whisper.node'
import { getModelPath, getVadModelPath, isModelDownloaded, isVadModelDownloaded, ensureVadModel, type WhisperModelSize } from './whisper-models'
import { convertToWhisperFormat, cleanupTempFile } from './audio-converter'

let activeContext: WhisperContext | null = null
let activeModelSize: WhisperModelSize | null = null
let activeStop: (() => Promise<void>) | null = null
let vadContext: WhisperVadContext | null = null

export interface WhisperTranscribeResult {
  text: string
  language: string
  segments: Array<{ text: string; t0: number; t1: number }>
}

/**
 * Use VAD to detect speech segments, then transcribe only those segments.
 * This prevents Whisper from hallucinating during silence.
 */
async function getVadSegments(
  wavPath: string,
  useGpu: boolean
): Promise<Array<{ t0: number; t1: number }> | null> {
  try {
    if (!isVadModelDownloaded()) {
      await ensureVadModel()
    }
    if (!isVadModelDownloaded()) {
      console.log('[Whisper] VAD model not available, skipping VAD')
      return null
    }

    if (!vadContext) {
      vadContext = await initWhisperVad({
        filePath: getVadModelPath(),
        useGpu: false  // VAD model is tiny (<1MB), CPU is fine and avoids Metal backend conflicts
      })
    }

    const segments = await vadContext.detectSpeechFile(wavPath, {
      threshold: 0.5,
      minSpeechDurationMs: 500,
      minSilenceDurationMs: 300,
      speechPadMs: 200
    })

    console.log(`[Whisper] VAD detected ${segments.length} speech segments`)
    return segments
  } catch (err) {
    console.warn('[Whisper] VAD failed, proceeding without:', err)
    return null
  }
}

export async function transcribeWithWhisper(
  audioFilePath: string,
  config: {
    modelSize: WhisperModelSize
    language: string
    useGpu: boolean
  },
  progressCallback?: (stage: string, progress: number) => void
): Promise<WhisperTranscribeResult> {
  if (!isModelDownloaded(config.modelSize)) {
    throw new Error(
      `Whisper model "${config.modelSize}" is not downloaded. Please download it in Settings first.`
    )
  }

  const modelPath = getModelPath(config.modelSize)
  let tempWavPath: string | null = null

  try {
    // Stage 1: Convert audio to whisper format
    progressCallback?.('converting', 5)
    tempWavPath = await convertToWhisperFormat(audioFilePath)

    // Stage 2: Run VAD to detect speech segments
    progressCallback?.('detecting_speech', 8)
    const vadSegments = await getVadSegments(tempWavPath, config.useGpu)

    // Stage 3: Initialize whisper context (reuse if same model)
    progressCallback?.('loading_model', 10)
    if (!activeContext || activeModelSize !== config.modelSize) {
      if (activeContext) await activeContext.release()
      activeContext = await initWhisper({
        filePath: modelPath,
        useGpu: config.useGpu
      })
      activeModelSize = config.modelSize
    }

    // Stage 4: Transcribe
    progressCallback?.('transcribing', 15)

    if (vadSegments && vadSegments.length > 0) {
      // Transcribe each speech segment separately to avoid hallucination in silence gaps
      return await transcribeWithVadSegments(tempWavPath, config, vadSegments, progressCallback)
    }

    // Fallback: transcribe the whole file if VAD unavailable
    return await transcribeFullFile(tempWavPath, config, progressCallback)
  } finally {
    if (tempWavPath) {
      cleanupTempFile(tempWavPath)
    }
  }
}

async function transcribeFullFile(
  wavPath: string,
  config: { language: string },
  progressCallback?: (stage: string, progress: number) => void
): Promise<WhisperTranscribeResult> {
  const { stop, promise } = activeContext!.transcribeFile(wavPath, {
    language: config.language === 'auto' ? undefined : config.language,
    temperature: 0.0,
    temperatureInc: 0.2,
    onProgress: (progress: number) => {
      const mapped = 15 + Math.round(progress * 0.75)
      progressCallback?.('transcribing', mapped)
    }
  })

  activeStop = stop
  const result: TranscribeResult = await promise
  activeStop = null

  if (result.isAborted) throw new Error('Transcription was cancelled')

  progressCallback?.('transcribing', 95)

  return {
    text: result.result,
    language: result.language || config.language,
    segments: result.segments
  }
}

async function transcribeWithVadSegments(
  wavPath: string,
  config: { language: string },
  vadSegments: Array<{ t0: number; t1: number }>,
  progressCallback?: (stage: string, progress: number) => void
): Promise<WhisperTranscribeResult> {
  const allSegments: Array<{ text: string; t0: number; t1: number }> = []
  const textParts: string[] = []
  let detectedLanguage = config.language

  for (let i = 0; i < vadSegments.length; i++) {
    const vad = vadSegments[i]
    const progressBase = 15 + Math.round((i / vadSegments.length) * 75)
    progressCallback?.('transcribing', progressBase)

    // Convert VAD timestamps (centiseconds) to seconds for whisper offset/duration
    const offsetMs = vad.t0 * 10
    const durationMs = (vad.t1 - vad.t0) * 10
    const offsetSec = Math.floor(offsetMs / 1000)
    const durationSec = Math.ceil(durationMs / 1000) + 1 // +1s padding

    const { stop, promise } = activeContext!.transcribeFile(wavPath, {
      language: config.language === 'auto' ? undefined : config.language,
      temperature: 0.0,
      temperatureInc: 0.2,
      offset: offsetSec * 1000,    // whisper expects milliseconds
      duration: durationSec * 1000,
      onProgress: (progress: number) => {
        const segProgress = progressBase + Math.round((progress / 100) * (75 / vadSegments.length))
        progressCallback?.('transcribing', Math.min(segProgress, 90))
      }
    })

    activeStop = stop
    const result: TranscribeResult = await promise
    activeStop = null

    if (result.isAborted) throw new Error('Transcription was cancelled')

    if (result.result.trim()) {
      textParts.push(result.result.trim())
      allSegments.push(...result.segments)
    }
    if (result.language) detectedLanguage = result.language
  }

  progressCallback?.('transcribing', 95)

  return {
    text: textParts.join(' '),
    language: detectedLanguage,
    segments: allSegments
  }
}

export async function cancelWhisperTranscription(): Promise<void> {
  if (activeStop) {
    await activeStop()
    activeStop = null
  }
}

export async function releaseWhisperContext(): Promise<void> {
  if (activeContext) {
    await activeContext.release()
    activeContext = null
    activeModelSize = null
  }
  if (vadContext) {
    await vadContext.release()
    vadContext = null
  }
}
