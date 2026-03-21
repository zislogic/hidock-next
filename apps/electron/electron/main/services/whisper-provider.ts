import { initWhisper } from '@fugood/whisper.node'
import type { WhisperContext, TranscribeResult } from '@fugood/whisper.node'
import { getModelPath, isModelDownloaded, type WhisperModelSize } from './whisper-models'
import { convertToWhisperFormat, cleanupTempFile } from './audio-converter'

let activeContext: WhisperContext | null = null
let activeStop: (() => Promise<void>) | null = null

export interface WhisperTranscribeResult {
  text: string
  language: string
  segments: Array<{ text: string; t0: number; t1: number }>
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

    // Stage 2: Initialize whisper context (reuse if same model)
    progressCallback?.('loading_model', 10)
    if (!activeContext) {
      activeContext = await initWhisper({
        filePath: modelPath,
        useGpu: config.useGpu
      })
    }

    // Stage 3: Transcribe
    progressCallback?.('transcribing', 15)
    const { stop, promise } = activeContext.transcribeFile(tempWavPath, {
      language: config.language === 'auto' ? undefined : config.language,
      temperature: 0.0,
      onProgress: (progress: number) => {
        // Map whisper's 0-100 to our 15-90 range
        const mapped = 15 + Math.round(progress * 0.75)
        progressCallback?.('transcribing', mapped)
      }
    })

    activeStop = stop

    const result: TranscribeResult = await promise
    activeStop = null

    if (result.isAborted) {
      throw new Error('Transcription was cancelled')
    }

    progressCallback?.('transcribing', 95)

    return {
      text: result.result,
      language: result.language || config.language,
      segments: result.segments
    }
  } finally {
    if (tempWavPath) {
      cleanupTempFile(tempWavPath)
    }
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
  }
}
