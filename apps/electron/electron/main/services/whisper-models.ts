import { app } from 'electron'
import { join } from 'path'
import { existsSync, mkdirSync, unlinkSync, readdirSync, createWriteStream, renameSync } from 'fs'
import https from 'https'
import { IncomingMessage } from 'http'

export type WhisperModelSize = 'tiny' | 'base' | 'small' | 'medium' | 'large-v3-turbo' | 'large-v3'

export interface ModelInfo {
  size: WhisperModelSize
  filename: string
  url: string
  approxMB: number
}

export const WHISPER_MODELS: Record<WhisperModelSize, ModelInfo> = {
  tiny: {
    size: 'tiny',
    filename: 'ggml-tiny.bin',
    url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
    approxMB: 75
  },
  base: {
    size: 'base',
    filename: 'ggml-base.bin',
    url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
    approxMB: 142
  },
  small: {
    size: 'small',
    filename: 'ggml-small.bin',
    url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
    approxMB: 466
  },
  medium: {
    size: 'medium',
    filename: 'ggml-medium.bin',
    url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin',
    approxMB: 1500
  },
  'large-v3-turbo': {
    size: 'large-v3-turbo',
    filename: 'ggml-large-v3-turbo.bin',
    url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin',
    approxMB: 1600
  },
  'large-v3': {
    size: 'large-v3',
    filename: 'ggml-large-v3.bin',
    url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin',
    approxMB: 3100
  }
}

// VAD model for voice activity detection (auto-downloaded alongside whisper models)
export const VAD_MODEL = {
  filename: 'ggml-silero-vad.bin',
  url: 'https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin',
  approxMB: 1
}

export function getVadModelPath(): string {
  return join(getModelsDir(), VAD_MODEL.filename)
}

export function isVadModelDownloaded(): boolean {
  return existsSync(getVadModelPath())
}

let activeDownloadAbort: AbortController | null = null

export function getModelsDir(): string {
  const dir = join(app.getPath('userData'), 'whisper-models')
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true })
  }
  return dir
}

export function getModelPath(size: WhisperModelSize): string {
  return join(getModelsDir(), WHISPER_MODELS[size].filename)
}

export function isModelDownloaded(size: WhisperModelSize): boolean {
  return existsSync(getModelPath(size))
}

export function getDownloadedModels(): WhisperModelSize[] {
  const dir = getModelsDir()
  const files = readdirSync(dir)
  return (Object.keys(WHISPER_MODELS) as WhisperModelSize[]).filter((size) =>
    files.includes(WHISPER_MODELS[size].filename)
  )
}

export function deleteModel(size: WhisperModelSize): boolean {
  const path = getModelPath(size)
  if (existsSync(path)) {
    unlinkSync(path)
    return true
  }
  return false
}

export function cancelModelDownload(): void {
  if (activeDownloadAbort) {
    activeDownloadAbort.abort()
    activeDownloadAbort = null
  }
}

async function downloadFile(
  url: string,
  destPath: string,
  abort: AbortController,
  onProgress?: (progress: number, bytesDownloaded: number, totalBytes: number) => void
): Promise<void> {
  const tempPath = destPath + '.download'

  if (existsSync(destPath)) return

  await new Promise<void>((resolve, reject) => {
    const follow = (followUrl: string, redirectCount = 0): void => {
      if (redirectCount > 5) {
        reject(new Error('Too many redirects'))
        return
      }

      const request = https.get(followUrl, { signal: abort.signal }, (response: IncomingMessage) => {
        if (response.statusCode && response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
          response.resume()
          follow(response.headers.location, redirectCount + 1)
          return
        }

        if (response.statusCode !== 200) {
          reject(new Error(`Download failed with status ${response.statusCode}`))
          return
        }

        const totalBytes = parseInt(response.headers['content-length'] || '0', 10)
        let bytesDownloaded = 0

        const file = createWriteStream(tempPath)
        response.pipe(file)

        response.on('data', (chunk: Buffer) => {
          bytesDownloaded += chunk.length
          if (onProgress && totalBytes > 0) {
            onProgress(Math.round((bytesDownloaded / totalBytes) * 100), bytesDownloaded, totalBytes)
          }
        })

        file.on('finish', () => {
          file.close(() => {
            renameSync(tempPath, destPath)
            resolve()
          })
        })

        file.on('error', (err) => {
          file.close()
          cleanup(tempPath)
          reject(err)
        })
      })

      request.on('error', (err) => {
        cleanup(tempPath)
        reject(err)
      })
    }

    follow(url)
  })
}

export async function downloadModel(
  size: WhisperModelSize,
  onProgress?: (progress: number, bytesDownloaded: number, totalBytes: number) => void
): Promise<string> {
  const model = WHISPER_MODELS[size]
  const destPath = getModelPath(size)

  if (existsSync(destPath)) {
    // Still ensure VAD model is downloaded
    await ensureVadModel()
    return destPath
  }

  const abort = new AbortController()
  activeDownloadAbort = abort

  try {
    await downloadFile(model.url, destPath, abort, onProgress)
    // Auto-download VAD model alongside whisper model (small, ~2MB)
    await ensureVadModel()
    return destPath
  } catch (err) {
    cleanup(destPath + '.download')
    throw err
  } finally {
    if (activeDownloadAbort === abort) {
      activeDownloadAbort = null
    }
  }
}

export async function ensureVadModel(): Promise<void> {
  const vadPath = getVadModelPath()
  if (existsSync(vadPath)) return
  console.log('[Whisper] Auto-downloading VAD model...')
  const abort = new AbortController()
  try {
    await downloadFile(VAD_MODEL.url, vadPath, abort)
    console.log('[Whisper] VAD model downloaded')
  } catch (err) {
    console.warn('[Whisper] Failed to download VAD model:', err)
  }
}

function cleanup(path: string): void {
  try {
    if (existsSync(path)) unlinkSync(path)
  } catch { /* ignore cleanup errors */ }
}
