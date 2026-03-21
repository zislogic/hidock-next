import { app } from 'electron'
import { join } from 'path'
import { existsSync, mkdirSync, unlinkSync, readdirSync, createWriteStream, renameSync } from 'fs'
import https from 'https'
import { IncomingMessage } from 'http'

export type WhisperModelSize = 'tiny' | 'base' | 'small' | 'medium' | 'large-v3'

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
  'large-v3': {
    size: 'large-v3',
    filename: 'ggml-large-v3.bin',
    url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin',
    approxMB: 3100
  }
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

export async function downloadModel(
  size: WhisperModelSize,
  onProgress?: (progress: number, bytesDownloaded: number, totalBytes: number) => void
): Promise<string> {
  const model = WHISPER_MODELS[size]
  const destPath = getModelPath(size)
  const tempPath = destPath + '.download'

  if (existsSync(destPath)) {
    return destPath
  }

  const abort = new AbortController()
  activeDownloadAbort = abort

  try {
    await new Promise<void>((resolve, reject) => {
      const follow = (url: string, redirectCount = 0): void => {
        if (redirectCount > 5) {
          reject(new Error('Too many redirects'))
          return
        }

        const request = https.get(url, { signal: abort.signal }, (response: IncomingMessage) => {
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

      follow(model.url)
    })

    return destPath
  } catch (err) {
    cleanup(tempPath)
    throw err
  } finally {
    if (activeDownloadAbort === abort) {
      activeDownloadAbort = null
    }
  }
}

function cleanup(path: string): void {
  try {
    if (existsSync(path)) unlinkSync(path)
  } catch { /* ignore cleanup errors */ }
}
