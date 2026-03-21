import { execFile } from 'child_process'
import { tmpdir } from 'os'
import { join } from 'path'
import { existsSync, unlinkSync } from 'fs'
import { randomUUID } from 'crypto'

function getFfmpegPath(): string {
  // ffmpeg-static provides the platform-specific binary path
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  return require('ffmpeg-static') as string
}

/**
 * Convert any supported audio file to 16kHz mono WAV (whisper.cpp requirement).
 * Returns the path to the converted temp file. Caller must clean up with cleanupTempFile().
 */
export async function convertToWhisperFormat(inputPath: string): Promise<string> {
  const outputPath = join(tmpdir(), `hidock-whisper-${randomUUID()}.wav`)
  const ffmpegPath = getFfmpegPath()

  return new Promise<string>((resolve, reject) => {
    execFile(
      ffmpegPath,
      [
        '-i', inputPath,
        '-ar', '16000',
        '-ac', '1',
        '-c:a', 'pcm_s16le',
        '-f', 'wav',
        '-y',
        outputPath
      ],
      { timeout: 120_000 },
      (error, _stdout, stderr) => {
        if (error) {
          cleanupTempFile(outputPath)
          reject(new Error(`Audio conversion failed: ${stderr || error.message}`))
          return
        }
        resolve(outputPath)
      }
    )
  })
}

export function cleanupTempFile(path: string): void {
  try {
    if (existsSync(path)) unlinkSync(path)
  } catch { /* ignore cleanup errors */ }
}
