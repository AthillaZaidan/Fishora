import { ApiError, kindFromResponse } from './errors'

export { ApiError }
export type { ApiErrorKind } from './errors'

const PUBLIC_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000'
// A server render inside a container cannot use the browser's URL (localhost is
// the container itself), so the server may read an internal one at runtime.
const BASE =
  typeof window === 'undefined'
    ? (process.env.FISHORA_INTERNAL_API_BASE_URL ?? PUBLIC_BASE)
    : PUBLIC_BASE
const DEFAULT_TIMEOUT_MS = 35_000 // above the backend's 30s CV timeout

export interface ApiFetchOptions extends RequestInit {
  timeoutMs?: number
}

export async function apiFetch<T>(
  path: string,
  { timeoutMs = DEFAULT_TIMEOUT_MS, signal, ...init }: ApiFetchOptions = {}
): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const combined =
    signal && typeof AbortSignal.any === 'function'
      ? AbortSignal.any([signal, controller.signal])
      : controller.signal

  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, { credentials: 'include', ...init, signal: combined })
  } catch (cause) {
    // Aborted means timeout, anything else means the network is gone.
    const aborted = cause instanceof DOMException && cause.name === 'AbortError'
    throw new ApiError(aborted ? 'timeout' : 'offline', 0)
  } finally {
    clearTimeout(timer)
  }

  if (!response.ok) {
    let detail = ''
    let chunkIds: string[] | undefined
    let highest: string | undefined
    try {
      const body = await response.json()
      detail = typeof body?.detail === 'string' ? body.detail : ''
      chunkIds = Array.isArray(body?.retrieved_chunk_ids) ? body.retrieved_chunk_ids : undefined
      highest =
        typeof body?.current_highest_per_kg === 'string' ? body.current_highest_per_kg : undefined
    } catch {
      // Non-JSON error body: the status carries enough.
    }
    throw new ApiError(kindFromResponse(response.status, detail, highest), response.status, chunkIds, highest)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}
