/** Closed set, mirroring the exception handlers in apps/main_api/main.py. */
export type ApiErrorKind =
  | 'offline'
  | 'timeout'
  | 'image_invalid'
  | 'image_too_large'
  | 'not_found'
  | 'not_verified'
  | 'forbidden'
  | 'lot_not_allocated'
  | 'unsupported_species'
  | 'cv_label_unsupported'
  | 'cv_unavailable'
  | 'photo_not_fish'
  | 'photo_unknown_species'
  | 'generation_unavailable'
  | 'generation_invalid'
  | 'outbid'
  | 'server'

// Kept separate from the server's `detail`, which can carry internal hostnames.
const MESSAGES: Record<ApiErrorKind, string> = {
  offline: 'No connection. What you have entered is still saved.',
  timeout: 'The request took too long. Try again.',
  image_invalid: 'Image format not supported. Use JPG or PNG.',
  image_too_large: 'Image is too large. The maximum is 10 MB.',
  not_found: 'Not found.',
  not_verified: 'The species has not been verified yet. Confirm it before continuing.',
  forbidden: 'Your account is not allowed to do this.',
  lot_not_allocated: 'This lot has not been allocated yet.',
  unsupported_species: 'This species is not supported yet.',
  cv_label_unsupported: 'The model returned a species that is not supported yet.',
  cv_unavailable: 'The identification service is unavailable right now.',
  photo_not_fish: 'This photo does not look like a fish. Retake it: whole fish, plain background, enough light.',
  photo_unknown_species:
    'This fish was not recognised as one of the 11 supported species. Retake the photo, or pick the species if it is in the list.',
  generation_unavailable: 'Knowledge card generation is unavailable right now.',
  generation_invalid: 'The knowledge card failed validation.',
  outbid: 'Your bid must be higher than the current highest bid.',
  server: 'Something went wrong. Try again.',
}

/** The user-facing copy for a kind, for callers that build an error locally. */
export function messageFor(kind: ApiErrorKind): string {
  return MESSAGES[kind]
}

/** Only these get a Retry button. */
const RETRYABLE: ReadonlySet<ApiErrorKind> = new Set([
  'offline', 'timeout', 'cv_unavailable', 'generation_unavailable', 'server',
])

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status: number
  readonly retryable: boolean
  readonly userMessage: string
  /** Generation failures only. For diagnosis, never for display. */
  readonly retrievedChunkIds?: string[]
  readonly currentHighestPerKg?: string

  constructor(
    kind: ApiErrorKind,
    status: number,
    retrievedChunkIds?: string[],
    currentHighestPerKg?: string
  ) {
    super(`ApiError(${kind}) status=${status}`)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
    this.retryable = RETRYABLE.has(kind)
    this.userMessage = MESSAGES[kind]
    this.retrievedChunkIds = retrievedChunkIds
    this.currentHighestPerKg = currentHighestPerKg
  }
}

export function kindFromResponse(
  status: number,
  detail: string,
  currentHighestPerKg?: string
): ApiErrorKind {
  switch (status) {
    case 400: return 'image_invalid'
    case 413: return 'image_too_large'
    case 404: return 'not_found'
    case 409: return currentHighestPerKg ? 'outbid' : 'not_verified'
    case 422:
      // The CV gates refused the photo; the detail text says which gate.
      if (detail.includes('photo rejected: not a fish')) return 'photo_not_fish'
      if (detail.includes('photo rejected: unknown species')) return 'photo_unknown_species'
      return 'unsupported_species'
    case 503: return 'cv_unavailable'
    case 502:
      // Three 502s share the status; detail text is the only discriminator.
      if (detail.includes('failed validation')) return 'generation_invalid'
      if (detail.includes('unsupported species label')) return 'cv_label_unsupported'
      return 'generation_unavailable'
    default: return 'server'
  }
}
