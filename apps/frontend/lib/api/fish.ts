import { apiFetch, type ApiFetchOptions } from './client'

export type VerificationStatus = 'pending' | 'confirmed' | 'corrected'
export type TaxonomyStatus =
  | 'VERIFIED_TAXONOMY' | 'MIXED_TAXONOMY'
  | 'MEDIUM_CONFIDENCE_LABEL_AMBIGUITY' | 'TAXONOMY_REVIEW_REQUIRED'

export interface SpeciesCandidate {
  species_id: string
  normalized_label: string
  confidence: number
}

export interface IdentificationResult {
  prediction_id: string
  model_version: string
  status: 'confident_prediction' | 'low_confidence_human_verification_required'
  prediction: SpeciesCandidate
  top_candidates: SpeciesCandidate[]
  threshold: number
  verification_status: 'pending'
}

export interface SourceMetadata {
  source_id: string
  title: string
  source_type: string
  url: string
  publisher: string
  reviewed_at: string | null
  verification_status: 'verified'
}

export interface KnowledgeCard {
  common_name: string
  scientific_name: string | null
  taxonomy_status: TaxonomyStatus
  physical_characteristics: string | null
  taste: string | null
  texture: string | null
  processing_methods: string[]
  commercial_uses: string[]
  similar_or_substitute_species: string[]
  potential_buyer_segments: string[]
  limitations: string[]
  sources: SourceMetadata[]
}

export interface KnowledgeResponse {
  prediction_id: string
  species_id: string
  card: KnowledgeCard
}

/**
 * The 202 the endpoint returns while the agent graph is still running.
 *
 * It is a success status with no card in it. Typing this path as
 * KnowledgeResponse let a caller read `.card` off it and crash, so the union
 * makes the compiler demand the check.
 */
export interface KnowledgePending {
  detail: string
  job_id: string
  status: 'processing'
}

export type KnowledgeResult = KnowledgeResponse | KnowledgePending

export function hasCard(result: KnowledgeResult): result is KnowledgeResponse {
  return 'card' in result && Boolean(result.card)
}

// Each call takes `init` so a server action can forward the operator's
// session cookie: these routes require an operator, and a server-side fetch
// carries no browser cookies of its own.
export function identifyFish(file: File, signal?: AbortSignal, init: ApiFetchOptions = {}) {
  const body = new FormData()
  // Field name must be `file`: it matches the FastAPI File(...) parameter.
  body.append('file', file)
  // No Content-Type header: setting it by hand omits the multipart boundary.
  return apiFetch<IdentificationResult>('/api/v1/fish/identify', {
    ...init, method: 'POST', body, signal,
  })
}

export function verifySpecies(predictionId: string, verifiedSpeciesId: string, init: ApiFetchOptions = {}) {
  return apiFetch<{
    prediction_id: string
    predicted_species_id: string
    verified_species_id: string
    verification_status: 'confirmed' | 'corrected'
  }>('/api/v1/fish/verify', {
    ...init,
    method: 'POST',
    headers: { ...init.headers, 'content-type': 'application/json' },
    body: JSON.stringify({ prediction_id: predictionId, verified_species_id: verifiedSpeciesId }),
  })
}

export interface ManualEntryResult {
  prediction_id: string
  model_version: string
  verified_species_id: string
  normalized_label: string
  verification_status: 'confirmed' | 'corrected'
}

/** Operator names the species themselves. Used when identification is down. */
export function declareSpeciesManually(file: File, speciesId: string, init: ApiFetchOptions = {}) {
  const body = new FormData()
  body.append('file', file)
  body.append('species_id', speciesId)
  return apiFetch<ManualEntryResult>('/api/v1/fish/manual', { ...init, method: 'POST', body })
}

export function getKnowledge(predictionId: string, init: ApiFetchOptions = {}) {
  // Longer budget: this path runs retrieval plus generation.
  return apiFetch<KnowledgeResult>(
    `/api/v1/predictions/${encodeURIComponent(predictionId)}/knowledge`,
    { ...init, timeoutMs: 70_000 }
  )
}
