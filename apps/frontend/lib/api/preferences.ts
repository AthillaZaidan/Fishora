import { ApiError, apiFetch, type ApiFetchOptions } from './client'
import type { Lot, MatchReason } from './commerce'

/** The saved buying profile. Decimals arrive as strings, as FastAPI sends them. */
export interface BuyerPreferences {
  buyer_id: string
  business_type: string
  intended_uses: string[]
  characteristics: string[]
  max_price_per_kg: string | null
  min_quantity_kg: string | null
  latitude: number
  longitude: number
}

export type PreferenceDraft = Omit<BuyerPreferences, 'buyer_id'>

export interface PreferencePreview {
  /** Active lots landed within the radius of the draft location. */
  in_range: number
  /** Of those, the ones scoring at or above the match threshold. */
  matched: number
  match_threshold: number
  radius_km: number
  nearest_landing_point: { id: string; name: string; distance_km: number } | null
}

export interface Recommendation {
  lot: Lot
  score: number
  /** Set by the API from its own threshold. Absent from older responses. */
  matched?: boolean
  reasons: MatchReason[]
}

export interface Recommendations {
  items: Recommendation[]
  profile_missing: boolean
  match_threshold?: number
}

/**
 * The "Matched for you" threshold, used only when a response does not carry
 * its own. The API owns the number (services/matching.py MATCH_THRESHOLD).
 */
export const MATCH_THRESHOLD_FALLBACK = 0.6

type RecommendationLike = {
  items: { lot: { id: string }; score: number; matched?: boolean }[]
  match_threshold?: number
}

/** Ids of the lots the API counts as matched, so every screen badges the same set. */
export function matchedLotIds(recommendations: RecommendationLike): string[] {
  const threshold = recommendations.match_threshold ?? MATCH_THRESHOLD_FALLBACK
  return recommendations.items
    .filter((item) => item.matched ?? item.score >= threshold)
    .map((item) => item.lot.id)
}

function path(buyerId: string, suffix: string) {
  return `/api/v1/buyers/${encodeURIComponent(buyerId)}/${suffix}`
}

/**
 * The saved profile, or null when the buyer has not saved one yet.
 * `init` lets a Server Component forward the session cookie.
 */
export async function getPreferences(
  buyerId: string,
  init?: ApiFetchOptions
): Promise<BuyerPreferences | null> {
  try {
    return await apiFetch<BuyerPreferences>(path(buyerId, 'preferences'), init)
  } catch (cause) {
    if (cause instanceof ApiError && cause.status === 404) return null
    throw cause
  }
}

export function savePreferences(buyerId: string, draft: PreferenceDraft) {
  return apiFetch<BuyerPreferences>(path(buyerId, 'preferences'), {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(draft),
  })
}

/** What the draft would match if saved. Stores nothing. */
export function previewPreferences(buyerId: string, draft: PreferenceDraft, signal?: AbortSignal) {
  return apiFetch<PreferencePreview>(path(buyerId, 'preferences/preview'), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(draft),
    signal,
  })
}

export function getRecommendations(buyerId: string, init?: ApiFetchOptions) {
  return apiFetch<Recommendations>(path(buyerId, 'recommendations'), init)
}
