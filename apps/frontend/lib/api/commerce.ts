import { apiFetch } from './client'
import type { components } from './schema'
import type { KnowledgeCard } from './fish'

export type Lot = components['schemas']['LotResponse']
export type Bid = components['schemas']['BidResponse']
export type MatchReason = components['schemas']['MatchReasonResponse']
// The generated schema types taxonomy_status as a plain string because the
// FastAPI model does. Components switch on the narrowed union, so the narrowing
// happens once here rather than at every call site.
export type DiscoverResponse = Omit<
  components['schemas']['DiscoverResponse'],
  'card'
> & { card: KnowledgeCard }
export type PreferenceRequest = components['schemas']['PreferenceRequest']

export type Review = components['schemas']['ReviewResponse']
export type ReviewPayload = components['schemas']['ReviewRequest']

export type SearchResult = components['schemas']['SearchResponse']

export function listLots(query = '') {
  return apiFetch<Lot[]>(`/api/v1/lots${query ? `?${query}` : ''}`)
}

/** Open lots by name or card characteristics, plus the similar fish the match's card names. */
export function searchLots(query: string) {
  return apiFetch<SearchResult>(`/api/v1/lots/search?${query}`)
}

export function getLot(id: string) {
  return apiFetch<Lot>(`/api/v1/lots/${encodeURIComponent(id)}`)
}

export function listBids(lotId: string) {
  return apiFetch<Bid[]>(`/api/v1/lots/${encodeURIComponent(lotId)}/bids`)
}

export function placeBid(lotId: string, amountPerKg: string) {
  return apiFetch<Bid>(`/api/v1/lots/${encodeURIComponent(lotId)}/bids`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ amount_per_kg: amountPerKg }),
  })
}

export function allocateLot(lotId: string) {
  return apiFetch<{ id: string; status: 'allocated'; allocated_buyer_id: string }>(
    `/api/v1/lots/${encodeURIComponent(lotId)}/allocate`,
    { method: 'POST' }
  )
}

export function submitReview(lotId: string, payload: ReviewPayload) {
  return apiFetch<Review>(`/api/v1/lots/${encodeURIComponent(lotId)}/review`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

/** Public, and keyed on species: other buyers' experience of the same fish. */
export function listReviews(lotId: string) {
  return apiFetch<Review[]>(`/api/v1/lots/${encodeURIComponent(lotId)}/reviews`)
}

export function savePreferences(buyerId: string, payload: PreferenceRequest) {
  return apiFetch(`/api/v1/buyers/${encodeURIComponent(buyerId)}/preferences`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function getRecommendations(buyerId: string) {
  return apiFetch<{
    items: { lot: Lot; score: number; reasons: MatchReason[] }[]
    profile_missing: boolean
  }>(`/api/v1/buyers/${encodeURIComponent(buyerId)}/recommendations`)
}

export function getDiscover(slug: string) {
  return apiFetch<DiscoverResponse>(`/api/v1/discover/${encodeURIComponent(slug)}`)
}

export function login(username: string, password: string) {
  return apiFetch<{ id: string; role: string; name: string; username: string }>(
    '/api/v1/auth/login',
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ username, password }),
    }
  )
}

export function logout() {
  return apiFetch<{ ok: boolean }>('/api/v1/auth/logout', { method: 'POST' })
}

export function getMe() {
  return apiFetch<{ id: string; role: string; name: string; username: string }>('/api/v1/auth/me')
}

/** One catch as `lot_count` lots of `quantity_kg` each; every lot is its own auction. */
export function publishLot(payload: {
  prediction_id: string
  quantity_kg: string
  lot_count: number
  starting_price_per_kg: string
  size_category: 'S' | 'M' | 'L'
  landing_point_id: string
  auction_minutes?: number
}) {
  return apiFetch<Lot[]>('/api/v1/lots', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function closeLot(lotId: string) {
  return apiFetch<Lot>(`/api/v1/lots/${encodeURIComponent(lotId)}/close`, { method: 'POST' })
}
