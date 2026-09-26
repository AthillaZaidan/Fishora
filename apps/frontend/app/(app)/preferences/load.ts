import 'server-only'

import { cookies } from 'next/headers'
import {
  getPreferences,
  getRecommendations,
  matchedLotIds,
  type BuyerPreferences,
} from '@/lib/api/preferences'

export interface SavedProfile {
  preferences: BuyerPreferences | null
  /** What the saved profile matches now; null when there is no profile or the count failed. */
  counts: { matched: number; inRange: number } | null
  /** The profile could not be read. The form must not open blank and overwrite it. */
  loadFailed: boolean
}

// A Server Component fetch carries no browser cookies, so the session is
// forwarded by hand, as lib/api/server.ts does for its own calls.
async function session() {
  const jar = await cookies()
  const cookie = jar
    .getAll()
    .map((entry) => `${entry.name}=${entry.value}`)
    .join('; ')
  return cookie ? { headers: { cookie } } : {}
}

export async function loadSavedProfile(buyerId: string): Promise<SavedProfile> {
  const init = await session()
  let preferences: BuyerPreferences | null
  try {
    preferences = await getPreferences(buyerId, init)
  } catch {
    return { preferences: null, counts: null, loadFailed: true }
  }
  if (preferences === null) return { preferences, counts: null, loadFailed: false }

  try {
    const recommendations = await getRecommendations(buyerId, init)
    return {
      preferences,
      counts: {
        matched: matchedLotIds(recommendations).length,
        inRange: recommendations.items.length,
      },
      loadFailed: false,
    }
  } catch {
    return { preferences, counts: null, loadFailed: false }
  }
}
