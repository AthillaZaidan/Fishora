import { MarketplaceView } from '@/components/marketplace/marketplace-view'
import { listLots, searchLots, type Lot, type SearchResult } from '@/lib/api/commerce'
import { matchedLotIds } from '@/lib/api/preferences'
import { getMeAsServer, getRecommendationsAsServer } from '@/lib/api/server'
import { lotApiQuery, parseFilters, searchApiQuery } from '@/lib/marketplace-filters'

// A lot is badged "Matched for you" when the API marks it matched: a score of
// at least MATCH_THRESHOLD (0.6) in services/matching.py, whose weights over
// use, characteristics, business type, price, volume and distance sum to 1.
// Price, volume and distance alone reach 0.45, so a badge always means the
// fish itself fits. The threshold lives in the API so the lot page, this grid
// and the preferences count cannot disagree; matchedLotIds falls back to 0.6
// only for a response that does not carry it.

export default async function MarketplacePage({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>
}) {
  const params = await searchParams
  // Same builders the client uses, so a refresh cannot change the result set.
  const filters = parseFilters(new URLSearchParams(
    Object.entries(params).flatMap(([key, value]) =>
      typeof value === 'string' ? [[key, value] as [string, string]] : []
    )
  ).toString())

  // Fetch first, render after: JSX built inside a try/catch is not protected by it.
  let lots: Lot[] = []
  let search: SearchResult | null = null
  try {
    if (filters.query.trim()) {
      search = await searchLots(searchApiQuery(filters))
      lots = search.matches
    } else {
      lots = await listLots(lotApiQuery(filters))
    }
  } catch {
    lots = []
  }

  // Only a signed-in buyer with a preference profile has matches. Anyone else
  // sees the plain grid rather than an empty promise.
  let matchedIds: string[] = []
  let buyerId: string | null = null
  try {
    const me = await getMeAsServer()
    if (me.role === 'buyer') {
      buyerId = me.id
      matchedIds = matchedLotIds(await getRecommendationsAsServer(me.id))
    }
  } catch {
    // Anonymous or no profile: no badges.
  }

  return (
    <MarketplaceView
      lots={lots}
      similar={search?.similar ?? []}
      similarTo={search?.similar_to ?? []}
      matchedIds={matchedIds}
      buyerId={buyerId}
      inventoryEmpty={lots.length === 0 && !filters.query && !filters.minPrice && !filters.maxPrice}
    />
  )
}
