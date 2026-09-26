import { MarketplaceView } from '@/components/marketplace/marketplace-view'
import { listLots, searchLots, type Lot, type SearchResult } from '@/lib/api/commerce'
import { getMeAsServer, getRecommendationsAsServer } from '@/lib/api/server'
import { lotApiQuery, parseFilters, searchApiQuery } from '@/lib/marketplace-filters'

// A lot is badged "Matched for you" from this score up. The weights in
// services/matching.py sum to 1, so 0.6 means most of what the buyer asked for
// (use, characteristics, price, volume, distance) holds, not just the distance.
const MATCH_BADGE_MIN_SCORE = 0.6

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
  try {
    const me = await getMeAsServer()
    if (me.role === 'buyer') {
      const { items } = await getRecommendationsAsServer(me.id)
      matchedIds = items.filter((item) => item.score >= MATCH_BADGE_MIN_SCORE).map((item) => item.lot.id)
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
      inventoryEmpty={lots.length === 0 && !filters.query && !filters.minPrice && !filters.maxPrice}
    />
  )
}
