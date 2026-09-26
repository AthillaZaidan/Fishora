/**
 * Market testing cut the filters to price alone; the search box covers species
 * and characteristics, and "Matched for you" moved onto the cards themselves.
 */
export interface MarketplaceFilters {
  query: string
  minPrice: string
  maxPrice: string
  /** Only lots marked "Matched for you". Applied in the browser: the match
   *  set comes from the buyer's recommendations, not from the lots query. */
  matchedOnly: boolean
}

export const EMPTY_FILTERS: MarketplaceFilters = {
  query: '',
  minPrice: '',
  maxPrice: '',
  matchedOnly: false,
}

export function parseFilters(search: string): MarketplaceFilters {
  const params = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search)
  return {
    query: params.get('q') ?? '',
    minPrice: params.get('min_price') ?? '',
    maxPrice: params.get('max_price') ?? '',
    matchedOnly: params.get('matched') === '1',
  }
}

export function serializeFilters(filters: MarketplaceFilters): string {
  const params = new URLSearchParams()
  if (filters.query.trim()) params.set('q', filters.query.trim())
  if (filters.minPrice) params.set('min_price', filters.minPrice)
  if (filters.maxPrice) params.set('max_price', filters.maxPrice)
  if (filters.matchedOnly) params.set('matched', '1')
  return params.toString()
}

/** Counts the filter panel only; the search box shows its own state. */
export function activeFilterCount(filters: MarketplaceFilters): number {
  return filters.minPrice || filters.maxPrice ? 1 : 0
}

/** The API query for a set of filters. Shared by the server page and the client
 *  poll: if these drift, polling replaces filtered results with unfiltered. */
export function lotApiQuery(filters: MarketplaceFilters): string {
  const params = new URLSearchParams()
  if (filters.minPrice) params.set('min_price', filters.minPrice)
  if (filters.maxPrice) params.set('max_price', filters.maxPrice)
  params.set('status', 'active')
  return params.toString()
}

/** The search endpoint's query: the text plus the same price range. */
export function searchApiQuery(filters: MarketplaceFilters): string {
  const params = new URLSearchParams()
  params.set('q', filters.query.trim())
  if (filters.minPrice) params.set('min_price', filters.minPrice)
  if (filters.maxPrice) params.set('max_price', filters.maxPrice)
  return params.toString()
}
