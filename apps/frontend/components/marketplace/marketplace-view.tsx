'use client'

import Link from 'next/link'
import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { Fish, Funnel, MagnifyingGlass, Sparkle, Star } from '@phosphor-icons/react/dist/ssr'
import { Button } from '@/components/common/button'
import { EmptyState } from '@/components/common/empty-state'
import { LotCard } from '@/components/lot/lot-card'
import { FilterRail } from '@/components/marketplace/filter-rail'
import { FilterSheet } from '@/components/marketplace/filter-sheet'
import { listLots } from '@/lib/api/commerce'
import { getRecommendations, matchedLotIds } from '@/lib/api/preferences'
import {
  activeFilterCount,
  EMPTY_FILTERS,
  lotApiQuery,
  parseFilters,
  serializeFilters,
  type MarketplaceFilters,
} from '@/lib/marketplace-filters'
import type { components } from '@/lib/api/schema'

type Lot = components['schemas']['LotResponse']

const POLL_MS = 15_000

// Mirrors the server query in the marketplace page. Polling with anything else
// would answer a filtered grid with the whole open market.

interface PollSnapshot {
  lots: Lot[]
  fresh: number
  /** Matched lot ids as of this poll; null until recommendations have been read. */
  matched: string[] | null
}

interface PollResult {
  lots: Lot[]
  matched: string[] | null
}

// An external store, not state in an effect: the interval, the tab visibility
// and the fetch all live outside React, and useSyncExternalStore is how this
// codebase reads such values.
function createLotPoll(load: (query: string) => Promise<PollResult>) {
  const listeners = new Set<() => void>()
  let snapshot: PollSnapshot | null = null
  let seen = new Set<string>()
  let fresh = 0
  let query = ''
  let timer: ReturnType<typeof setInterval> | null = null

  const emit = () => {
    for (const listener of listeners) listener()
  }

  const poll = () => {
    const asked = query
    load(asked)
      .then((result) => {
        // A late reply from a filter the buyer already left must not land.
        if (asked !== query) return
        const { lots } = result
        // A failed recommendations read keeps the last known badges rather
        // than dropping them all for fifteen seconds.
        const matched = result.matched ?? snapshot?.matched ?? null
        const arrived = lots.filter((lot) => !seen.has(lot.id))
        for (const lot of arrived) seen.add(lot.id)
        fresh += arrived.length
        const same =
          snapshot !== null &&
          snapshot.fresh === fresh &&
          snapshot.lots.length === lots.length &&
          snapshot.lots.every((lot, index) => lot.id === lots[index].id) &&
          (snapshot.matched ?? []).join(',') === (matched ?? []).join(',')
        if (same) return
        snapshot = { lots, fresh, matched }
        emit()
      })
      .catch(() => {
        // A dropped poll keeps the last good grid. Blanking it would be worse
        // than being fifteen seconds stale.
      })
  }

  const stopTimer = () => {
    if (timer === null) return
    clearInterval(timer)
    timer = null
  }

  const resume = (immediate: boolean) => {
    if (document.visibilityState === 'hidden') {
      stopTimer()
      return
    }
    if (timer !== null) return
    timer = setInterval(poll, POLL_MS)
    // Coming back to a parked tab, the grid is as stale as the time away.
    if (immediate) poll()
  }

  const onVisible = () => resume(true)

  return {
    read: () => snapshot,
    dismiss: () => {
      if (fresh === 0) return
      fresh = 0
      if (snapshot !== null) snapshot = { ...snapshot, fresh: 0 }
      emit()
    },
    start(nextQuery: string, seedIds: string[], listener: () => void) {
      if (nextQuery !== query) {
        query = nextQuery
        snapshot = null
        fresh = 0
      }
      seen = new Set([...seedIds, ...(snapshot?.lots ?? []).map((lot) => lot.id)])
      listeners.add(listener)
      resume(false)
      document.addEventListener('visibilitychange', onVisible)
      window.addEventListener('focus', onVisible)
      return () => {
        listeners.delete(listener)
        document.removeEventListener('visibilitychange', onVisible)
        window.removeEventListener('focus', onVisible)
        if (listeners.size === 0) stopTimer()
      }
    },
  }
}

const noSnapshot = () => null

const SEARCH_DEBOUNCE_MS = 300

export function MarketplaceView({
  lots,
  similar = [],
  similarTo = [],
  matchedIds = [],
  buyerId = null,
  inventoryEmpty,
}: {
  lots: Lot[]
  /** Other open lots the searched fish's knowledge card names as similar. */
  similar?: Lot[]
  similarTo?: string[]
  /** Lots that fit the signed-in buyer's preferences. */
  matchedIds?: string[]
  /** The signed-in buyer, so polling can refresh their matches with the lots. */
  buyerId?: string | null
  inventoryEmpty: boolean
}) {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const search = searchParams.toString()
  const filters = useMemo(() => parseFilters(search), [search])
  const [sheetOpen, setSheetOpen] = useState(false)
  const [typed, setTyped] = useState(filters.query)
  const count = activeFilterCount(filters)
  const searching = filters.query.trim() !== ''

  const apply = useCallback(
    (next: MarketplaceFilters) => {
      const query = serializeFilters(next)
      router.replace(query ? `${pathname}?${query}` : pathname)
    },
    [router, pathname]
  )

  // Results follow the typing, a beat behind it, so every keystroke does not
  // become a request.
  useEffect(() => {
    if (typed.trim() === filters.query.trim()) return
    const timer = setTimeout(() => apply({ ...filters, query: typed }), SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [typed, filters, apply])

  // Lots and matches are read together on every tick: a lot that arrives by
  // polling is badged only once the recommendations that score it are in, so
  // the badges never lag the grid or disagree with the lot page.
  const poll = useMemo(
    () =>
      createLotPoll(async (query) => {
        const lots = await listLots(query)
        if (!buyerId) return { lots, matched: null }
        try {
          return { lots, matched: matchedLotIds(await getRecommendations(buyerId)) }
        } catch {
          return { lots, matched: null }
        }
      }),
    [buyerId]
  )
  const pollQuery = useMemo(() => lotApiQuery(filters), [filters])
  // A joined string, so a re-render with an equal list does not resubscribe.
  const seed = useMemo(() => lots.map((lot) => lot.id).join(','), [lots])
  const subscribe = useCallback(
    (onChange: () => void) =>
      // Search results come with their similar-fish list from the server; the
      // open-market poll would replace them with unsearched lots.
      searching ? () => {} : poll.start(pollQuery, seed ? seed.split(',') : [], onChange),
    [poll, searching, pollQuery, seed]
  )
  const live = useSyncExternalStore(subscribe, poll.read, noSnapshot)
  const current = live?.lots ?? lots
  const fresh = live?.fresh ?? 0
  const liveMatched = live?.matched ?? null
  const matched = useMemo(() => new Set(liveMatched ?? matchedIds), [liveMatched, matchedIds])

  const visible = useMemo(() => {
    return current.filter((lot) => {
      if (filters.matchedOnly && !matched.has(lot.id)) return false
      if (filters.minPrice && Number(lot.starting_price_per_kg) < Number(filters.minPrice)) return false
      if (filters.maxPrice && Number(lot.starting_price_per_kg) > Number(filters.maxPrice)) return false
      return true
    })
  }, [current, filters, matched])

  const grid = (items: Lot[]) => (
    <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {items.map((lot, index) => (
        <Link key={lot.id} href={`/marketplace/${lot.id}`}>
          <LotCard lot={lot} matched={matched.has(lot.id)} priority={index === 0} />
        </Link>
      ))}
    </div>
  )

  return (
    <div className="pb-24 lg:pb-8">
      {/* The heading spans both columns. Inside the grid column it would have
          pushed the cards down while the filter rail beside them started at
          the top of the page. */}
      <header>
        <h1 className="text-h1 text-ink">All lots</h1>
        <p className="text-body-sm mt-1 max-w-[52ch] text-ink-muted">
          Every auction lot that is still open.{buyerId ? ' Lots that fit your buying profile are marked “Matched for you”.' : ''}
        </p>
      </header>

      <div className="mt-4 flex gap-8 lg:mt-6">
        <FilterRail filters={filters} onChange={apply} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 border-b border-line pb-3">
            <label className="relative min-w-0 flex-1">
              <span className="sr-only">Search fish</span>
              <MagnifyingGlass
                size={18}
                className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-ink-muted"
                aria-hidden
              />
              <input
                type="search"
                value={typed}
                onChange={(event) => setTyped(event.target.value)}
                placeholder="Search by fish name, taste, texture or use"
                className="text-body min-h-11 w-full rounded-full border border-line-input bg-transparent pr-4 pl-10 text-ink"
              />
            </label>
            {buyerId && (
              <button
                type="button"
                aria-pressed={filters.matchedOnly}
                onClick={() => apply({ ...filters, matchedOnly: !filters.matchedOnly })}
                className={`text-body-sm flex min-h-11 shrink-0 items-center gap-1.5 rounded-full border px-3 ${
                  filters.matchedOnly ? 'border-ink bg-bg-sunken text-ink' : 'border-line text-ink-muted hover:text-ink'
                }`}
              >
                <Star size={14} weight="fill" className="text-accent" aria-hidden />
                Matched for you
              </button>
            )}
            {fresh > 0 && (
              <button
                type="button"
                onClick={poll.dismiss}
                className="text-body-sm min-h-11 shrink-0 rounded-full border border-line px-3 text-ink"
              >
                {fresh} new {fresh === 1 ? 'lot' : 'lots'}
              </button>
            )}
            <div className="shrink-0 lg:hidden">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                icon={<Funnel size={16} />}
                onClick={() => setSheetOpen(true)}
              >
                Price{count ? ` ${count}` : ''}
              </Button>
            </div>
          </div>

          {inventoryEmpty && current.length === 0 ? (
            <EmptyState
              icon={Fish}
              message="No open lots yet."
              action={
                <Button type="button" onClick={() => router.refresh()}>
                  Reload
                </Button>
              }
            />
          ) : visible.length === 0 && filters.matchedOnly && !searching ? (
            <EmptyState
              icon={Star}
              message={
                buyerId
                  ? 'No open lots match your buying profile right now.'
                  : 'Sign in as a buyer and save a buying profile to see lots matched for you.'
              }
              action={
                <div className="flex flex-wrap justify-center gap-2">
                  <Link href={buyerId ? '/preferences' : '/account?next=%2Fpreferences'}>
                    <Button type="button">{buyerId ? 'Edit buying profile' : 'Sign in'}</Button>
                  </Link>
                  <Button type="button" variant="secondary" onClick={() => apply({ ...filters, matchedOnly: false })}>
                    Show all lots
                  </Button>
                </div>
              }
            />
          ) : visible.length === 0 ? (
            <EmptyState
              icon={searching ? MagnifyingGlass : Funnel}
              message={searching ? `No lots for "${filters.query}".` : 'No lots in this price range.'}
              action={
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => {
                    setTyped('')
                    apply(EMPTY_FILTERS)
                  }}
                >
                  Clear search and filters
                </Button>
              }
            />
          ) : (
            grid(visible)
          )}

          {similar.length > 0 && (
            <section className="mt-10" aria-labelledby="similar-heading">
              <h2 id="similar-heading" className="text-h3 flex items-center gap-2 text-ink">
                <Sparkle size={18} weight="fill" className="text-accent" aria-hidden />
                Similar fish on sale
              </h2>
              <p className="text-body-sm mt-1 max-w-[60ch] text-ink-muted">
                Besides {similarTo.join(', ')}, these fish are on sale too. The similarity comes from the
                knowledge {similarTo.length > 1 ? 'cards of those fish' : 'card of that fish'}.
              </p>
              {grid(similar)}
            </section>
          )}
        </div>
      </div>
      <FilterSheet
        open={sheetOpen}
        onClose={() => setSheetOpen(false)}
        filters={filters}
        onChange={apply}
        resultCount={visible.length}
      />
    </div>
  )
}