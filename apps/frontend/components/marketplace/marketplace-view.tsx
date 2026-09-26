'use client'

import Link from 'next/link'
import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { Fish, Funnel, MagnifyingGlass, Sparkle } from '@phosphor-icons/react/dist/ssr'
import { Button } from '@/components/common/button'
import { EmptyState } from '@/components/common/empty-state'
import { LotCard } from '@/components/lot/lot-card'
import { FilterRail } from '@/components/marketplace/filter-rail'
import { FilterSheet } from '@/components/marketplace/filter-sheet'
import { listLots } from '@/lib/api/commerce'
import {
  activeFilterCount,
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
}

// An external store, not state in an effect: the interval, the tab visibility
// and the fetch all live outside React, and useSyncExternalStore is how this
// codebase reads such values.
function createLotPoll(load: (query: string) => Promise<Lot[]>) {
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
      .then((lots) => {
        // A late reply from a filter the buyer already left must not land.
        if (asked !== query) return
        const arrived = lots.filter((lot) => !seen.has(lot.id))
        for (const lot of arrived) seen.add(lot.id)
        fresh += arrived.length
        const same =
          snapshot !== null &&
          snapshot.fresh === fresh &&
          snapshot.lots.length === lots.length &&
          snapshot.lots.every((lot, index) => lot.id === lots[index].id)
        if (same) return
        snapshot = { lots, fresh }
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
      if (snapshot !== null) snapshot = { lots: snapshot.lots, fresh: 0 }
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
  inventoryEmpty,
}: {
  lots: Lot[]
  /** Other open lots the searched fish's knowledge card names as similar. */
  similar?: Lot[]
  similarTo?: string[]
  /** Lots that fit the signed-in buyer's preferences. */
  matchedIds?: string[]
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
  const matched = useMemo(() => new Set(matchedIds), [matchedIds])

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

  const poll = useMemo(() => createLotPoll(listLots), [])
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

  const visible = useMemo(() => {
    return current.filter((lot) => {
      if (filters.minPrice && Number(lot.starting_price_per_kg) < Number(filters.minPrice)) return false
      if (filters.maxPrice && Number(lot.starting_price_per_kg) > Number(filters.maxPrice)) return false
      return true
    })
  }, [current, filters])

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
          Semua lot lelang yang masih aktif.
        </p>
      </header>

      <div className="mt-4 flex gap-8 lg:mt-6">
        <FilterRail filters={filters} onChange={apply} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 border-b border-line pb-3">
            <label className="relative min-w-0 flex-1">
              <span className="sr-only">Cari ikan</span>
              <MagnifyingGlass
                size={18}
                className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-ink-muted"
                aria-hidden
              />
              <input
                type="search"
                value={typed}
                onChange={(event) => setTyped(event.target.value)}
                placeholder="Cari nama ikan, rasa, tekstur, atau olahan"
                className="text-body min-h-11 w-full rounded-full border border-line-input bg-transparent pr-4 pl-10 text-ink"
              />
            </label>
            {fresh > 0 && (
              <button
                type="button"
                onClick={poll.dismiss}
                className="text-body-sm min-h-11 shrink-0 rounded-full border border-line px-3 text-ink"
              >
                {fresh} lot baru
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
                Harga{count ? ` ${count}` : ''}
              </Button>
            </div>
          </div>

          {inventoryEmpty && current.length === 0 ? (
            <EmptyState icon={Fish} message="Belum ada lot aktif." action={<Button type="button">Muat ulang</Button>} />
          ) : visible.length === 0 ? (
            <EmptyState
              icon={searching ? MagnifyingGlass : Funnel}
              message={searching ? `Tidak ada lot untuk "${filters.query}".` : 'Tidak ada lot di rentang harga ini.'}
              action={
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => {
                    setTyped('')
                    apply({ query: '', minPrice: '', maxPrice: '' })
                  }}
                >
                  Hapus pencarian dan filter
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
                Rekomendasi ikan serupa
              </h2>
              <p className="text-body-sm mt-1 max-w-[60ch] text-ink-muted">
                Selain {similarTo.join(', ')}, ikan ini juga sedang dijual. Kemiripannya diambil dari kartu
                pengetahuan {similarTo.length > 1 ? 'ikan-ikan tersebut' : 'ikan tersebut'}.
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