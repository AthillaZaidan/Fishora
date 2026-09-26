'use client'

import { useCallback, useEffect, useState } from 'react'
import Link from 'next/link'
import { QrCode, Trophy } from '@phosphor-icons/react/dist/ssr'
import { KnowledgeCardView } from '@/components/fish/knowledge-card'
import { ReviewsRatings } from '@/components/fish/reviews-ratings'
import { SpeciesArt } from '@/components/fish/species-art'
import { SpeciesHeader } from '@/components/fish/species-header'
import { Button } from '@/components/common/button'
import { Field } from '@/components/common/field'
import { Sheet } from '@/components/common/sheet'
import { ReviewForm } from '@/components/buyer/review-form'
import { QrSheet } from '@/components/qr/qr-sheet'
import { MatchReasons } from '@/components/lot/match-reasons'
import { BidHistory } from '@/components/lot/bid-history'
import { Countdown, useNow } from '@/components/lot/countdown'
import { ApiError } from '@/lib/api/errors'
import {
  getLot,
  listBids,
  lotPhotoUrl,
  placeBid,
  type Bid,
  type Lot,
  type Review,
} from '@/lib/api/commerce'
import { kilograms, rupiah, rupiahPerKg } from '@/lib/format'
import { Z } from '@/lib/z'
import type { components } from '@/lib/api/schema'
import type { KnowledgeCard } from '@/lib/api/fish'

type Reason = components['schemas']['MatchReasonResponse']

/** Rupiah per kg a new bid must add to the highest one. The API only requires
 *  "more than the highest"; this is the step the form suggests and enforces. */
export const BID_INCREMENT = 1000
/** How often a live lot re-reads its price and bids. */
const POLL_MS = 5000

export function LotDetail({
  lot: initialLot,
  card,
  reasons,
  reviews,
  bids = [],
  canReview = false,
  viewer = 'guest',
  photoUrl,
}: {
  lot: Lot
  /** Null when the lot has no readable card; shown as missing, never as verified. */
  card: KnowledgeCard | null
  reasons: Reason[]
  reviews: Review[]
  /** Newest last or first: BidHistory orders them. */
  bids?: Bid[]
  /** Only the buyer holding the allocation may write one, or print its QR card. */
  canReview?: boolean
  /** Who is looking: only a buyer can bid; a guest is asked to sign in first. */
  viewer?: 'buyer' | 'operator' | 'guest'
  /** A real photograph when one exists. Falls back to the species composition. */
  photoUrl?: string
}) {
  const [lot, setLot] = useState(initialLot)
  const [history, setHistory] = useState(bids)
  const [posted, setPosted] = useState<Review[]>([])
  const [amount, setAmount] = useState('')
  const [sheetOpen, setSheetOpen] = useState(false)
  const [qrOpen, setQrOpen] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const now = useNow()

  const label = lot.species_id.replace('species_', '')
  const starting = Number(lot.starting_price_per_kg)
  const highest = lot.current_highest_per_kg == null ? null : Number(lot.current_highest_per_kg)
  // The first bid may equal the starting price; every later one must beat the
  // highest, by the suggested step.
  const minBid = highest === null ? starting : highest + BID_INCREMENT
  // The clock closes bidding the moment the countdown reaches zero, before the
  // next poll brings the server's "closed". Null on the server render.
  const timeUp = now !== null && now >= new Date(lot.auction_ends_at).getTime()
  const live = lot.status === 'active' && !timeUp
  const won = lot.won_by_you === true || canReview

  const refresh = useCallback(async (): Promise<Lot | null> => {
    try {
      const [nextLot, nextBids] = await Promise.all([getLot(initialLot.id), listBids(initialLot.id)])
      setLot(nextLot)
      setHistory(nextBids)
      return nextLot
    } catch {
      // A missed poll keeps the last known state; the next one tries again.
      return null
    }
  }, [initialLot.id])

  // Poll while the server still says active. Once the end passes, the next
  // read comes back closed (the API closes on read) and polling stops.
  useEffect(() => {
    if (lot.status !== 'active') return
    const id = setInterval(() => void refresh(), POLL_MS)
    return () => clearInterval(id)
  }, [lot.status, refresh])

  const openBid = () => {
    setAmount(String(minBid))
    setError('')
    setSheetOpen(true)
  }

  const submit = async () => {
    const value = Number(amount)
    if (!amount.trim() || !Number.isFinite(value) || value <= 0) {
      setError('Enter a price per kg.')
      return
    }
    if (value < minBid) {
      setError(`The minimum bid is ${rupiahPerKg(minBid)}.`)
      return
    }
    setBusy(true)
    setError('')
    try {
      const bid = await placeBid(lot.id, String(value))
      setHistory((current) => [bid, ...current.filter((item) => item.id !== bid.id)])
      setLot((current) => ({ ...current, current_highest_per_kg: bid.amount_per_kg }))
      setSheetOpen(false)
      void refresh()
    } catch (cause) {
      if (cause instanceof ApiError && cause.kind === 'outbid') {
        // Someone bid first (or the amount was under the starting price).
        // Re-read, then offer the new minimum.
        const next = await refresh()
        const nextHighest = next?.current_highest_per_kg == null ? null : Number(next.current_highest_per_kg)
        const nextMin = nextHighest === null ? starting : nextHighest + BID_INCREMENT
        setAmount(String(nextMin))
        setError(
          nextHighest === null
            ? `The first bid must be at least ${rupiahPerKg(starting)}.`
            : `Outbid: the highest bid is now ${rupiahPerKg(nextHighest)}. The minimum is ${rupiahPerKg(nextMin)}.`
        )
      } else if (cause instanceof ApiError && cause.status === 409) {
        void refresh()
        setError('This auction has ended. No more bids are accepted.')
      } else if (cause instanceof ApiError && cause.status === 401) {
        setError('Your session has ended. Sign in again as a buyer to bid.')
      } else if (cause instanceof ApiError && cause.status === 403) {
        setError('Only buyer accounts can bid.')
      } else if (cause instanceof ApiError && cause.status === 422) {
        setError('Enter a valid price per kg.')
      } else if (cause instanceof ApiError && (cause.kind === 'offline' || cause.kind === 'timeout')) {
        setError('The bid did not reach Fishora. Check your connection and try again.')
      } else {
        setError('Could not place the bid. Try again.')
      }
    } finally {
      setBusy(false)
    }
  }

  const finalPrice = highest === null ? null : rupiahPerKg(highest)
  const actions = live ? (
    <>
      <div className="min-w-0">
        <p className="text-label text-ink-muted">{highest === null ? 'Starting price' : 'Highest bid'}</p>
        <p className="text-num-lg tabular-nums text-ink">{rupiahPerKg(highest ?? starting)}</p>
        <p className="text-body-sm text-ink-muted">
          <Countdown endsAt={lot.auction_ends_at} /> left · next bid from{' '}
          <span className="tabular-nums">{rupiahPerKg(minBid)}</span>
        </p>
      </div>
      {viewer === 'buyer' ? (
        <Button type="button" onClick={openBid}>
          Place a bid
        </Button>
      ) : viewer === 'guest' ? (
        // Anyone can read a lot; bidding needs a buyer account.
        <Link href={`/account?next=${encodeURIComponent(`/marketplace/${lot.id}`)}`}>
          <Button type="button">Sign in to bid</Button>
        </Link>
      ) : null}
    </>
  ) : lot.status === 'allocated' && won ? (
    <div className="flex items-center gap-3">
      <Trophy size={24} weight="fill" className="shrink-0 text-accent" aria-hidden />
      <div>
        <p className="text-h3 text-ink">Won by you</p>
        {finalPrice && <p className="text-body-sm text-ink-muted">Your winning bid: {finalPrice}</p>}
      </div>
    </div>
  ) : lot.status === 'allocated' ? (
    <div>
      <p className="text-h3 text-ink">Allocated</p>
      <p className="text-body-sm text-ink-muted">
        This lot has been sold{finalPrice ? ` at ${finalPrice}` : ''}.
      </p>
    </div>
  ) : (
    <div>
      <p className="text-h3 text-ink">Auction ended</p>
      <p className="text-body-sm text-ink-muted">
        {finalPrice
          ? `Highest bid ${finalPrice}. The fisher is choosing the winner.`
          : 'No bids were placed.'}
      </p>
    </div>
  )

  return (
    <div data-page="lot-detail" className="mx-auto flex w-full max-w-[720px] flex-col gap-6 px-4 pb-28 lg:pb-8">
      <SpeciesHeader label={label} verified />
      {reasons.length > 0 && <MatchReasons reasons={reasons} />}
      {(photoUrl ?? lotPhotoUrl(lot)) ? (
        // eslint-disable-next-line @next/next/no-img-element -- served by the API, another origin
        <img src={photoUrl ?? lotPhotoUrl(lot)} alt="Catch photo" className="aspect-[4/3] w-full rounded-2xl object-cover" />
      ) : (
        <SpeciesArt label={label} className="aspect-[4/3] w-full rounded-2xl" />
      )}
      <dl className="grid grid-cols-2 gap-3">
        <div>
          <dt className="text-label text-ink-muted">Volume</dt>
          <dd className="text-num-sm tabular-nums text-ink">{kilograms(Number(lot.quantity_kg))}</dd>
        </div>
        <div>
          <dt className="text-label text-ink-muted">Starting price</dt>
          <dd className="text-num-sm tabular-nums text-ink">{rupiahPerKg(starting)}</dd>
        </div>
        {lot.batch_size > 1 && (
          <div>
            <dt className="text-label text-ink-muted">Lot</dt>
            <dd className="text-num-sm tabular-nums text-ink">
              {lot.batch_index} of {lot.batch_size}
            </dd>
          </div>
        )}
      </dl>
      <div className="hidden items-center justify-between gap-3 rounded-2xl border border-line px-5 py-4 lg:flex">
        {actions}
      </div>
      {won && (
        // Winning the lot is what grants the card: the restaurant prints it for
        // its diners, and the code opens this fish in Fishora.
        <section className="flex flex-col gap-3 rounded-2xl border border-line px-5 py-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-h3 text-ink">Fishora QR for your customers</h2>
            <p className="text-body-sm mt-1 max-w-[52ch] text-ink-muted">
              You won this lot. Print a fish card with a QR code to display in your restaurant; customers who
              scan it can explore this fish in Fishora.
            </p>
          </div>
          <Button type="button" icon={<QrCode size={18} />} onClick={() => setQrOpen(true)}>
            Open Fishora QR
          </Button>
        </section>
      )}
      {card ? (
        <KnowledgeCardView card={card} label={label} />
      ) : (
        <section className="rounded-2xl bg-surface px-5 py-5">
          <h2 className="text-body-sm text-ink-muted">Knowledge not yet available</h2>
          <p className="text-body-sm mt-1 text-ink-muted">
            The knowledge card for this lot could not be loaded. The lot itself is unaffected.
          </p>
        </section>
      )}
      <ReviewsRatings reviews={[...posted, ...reviews]} />
      <section className="flex flex-col gap-3">
        <div className="flex items-baseline justify-between gap-3">
          <h2 className="text-h3 text-ink">Bid history</h2>
          {live && <p className="text-body-sm text-ink-muted">Updates every few seconds</p>}
        </div>
        <BidHistory bids={history} />
      </section>
      {won && <ReviewForm lotId={lot.id} onSubmitted={(review) => setPosted((current) => [review, ...current])} />}

      {/* The same actions twice: a bar pinned to the bottom on a phone, and a
          panel under the lot facts on a wide screen, where the bar was hidden
          and left no way to bid or to sign in to bid. */}
      <div
        className="fixed inset-x-0 bottom-0 flex items-center justify-between gap-3 border-t border-line bg-surface px-4 py-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] lg:hidden"
        style={{ zIndex: Z.actionBar }}
      >
        {actions}
      </div>

      <Sheet
        // Closes itself when the auction ends under it.
        open={sheetOpen && live}
        onClose={() => setSheetOpen(false)}
        title="Place a bid"
        footer={
          <Button block type="button" loading={busy} onClick={submit}>
            Send bid
          </Button>
        }
      >
        <Field
          label="Price per kg"
          prefix="Rp"
          inputMode="numeric"
          value={amount}
          onChange={(event) => setAmount(event.target.value.replace(/[^\d]/g, ''))}
          helper={
            highest === null
              ? `No bids yet. The first bid must be at least ${rupiah(starting)}.`
              : `Minimum ${rupiah(minBid)}: the highest bid plus ${rupiah(BID_INCREMENT)}.`
          }
          error={error || undefined}
        />
      </Sheet>
      {qrOpen && (
        <QrSheet open onClose={() => setQrOpen(false)} lot={lot} card={card && card.common_name ? card : null} />
      )}
    </div>
  )
}
