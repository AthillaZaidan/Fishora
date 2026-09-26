'use client'

import { useState } from 'react'
import Link from 'next/link'
import { QrCode } from '@phosphor-icons/react/dist/ssr'
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
import { Countdown } from '@/components/lot/countdown'
import { ApiError } from '@/lib/api/errors'
import { placeBid, type Bid, type Review } from '@/lib/api/commerce'
import { kilograms, rupiahPerKg } from '@/lib/format'
import { Z } from '@/lib/z'
import type { components } from '@/lib/api/schema'
import type { KnowledgeCard } from '@/lib/api/fish'

type Lot = components['schemas']['LotResponse']
type Reason = components['schemas']['MatchReasonResponse']

export function LotDetail({
  lot,
  card,
  reasons,
  reviews,
  bids = [],
  canReview = false,
  viewer = 'guest',
  photoUrl,
}: {
  lot: Lot
  card: KnowledgeCard
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
  const [posted, setPosted] = useState<Review[]>([])
  const [placed, setPlaced] = useState<Bid[]>([])
  const [highest, setHighest] = useState(Number(lot.current_highest_per_kg ?? lot.starting_price_per_kg))
  const [amount, setAmount] = useState(String(highest + 1000))
  const [sheetOpen, setSheetOpen] = useState(false)
  const [qrOpen, setQrOpen] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const closed = lot.status !== 'active'
  const label = lot.species_id.replace('species_', '')

  const submit = async () => {
    const value = Number(amount)
    if (value <= highest) {
      setError('Penawaran harus di atas harga tertinggi saat ini.')
      return
    }
    setBusy(true)
    setError('')
    try {
      const bid = await placeBid(lot.id, amount)
      setHighest(Number(bid.amount_per_kg))
      setPlaced((current) => [bid, ...current])
      setSheetOpen(false)
    } catch (cause) {
      if (cause instanceof ApiError && cause.kind === 'outbid' && cause.currentHighestPerKg) {
        const next = Number(cause.currentHighestPerKg)
        setHighest(next)
        setAmount(String(next + 1000))
        setError(`Harga tertinggi sekarang ${rupiahPerKg(next)}`)
      } else if (cause instanceof ApiError) {
        setError(cause.userMessage)
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div data-page="lot-detail" className="flex flex-col gap-6 px-4 pb-28 lg:pb-8">
      <SpeciesHeader label={label} verified />
      {reasons.length > 0 && <MatchReasons reasons={reasons} />}
      {photoUrl ? (
        // eslint-disable-next-line @next/next/no-img-element -- operator upload, arbitrary origin
        <img src={photoUrl} alt="" className="aspect-[4/3] w-full rounded-2xl object-cover" />
      ) : (
        <SpeciesArt
          label={lot.species_id.replace('species_', '')}
          className="aspect-[4/3] w-full rounded-2xl"
        />
      )}
      <dl className="grid grid-cols-2 gap-3">
        <div>
          <dt className="text-label text-ink-muted">Volume</dt>
          <dd className="text-num-sm tabular-nums text-ink">{kilograms(Number(lot.quantity_kg))}</dd>
        </div>
        <div>
          <dt className="text-label text-ink-muted">Harga awal</dt>
          <dd className="text-num-sm tabular-nums text-ink">{rupiahPerKg(Number(lot.starting_price_per_kg))}</dd>
        </div>
        {lot.batch_size > 1 && (
          <div>
            <dt className="text-label text-ink-muted">Lot</dt>
            <dd className="text-num-sm tabular-nums text-ink">
              {lot.batch_index} dari {lot.batch_size}
            </dd>
          </div>
        )}
      </dl>
      {canReview && (
        // Winning the lot is what grants the card: the restaurant prints it for
        // its diners, and the code opens this fish in Fishora.
        <section className="flex flex-col gap-3 rounded-2xl border border-line px-5 py-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-h3 text-ink">Fishora QR untuk pelanggan Anda</h2>
            <p className="text-body-sm mt-1 max-w-[52ch] text-ink-muted">
              Anda memenangkan lot ini. Cetak kartu ikan dengan kode QR untuk dipajang di restoran; pelanggan
              yang memindainya bisa menjelajahi ikan ini di Fishora.
            </p>
          </div>
          <Button type="button" icon={<QrCode size={18} />} onClick={() => setQrOpen(true)}>
            Buka Fishora QR
          </Button>
        </section>
      )}
      <KnowledgeCardView card={card} label={label} />
      <ReviewsRatings reviews={[...posted, ...reviews]} />
      <section className="flex flex-col gap-3">
        <h2 className="text-h3 text-ink">Riwayat penawaran</h2>
        <BidHistory bids={[...placed, ...bids]} />
      </section>
      {canReview && <ReviewForm lotId={lot.id} onSubmitted={(review) => setPosted((current) => [review, ...current])} />}

      <div
        className="fixed inset-x-0 bottom-0 flex items-center justify-between gap-3 border-t border-line bg-surface px-4 py-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] lg:hidden"
        style={{ zIndex: Z.actionBar }}
      >
        {closed ? (
          <p className="text-body text-ink">Lelang selesai. {rupiahPerKg(highest)}</p>
        ) : (
          <>
            <div>
              <p className="text-num-lg tabular-nums text-ink">{rupiahPerKg(highest)}</p>
              <Countdown endsAt={lot.auction_ends_at} />
            </div>
            {viewer === 'buyer' ? (
              <Button type="button" onClick={() => setSheetOpen(true)}>
                Ajukan penawaran
              </Button>
            ) : viewer === 'guest' ? (
              // Anyone can read a lot; bidding needs a buyer account.
              <Link href={`/account?next=${encodeURIComponent(`/marketplace/${lot.id}`)}`}>
                <Button type="button">Masuk untuk menawar</Button>
              </Link>
            ) : null}
          </>
        )}
      </div>

      <Sheet
        open={sheetOpen}
        onClose={() => setSheetOpen(false)}
        title="Ajukan penawaran"
        footer={
          <Button block type="button" loading={busy} onClick={submit}>
            Kirim
          </Button>
        }
      >
        <Field
          label="Harga per kg"
          prefix="Rp"
          value={amount}
          onChange={(event) => setAmount(event.target.value)}
          error={error || undefined}
        />
      </Sheet>
      {qrOpen && <QrSheet open onClose={() => setQrOpen(false)} lot={lot} card={card.common_name ? card : null} />}
    </div>
  )
}
