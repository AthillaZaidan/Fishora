'use client'

import { useState } from 'react'
import { Button } from '@/components/common/button'
import { Sheet } from '@/components/common/sheet'
import { QrSheet } from '@/components/qr/qr-sheet'
import { BidHistory } from '@/components/lot/bid-history'
import { LotCard } from '@/components/lot/lot-card'
import { Skeleton } from '@/components/common/skeleton'
import { ApiError } from '@/lib/api/errors'
import { allocateLot, closeLot, listBids, type Bid, type Lot } from '@/lib/api/commerce'
import { rupiahPerKg } from '@/lib/format'

type Pending = { lot: Lot; kind: 'allocate' | 'close' }

const STATUS_LABEL: Record<string, string> = {
  draft: 'Draft',
  active: 'Live',
  closed: 'Closed',
  allocated: 'Allocated',
}

function failure(cause: unknown, kind: Pending['kind']): string {
  if (cause instanceof ApiError && cause.status === 409) {
    return kind === 'allocate'
      ? 'This lot cannot be allocated: it is still live, has no bids, or was already allocated.'
      : 'This lot is already allocated and cannot be closed.'
  }
  if (cause instanceof ApiError && cause.status === 403) return 'Only the fisher who listed this lot can do that.'
  if (cause instanceof ApiError && cause.status === 401) return 'Your session has ended. Sign in again.'
  if (cause instanceof ApiError && (cause.kind === 'offline' || cause.kind === 'timeout')) {
    return 'No connection to Fishora. Try again.'
  }
  return 'Something went wrong. Try again.'
}

export function OperatorLots({ lots }: { lots: Lot[] }) {
  const [items, setItems] = useState(lots)
  const [pending, setPending] = useState<Pending | null>(null)
  // The bids behind the allocation dialog: the winner shown is the API's, not a guess.
  const [pendingBids, setPendingBids] = useState<Bid[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [actionError, setActionError] = useState('')
  const [outcome, setOutcome] = useState<Record<string, string>>({})
  const [qr, setQr] = useState<Lot | null>(null)
  const [monitored, setMonitored] = useState<Lot | null>(null)
  const [bids, setBids] = useState<Bid[] | null>(null)
  const closing = pending?.kind === 'close'
  const allocating = pending?.kind === 'allocate'

  const update = (id: string, patch: Partial<Lot>) =>
    setItems((current) => current.map((item) => (item.id === id ? { ...item, ...patch } : item)))

  // On demand, one lot at a time: this page lists many lots and most of their
  // histories are never opened.
  const openBids = async (lot: Lot) => {
    setMonitored(lot)
    setBids(null)
    try {
      setBids(await listBids(lot.id))
    } catch {
      setBids([])
    }
  }

  const ask = async (lot: Lot, kind: Pending['kind']) => {
    setPending({ lot, kind })
    setActionError('')
    setPendingBids(null)
    if (kind !== 'allocate') return
    try {
      setPendingBids(await listBids(lot.id))
    } catch {
      setPendingBids([])
    }
  }

  // Highest amount, earliest on a tie: the rule the API allocates by.
  const leading = pendingBids?.length
    ? [...pendingBids].sort(
        (a, b) => Number(b.amount_per_kg) - Number(a.amount_per_kg) || a.created_at.localeCompare(b.created_at)
      )[0]
    : null

  const confirm = async () => {
    if (!pending) return
    setBusy(true)
    setActionError('')
    try {
      if (pending.kind === 'close') {
        const closed = await closeLot(pending.lot.id)
        update(pending.lot.id, { status: closed.status })
        setPending(null)
        return
      }
      const result = await allocateLot(pending.lot.id)
      update(pending.lot.id, {
        status: 'allocated',
        allocated_buyer_id: result.allocated_buyer_id,
        current_highest_per_kg: result.current_highest_per_kg ?? pending.lot.current_highest_per_kg,
      })
      const amount = result.winning_amount_per_kg ?? result.current_highest_per_kg
      setOutcome((current) => ({
        ...current,
        [pending.lot.id]: `Allocated to ${result.winner_label ?? 'the highest bidder'}${
          amount ? ` at ${rupiahPerKg(Number(amount))}` : ''
        }.`,
      }))
      setPending(null)
    } catch (cause) {
      setActionError(failure(cause, pending.kind))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <h1 className="text-h1 text-ink">My lots</h1>
      {items.length === 0 ? (
        <p className="text-body-sm mt-4 text-ink-muted">No lots yet. Publish a catch from Identify.</p>
      ) : (
        // The same card grid buyers browse, so the operator sees each lot the
        // way it is listed, with its controls underneath.
        <ul className="mt-6 grid grid-cols-1 gap-x-4 gap-y-8 sm:grid-cols-2 xl:grid-cols-3">
          {items.map((lot, index) => (
            <li key={lot.id} className="flex flex-col gap-3">
              <LotCard lot={lot} priority={index === 0} />
              <p className="text-body-sm px-1 text-ink-muted">
                Status: <span className="text-ink">{STATUS_LABEL[lot.status] ?? lot.status}</span> · Starting price{' '}
                {rupiahPerKg(Number(lot.starting_price_per_kg))}
              </p>
              {outcome[lot.id] && <p className="text-body-sm px-1 text-ink">{outcome[lot.id]}</p>}
              <div className="flex flex-wrap gap-2">
                {lot.status === 'active' && (
                  <Button type="button" size="sm" onClick={() => ask(lot, 'close')}>
                    Close auction
                  </Button>
                )}
                {lot.status === 'closed' && (
                  <Button type="button" size="sm" onClick={() => ask(lot, 'allocate')}>
                    Allocate to winning bidder
                  </Button>
                )}
                <Button type="button" size="sm" variant="secondary" onClick={() => openBids(lot)}>
                  View bids
                </Button>
                {/* Available at every status, not only once allocated. The QR
                    points at the public page for the lot, which an operator has
                    reason to show a buyer while the auction is still running. */}
                <Button type="button" size="sm" variant="secondary" onClick={() => setQr(lot)}>
                  Make QR
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
      <Sheet
        open={Boolean(pending)}
        onClose={() => {
          if (!busy) setPending(null)
        }}
        title={closing ? 'Close this auction?' : 'Confirm allocation'}
        footer={
          <Button
            block
            type="button"
            loading={busy}
            disabled={allocating && (pendingBids === null || pendingBids.length === 0)}
            onClick={confirm}
          >
            {closing ? 'Close auction' : 'Allocate'}
          </Button>
        }
      >
        {closing && pending && (
          <p className="text-body text-ink">Close this auction now? Buyers will not be able to place new bids.</p>
        )}
        {allocating && pending && (
          pendingBids === null ? (
            <Skeleton className="h-10 w-full" />
          ) : leading ? (
            <p className="text-body text-ink">
              Allocate this lot to {leading.bidder} for {rupiahPerKg(Number(leading.amount_per_kg))}, the
              highest bid?
            </p>
          ) : (
            <p className="text-body text-ink">This lot has no bids, so there is no one to allocate it to.</p>
          )
        )}
        {actionError && <p className="text-body-sm mt-3 text-state-error">{actionError}</p>}
      </Sheet>
      <Sheet open={Boolean(monitored)} onClose={() => setMonitored(null)} title="Bid history">
        {bids === null ? (
          <div className="flex flex-col gap-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : (
          <BidHistory bids={bids} />
        )}
      </Sheet>
      {qr && <QrSheet open onClose={() => setQr(null)} lot={qr} />}
    </>
  )
}
