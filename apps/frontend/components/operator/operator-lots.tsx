'use client'

import { useState } from 'react'
import { Button } from '@/components/common/button'
import { Sheet } from '@/components/common/sheet'
import { QrSheet } from '@/components/qr/qr-sheet'
import { BidHistory } from '@/components/lot/bid-history'
import { LotCard } from '@/components/lot/lot-card'
import { Skeleton } from '@/components/common/skeleton'
import { allocateLot, closeLot, listBids, type Bid } from '@/lib/api/commerce'
import { rupiahPerKg } from '@/lib/format'
import type { components } from '@/lib/api/schema'

type Lot = components['schemas']['LotResponse']
type Pending = { lot: Lot; kind: 'allocate' | 'close' }

const STATUS_LABEL: Record<string, string> = {
  draft: 'Draf',
  active: 'Berlangsung',
  closed: 'Ditutup',
  allocated: 'Dialokasikan',
}

export function OperatorLots({ lots }: { lots: Lot[] }) {
  const [items, setItems] = useState(lots)
  const [pending, setPending] = useState<Pending | null>(null)
  const [qr, setQr] = useState<Lot | null>(null)
  const [monitored, setMonitored] = useState<Lot | null>(null)
  const [bids, setBids] = useState<Bid[] | null>(null)
  const closing = pending?.kind === 'close'
  const allocating = pending?.kind === 'allocate'

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

  return (
    <>
      <h1 className="text-h1 text-ink">Lot saya</h1>
      {items.length === 0 ? (
        <p className="text-body-sm mt-4 text-ink-muted">Belum ada lot. Terbitkan tangkapan dari menu Identify.</p>
      ) : (
        // The same card grid buyers browse, so the operator sees each lot the
        // way it is listed, with its controls underneath.
        <ul className="mt-6 grid grid-cols-1 gap-x-4 gap-y-8 sm:grid-cols-2 xl:grid-cols-3">
          {items.map((lot, index) => (
            <li key={lot.id} className="flex flex-col gap-3">
              <LotCard lot={lot} priority={index === 0} />
              <p className="text-body-sm px-1 text-ink-muted">
                Status: <span className="text-ink">{STATUS_LABEL[lot.status] ?? lot.status}</span> · Harga awal{' '}
                {rupiahPerKg(Number(lot.starting_price_per_kg))}
              </p>
              <div className="flex flex-wrap gap-2">
                {lot.status === 'active' && (
                  <Button type="button" size="sm" onClick={() => setPending({ lot, kind: 'close' })}>
                    Tutup lelang
                  </Button>
                )}
                {lot.status === 'closed' && (
                  <Button type="button" size="sm" onClick={() => setPending({ lot, kind: 'allocate' })}>
                    Allocate to winning bidder
                  </Button>
                )}
                <Button type="button" size="sm" variant="secondary" onClick={() => openBids(lot)}>
                  Lihat penawaran
                </Button>
                {/* Available at every status, not only once allocated. The QR
                    points at the public page for the lot, which an operator has
                    reason to show a buyer while the auction is still running. */}
                <Button type="button" size="sm" variant="secondary" onClick={() => setQr(lot)}>
                  Buat QR
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
      <Sheet
        open={Boolean(pending)}
        onClose={() => setPending(null)}
        title={closing ? 'Konfirmasi tutup lelang' : 'Konfirmasi alokasi'}
        footer={
          <Button
            block
            type="button"
            onClick={async () => {
              if (!pending) return
              if (pending.kind === 'close') {
                await closeLot(pending.lot.id)
                setItems((current) =>
                  current.map((item) =>
                    item.id === pending.lot.id ? { ...item, status: 'closed' } : item
                  )
                )
                setPending(null)
                return
              }
              await allocateLot(pending.lot.id)
              setQr(pending.lot)
              setPending(null)
            }}
          >
            {closing ? 'Konfirmasi tutup' : 'Konfirmasi'}
          </Button>
        }
      >
        {closing && pending && (
          <p className="text-body text-ink">
            Tutup lelang ini sekarang? Pembeli tidak bisa mengajukan tawaran baru.
          </p>
        )}
        {allocating && pending && (
          <p className="text-body text-ink">
            Alokasikan ke Dewi Anggraini sebesar {rupiahPerKg(Number(pending.lot.current_highest_per_kg ?? pending.lot.starting_price_per_kg))}?
          </p>
        )}
      </Sheet>
      <Sheet
        open={Boolean(monitored)}
        onClose={() => setMonitored(null)}
        title="Riwayat penawaran"
      >
        {bids === null ? (
          <div className="flex flex-col gap-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : (
          <BidHistory bids={bids} />
        )}
      </Sheet>
      {qr && (
        <QrSheet open onClose={() => setQr(null)} lot={qr} />
      )}
    </>
  )
}
