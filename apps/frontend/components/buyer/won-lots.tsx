'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { QrCode } from '@phosphor-icons/react/dist/ssr'
import { Button } from '@/components/common/button'
import { Skeleton } from '@/components/common/skeleton'
import { LotCard } from '@/components/lot/lot-card'
import { QrSheet } from '@/components/qr/qr-sheet'
import { listWonLots, type Lot } from '@/lib/api/commerce'

/**
 * The lots this buyer won. Each one unlocks a printable Fishora QR card.
 *
 * Read from the buyer-scoped endpoint (GET /api/v1/lots?won=1) in the browser,
 * where the session cookie goes with the request. The API no longer tells
 * anyone else who won a lot, so filtering a public list cannot find them.
 * `lots` is kept as optional initial data for callers that still pass it.
 */
export function WonLots({ lots: initial }: { lots?: Lot[] }) {
  const [lots, setLots] = useState<Lot[] | null>(initial && initial.length > 0 ? initial : null)
  const [failed, setFailed] = useState(false)
  const [qr, setQr] = useState<Lot | null>(null)

  useEffect(() => {
    let cancelled = false
    listWonLots()
      .then((data) => {
        if (!cancelled) setLots(data)
      })
      .catch(() => {
        if (!cancelled) setFailed(true)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <section className="mt-10" aria-labelledby="won-lots-heading">
      <h2 id="won-lots-heading" className="text-h2 text-ink">
        Lots you won
      </h2>
      <p className="text-body-sm mt-1 max-w-[56ch] text-ink-muted">
        Every lot you win comes with a Fishora QR: a fish card to print and display for your customers.
      </p>
      {lots === null ? (
        failed ? (
          <p className="text-body-sm mt-4 text-state-error">Could not load your won lots. Reload to try again.</p>
        ) : (
          <div className="mt-4 grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3">
            <Skeleton className="aspect-[4/3] w-full" />
          </div>
        )
      ) : lots.length === 0 ? (
        <p className="text-body-sm mt-4 text-ink-muted">No lots have been allocated to you yet.</p>
      ) : (
        <ul className="mt-4 grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3">
          {lots.map((lot) => (
            <li key={lot.id} className="flex flex-col gap-3">
              <Link href={`/marketplace/${lot.id}`}>
                <LotCard lot={lot} />
              </Link>
              <Button type="button" variant="secondary" icon={<QrCode size={18} />} onClick={() => setQr(lot)}>
                Fishora QR
              </Button>
            </li>
          ))}
        </ul>
      )}
      {qr && <QrSheet open onClose={() => setQr(null)} lot={qr} />}
    </section>
  )
}
