'use client'

import Link from 'next/link'
import { useState } from 'react'
import { QrCode } from '@phosphor-icons/react/dist/ssr'
import { Button } from '@/components/common/button'
import { LotCard } from '@/components/lot/lot-card'
import { QrSheet } from '@/components/qr/qr-sheet'
import type { Lot } from '@/lib/api/commerce'

/** The lots this buyer won. Each one unlocks a printable Fishora QR card. */
export function WonLots({ lots }: { lots: Lot[] }) {
  const [qr, setQr] = useState<Lot | null>(null)

  return (
    <section className="mt-10" aria-labelledby="won-lots-heading">
      <h2 id="won-lots-heading" className="text-h2 text-ink">
        Lot yang Anda menangkan
      </h2>
      <p className="text-body-sm mt-1 max-w-[56ch] text-ink-muted">
        Setiap lot yang Anda menangkan punya Fishora QR: kartu ikan untuk dicetak dan dipajang bagi pelanggan.
      </p>
      {lots.length === 0 ? (
        <p className="text-body-sm mt-4 text-ink-muted">Belum ada lot yang dialokasikan kepada Anda.</p>
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
