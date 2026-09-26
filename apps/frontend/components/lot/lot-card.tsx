import Image from 'next/image'
import { Star } from '@phosphor-icons/react/dist/ssr'
import { Countdown } from '@/components/lot/countdown'
import { kilograms, rupiahPerKg } from '@/lib/format'
import { resolveSpecies } from '@/lib/species'
import { SpeciesArt } from '@/components/fish/species-art'
import type { components } from '@/lib/api/schema'

export type Lot = components['schemas']['LotResponse']

export interface LotCardProps {
  lot: Lot
  /** A real photograph when one exists. Falls back to the species composition. */
  photoUrl?: string
  /** Set on the first card in a grid: it is the LCP element. */
  priority?: boolean
  /** The lot fits the signed-in buyer's preferences. */
  matched?: boolean
}

export function LotCard({ lot, photoUrl, matched = false, priority = false }: LotCardProps) {
  const label = lot.species_id.replace('species_', '')
  const names = resolveSpecies(label)
  const live = lot.status === 'active'

  return (
    <article className="flex flex-col gap-3 lg:transition-transform lg:hover:-translate-y-[2px]">
      <div className="relative aspect-[4/3] overflow-hidden rounded-[var(--radius-card)] bg-bg-sunken">
        {photoUrl ? (
          <Image src={photoUrl} alt={names.commonName} fill className="object-cover" sizes="(max-width: 640px) 100vw, 33vw" />
        ) : (
          <SpeciesArt label={label} className="absolute inset-0" priority={priority} />
        )}
        {matched && (
          <p className="text-body-sm absolute top-3 left-3 flex items-center gap-1 rounded-full bg-surface/90 px-2.5 py-1 text-ink backdrop-blur-sm">
            <Star size={14} weight="fill" className="text-accent" aria-hidden />
            Matched for you
          </p>
        )}
      </div>
      <div className="flex flex-col gap-1 px-1">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-h3 text-ink">{names.commonName}</h2>
          {live ? (
            <p className="text-body-sm flex items-center gap-1.5 text-ink">
              <span className="size-1.5 rounded-full bg-accent" aria-hidden />
              Live
            </p>
          ) : (
            <p className="text-body-sm text-ink-muted">Ended</p>
          )}
        </div>
        <p className="text-num-sm tabular-nums text-ink-muted">
          {kilograms(Number(lot.quantity_kg))}
          {lot.batch_size > 1 && ` · Lot ${lot.batch_index} of ${lot.batch_size}`}
        </p>
        <p className="text-num-sm tabular-nums text-ink">
          {rupiahPerKg(Number(lot.current_highest_per_kg ?? lot.starting_price_per_kg))}
        </p>
        {live && <Countdown endsAt={lot.auction_ends_at} />}
      </div>
    </article>
  )
}
