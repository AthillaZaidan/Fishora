import { Star } from '@phosphor-icons/react/dist/ssr'
import { Countdown } from '@/components/lot/countdown'
import { kilograms, rupiahPerKg } from '@/lib/format'
import { resolveSpecies } from '@/lib/species'
import { SpeciesArt } from '@/components/fish/species-art'
import { lotPhotoUrl } from '@/lib/api/commerce'
import type { components } from '@/lib/api/schema'

export type Lot = components['schemas']['LotResponse']

export interface LotCardProps {
  lot: Lot
  /** Overrides the lot's own catch photo. Without either, the species picture is shown. */
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
  // The operator's own catch photo when the lot has one, so every lot shows
  // its fish rather than one picture per species.
  const photo = photoUrl ?? lotPhotoUrl(lot as { photo_url?: string | null })

  return (
    <article className="flex flex-col gap-3 lg:transition-transform lg:hover:-translate-y-[2px]">
      <div className="relative aspect-[4/3] overflow-hidden rounded-[var(--radius-card)] bg-bg-sunken">
        {photo ? (
          // A plain img: the photo is served by the API on another origin, which
          // next/image would refuse without listing that host in next.config.
          // eslint-disable-next-line @next/next/no-img-element
          <img src={photo} alt={names.commonName} loading={priority ? 'eager' : 'lazy'} className="absolute inset-0 size-full object-cover" />
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
