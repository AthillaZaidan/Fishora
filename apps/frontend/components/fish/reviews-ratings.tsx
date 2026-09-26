import { ChatCircle, Star } from '@phosphor-icons/react/dist/ssr'
import { EmptyState } from '@/components/common/empty-state'
import { normaliseDashes } from '@/lib/format'
import type { Review } from '@/lib/api/commerce'

// The unverified surface: what other buyers report, never styled like the
// verified knowledge card. Reviews are keyed to the species, so a rating earned
// on one lot shows on every lot of that fish.
export function ReviewsRatings({ reviews }: { reviews: Review[] }) {
  const average =
    reviews.length === 0
      ? null
      : reviews.reduce((sum, review) => sum + review.processing_suitability, 0) / reviews.length

  return (
    <section className="rounded-2xl bg-bg-sunken px-5 py-5" aria-labelledby="reviews-heading">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="reviews-heading" className="text-h3 text-ink">
          Ulasan & rating
        </h2>
        {average !== null && (
          <p className="text-body-sm flex items-center gap-1.5 text-ink">
            <Stars value={average} />
            <span className="tabular-nums">{average.toLocaleString('id-ID', { maximumFractionDigits: 1 })}</span>
            <span className="text-ink-muted">· {reviews.length} ulasan</span>
          </p>
        )}
      </div>
      <p className="text-body-sm mt-1 text-ink-muted">
        Dari pembeli lain yang sudah memakai ikan ini. Bukan pengetahuan terverifikasi.
      </p>
      {reviews.length === 0 ? (
        <EmptyState icon={ChatCircle} message="Belum ada ulasan untuk ikan ini." />
      ) : (
        <ul className="mt-4 flex flex-col divide-y divide-line">
          {reviews.map((review) => (
            <li key={review.id} className="flex flex-col gap-1 py-3 first:pt-0 last:pb-0">
              <div className="flex items-center justify-between gap-3">
                <p className="text-label text-ink">{normaliseDashes(review.actual_use)}</p>
                <Stars value={review.processing_suitability} />
              </div>
              {review.comment && (
                <p className="text-body-sm text-ink-muted">{normaliseDashes(review.comment)}</p>
              )}
              {review.substitute_acceptance === true && (
                <p className="text-body-sm text-ink-muted">Bisa dipakai sebagai pengganti</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

/** Five stars, filled to the nearest whole star, with the number for screen readers. */
export function Stars({ value }: { value: number }) {
  const filled = Math.round(value)
  return (
    <span className="flex items-center gap-0.5" role="img" aria-label={`Rating ${value.toLocaleString('id-ID', { maximumFractionDigits: 1 })} dari 5`}>
      {[1, 2, 3, 4, 5].map((index) => (
        <Star
          key={index}
          size={14}
          weight={index <= filled ? 'fill' : 'regular'}
          className={index <= filled ? 'text-accent' : 'text-ink-faint'}
          aria-hidden
        />
      ))}
    </span>
  )
}
