import { CheckCircle, XCircle } from '@phosphor-icons/react/dist/ssr'
import type { components } from '@/lib/api/schema'

export type MatchReason = components['schemas']['MatchReasonResponse']

/**
 * How a lot compares with the signed-in buyer's saved profile, one line per
 * criterion. Met criteria first, so the reason a lot was matched is what a
 * buyer reads before what it misses.
 */
export function MatchReasons({
  reasons,
  heading = 'Why this matches you',
  intro = 'Compared with the buying profile you saved: your location, business type, uses, qualities, price and volume.',
}: {
  reasons: MatchReason[]
  heading?: string
  intro?: string | null
}) {
  const ordered = [...reasons].sort((a, b) => Number(b.met) - Number(a.met))
  const met = reasons.filter((reason) => reason.met).length
  return (
    <section aria-label={heading} className="flex flex-col gap-3">
      <div>
        <h2 className="text-h3 text-ink">{heading}</h2>
        {intro && <p className="text-body-sm mt-1 text-ink-muted">{intro}</p>}
        <p className="text-num-sm mt-1 tabular-nums text-ink-muted">
          {met} of {reasons.length} criteria met
        </p>
      </div>
      <ul className="flex flex-col gap-2">
        {ordered.map((reason) => {
          const Icon = reason.met ? CheckCircle : XCircle
          return (
            <li key={reason.criterion} className="flex items-start gap-2">
              <Icon
                size={20}
                weight={reason.met ? 'fill' : 'regular'}
                className={reason.met ? 'text-verified' : 'text-ink-faint'}
                data-met={reason.met ? 'true' : 'false'}
                aria-hidden
              />
              <div>
                <p className="text-body-sm text-ink">
                  <span className="sr-only">{reason.met ? 'Met: ' : 'Not met: '}</span>
                  {reason.detail}
                </p>
                {reason.value && (
                  <p className="text-num-sm tabular-nums text-ink-muted">{reason.value}</p>
                )}
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
