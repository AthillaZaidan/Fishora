import { dateTime, rupiahPerKg } from '@/lib/format'
import type { Bid } from '@/lib/api/commerce'

/**
 * Bids newest first, each under the label the API gives its buyer: "Bidder 1"
 * for whoever bid first, "You" for the viewer. Buyer ids are not public; the
 * lot's operator also gets the id, shown after the label.
 * Heading-free, so the caller can title it or let a Sheet header do that.
 */
export function BidHistory({ bids }: { bids: Bid[] }) {
  const ordered = [...bids].sort((a, b) => b.created_at.localeCompare(a.created_at))

  return (
    <>
      {ordered.length === 0 ? (
        <p className="text-body-sm text-ink-muted">No bids on this lot yet.</p>
      ) : (
        <ol className="flex flex-col divide-y divide-line">
          {ordered.map((bid, index) => (
            <li key={bid.id} className="flex items-baseline justify-between gap-3 py-2">
              <div className="min-w-0">
                <p className="text-num-sm tabular-nums text-ink">
                  {rupiahPerKg(Number(bid.amount_per_kg))}
                  {index === 0 && <span className="text-body-sm ml-2 text-ink-muted">Highest</span>}
                </p>
                <p className={['truncate text-body-sm', bid.is_you ? 'text-ink' : 'text-ink-muted'].join(' ')}>
                  {bid.bidder}
                  {bid.buyer_id && !bid.is_you && <span className="text-num-sm"> · {bid.buyer_id}</span>}
                </p>
              </div>
              <p className="shrink-0 text-num-sm tabular-nums text-ink-muted">
                {dateTime(bid.created_at)}
              </p>
            </li>
          ))}
        </ol>
      )}
    </>
  )
}
