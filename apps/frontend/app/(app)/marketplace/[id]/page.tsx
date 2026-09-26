import { cookies } from 'next/headers'
import { notFound } from 'next/navigation'
import { LotDetail } from '@/components/lot/lot-detail'
import type { ApiFetchOptions } from '@/lib/api/client'
import {
  getDiscover,
  getLot,
  listBids,
  listReviews,
  lotPhotoUrl,
  type Bid,
  type Lot,
  type MatchReason,
  type Review,
} from '@/lib/api/commerce'
import { ApiError } from '@/lib/api/errors'
import { getMeAsServer, getRecommendationsAsServer } from '@/lib/api/server'
import type { KnowledgeCard } from '@/lib/api/fish'

/**
 * A Server Component fetch carries no browser cookies, and what this page shows
 * depends on who is asking: whether the viewer won the lot, and which bids in
 * the history are theirs. So the lot and its bids are read with the session.
 */
async function withSession(): Promise<ApiFetchOptions> {
  const jar = await cookies()
  const cookie = jar
    .getAll()
    .map((entry) => `${entry.name}=${entry.value}`)
    .join('; ')
  return cookie ? { headers: { cookie } } : {}
}

export default async function LotPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  const session = await withSession()

  let lot: Lot
  try {
    lot = await getLot(id, session)
  } catch (cause) {
    // A mistyped or stale link is a 404 page, not the error screen. Anything
    // else (API down) is a real failure and stays one.
    if (cause instanceof ApiError && cause.kind === 'not_found') notFound()
    throw cause
  }

  // No card is shown as missing, never as an empty card marked verified.
  let card: KnowledgeCard | null = null
  try {
    card = (await getDiscover(lot.public_slug)).card
  } catch {
    // Snapshot is optional on a live lot that has not finished generation.
  }

  // Fetch first, render after: JSX built inside a try/catch is not protected.
  let reviews: Review[] = []
  try {
    reviews = await listReviews(id)
  } catch {
    // A missing signal list must not take the lot page down with it.
  }

  let bids: Bid[] = []
  try {
    bids = await listBids(id, session)
  } catch {
    // Same: a lot with an unreadable bid list is still worth reading about.
  }

  let viewer: 'buyer' | 'operator' | 'guest' = 'guest'
  // Explainability is per buyer, so it exists only for a signed-in one. An
  // operator or a visitor gets nothing here rather than an empty panel.
  let reasons: MatchReason[] = []
  try {
    const me = await getMeAsServer()
    viewer = me.role === 'buyer' ? 'buyer' : 'operator'
    if (me.role === 'buyer') {
      const { items } = await getRecommendationsAsServer(me.id)
      reasons = items.find((item) => item.lot.id === lot.id)?.reasons ?? []
    }
  } catch {
    // Anonymous or expired session: the list alone, with no form.
  }

  return (
    <LotDetail
      lot={lot}
      card={card}
      reasons={reasons}
      reviews={reviews}
      bids={bids}
      // The API sets won_by_you only for the buyer the lot went to.
      canReview={viewer === 'buyer' && lot.won_by_you === true}
      viewer={viewer}
      photoUrl={lotPhotoUrl(lot)}
    />
  )
}
