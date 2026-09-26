import { LotDetail } from '@/components/lot/lot-detail'
import {
  getDiscover,
  getLot,
  listBids,
  listReviews,
  type Bid,
  type MatchReason,
  type Review,
} from '@/lib/api/commerce'
import { getMeAsServer, getRecommendationsAsServer } from '@/lib/api/server'
import type { KnowledgeCard } from '@/lib/api/fish'

const EMPTY_CARD: KnowledgeCard = {
  common_name: '',
  scientific_name: null,
  taxonomy_status: 'VERIFIED_TAXONOMY',
  physical_characteristics: null,
  taste: null,
  texture: null,
  processing_methods: [],
  commercial_uses: [],
  similar_or_substitute_species: [],
  potential_buyer_segments: [],
  limitations: [],
  sources: [],
}

export default async function LotPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  const lot = await getLot(id)
  let card = EMPTY_CARD
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
    bids = await listBids(id)
  } catch {
    // Same: a lot with an unreadable bid list is still worth reading about.
  }

  let canReview = false
  let viewer: 'buyer' | 'operator' | 'guest' = 'guest'
  // Explainability is per buyer, so it exists only for a signed-in one. An
  // operator or a visitor gets nothing here rather than an empty panel.
  let reasons: MatchReason[] = []
  try {
    const me = await getMeAsServer()
    viewer = me.role === 'buyer' ? 'buyer' : 'operator'
    canReview = me.role === 'buyer' && lot.allocated_buyer_id === me.id
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
      canReview={canReview}
      viewer={viewer}
    />
  )
}
