import Link from 'next/link'
import { notFound } from 'next/navigation'
import { Button } from '@/components/common/button'
import { KnowledgeCardView } from '@/components/fish/knowledge-card'
import { ReviewsRatings } from '@/components/fish/reviews-ratings'
import { SpeciesArt } from '@/components/fish/species-art'
import { SpeciesHeader } from '@/components/fish/species-header'
import { getDiscover, listReviews, type Review } from '@/lib/api/commerce'
import { getMeAsServer } from '@/lib/api/server'

export const dynamic = 'force-dynamic'

/**
 * Where a diner lands after scanning the Fishora QR a restaurant printed.
 *
 * The same fish page a buyer sees, inside the same app shell, minus bidding:
 * the knowledge card, protein, and other buyers' ratings. Bidding needs a buyer
 * account, so a signed-out visitor is offered sign-in instead. The URL is the
 * one printed on the card, so it must keep working for as long as cards exist.
 */
export default async function DiscoverPage({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  let data
  try {
    data = await getDiscover(slug)
  } catch {
    notFound()
  }
  const label = data.species_id.replace('species_', '')

  let reviews: Review[] = []
  try {
    reviews = await listReviews(data.lot_id)
  } catch {
    // The fish page stands without its ratings.
  }

  let signedIn = false
  try {
    await getMeAsServer()
    signedIn = true
  } catch {
    // A diner, most likely.
  }

  return (
    <div data-page="discover" className="mx-auto flex max-w-[720px] flex-col gap-6 pb-8">
      <SpeciesHeader label={label} scientificName={data.card.scientific_name} verified />
      <SpeciesArt label={label} className="aspect-[4/3] w-full rounded-2xl" />
      <KnowledgeCardView card={data.card} label={label} />
      <ReviewsRatings reviews={reviews} />
      <section className="flex flex-col gap-3 rounded-2xl border border-line px-5 py-5">
        <h2 className="text-h3 text-ink">Jelajahi Fishora</h2>
        <p className="text-body-sm text-ink-muted">
          Lihat ikan lain yang sedang dilelang langsung dari tempat pendaratan.
          {!signedIn && ' Untuk mengajukan penawaran, masuk dengan akun pembeli.'}
        </p>
        <div className="flex flex-wrap gap-2">
          <Link href="/marketplace">
            <Button type="button">Lihat semua lot</Button>
          </Link>
          {!signedIn && (
            <Link href="/account?next=/marketplace">
              <Button type="button" variant="secondary">
                Masuk untuk menawar
              </Button>
            </Link>
          )}
        </div>
      </section>
    </div>
  )
}
